/// Phase 6: Prediction Edges — self-supervised next-unit prediction (§C §8-9).
///
/// A prediction edge represents: "given context unit C, the next unit is N".
/// Edges accumulate evidence over training and are used during generation.
use std::collections::HashMap;
use crate::units::UnitId;
use crate::chunks::STRENGTH_DECAY;

pub const PREDICTION_REWARD: f64 = 1.0;

/// A single directed prediction edge: context → next_unit.
#[derive(Debug, Clone)]
pub struct PredictionEdge {
    pub context: UnitId,
    pub next_unit: UnitId,
    pub success_count: u32,
    pub total_count: u32,
    pub strength: f64,
}

impl PredictionEdge {
    pub fn new(context: UnitId, next_unit: UnitId) -> Self {
        Self {
            context,
            next_unit,
            success_count: 0,
            total_count: 0,
            strength: 0.0,
        }
    }

    pub fn record_success(&mut self) {
        self.success_count += 1;
        self.total_count += 1;
        self.strength = STRENGTH_DECAY * self.strength + PREDICTION_REWARD;
    }

    pub fn record_failure(&mut self) {
        self.total_count += 1;
        self.strength = STRENGTH_DECAY * self.strength;
    }

    pub fn confidence(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.total_count as f64
        }
    }

    /// Prediction score used for ranking (§C §7).
    pub fn score(&self) -> f64 {
        self.confidence() + self.strength.min(100.0) / 100.0
    }
}

/// Stores all prediction edges indexed by (context, next_unit).
#[derive(Debug, Default)]
pub struct PredictionStore {
    /// (context, next_unit) → edge
    edges: HashMap<(UnitId, UnitId), PredictionEdge>,
}

impl PredictionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `next_unit` followed `context` (success = correctly predicted).
    pub fn observe(&mut self, context: UnitId, next_unit: UnitId, predicted: bool) {
        let edge = self
            .edges
            .entry((context, next_unit))
            .or_insert_with(|| PredictionEdge::new(context, next_unit));
        if predicted {
            edge.record_success();
        } else {
            edge.record_failure();
        }
    }

    /// Learn from a sequence of units: every adjacent pair (units[i], units[i+1])
    /// is recorded as a successful observation (self-supervised).
    pub fn learn_sequence(&mut self, units: &[UnitId]) {
        for window in units.windows(2) {
            self.observe(window[0], window[1], true);
        }
    }

    /// Return candidates for the next unit given `context`, ranked by score.
    pub fn predict(&self, context: UnitId) -> Vec<(&PredictionEdge, f64)> {
        let mut candidates: Vec<(&PredictionEdge, f64)> = self
            .edges
            .iter()
            .filter(|((ctx, _), _)| *ctx == context)
            .map(|(_, edge)| (edge, edge.score()))
            .collect();
        // Highest score first
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Top-1 prediction for `context`.  Returns None if no edges exist.
    pub fn top1(&self, context: UnitId) -> Option<UnitId> {
        self.predict(context).into_iter().next().map(|(e, _)| e.next_unit)
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Iterate all edges for serialisation.
    pub fn iter_all(&self) -> impl Iterator<Item = &PredictionEdge> {
        self.edges.values()
    }

    /// Get or create an edge for mutation during deserialisation.
    pub fn get_or_create(&mut self, context: UnitId, next_unit: UnitId) -> &mut PredictionEdge {
        self.edges
            .entry((context, next_unit))
            .or_insert_with(|| PredictionEdge::new(context, next_unit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::UnitId;

    fn p(id: u32) -> UnitId { UnitId::primitive(id) }

    #[test]
    fn test_observe_creates_edge() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2), true);
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_predict_returns_candidate() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2), true);
        let preds = store.predict(p(1));
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0.next_unit, p(2));
    }

    #[test]
    fn test_top1() {
        let mut store = PredictionStore::new();
        // p(2) is seen more often after p(1)
        for _ in 0..10 { store.observe(p(1), p(2), true); }
        for _ in 0..3  { store.observe(p(1), p(3), true); }
        assert_eq!(store.top1(p(1)), Some(p(2)));
    }

    #[test]
    fn test_no_prediction_for_unseen_context() {
        let store = PredictionStore::new();
        assert!(store.top1(p(99)).is_none());
    }

    #[test]
    fn test_learn_sequence() {
        let mut store = PredictionStore::new();
        let seq = vec![p(1), p(2), p(3), p(2), p(3)];
        store.learn_sequence(&seq);
        // Should have edges: 1→2, 2→3, 3→2, 2→3 (deduplicated into one edge)
        assert!(store.edge_count() >= 3);
        assert_eq!(store.top1(p(2)), Some(p(3)));
    }

    #[test]
    fn test_strength_increases_with_repetition() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2), true);
        let s1 = store.edges[&(p(1), p(2))].strength;
        store.observe(p(1), p(2), true);
        let s2 = store.edges[&(p(1), p(2))].strength;
        assert!(s2 > s1);
    }

    #[test]
    fn test_confidence_tracks_accuracy() {
        let mut store = PredictionStore::new();
        for _ in 0..8 { store.observe(p(1), p(2), true); }
        for _ in 0..2 { store.observe(p(1), p(2), false); }
        let edge = &store.edges[&(p(1), p(2))];
        assert!((edge.confidence() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_ranked_by_score() {
        let mut store = PredictionStore::new();
        // p(3) has higher confidence after p(1) than p(4)
        for _ in 0..20 { store.observe(p(1), p(3), true); }
        for _ in 0..5  { store.observe(p(1), p(4), true); }
        for _ in 0..5  { store.observe(p(1), p(4), false); }
        let ranked = store.predict(p(1));
        assert_eq!(ranked[0].0.next_unit, p(3));
    }
}
