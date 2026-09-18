/// Phase 6 / v0.2-v0.3: Prediction Edges — self-supervised next-unit prediction.
///
/// v0.3: Contextual Avoidance — negative feedback accumulates in `avoidance` (§19),
/// keeping positive and negative signals separate.
/// Route Score = Likelihood + Positive Value − Avoidance (§20).
use std::collections::HashMap;
use crate::units::UnitId;
use crate::chunks::STRENGTH_DECAY;

pub const PREDICTION_REWARD: f64 = 1.0;

/// A single directed prediction edge: context → next_unit.
/// v0.3 fields: usage_strength + feedback_value (positive) + avoidance (negative)
#[derive(Debug, Clone)]
pub struct PredictionEdge {
    pub context: UnitId,
    pub next_unit: UnitId,
    /// How many times this edge was observed during exposure.
    pub use_count: u32,
    /// Decaying sum of usage events.
    pub usage_strength: f64,
    /// Accumulated POSITIVE feedback signal only: V_new = decay * V_old + r (r > 0)
    pub feedback_value: f64,
    /// Number of feedback events applied to this edge.
    pub feedback_count: u32,
    /// Accumulated NEGATIVE feedback magnitude (§19): A_new = decay * A_old + |r| (r < 0)
    pub avoidance: f64,
}

impl PredictionEdge {
    pub fn new(context: UnitId, next_unit: UnitId) -> Self {
        Self {
            context,
            next_unit,
            use_count: 0,
            usage_strength: 0.0,
            feedback_value: 0.0,
            feedback_count: 0,
            avoidance: 0.0,
        }
    }

    /// Record one observation (positive exposure).
    pub fn record_usage(&mut self) {
        self.use_count += 1;
        self.usage_strength = STRENGTH_DECAY * self.usage_strength + PREDICTION_REWARD;
    }

    /// Apply one feedback credit r to this edge (v0.3: splits on sign).
    /// Positive r → feedback_value; negative r → avoidance (|r|).
    pub fn apply_feedback(&mut self, r: f64, decay: f64) {
        if r >= 0.0 {
            self.feedback_value = decay * self.feedback_value + r;
        } else {
            self.avoidance = decay * self.avoidance + (-r);
        }
        self.feedback_count += 1;
    }

    /// Binary confidence: 1.0 if this edge has been observed at all.
    pub fn confidence(&self) -> f64 {
        if self.use_count > 0 { 1.0 } else { 0.0 }
    }

    /// Route score (v0.3): Likelihood + Positive Value − Avoidance (§20).
    pub fn score(&self) -> f64 {
        let strength_norm = (self.usage_strength / 100.0).min(1.0);
        let pos_value = (self.feedback_value / 10.0).min(1.0);
        let avoid_norm = (self.avoidance / 10.0).min(1.0);
        self.confidence() + strength_norm + pos_value - avoid_norm
    }
}

/// Stores all prediction edges indexed by (context, next_unit).
///
/// Phase 5: secondary source_index maps context → Vec<next_unit> so that
/// `predict()` is O(K) fan-out instead of O(N_total) full-scan.
#[derive(Debug, Default)]
pub struct PredictionStore {
    /// (context, next_unit) → edge  (O(1) point lookup)
    edges: HashMap<(UnitId, UnitId), PredictionEdge>,
    /// context → list of known successors  (Phase 5 source index)
    source_index: HashMap<UnitId, Vec<UnitId>>,
}

impl PredictionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `next_unit` followed `context` (positive exposure).
    pub fn observe(&mut self, context: UnitId, next_unit: UnitId) {
        let edge = self
            .edges
            .entry((context, next_unit))
            .or_insert_with(|| PredictionEdge::new(context, next_unit));
        edge.record_usage();
        // Maintain source index — only add if this is a new (context, next_unit) pair.
        let successors = self.source_index.entry(context).or_default();
        if !successors.contains(&next_unit) {
            successors.push(next_unit);
        }
    }

    /// Learn from a sequence of units: every adjacent pair (units[i], units[i+1])
    /// is recorded as a positive observation (self-supervised).
    pub fn learn_sequence(&mut self, units: &[UnitId]) {
        for window in units.windows(2) {
            self.observe(window[0], window[1]);
        }
    }

    /// Return candidates for the next unit given `context`, ranked by score.
    ///
    /// Phase 5: O(K) via source_index instead of O(N_total) full scan.
    pub fn predict(&self, context: UnitId) -> Vec<(&PredictionEdge, f64)> {
        let Some(successors) = self.source_index.get(&context) else {
            return Vec::new();
        };
        let mut candidates: Vec<(&PredictionEdge, f64)> = successors
            .iter()
            .filter_map(|&nxt| self.edges.get(&(context, nxt)))
            .map(|edge| (edge, edge.score()))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Top-1 prediction for `context`.  Returns None if no edges exist.
    pub fn top1(&self, context: UnitId) -> Option<UnitId> {
        self.predict(context).into_iter().next().map(|(e, _)| e.next_unit)
    }

    /// Top-1 prediction with its score.
    pub fn top1_with_score(&self, context: UnitId) -> Option<(UnitId, f64)> {
        self.predict(context).into_iter().next().map(|(e, s)| (e.next_unit, s))
    }

    /// Apply feedback credit r to the edge (context, next_unit) if it exists.
    pub fn apply_feedback_to_edge(
        &mut self,
        context: UnitId,
        next_unit: UnitId,
        r: f64,
        decay: f64,
    ) {
        if let Some(edge) = self.edges.get_mut(&(context, next_unit)) {
            edge.apply_feedback(r, decay);
        }
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
        let is_new = !self.edges.contains_key(&(context, next_unit));
        let edge = self.edges
            .entry((context, next_unit))
            .or_insert_with(|| PredictionEdge::new(context, next_unit));
        if is_new {
            let successors = self.source_index.entry(context).or_default();
            if !successors.contains(&next_unit) {
                successors.push(next_unit);
            }
        }
        edge
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
        store.observe(p(1), p(2));
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_predict_returns_candidate() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2));
        let preds = store.predict(p(1));
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0.next_unit, p(2));
    }

    #[test]
    fn test_top1() {
        let mut store = PredictionStore::new();
        for _ in 0..10 { store.observe(p(1), p(2)); }
        for _ in 0..3  { store.observe(p(1), p(3)); }
        assert_eq!(store.top1(p(1)), Some(p(2)));
    }

    #[test]
    fn test_top1_with_score() {
        let mut store = PredictionStore::new();
        for _ in 0..10 { store.observe(p(1), p(2)); }
        let (unit, score) = store.top1_with_score(p(1)).unwrap();
        assert_eq!(unit, p(2));
        assert!(score > 0.0);
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
        assert!(store.edge_count() >= 3);
        assert_eq!(store.top1(p(2)), Some(p(3)));
    }

    #[test]
    fn test_strength_increases_with_repetition() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2));
        let s1 = store.edges[&(p(1), p(2))].usage_strength;
        store.observe(p(1), p(2));
        let s2 = store.edges[&(p(1), p(2))].usage_strength;
        assert!(s2 > s1);
    }

    #[test]
    fn test_confidence_binary() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2));
        let edge = &store.edges[&(p(1), p(2))];
        assert_eq!(edge.confidence(), 1.0);
        assert_eq!(edge.use_count, 1);
    }

    #[test]
    fn test_ranked_by_strength() {
        let mut store = PredictionStore::new();
        for _ in 0..20 { store.observe(p(1), p(3)); }
        for _ in 0..5  { store.observe(p(1), p(4)); }
        let ranked = store.predict(p(1));
        assert_eq!(ranked[0].0.next_unit, p(3));
    }

    #[test]
    fn test_apply_feedback_to_edge() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2));
        store.apply_feedback_to_edge(p(1), p(2), 1.0, 0.99);
        let edge = &store.edges[&(p(1), p(2))];
        assert!((edge.feedback_value - 1.0).abs() < 1e-9);
        assert_eq!(edge.feedback_count, 1);
    }

    #[test]
    fn test_feedback_affects_score() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2));
        store.observe(p(1), p(3));
        // Penalize p(2)
        for _ in 0..10 { store.apply_feedback_to_edge(p(1), p(2), -1.0, 0.99); }
        let ranked = store.predict(p(1));
        assert_eq!(ranked[0].0.next_unit, p(3), "p(3) should rank higher after p(2) is penalized");
    }
}
