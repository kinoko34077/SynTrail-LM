/// Phase 6 / v0.2: Prediction Edges — self-supervised next-unit prediction.
///
/// v0.2 separates usage (exposure) from evaluation (feedback).
/// All learning calls are now positive exposures (no failure path).
use std::collections::HashMap;
use crate::units::UnitId;
use crate::chunks::STRENGTH_DECAY;

pub const PREDICTION_REWARD: f64 = 1.0;

/// A single directed prediction edge: context → next_unit.
/// v0.2 fields: usage_strength (from exposure) + feedback_value (from ±1 feedback)
#[derive(Debug, Clone)]
pub struct PredictionEdge {
    pub context: UnitId,
    pub next_unit: UnitId,
    /// How many times this edge was observed during exposure.
    pub use_count: u32,
    /// Decaying sum of usage events.
    pub usage_strength: f64,
    /// Accumulated feedback signal: V_new = decay * V_old + r
    pub feedback_value: f64,
    /// Number of feedback events applied to this edge.
    pub feedback_count: u32,
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
        }
    }

    /// Record one observation (positive exposure).
    pub fn record_usage(&mut self) {
        self.use_count += 1;
        self.usage_strength = STRENGTH_DECAY * self.usage_strength + PREDICTION_REWARD;
    }

    /// Apply one feedback credit r to this edge.
    /// V_new = decay * V_old + r
    pub fn apply_feedback(&mut self, r: f64, decay: f64) {
        self.feedback_value = decay * self.feedback_value + r;
        self.feedback_count += 1;
    }

    /// Binary confidence: 1.0 if this edge has been observed at all.
    pub fn confidence(&self) -> f64 {
        if self.use_count > 0 { 1.0 } else { 0.0 }
    }

    /// Prediction score used for ranking.
    /// Higher usage_strength and positive feedback_value raise the score.
    pub fn score(&self) -> f64 {
        let strength_norm = (self.usage_strength / 100.0).min(1.0);
        let feedback_term = (self.feedback_value / 10.0).clamp(-1.0, 1.0);
        self.confidence() + strength_norm + feedback_term
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

    /// Record that `next_unit` followed `context` (positive exposure).
    pub fn observe(&mut self, context: UnitId, next_unit: UnitId) {
        let edge = self
            .edges
            .entry((context, next_unit))
            .or_insert_with(|| PredictionEdge::new(context, next_unit));
        edge.record_usage();
    }

    /// Learn from a sequence of units: every adjacent pair (units[i], units[i+1])
    /// is recorded as a positive observation (self-supervised).
    pub fn learn_sequence(&mut self, units: &[UnitId]) {
        for window in units.windows(2) {
            self.observe(window[0], window[1]);
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
