/// Prediction Edges — self-supervised next-unit prediction.
///
/// v0.3: Contextual Avoidance — negative feedback accumulates in `avoidance`,
/// keeping positive and negative signals separate.
/// Route Score = Likelihood + Positive Value − Avoidance.
///
/// Phase 5: secondary source_index for O(K) fan-out.
/// Phase 9: last_used_tick + lazy decay.
/// Phase 15: flat arena storage — edges: Vec<PredictionEdge> + edge_index for
///           O(1) point lookup, source_index stores Vec<usize> (arena indices)
///           instead of Vec<UnitId> for direct array access in predict().
use std::collections::HashMap;
use crate::units::UnitId;
use crate::chunks::STRENGTH_DECAY;

pub const PREDICTION_REWARD: f64 = 1.0;

/// A single directed prediction edge: context → next_unit.
///
/// Phase 9: last_used_tick enables lazy decay.
/// Phase C (spec §15): external_route_evidence and usage_strength (practice_confidence) are separate.
///   - external_route_evidence: updated ONLY by Experience (expose_external).
///     Represents "how often this transition was observed in the real world."
///   - usage_strength (practice_confidence): updated by both Experience AND Replay.
///     Represents "how well-practiced this route is."
#[derive(Debug, Clone)]
pub struct PredictionEdge {
    pub context: UnitId,
    pub next_unit: UnitId,
    pub use_count: u32,
    /// Practice confidence — updated by both Experience and Replay.
    pub usage_strength: f64,
    /// External Route Evidence — updated ONLY by Experience (expose_external).
    /// Never modified by Replay. Represents real-world observation frequency.
    pub external_route_evidence: f64,
    pub last_used_tick: u64,
    pub feedback_value: f64,
    pub feedback_count: u32,
    pub avoidance: f64,
}

impl PredictionEdge {
    pub fn new(context: UnitId, next_unit: UnitId) -> Self {
        Self {
            context,
            next_unit,
            use_count: 0,
            usage_strength: 0.0,
            external_route_evidence: 0.0,
            last_used_tick: 0,
            feedback_value: 0.0,
            feedback_count: 0,
            avoidance: 0.0,
        }
    }

    /// Phase 9 Lazy Decay: s_now = s_stored · λ^Δt + reward
    pub fn record_usage_at(&mut self, tick: u64) {
        let elapsed = tick.saturating_sub(self.last_used_tick);
        let decayed = if elapsed > 0 {
            self.usage_strength * STRENGTH_DECAY.powi(elapsed.min(u32::MAX as u64) as i32)
        } else {
            self.usage_strength
        };
        self.usage_strength = decayed + PREDICTION_REWARD;
        self.use_count += 1;
        self.last_used_tick = tick;
    }

    pub fn lazy_strength(&self, current_tick: u64) -> f64 {
        let elapsed = current_tick.saturating_sub(self.last_used_tick);
        if elapsed == 0 {
            self.usage_strength
        } else {
            self.usage_strength * STRENGTH_DECAY.powi(elapsed.min(u32::MAX as u64) as i32)
        }
    }

    #[inline]
    pub fn record_usage(&mut self) {
        self.usage_strength = STRENGTH_DECAY * self.usage_strength + PREDICTION_REWARD;
        self.use_count += 1;
        self.last_used_tick += 1;
    }

    pub fn apply_feedback(&mut self, r: f64, decay: f64) {
        if r >= 0.0 {
            self.feedback_value = decay * self.feedback_value + r;
        } else {
            self.avoidance = decay * self.avoidance + (-r);
        }
        self.feedback_count += 1;
    }

    pub fn confidence(&self) -> f64 {
        if self.use_count > 0 { 1.0 } else { 0.0 }
    }

    pub fn score(&self) -> f64 {
        let strength_norm = (self.usage_strength / 100.0).min(1.0);
        let pos_value = (self.feedback_value / 10.0).min(1.0);
        let avoid_norm = (self.avoidance / 10.0).min(1.0);
        self.confidence() + strength_norm + pos_value - avoid_norm
    }
}

/// Stores all prediction edges.
///
/// Phase 15 (Flat Arena): edges are stored in a contiguous Vec<PredictionEdge>.
/// `edge_index` maps (context, next_unit) → Vec index for O(1) point lookup.
/// `source_index` maps context → Vec<usize> (arena indices) so that `predict()`
/// walks the flat edge array directly without secondary HashMap lookups.
#[derive(Debug, Default)]
pub struct PredictionStore {
    /// Flat arena — all edges contiguous for cache locality.
    edges: Vec<PredictionEdge>,
    /// (context, next_unit) → index into `edges`.
    edge_index: HashMap<(UnitId, UnitId), usize>,
    /// context → list of arena indices of known successor edges (Phase 5/15).
    source_index: HashMap<UnitId, Vec<usize>>,
}

impl PredictionStore {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Insert a new edge and update both indices.  Caller must verify the
    /// edge does not already exist.
    fn insert_new(&mut self, context: UnitId, next_unit: UnitId) -> usize {
        let idx = self.edges.len();
        self.edges.push(PredictionEdge::new(context, next_unit));
        self.edge_index.insert((context, next_unit), idx);
        self.source_index.entry(context).or_default().push(idx);
        idx
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Record an edge usage from Replay (practice only): updates usage_strength.
    /// Does NOT update external_route_evidence.
    pub fn observe_at(&mut self, context: UnitId, next_unit: UnitId, tick: u64) {
        let idx = if let Some(&i) = self.edge_index.get(&(context, next_unit)) {
            i
        } else {
            self.insert_new(context, next_unit)
        };
        self.edges[idx].record_usage_at(tick);
    }

    pub fn observe(&mut self, context: UnitId, next_unit: UnitId) {
        self.observe_at(context, next_unit, 0);
    }

    /// Record an edge from an external Experience: updates BOTH usage_strength
    /// AND external_route_evidence. Only call from expose_external paths.
    pub fn observe_external_at(&mut self, context: UnitId, next_unit: UnitId, tick: u64) {
        let idx = if let Some(&i) = self.edge_index.get(&(context, next_unit)) {
            i
        } else {
            self.insert_new(context, next_unit)
        };
        self.edges[idx].record_usage_at(tick);
        self.edges[idx].external_route_evidence += 1.0;
    }

    /// Update practice_confidence only (Replay path). Does not touch external_route_evidence.
    pub fn learn_sequence_at(&mut self, units: &[UnitId], tick: u64) {
        for window in units.windows(2) {
            self.observe_at(window[0], window[1], tick);
        }
    }

    pub fn learn_sequence(&mut self, units: &[UnitId]) {
        self.learn_sequence_at(units, 0);
    }

    /// Update both practice_confidence AND external_route_evidence (Experience path).
    pub fn learn_sequence_external_at(&mut self, units: &[UnitId], tick: u64) {
        for window in units.windows(2) {
            self.observe_external_at(window[0], window[1], tick);
        }
    }

    /// Return candidates for the next unit given `context`, ranked by score.
    ///
    /// Phase 5/15: O(K) via source_index; each entry is a direct arena index.
    pub fn predict(&self, context: UnitId) -> Vec<(&PredictionEdge, f64)> {
        let Some(indices) = self.source_index.get(&context) else {
            return Vec::new();
        };
        let mut candidates: Vec<(&PredictionEdge, f64)> = indices
            .iter()
            .map(|&i| &self.edges[i])
            .map(|edge| (edge, edge.score()))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    pub fn top1(&self, context: UnitId) -> Option<UnitId> {
        self.predict(context).into_iter().next().map(|(e, _)| e.next_unit)
    }

    pub fn top1_with_score(&self, context: UnitId) -> Option<(UnitId, f64)> {
        self.predict(context).into_iter().next().map(|(e, s)| (e.next_unit, s))
    }

    /// Return up to `k` best candidates for `context`, ranked by score.
    pub fn top_k_with_score(&self, context: UnitId, k: usize) -> Vec<(UnitId, f64)> {
        let mut results = self.predict(context);
        results.truncate(k);
        results.into_iter().map(|(e, s)| (e.next_unit, s)).collect()
    }

    pub fn apply_feedback_to_edge(
        &mut self,
        context: UnitId,
        next_unit: UnitId,
        r: f64,
        decay: f64,
    ) {
        if let Some(&idx) = self.edge_index.get(&(context, next_unit)) {
            self.edges[idx].apply_feedback(r, decay);
        }
    }

    /// Return the lazily-decayed strength of a specific edge at `tick`.
    pub fn edge_lazy_strength(&self, context: UnitId, next_unit: UnitId, tick: u64) -> Option<f64> {
        self.edge_index.get(&(context, next_unit))
            .map(|&i| self.edges[i].lazy_strength(tick))
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Iterate all edges for serialisation.
    pub fn iter_all(&self) -> impl Iterator<Item = &PredictionEdge> {
        self.edges.iter()
    }

    /// Get or create an edge for mutation during deserialisation.
    pub fn get_or_create(&mut self, context: UnitId, next_unit: UnitId) -> &mut PredictionEdge {
        let idx = if let Some(&i) = self.edge_index.get(&(context, next_unit)) {
            i
        } else {
            self.insert_new(context, next_unit)
        };
        &mut self.edges[idx]
    }

    /// Read-only point lookup — used by unit tests.
    pub(crate) fn edge_for(&self, context: UnitId, next_unit: UnitId) -> Option<&PredictionEdge> {
        self.edge_index.get(&(context, next_unit)).map(|&i| &self.edges[i])
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
        let s1 = store.edge_for(p(1), p(2)).unwrap().usage_strength;
        store.observe(p(1), p(2));
        let s2 = store.edge_for(p(1), p(2)).unwrap().usage_strength;
        assert!(s2 > s1);
    }

    #[test]
    fn test_confidence_binary() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2));
        let edge = store.edge_for(p(1), p(2)).unwrap();
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
        let edge = store.edge_for(p(1), p(2)).unwrap();
        assert!((edge.feedback_value - 1.0).abs() < 1e-9);
        assert_eq!(edge.feedback_count, 1);
    }

    #[test]
    fn test_feedback_affects_score() {
        let mut store = PredictionStore::new();
        store.observe(p(1), p(2));
        store.observe(p(1), p(3));
        for _ in 0..10 { store.apply_feedback_to_edge(p(1), p(2), -1.0, 0.99); }
        let ranked = store.predict(p(1));
        assert_eq!(ranked[0].0.next_unit, p(3), "p(3) should rank higher after p(2) is penalized");
    }
}
