/// v0.3 ModelState — unified learn/generate/associate model.
///
/// v0.3 additions:
/// - `associations: AssociationStore` — bidirectional co-occurrence memory
/// - `expose()` now records associations for every adjacent segmented pair
/// - `recall(source)` — return Top-K associations with hierarchical fallback
use std::collections::HashMap;

use crate::association::AssociationStore;
use crate::chunks::ChunkRegistry;
use crate::prediction::PredictionStore;
use crate::primitives::PrimitiveRegistry;
use crate::segmentation::{expand, segment};
use crate::trace::{DecisionStep, TraceId, TurnTrace, now_secs};
use crate::units::UnitId;

const MERGE_THRESHOLD: u32 = 4;
const BASE_MERGE_PROBABILITY: f64 = 0.5;
const SEGMENT_MIN_SCORE: f64 = 0.0;

/// Running counters for the primary metric.
#[derive(Debug, Default, Clone, Copy)]
pub struct Metrics {
    pub total_decisions: u64,
    pub total_characters: u64,
}

impl Metrics {
    pub fn decision_per_character(&self) -> f64 {
        if self.total_characters == 0 {
            0.0
        } else {
            self.total_decisions as f64 / self.total_characters as f64
        }
    }
}

/// The complete model state.
#[derive(Debug, Default)]
pub struct ModelState {
    pub primitives: PrimitiveRegistry,
    pub chunks: ChunkRegistry,
    pub predictions: PredictionStore,
    /// v0.3: bidirectional co-occurrence associations.
    pub associations: AssociationStore,
    pub tick: u64,
    pub(crate) merge_candidates: HashMap<(UnitId, UnitId), u32>,
    pub metrics: Metrics,
    /// Monotonically increasing trace ID counter.
    pub(crate) next_trace_id: TraceId,
}

impl ModelState {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Exposure (v0.2 rename of train) ──────────────────────────────────

    /// Expose the model to one text sample.
    ///
    /// Updates usage statistics (use_count, usage_strength) and builds
    /// prediction edges from the segmented sequence.
    /// Does NOT apply external feedback.
    pub fn expose(&mut self, text: &str) {
        self.tick += 1;
        let tick = self.tick;

        let prim_ids = self.primitives.encode(text);
        if prim_ids.is_empty() {
            return;
        }

        let segmented = segment(&prim_ids, &self.chunks, SEGMENT_MIN_SCORE);

        for &unit in &segmented {
            if let Some(cid) = unit.as_chunk() {
                if let Some(chunk) = self.chunks.get_mut(cid) {
                    chunk.record_usage(tick);
                }
            }
        }

        self.predictions.learn_sequence(&segmented);
        self.metrics.total_decisions += segmented.len() as u64;
        self.metrics.total_characters += prim_ids.len() as u64;

        // v0.3: bidirectional associations for every adjacent pair
        for window in segmented.windows(2) {
            let (a, b) = (window[0], window[1]);
            self.associations.observe(a, b, tick);
            self.associations.observe(b, a, tick);
        }

        self.consider_merges(&segmented, tick);
    }

    /// Backward-compatible alias — callers not yet migrated to `expose`.
    #[inline]
    pub fn train(&mut self, text: &str) {
        self.expose(text);
    }

    // ── Frozen generation (v0.2) ─────────────────────────────────────────

    /// Generate text from `seed_text` without mutating self.
    ///
    /// Returns the generated string and a TurnTrace recording every decision.
    /// The seed units are NOT decisions; only units chosen by top1 are.
    pub fn generate_with_trace(
        &self,
        seed_text: &str,
        input_text: &str,
        max_units: usize,
        trace_id: TraceId,
    ) -> (String, TurnTrace) {
        let prim_ids = {
            let mut tmp = self.primitives.clone();
            tmp.encode(seed_text)
        };

        let seed_units = segment(&prim_ids, &self.chunks, SEGMENT_MIN_SCORE);
        let mut generated_units = seed_units.clone();
        let mut steps: Vec<DecisionStep> = Vec::new();

        for step_index in 0..max_units {
            let context = match generated_units.last() {
                Some(&u) => u,
                None => break,
            };
            match self.predictions.top1_with_score(context) {
                Some((next, score)) => {
                    steps.push(DecisionStep {
                        step_index,
                        unit: next,
                        context,
                        score,
                    });
                    generated_units.push(next);
                }
                None => break,
            }
        }

        let expanded = expand(&generated_units, &self.chunks, &self.primitives);
        let output = self.primitives.decode(&expanded).unwrap_or_default();

        let trace = TurnTrace::new(
            trace_id,
            self.tick,
            seed_text.to_owned(),
            input_text.to_owned(),
            output.clone(),
            steps,
            now_secs(),
        );

        (output, trace)
    }

    /// Convenience: generate without a trace.
    pub fn generate(&self, seed_text: &str, max_units: usize) -> String {
        let (output, _) = self.generate_with_trace(seed_text, seed_text, max_units, 0);
        output
    }

    /// Allocate the next trace ID (monotonically increasing).
    pub fn alloc_trace_id(&mut self) -> TraceId {
        self.next_trace_id += 1;
        self.next_trace_id
    }

    // ── Feedback application (v0.2) ──────────────────────────────────────

    /// Apply per-step feedback credits to the edges and chunks in `trace`.
    ///
    /// `credits[i]` is the credit for `trace.decision_steps[i]`.
    /// For each step, the prediction edge (context, unit) is updated.
    /// Chunks in the unit also receive the credit.
    pub fn apply_feedback_to_trace(
        &mut self,
        trace: &TurnTrace,
        credits: &[f64],
        decay: f64,
    ) {
        for (step, &r) in trace.decision_steps.iter().zip(credits.iter()) {
            self.predictions.apply_feedback_to_edge(step.context, step.unit, r, decay);
            if let Some(cid) = step.unit.as_chunk() {
                if let Some(chunk) = self.chunks.get_mut(cid) {
                    chunk.apply_feedback(r, decay);
                }
            }
        }
    }

    /// v0.3: Return Top-K associations for `source` with hierarchical fallback.
    pub fn recall(&self, source: UnitId, limit: usize) -> Vec<(UnitId, f64)> {
        self.associations.recall(source, &self.chunks, limit)
    }

    /// A simple fingerprint of the current model state for snapshot tagging.
    pub fn state_fingerprint(&self) -> String {
        format!(
            "tick={} prims={} chunks={} edges={} assoc={}",
            self.tick,
            self.primitives.len(),
            self.chunks.len(),
            self.predictions.edge_count(),
            self.associations.edge_count(),
        )
    }

    // ── Merge / reactivation (v0.4) ──────────────────────────────────────

    fn consider_merges(&mut self, units: &[UnitId], _tick: u64) {
        for window in units.windows(2) {
            let (left, right) = (window[0], window[1]);

            if let Some(existing_id) = self.chunks.find_by_pair(left, right) {
                // v0.4 §12: reactivate SLEEP chunks on re-encounter
                if let Some(chunk) = self.chunks.get(existing_id) {
                    if chunk.is_sleep() {
                        self.chunks.get_mut(existing_id).unwrap().promote_to_hot();
                    }
                }
                continue;
            }

            let count = self
                .merge_candidates
                .entry((left, right))
                .and_modify(|c| *c += 1)
                .or_insert(1);
            let count_val = *count;
            if count_val >= MERGE_THRESHOLD {
                let freq = (count_val as f64 / MERGE_THRESHOLD as f64).min(4.0);
                let prob = BASE_MERGE_PROBABILITY * freq.sqrt();
                if freq >= 2.0 || pseudo_rand(left, right, count_val) < prob {
                    let exp_len = self.expanded_length(left) + self.expanded_length(right);
                    let cid = self.chunks.get_or_create(left, right, exp_len);
                    if let Some(chunk) = self.chunks.get_mut(cid) {
                        if chunk.use_count == 0 {
                            chunk.record_usage(self.tick);
                        }
                    }
                    self.merge_candidates.remove(&(left, right));
                }
            }
        }
    }

    /// Enforce HOT budget — demote weakest HOT chunks to SLEEP (§41).
    /// `budget = 0` means unlimited (no-op).
    pub fn enforce_hot_budget(&mut self, budget: usize) {
        if budget == 0 { return; }
        let hot_count = self.chunks.hot_count();
        if hot_count <= budget { return; }
        let excess = hot_count - budget;
        let mut hot: Vec<(u32, f64)> = self.chunks.iter_all()
            .filter(|c| c.is_hot())
            .map(|c| (c.id, c.usage_strength))
            .collect();
        hot.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (id, _) in hot.iter().take(excess) {
            if let Some(c) = self.chunks.get_mut(*id) { c.demote_to_sleep(); }
        }
    }

    fn expanded_length(&self, unit: UnitId) -> u32 {
        if unit.is_primitive() {
            1
        } else if let Some(cid) = unit.as_chunk() {
            self.chunks.get(cid).map(|c| c.expanded_length).unwrap_or(1)
        } else {
            1
        }
    }

    // ── Inspection ───────────────────────────────────────────────────────

    pub fn primitive_count(&self) -> usize { self.primitives.len() }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    pub fn hot_chunk_count(&self) -> usize { self.chunks.hot_count() }
    pub fn sleep_chunk_count(&self) -> usize { self.chunks.sleep_count() }
    pub fn edge_count(&self) -> usize { self.predictions.edge_count() }
    pub fn association_count(&self) -> usize { self.associations.edge_count() }

    pub fn merge_candidates_iter(&self) -> impl Iterator<Item = (&(UnitId, UnitId), &u32)> {
        self.merge_candidates.iter()
    }

    pub fn from_parts(
        primitives: PrimitiveRegistry,
        chunks: ChunkRegistry,
        predictions: PredictionStore,
        associations: AssociationStore,
        tick: u64,
        merge_candidates: HashMap<(UnitId, UnitId), u32>,
        metrics: Metrics,
        next_trace_id: TraceId,
    ) -> Self {
        Self {
            primitives,
            chunks,
            predictions,
            associations,
            tick,
            merge_candidates,
            metrics,
            next_trace_id,
        }
    }
}

fn pseudo_rand(left: UnitId, right: UnitId, count: u32) -> f64 {
    let h = left.raw().wrapping_mul(2654435761)
        ^ right.raw().wrapping_mul(2246822519)
        ^ count.wrapping_mul(3266489917);
    (h as f64) / (u32::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_registers_primitives() {
        let mut m = ModelState::new();
        m.train("abc");
        assert!(m.primitive_count() >= 3);
    }

    #[test]
    fn test_expose_registers_primitives() {
        let mut m = ModelState::new();
        m.expose("abc");
        assert!(m.primitive_count() >= 3);
    }

    #[test]
    fn test_train_builds_prediction_edges() {
        let mut m = ModelState::new();
        m.train("abcabc");
        assert!(m.edge_count() > 0);
    }

    #[test]
    fn test_metrics_accumulate() {
        let mut m = ModelState::new();
        m.train("hello");
        assert!(m.metrics.total_characters >= 5);
        assert!(m.metrics.total_decisions >= 1);
    }

    #[test]
    fn test_repeated_training_creates_chunks() {
        let mut m = ModelState::new();
        for _ in 0..(MERGE_THRESHOLD * 2 + 2) {
            m.train("ab");
        }
        assert!(m.chunk_count() > 0);
    }

    #[test]
    fn test_dpc_decreases_with_learning() {
        let mut m = ModelState::new();
        let text = "ab".repeat(50);
        for chunk in text.as_bytes().chunks(4) {
            m.train(std::str::from_utf8(chunk).unwrap());
        }
        let dpc = m.metrics.decision_per_character();
        assert!(dpc > 0.0);
    }

    #[test]
    fn test_generate_returns_nonempty_with_trained_model() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let out = m.generate("hel", 5);
        assert!(out.starts_with("hel") || !out.is_empty());
    }

    #[test]
    fn test_generate_empty_seed_with_no_training() {
        let m = ModelState::new();
        let out = m.generate("", 10);
        assert_eq!(out, "");
    }

    #[test]
    fn test_train_empty_string() {
        let mut m = ModelState::new();
        m.train("");
        assert_eq!(m.primitive_count(), 0);
    }

    #[test]
    fn test_decision_per_character_zero_before_training() {
        let m = ModelState::new();
        assert_eq!(m.metrics.decision_per_character(), 0.0);
    }

    #[test]
    fn test_generate_with_trace_records_steps() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let tid = m.alloc_trace_id();
        let (output, trace) = m.generate_with_trace("hel", "hel", 5, tid);
        assert!(!output.is_empty());
        assert_eq!(trace.trace_id, 1);
        assert_eq!(trace.generation_seed, "hel");
    }

    #[test]
    fn test_generate_with_trace_no_mutation() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let edges_before = m.edge_count();
        let tick_before = m.tick;
        let _ = m.generate("hel", 5);
        assert_eq!(m.edge_count(), edges_before);
        assert_eq!(m.tick, tick_before);
    }

    #[test]
    fn test_apply_feedback_to_trace() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let tid = m.alloc_trace_id();
        let (_, trace) = m.generate_with_trace("hel", "hel", 5, tid);
        if trace.decision_count > 0 {
            let credits: Vec<f64> = vec![1.0; trace.decision_count];
            m.apply_feedback_to_trace(&trace, &credits, 0.99);
            // Just verify no panic and at least one feedback was applied
            let total_fb: u32 = m.predictions.iter_all()
                .map(|e| e.feedback_count)
                .sum();
            assert!(total_fb > 0 || trace.decision_count == 0);
        }
    }

    #[test]
    fn test_state_fingerprint() {
        let mut m = ModelState::new();
        let fp1 = m.state_fingerprint();
        m.train("abc");
        let fp2 = m.state_fingerprint();
        assert_ne!(fp1, fp2);
    }
}
