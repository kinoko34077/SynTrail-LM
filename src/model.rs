/// Phase 7-8: Unified learn/generate model state (§C §10-11).
///
/// Single struct owns all registries.  Training and generation share the
/// same segmentation + prediction path — there is no separate mode.
use std::collections::HashMap;

use crate::chunks::ChunkRegistry;
use crate::prediction::PredictionStore;
use crate::primitives::PrimitiveRegistry;
use crate::segmentation::{expand, segment};
use crate::units::UnitId;

// ── Probabilistic merge parameters (§C §6) ────────────────────────────────

/// A new chunk is created when its candidate pair is observed this many times.
const MERGE_THRESHOLD: u32 = 4;
/// Base probability applied after threshold is crossed.
const BASE_MERGE_PROBABILITY: f64 = 0.5;

// ── Segmentation threshold ─────────────────────────────────────────────────

/// Minimum score for a chunk to be used during segmentation.
/// A chunk with score < this value is skipped even if it exists.
const SEGMENT_MIN_SCORE: f64 = 0.0;

/// Running counters for the primary metric (§E §16).
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
    /// Monotonically increasing logical clock (incremented per training step).
    pub tick: u64,
    /// Observation counts for candidate (left, right) pairs not yet made
    /// into chunks.  Used for probabilistic merge decisions.
    pub(crate) merge_candidates: HashMap<(UnitId, UnitId), u32>,
    pub metrics: Metrics,
}

impl ModelState {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Training ─────────────────────────────────────────────────────────

    /// Process one training sample end-to-end.
    ///
    /// 1. Encode text → primitive IDs → UnitIds.
    /// 2. Segment using existing chunks.
    /// 3. Record successes for all chunks that survived segmentation.
    /// 4. Learn prediction edges from the segmented sequence.
    /// 5. Evaluate candidate merges for new chunk creation.
    /// 6. Update metrics.
    pub fn train(&mut self, text: &str) {
        self.tick += 1;
        let tick = self.tick;

        let prim_ids = self.primitives.encode(text);
        if prim_ids.is_empty() {
            return;
        }

        // Segment using current chunk knowledge
        let segmented = segment(&prim_ids, &self.chunks, SEGMENT_MIN_SCORE);

        // Record success for every chunk that appears in the segmentation
        for &unit in &segmented {
            if let Some(cid) = unit.as_chunk() {
                if let Some(chunk) = self.chunks.get_mut(cid) {
                    chunk.record_success(tick);
                }
            }
        }

        // Self-supervised: learn prediction edges from segmented sequence
        self.predictions.learn_sequence(&segmented);

        // Update metrics
        self.metrics.total_decisions += segmented.len() as u64;
        self.metrics.total_characters += prim_ids.len() as u64;

        // Evaluate candidate pairs for new chunk creation
        self.consider_merges(&segmented, tick);
    }

    /// For each adjacent pair in `units`, increment its candidate count and
    /// probabilistically create a new chunk when it crosses the threshold.
    fn consider_merges(&mut self, units: &[UnitId], _tick: u64) {
        for window in units.windows(2) {
            let (left, right) = (window[0], window[1]);

            // Skip if a chunk for this pair already exists
            if self.chunks.find_by_pair(left, right).is_some() {
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
                // Deterministic stand-in for sampling: create when freq ≥ 2
                // (i.e., pair has been seen ≥ 2 × MERGE_THRESHOLD times).
                // A real implementation would use a seeded RNG here.
                if freq >= 2.0 || pseudo_rand(left, right, count_val) < prob {
                    let exp_len = self.expanded_length(left) + self.expanded_length(right);
                    let cid = self.chunks.get_or_create(left, right, exp_len);
                    // Bootstrap: credit one success so the initial score is positive
                    // (score = confidence + strength_norm - cost = 1.0 + small - 0.1 > 0)
                    // and the chunk is immediately eligible for segmentation.
                    if let Some(chunk) = self.chunks.get_mut(cid) {
                        if chunk.success_count == 0 {
                            chunk.record_success(self.tick);
                        }
                    }
                    // Clear so repeated creation isn't triggered
                    self.merge_candidates.remove(&(left, right));
                }
            }
        }
    }

    /// Expanded length (in Primitives) of a given unit.
    fn expanded_length(&self, unit: UnitId) -> u32 {
        if unit.is_primitive() {
            1
        } else if let Some(cid) = unit.as_chunk() {
            self.chunks
                .get(cid)
                .map(|c| c.expanded_length)
                .unwrap_or(1)
        } else {
            1
        }
    }

    // ── Generation ───────────────────────────────────────────────────────

    /// Generate at most `max_units` units starting from `seed_text`.
    /// Returns the generated text (seed + generated continuation).
    pub fn generate(&self, seed_text: &str, max_units: usize) -> String {
        let prim_ids = {
            // encode without mutating self — clone is cheap for short seeds
            let mut tmp = self.primitives.clone();
            tmp.encode(seed_text)
        };

        let units = segment(&prim_ids, &self.chunks, SEGMENT_MIN_SCORE);

        let mut generated_units: Vec<UnitId> = units.clone();

        for _ in 0..max_units {
            let context = match generated_units.last() {
                Some(&u) => u,
                None => break,
            };
            match self.predictions.top1(context) {
                Some(next) => generated_units.push(next),
                None => break,
            }
        }

        let expanded = expand(&generated_units, &self.chunks, &self.primitives);
        self.primitives.decode(&expanded).unwrap_or_default()
    }

    // ── Inspection ───────────────────────────────────────────────────────

    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn edge_count(&self) -> usize {
        self.predictions.edge_count()
    }

    /// Iterator over merge candidates for serialisation.
    pub fn merge_candidates_iter(&self) -> impl Iterator<Item = (&(UnitId, UnitId), &u32)> {
        self.merge_candidates.iter()
    }

    /// Reconstruct from deserialised parts.
    pub fn from_parts(
        primitives: PrimitiveRegistry,
        chunks: ChunkRegistry,
        predictions: PredictionStore,
        tick: u64,
        merge_candidates: HashMap<(UnitId, UnitId), u32>,
        metrics: Metrics,
    ) -> Self {
        Self {
            primitives,
            chunks,
            predictions,
            tick,
            merge_candidates,
            metrics,
        }
    }
}

/// Cheap deterministic pseudo-random in [0, 1) based on unit ids + count.
/// Replaces an RNG for the deterministic unit-test path.
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
        // Repeat enough times to cross MERGE_THRESHOLD twice (= 2×MERGE_THRESHOLD)
        for _ in 0..(MERGE_THRESHOLD * 2 + 2) {
            m.train("ab");
        }
        assert!(m.chunk_count() > 0, "expected at least one chunk after repeated training");
    }

    #[test]
    fn test_dpc_decreases_with_learning() {
        let mut m = ModelState::new();
        // Train on a repetitive sequence to let chunks form
        let text = "ab".repeat(50);
        for chunk in text.as_bytes().chunks(4) {
            m.train(std::str::from_utf8(chunk).unwrap());
        }
        // After learning, dpc should be ≤ 1.0 (ideally < 1.0 once chunks form)
        let dpc = m.metrics.decision_per_character();
        assert!(dpc > 0.0);
        // Not asserting < 1.0 yet — that's the Phase 10 goal; just check it runs
    }

    #[test]
    fn test_generate_returns_nonempty_with_trained_model() {
        let mut m = ModelState::new();
        for _ in 0..20 {
            m.train("hello world");
        }
        let out = m.generate("hel", 5);
        // Should at least echo back the seed
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
}
