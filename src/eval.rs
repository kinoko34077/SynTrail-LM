/// Phase 1 (v0.3): Frozen evaluation — measure model performance without
/// mutating any state (§46, EV-01).
///
/// Unlike the old `evaluate` CLI command which called `expose()`, this module
/// is strictly read-only.
use crate::model::ModelState;
use crate::segmentation::segment;

const SEGMENT_MIN_SCORE: f64 = 0.0;

/// Evaluation result from a single frozen pass over `text`.
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub line_count: usize,
    pub char_count: u64,
    pub decision_count: u64,
    pub dpc: f64,
    /// Correct top-1 predictions / total predictions attempted.
    pub correct_predictions: u64,
    pub total_predictions: u64,
    pub prediction_accuracy: f64,
}

/// Measure performance of `model` on `text` without any state mutation (EV-01).
///
/// For each line:
///   1. Encode to primitive IDs using a throw-away clone (never touches model.primitives)
///   2. Segment with current chunks (read-only)
///   3. Count decisions / chars
///   4. Check top-1 prediction accuracy over adjacent unit pairs
pub fn evaluate_frozen(model: &ModelState, text: &str) -> EvalResult {
    let mut char_count = 0u64;
    let mut decision_count = 0u64;
    let mut correct = 0u64;
    let mut total_preds = 0u64;
    let mut line_count = 0;

    for line in text.lines() {
        if line.is_empty() { continue; }
        line_count += 1;

        // Encode without registering new primitives
        let mut tmp_prims = model.primitives.clone();
        let prim_ids = tmp_prims.encode(line);
        char_count += prim_ids.len() as u64;

        let segmented = segment(&prim_ids, &model.chunks, SEGMENT_MIN_SCORE);
        decision_count += segmented.len() as u64;

        // Prediction accuracy: for each adjacent pair (ctx, next), check top1(ctx) == next
        for window in segmented.windows(2) {
            let (ctx, next) = (window[0], window[1]);
            total_preds += 1;
            if model.predictions.top1(ctx) == Some(next) {
                correct += 1;
            }
        }
    }

    let dpc = if char_count == 0 { 0.0 } else { decision_count as f64 / char_count as f64 };
    let prediction_accuracy = if total_preds == 0 { 0.0 } else { correct as f64 / total_preds as f64 };

    EvalResult {
        line_count,
        char_count,
        decision_count,
        dpc,
        correct_predictions: correct,
        total_predictions: total_preds,
        prediction_accuracy,
    }
}

/// Frozen evaluation of a single multi-line sample (TR-T06..TR-T08).
///
/// Unlike `evaluate_frozen`, this treats the entire `sample` string as one
/// sequence — `\n` characters are counted, segmented, and considered for
/// Prediction accuracy just like any other character.
/// The model is never mutated.
pub fn evaluate_sample_frozen(model: &ModelState, sample: &str) -> EvalResult {
    if sample.is_empty() {
        return EvalResult {
            line_count: 0,
            char_count: 0,
            decision_count: 0,
            dpc: 0.0,
            correct_predictions: 0,
            total_predictions: 0,
            prediction_accuracy: 0.0,
        };
    }

    // Encode without registering new primitives (clone discarded after)
    let mut tmp_prims = model.primitives.clone();
    let prim_ids = tmp_prims.encode(sample);
    let char_count = prim_ids.len() as u64;

    let segmented = segment(&prim_ids, &model.chunks, SEGMENT_MIN_SCORE);
    let decision_count = segmented.len() as u64;

    let mut correct = 0u64;
    let mut total_preds = 0u64;
    for window in segmented.windows(2) {
        let (ctx, next) = (window[0], window[1]);
        total_preds += 1;
        if model.predictions.top1(ctx) == Some(next) {
            correct += 1;
        }
    }

    let dpc = if char_count == 0 { 0.0 } else { decision_count as f64 / char_count as f64 };
    let prediction_accuracy =
        if total_preds == 0 { 0.0 } else { correct as f64 / total_preds as f64 };

    EvalResult {
        line_count: sample.lines().count(),
        char_count,
        decision_count,
        dpc,
        correct_predictions: correct,
        total_predictions: total_preds,
        prediction_accuracy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelState;

    #[test]
    fn ev01_evaluate_frozen_does_not_mutate() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let tick_before = m.tick;
        let edges_before = m.edge_count();
        let chunks_before = m.chunk_count();

        let _ = evaluate_frozen(&m, "hello world\nhello again");

        assert_eq!(m.tick, tick_before, "EV-01: tick changed");
        assert_eq!(m.edge_count(), edges_before, "EV-01: edges changed");
        assert_eq!(m.chunk_count(), chunks_before, "EV-01: chunks changed");
    }

    #[test]
    fn ev03_returns_dpc_and_accuracy() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let result = evaluate_frozen(&m, "hello world");
        assert!(result.dpc > 0.0, "EV-03: dpc should be positive");
        assert!(result.char_count > 0, "EV-03: chars should be counted");
    }

    #[test]
    fn ev01_empty_text_returns_zeroes() {
        let m = ModelState::new();
        let result = evaluate_frozen(&m, "");
        assert_eq!(result.dpc, 0.0);
        assert_eq!(result.char_count, 0);
        assert_eq!(result.prediction_accuracy, 0.0);
    }

    #[test]
    fn ev01_untrained_model_dpc_1() {
        let m = ModelState::new();
        let result = evaluate_frozen(&m, "abc");
        assert!((result.dpc - 1.0).abs() < 1e-9, "untrained: dpc should be 1.0, got {}", result.dpc);
    }

    // ── evaluate_sample_frozen tests (TR-T06..TR-T08) ─────────────────────

    #[test]
    fn tr_t06_sample_frozen_does_not_mutate() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.expose("hello\nworld\n"); }
        let tick_before = m.tick;
        let edges_before = m.edge_count();
        let chunks_before = m.chunk_count();
        let _ = evaluate_sample_frozen(&m, "hello\nworld\nagain");
        assert_eq!(m.tick, tick_before, "TR-T06: tick changed");
        assert_eq!(m.edge_count(), edges_before, "TR-T06: edges changed");
        assert_eq!(m.chunk_count(), chunks_before, "TR-T06: chunks changed");
    }

    #[test]
    fn tr_t07_sample_frozen_counts_newline_chars() {
        let m = ModelState::new();
        // "ab\ncd" has 5 chars including the newline
        let result = evaluate_sample_frozen(&m, "ab\ncd");
        assert_eq!(result.char_count, 5, "TR-T07: \\n must be counted");
    }

    #[test]
    fn tr_t08_sample_frozen_prediction_crosses_newline() {
        let mut m = ModelState::new();
        // Train with multi-line text so \n is a known primitive.
        for _ in 0..30 { m.expose("ab\ncd"); }
        let result = evaluate_sample_frozen(&m, "ab\ncd");
        // TR-T08 core: \n must not be excluded — char_count must be 5.
        assert_eq!(result.char_count, 5, "TR-T08: \\n excluded from char count");
        // There must be at least 1 decision (the whole string is processed).
        assert!(result.decision_count > 0, "TR-T08: no decisions");
        // If the model merged "ab\ncd" into one chunk, total_predictions may be 0 — that is correct.
        // If there are multiple units, there must be predictions.
        if result.decision_count > 1 {
            assert!(result.total_predictions > 0, "TR-T08: multi-unit but no predictions");
        }
    }

    #[test]
    fn tr_t06_empty_sample_returns_zeroes() {
        let m = ModelState::new();
        let r = evaluate_sample_frozen(&m, "");
        assert_eq!(r.char_count, 0);
        assert_eq!(r.dpc, 0.0);
    }
}
