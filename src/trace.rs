/// Phase 1: TurnTrace — records every generation decision (§7).
use crate::units::UnitId;

pub type TraceId = u64;

/// One model decision during generation: the unit that was selected at a
/// given step, and the context (previous unit) that triggered the prediction.
/// This is the feedback credit unit — NOT the internal sub-nodes of a Chunk.
#[derive(Debug, Clone)]
pub struct DecisionStep {
    /// 0-based position in the generation sequence (seed excluded).
    pub step_index: usize,
    /// The unit selected at this step (Primitive or Chunk).
    pub unit: UnitId,
    /// The context unit whose prediction edge led here.
    pub context: UnitId,
    /// Score of the chosen prediction edge.
    pub score: f64,
}

/// Complete trace of one generation call.
/// The seed units are NOT decisions; only units produced by top1() are.
#[derive(Debug, Clone)]
pub struct TurnTrace {
    pub trace_id: TraceId,
    /// Model tick before generation began.
    pub state_before_tick: u64,
    /// Seed text passed to generate.
    pub generation_seed: String,
    /// Input text for the turn (same as seed in chat mode).
    pub input_text: String,
    /// Full output text (seed + emitted, Completion mode).
    pub output_text: String,
    /// Only the emitted portion — excludes seed (Dialogue mode output).
    pub emitted_text: String,
    /// Decisions made during generation.
    pub decision_steps: Vec<DecisionStep>,
    /// decision_steps.len() — pre-computed for convenience.
    pub decision_count: usize,
    /// Unix timestamp seconds at creation.
    pub created_at: u64,
    /// True if generation stopped because EOS was predicted.
    pub stopped_by_eos: bool,
    /// True if generation stopped because a cycle was detected.
    pub stopped_by_cycle: bool,
}

impl TurnTrace {
    pub fn new(
        trace_id: TraceId,
        state_before_tick: u64,
        generation_seed: String,
        input_text: String,
        output_text: String,
        emitted_text: String,
        decision_steps: Vec<DecisionStep>,
        created_at: u64,
    ) -> Self {
        let decision_count = decision_steps.len();
        Self {
            trace_id,
            state_before_tick,
            generation_seed,
            input_text,
            output_text,
            emitted_text,
            decision_steps,
            decision_count,
            created_at,
            stopped_by_eos: false,
            stopped_by_cycle: false,
        }
    }
}

pub fn now_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
