/// Training scheduler — checkpoint state machine for a single Block.
///
/// Checkpoints: 4 → 8 → 16 → 32 exposures.
/// Decision points: 8, 16, 32 (at 4 we always continue per TR-ADAPT-02).
///
/// TR-T09: 4-exposure checkpoint never stops.
/// TR-T10: minimum 8 exposures guaranteed.
/// TR-T11: if improvement at 8 < 2 % → stop.
/// TR-T12: if improvement at 8 ≥ 2 % → continue to 16.
/// TR-T13: if improvement at 16 ≥ 1 % → continue to 32.
/// TR-T14: 32 exposures is the hard ceiling.
const CHECKPOINTS: [u32; 4] = [4, 8, 16, 32];
const EPSILON: f64 = 1e-10;

/// Outcome from `evaluate_checkpoint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointOutcome {
    /// Continue to the next checkpoint.
    Continue,
    /// Block is done; stop repeating.
    Finished,
}

pub struct TrainingScheduler {
    /// dpc measured *before* the first exposure of this block.
    pub pre_dpc: f64,
    /// Number of times `expose()` has been called for the current block.
    pub repeat_count: u32,
    /// Index into CHECKPOINTS (0..4).
    checkpoint_idx: usize,
    /// dpc recorded at each completed checkpoint (indexed by checkpoint_idx).
    checkpoint_dpcs: [f64; 4],
    /// True once this block has been declared finished.
    pub finished: bool,
}

impl TrainingScheduler {
    pub fn new(pre_dpc: f64) -> Self {
        Self {
            pre_dpc,
            repeat_count: 0,
            checkpoint_idx: 0,
            checkpoint_dpcs: [0.0; 4],
            finished: false,
        }
    }

    /// Restore mid-block state (used on Resume).
    pub fn resume(pre_dpc: f64, repeat_count: u32, checkpoint_dpcs: [f64; 4]) -> Self {
        // Determine which checkpoint_idx we're at based on repeat_count.
        let checkpoint_idx = CHECKPOINTS.iter().position(|&c| repeat_count < c).unwrap_or(4);
        let checkpoint_idx = checkpoint_idx.min(3);
        Self {
            pre_dpc,
            repeat_count,
            checkpoint_idx,
            checkpoint_dpcs,
            finished: false,
        }
    }

    pub fn checkpoint_dpcs(&self) -> [f64; 4] { self.checkpoint_dpcs }

    /// Next checkpoint target (number of exposures).
    pub fn next_checkpoint(&self) -> u32 { CHECKPOINTS[self.checkpoint_idx] }

    /// Maximum possible repeats.
    pub fn max_repeats() -> u32 { CHECKPOINTS[3] }

    /// Record one exposure.
    pub fn record_exposure(&mut self) { self.repeat_count += 1; }

    /// True when `repeat_count` has reached the current checkpoint.
    pub fn at_checkpoint(&self) -> bool { self.repeat_count >= self.next_checkpoint() }

    /// Called once `at_checkpoint()` is true. Provide current dpc.
    /// Returns whether training should continue for this block.
    pub fn evaluate_checkpoint(&mut self, current_dpc: f64) -> CheckpointOutcome {
        let idx = self.checkpoint_idx;
        self.checkpoint_dpcs[idx] = current_dpc;

        let keep_going = match idx {
            // After 4: always continue (TR-ADAPT-02 / TR-T09)
            0 => true,
            // After 8: ≥2 % improvement over checkpoint-4 dpc (TR-ADAPT-03 / TR-T11/T12)
            1 => {
                let prev = self.checkpoint_dpcs[0];
                let improvement = (prev - current_dpc) / prev.max(EPSILON);
                improvement >= 0.02
            }
            // After 16: ≥1 % improvement over checkpoint-8 dpc (TR-ADAPT-04 / TR-T13)
            2 => {
                let prev = self.checkpoint_dpcs[1];
                let improvement = (prev - current_dpc) / prev.max(EPSILON);
                improvement >= 0.01
            }
            // After 32: always stop (TR-ADAPT-05 / TR-T14)
            _ => false,
        };

        if !keep_going || idx >= 3 {
            self.finished = true;
            CheckpointOutcome::Finished
        } else {
            self.checkpoint_idx += 1;
            CheckpointOutcome::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_to_checkpoint(sched: &mut TrainingScheduler, n: u32) {
        for _ in 0..n { sched.record_exposure(); }
    }

    #[test]
    fn tr_t09_checkpoint4_never_stops() {
        let mut s = TrainingScheduler::new(0.9);
        run_to_checkpoint(&mut s, 4);
        assert!(s.at_checkpoint());
        let out = s.evaluate_checkpoint(0.89); // small improvement
        assert_eq!(out, CheckpointOutcome::Continue, "TR-T09: must continue after 4");
        assert!(!s.finished);
    }

    #[test]
    fn tr_t10_minimum_8_exposures() {
        let mut s = TrainingScheduler::new(0.9);
        run_to_checkpoint(&mut s, 4);
        s.evaluate_checkpoint(0.88);            // continue to 8
        run_to_checkpoint(&mut s, 4);           // now at 8
        assert!(s.at_checkpoint());
        assert_eq!(s.repeat_count, 8, "TR-T10: must reach 8");
    }

    #[test]
    fn tr_t11_stop_at_8_low_improvement() {
        let mut s = TrainingScheduler::new(0.9);
        run_to_checkpoint(&mut s, 4);
        s.evaluate_checkpoint(0.89);            // dpc at 4 = 0.89 (small drop from 0.9)
        run_to_checkpoint(&mut s, 4);
        // improvement = (0.89 - 0.889) / 0.89 ≈ 0.001 < 2 %
        let out = s.evaluate_checkpoint(0.889);
        assert_eq!(out, CheckpointOutcome::Finished, "TR-T11: must stop at 8 (< 2%)");
    }

    #[test]
    fn tr_t12_continue_to_16_if_improving() {
        let mut s = TrainingScheduler::new(0.9);
        run_to_checkpoint(&mut s, 4);
        s.evaluate_checkpoint(0.88);            // dpc at 4 = 0.88
        run_to_checkpoint(&mut s, 4);
        // improvement = (0.88 - 0.84) / 0.88 ≈ 4.5 % ≥ 2 %
        let out = s.evaluate_checkpoint(0.84);
        assert_eq!(out, CheckpointOutcome::Continue, "TR-T12: should go to 16");
    }

    #[test]
    fn tr_t13_continue_to_32_if_improving() {
        let mut s = TrainingScheduler::new(0.9);
        run_to_checkpoint(&mut s, 4);  s.evaluate_checkpoint(0.88);
        run_to_checkpoint(&mut s, 4);  s.evaluate_checkpoint(0.84); // → 16
        run_to_checkpoint(&mut s, 8);
        // improvement = (0.84 - 0.82) / 0.84 ≈ 2.4 % ≥ 1 %
        let out = s.evaluate_checkpoint(0.82);
        assert_eq!(out, CheckpointOutcome::Continue, "TR-T13: should go to 32");
    }

    #[test]
    fn tr_t14_max_32() {
        let mut s = TrainingScheduler::new(0.9);
        run_to_checkpoint(&mut s, 4);  s.evaluate_checkpoint(0.88);
        run_to_checkpoint(&mut s, 4);  s.evaluate_checkpoint(0.84);
        run_to_checkpoint(&mut s, 8);  s.evaluate_checkpoint(0.82);
        run_to_checkpoint(&mut s, 16);
        let out = s.evaluate_checkpoint(0.80);
        assert_eq!(out, CheckpointOutcome::Finished, "TR-T14: must stop at 32");
        assert_eq!(s.repeat_count, 32, "TR-T14: repeat_count must be 32");
    }
}
