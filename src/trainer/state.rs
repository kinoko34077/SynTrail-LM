/// Trainer progress persistence — saved to `<dataset-stem>.syntrail-trainer.json`.
///
/// Deliberately separate from the model file (§11 / §12).
/// Fingerprints guard against applying stale progress to a changed model or dataset.
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use super::adaptive::BlockLevel;

const STATE_VERSION: &str = "1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainerStatus {
    Idle,
    Ready,
    Running,
    Paused,
    Completed,
    Error,
}

impl Default for TrainerStatus { fn default() -> Self { Self::Idle } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerState {
    pub state_version: String,
    pub dataset_path: String,
    pub dataset_fingerprint: u64,
    pub normalized_length: usize,
    /// Byte cursor into the normalised source — start of next block to generate.
    pub cursor: usize,
    /// Byte start of the *current* (in-progress) block.
    pub current_block_start: usize,
    /// Byte end (exclusive) of the current block.
    pub current_block_end: usize,
    /// How many `expose()` calls have been done for the current block.
    pub current_block_repeat: u32,
    /// Checkpoint target we were at when paused (4 / 8 / 16 / 32).
    pub current_checkpoint: u32,
    /// dpc *before* the current block's first exposure.
    pub current_block_pre_dpc: f64,
    /// dpc recorded at completed checkpoints [4, 8, 16, 32] for the current block.
    pub checkpoint_dpcs: [f64; 4],
    pub block_level: BlockLevel,
    pub completed_block_count: usize,
    /// Last ≤8 pre-dpc values (oldest first).
    pub recent_pre_dpc: Vec<f64>,
    /// Last ≤8 repeats-used values (oldest first).
    pub recent_repeats_used: Vec<u32>,
    pub model_path: String,
    pub model_fingerprint: String,
    pub status: TrainerStatus,
    /// §32: Generation ID linking this trainer state to a specific model checkpoint.
    /// Both the model file and trainer state carry the same value when saved together.
    #[serde(default)]
    pub checkpoint_generation: u64,
}

impl TrainerState {
    pub fn new(
        dataset_path: &Path,
        dataset_fingerprint: u64,
        normalized_length: usize,
        model_path: &Path,
        model_fingerprint: String,
    ) -> Self {
        Self {
            state_version: STATE_VERSION.to_owned(),
            dataset_path: dataset_path.to_string_lossy().to_string(),
            dataset_fingerprint,
            normalized_length,
            cursor: 0,
            current_block_start: 0,
            current_block_end: 0,
            current_block_repeat: 0,
            current_checkpoint: 4,
            current_block_pre_dpc: 0.0,
            checkpoint_dpcs: [0.0; 4],
            block_level: BlockLevel::S,
            completed_block_count: 0,
            recent_pre_dpc: Vec::new(),
            recent_repeats_used: Vec::new(),
            model_path: model_path.to_string_lossy().to_string(),
            model_fingerprint,
            status: TrainerStatus::Ready,
            checkpoint_generation: 0,
        }
    }

    /// Path for the trainer state file derived from the dataset path.
    pub fn state_file_path(dataset_path: &Path) -> PathBuf {
        let stem = dataset_path.file_stem().unwrap_or_default();
        let dir = dataset_path.parent().unwrap_or_else(|| Path::new("."));
        dir.join(format!("{}.syntrail-trainer.json", stem.to_string_lossy()))
    }

    pub fn save(&self, dataset_path: &Path) -> Result<(), String> {
        let path = Self::state_file_path(dataset_path);
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, json).map_err(|e| e.to_string())
    }

    pub fn load(dataset_path: &Path) -> Result<Self, String> {
        let path = Self::state_file_path(dataset_path);
        if !path.exists() {
            return Err(format!("No trainer state at {}", path.display()));
        }
        let json = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let state: Self = serde_json::from_str(&json)
            .map_err(|e| format!("State parse error: {e}"))?;
        if state.state_version != STATE_VERSION {
            return Err(format!("Unsupported state version: {}", state.state_version));
        }
        Ok(state)
    }

    /// Verify that a resume is safe (TR-T19, TR-T20).
    pub fn verify_resume(
        &self,
        dataset_fingerprint: u64,
        model_fingerprint: &str,
    ) -> Result<(), String> {
        self.verify_resume_with_generation(dataset_fingerprint, model_fingerprint, None)
    }

    /// §32: Verify resume including checkpoint_generation pair check.
    /// Pass `model_generation: Some(gen)` to also verify model+state were saved together.
    pub fn verify_resume_with_generation(
        &self,
        dataset_fingerprint: u64,
        model_fingerprint: &str,
        model_generation: Option<u64>,
    ) -> Result<(), String> {
        if self.dataset_fingerprint != dataset_fingerprint {
            return Err(
                "Dataset has changed since last save (fingerprint mismatch). \
                 Cannot auto-resume — please Start from the beginning or select the original file."
                    .to_owned(),
            );
        }
        if self.model_fingerprint != model_fingerprint {
            return Err(
                "Model has changed since last save (fingerprint mismatch). \
                 Cannot auto-resume — the saved progress no longer matches the loaded model."
                    .to_owned(),
            );
        }
        // §32: generation check — skip when either side is zero (legacy files or generation unused).
        if let Some(model_gen) = model_generation {
            if self.checkpoint_generation != 0 && model_gen != 0
                && self.checkpoint_generation != model_gen
            {
                return Err(format!(
                    "Checkpoint generation mismatch (trainer={}, model={}). \
                     The model and trainer state files were not saved together.",
                    self.checkpoint_generation, model_gen
                ));
            }
        }
        Ok(())
    }

    /// Push a completed block's stats; keep at most 8.
    pub fn push_block_result(&mut self, pre_dpc: f64, repeats: u32) {
        self.recent_pre_dpc.push(pre_dpc);
        self.recent_repeats_used.push(repeats);
        if self.recent_pre_dpc.len() > 8 { self.recent_pre_dpc.remove(0); }
        if self.recent_repeats_used.len() > 8 { self.recent_repeats_used.remove(0); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_state() -> (TrainerState, NamedTempFile) {
        let dataset_file = NamedTempFile::new().unwrap();
        let model_file = NamedTempFile::new().unwrap();
        let s = TrainerState::new(
            dataset_file.path(),
            0xdeadbeef,
            1000,
            model_file.path(),
            "tick=0 prims=0 chunks=0 edges=0 assoc=0".to_owned(),
        );
        (s, dataset_file)
    }

    #[test]
    fn tr_t18_roundtrip() {
        let (mut state, dataset_file) = make_state();
        state.completed_block_count = 5;
        state.cursor = 1234;
        state.block_level = BlockLevel::M;
        state.recent_pre_dpc = vec![0.8, 0.75];
        state.recent_repeats_used = vec![8, 12];
        state.status = TrainerStatus::Paused;

        state.save(dataset_file.path()).unwrap();
        let loaded = TrainerState::load(dataset_file.path()).unwrap();

        assert_eq!(loaded.completed_block_count, 5, "TR-T18: block count");
        assert_eq!(loaded.cursor, 1234, "TR-T18: cursor");
        assert_eq!(loaded.block_level, BlockLevel::M, "TR-T18: level");
        assert_eq!(loaded.recent_pre_dpc, vec![0.8, 0.75], "TR-T18: pre_dpc");
        assert_eq!(loaded.recent_repeats_used, vec![8u32, 12], "TR-T18: repeats");
    }

    #[test]
    fn tr_t19_dataset_fingerprint_mismatch() {
        let (state, _) = make_state();
        let bad_fp = state.dataset_fingerprint + 1;
        let res = state.verify_resume(bad_fp, &state.model_fingerprint.clone());
        assert!(res.is_err(), "TR-T19: should reject dataset fingerprint mismatch");
    }

    #[test]
    fn tr_t20_model_fingerprint_mismatch() {
        let (state, _) = make_state();
        let res = state.verify_resume(state.dataset_fingerprint, "tick=999 prims=0 chunks=0 edges=0 assoc=0");
        assert!(res.is_err(), "TR-T20: should reject model fingerprint mismatch");
    }

    #[test]
    fn verify_ok_with_matching_fingerprints() {
        let (state, _) = make_state();
        let fp_copy = state.model_fingerprint.clone();
        assert!(state.verify_resume(state.dataset_fingerprint, &fp_copy).is_ok());
    }
}
