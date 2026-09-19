/// Trainer subsystem — adaptive multi-line exposure training engine.
///
/// Responsibilities are split as per §4:
///   dataset.rs    — file loading, encoding, normalisation
///   splitter.rs   — Block splitting preserving original text exactly
///   scheduler.rs  — checkpoint state machine (4→8→16→32 repeats)
///   adaptive.rs   — dpc-based stop/continue and block size decisions
///   state.rs      — TrainerState persistence and fingerprint verification
///   supervised.rs — §27/§28/§29: SupervisedSample + JSONL loader
pub mod adaptive;
pub mod dataset;
pub mod scheduler;
pub mod splitter;
pub mod state;
pub mod supervised;

pub use adaptive::{BlockLevel, LevelChange};
pub use dataset::Dataset;
pub use scheduler::TrainingScheduler;
pub use splitter::{Block, BlockSplitter};
pub use state::{TrainerState, TrainerStatus};
pub use supervised::{SupervisedDataset, SupervisedSample};
