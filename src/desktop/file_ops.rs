/// File Operations — FileKind, FileCommand, ModelFileService (§82-§86).
use std::path::{Path, PathBuf};

/// Classification of file kinds by extension (case-insensitive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    ModelJson,
    ModelDatabase,
    ModelBinary,
    DatasetText,
    /// §35: JSONL supervised dataset (§29 canonical import format).
    DatasetSupervised,
    TrainerState,
    Unknown,
}

impl FileKind {
    /// Detect the kind of a file from its path extension (case-insensitive).
    pub fn detect(path: &Path) -> Self {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        match ext.as_deref() {
            Some("json") => {
                // trainer state JSON has a .syntrail-trainer.json double extension
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.ends_with(".syntrail-trainer.json") {
                    FileKind::TrainerState
                } else {
                    FileKind::ModelJson
                }
            }
            Some("db") | Some("sqlite") => FileKind::ModelDatabase,
            Some("stm") => FileKind::ModelBinary,
            Some("txt") => FileKind::DatasetText,
            Some("jsonl") => FileKind::DatasetSupervised,
            _ => FileKind::Unknown,
        }
    }

    /// Returns true for any model file kind (JSON, DB, or binary).
    pub fn is_model(self) -> bool {
        matches!(self, FileKind::ModelJson | FileKind::ModelDatabase | FileKind::ModelBinary)
    }

    /// §35: Returns true for any dataset kind (text or supervised).
    pub fn is_dataset(self) -> bool {
        matches!(self, FileKind::DatasetText | FileKind::DatasetSupervised)
    }
}

/// Unified file command (§81).
/// Native Menu, Keyboard Shortcut, D&D, and toolbar all funnel into this.
#[derive(Debug, Clone)]
pub enum FileCommand {
    New,
    /// §27: Reset conversation history only; model state is preserved.
    NewConversation,
    Open,
    /// §39: Trainer-specific — open a text dataset file.
    OpenDataset,
    Save,
    SaveAs,
    /// Load a specific path (from D&D or a recent-files list).
    LoadPath(PathBuf),
}

/// Document state — re-exported from app (canonical definition lives there).
pub use crate::app::DocumentState;
