/// File Operations — FileKind, FileCommand, ModelFileService (§82-§86).
use std::path::{Path, PathBuf};

/// Classification of file kinds by extension (case-insensitive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    ModelJson,
    ModelDatabase,
    ModelBinary,
    DatasetText,
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
            _ => FileKind::Unknown,
        }
    }

    /// Returns true for any model file kind (JSON, DB, or binary).
    pub fn is_model(self) -> bool {
        matches!(self, FileKind::ModelJson | FileKind::ModelDatabase | FileKind::ModelBinary)
    }
}

/// Unified file command (§81).
/// Native Menu, Keyboard Shortcut, D&D, and toolbar all funnel into this.
#[derive(Debug, Clone)]
pub enum FileCommand {
    New,
    Open,
    Save,
    SaveAs,
    /// Load a specific path (from D&D or a recent-files list).
    LoadPath(PathBuf),
}

/// Document state — tracks current path and unsaved changes (§87).
#[derive(Debug, Default, Clone)]
pub struct DocumentState {
    pub path: Option<PathBuf>,
    pub dirty: bool,
}

impl DocumentState {
    pub fn new_unsaved() -> Self {
        Self { path: None, dirty: false }
    }

    pub fn from_path(path: PathBuf) -> Self {
        Self { path: Some(path), dirty: false }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Called after a successful save. Updates path and clears dirty flag.
    pub fn mark_saved(&mut self, path: PathBuf) {
        self.path = Some(path);
        self.dirty = false;
    }

    pub fn path_str(&self) -> &str {
        self.path.as_ref().and_then(|p| p.to_str()).unwrap_or("(unsaved)")
    }
}
