/// D&D common router — maps dropped paths to FileCommand (§95–96).
use std::path::PathBuf;
use super::file_ops::{FileCommand, FileKind};

/// Which file kinds a given application will accept via D&D.
pub struct AcceptedKinds {
    pub model: bool,
    pub dataset: bool,
}

impl AcceptedKinds {
    pub fn chat() -> Self {
        Self { model: true, dataset: false }
    }

    pub fn trainer() -> Self {
        Self { model: true, dataset: true }
    }
}

/// Result of classifying a set of dropped paths.
pub enum DropResult {
    /// Single file accepted; issue this command.
    Command(FileCommand),
    /// Multiple files of same kind — not supported.
    MultipleFiles,
    /// File kind not accepted by this app.
    Unsupported(PathBuf),
}

/// Route a list of dropped paths to a FileCommand for a given app context.
pub fn route_drop(paths: Vec<PathBuf>, accepted: &AcceptedKinds) -> DropResult {
    if paths.len() > 1 {
        return DropResult::MultipleFiles;
    }
    let Some(path) = paths.into_iter().next() else {
        return DropResult::MultipleFiles;
    };
    let kind = FileKind::detect(&path);
    match kind {
        FileKind::ModelJson | FileKind::ModelDatabase | FileKind::ModelBinary
            if accepted.model =>
        {
            DropResult::Command(FileCommand::LoadPath(path))
        }
        FileKind::DatasetText if accepted.dataset => {
            DropResult::Command(FileCommand::LoadPath(path))
        }
        _ => DropResult::Unsupported(path),
    }
}
