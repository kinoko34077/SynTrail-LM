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
    /// §43: Exactly one model + one dataset dropped together (Trainer only).
    ModelAndDataset { model: PathBuf, dataset: PathBuf },
    /// Multiple files of same kind or unsortable mix — not supported.
    MultipleFiles,
    /// File kind not accepted by this app.
    Unsupported(PathBuf),
}

/// Route a list of dropped paths to a FileCommand for a given app context.
pub fn route_drop(paths: Vec<PathBuf>, accepted: &AcceptedKinds) -> DropResult {
    // §43: Trainer model+dataset combo drop.
    if paths.len() == 2 && accepted.model && accepted.dataset {
        let kinds: Vec<_> = paths.iter().map(|p| FileKind::detect(p)).collect();
        let model_idx = kinds.iter().position(|k| k.is_model());
        let dataset_idx = kinds.iter().position(|k| k.is_dataset());
        if let (Some(mi), Some(di)) = (model_idx, dataset_idx) {
            if mi != di {
                let mut it = paths.into_iter();
                let (a, b) = (it.next().unwrap(), it.next().unwrap());
                let (model, dataset) = if mi == 0 { (a, b) } else { (b, a) };
                return DropResult::ModelAndDataset { model, dataset };
            }
        }
        return DropResult::MultipleFiles;
    }
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
        FileKind::DatasetText | FileKind::DatasetSupervised if accepted.dataset => {
            DropResult::Command(FileCommand::LoadPath(path))
        }
        _ => DropResult::Unsupported(path),
    }
}
