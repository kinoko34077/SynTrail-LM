/// File dialog wrappers — single source of truth for all GUI file operations (§35).
///
/// Both syntrail_gui and syntrail_trainer must call these functions instead of
/// inline rfd calls.  All dialog titles use Japanese (consistent with app locale).
#[cfg(feature = "gui")]
pub use gui::*;

#[cfg(feature = "gui")]
mod gui {
    use std::path::PathBuf;

    fn parent_of(path: &str) -> PathBuf {
        PathBuf::from(path)
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Open-model file picker (§35).
    pub fn open_model_dialog(current: &str) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("モデルを開く")
            .add_filter("SynTrail JSON", &["json"])
            .add_filter("SynTrail DB", &["db", "sqlite"])
            .add_filter("SynTrail STM", &["stm"])
            .add_filter("すべてのファイル", &["*"])
            .set_directory(parent_of(current))
            .pick_file()
    }

    /// Save-model file picker (§35, §6).
    pub fn save_model_dialog(current: &str) -> Option<PathBuf> {
        let cur = PathBuf::from(current);
        let name = cur.file_name().and_then(|n| n.to_str()).unwrap_or("model.stm");
        rfd::FileDialog::new()
            .set_title("名前を付けて保存")
            .add_filter("SynTrail STM (binary)", &["stm"])
            .add_filter("SynTrail JSON", &["json"])
            .add_filter("SynTrail DB", &["db", "sqlite"])
            .set_file_name(name)
            .set_directory(parent_of(current))
            .save_file()
    }

    /// Open-dataset file picker (§35).
    pub fn open_dataset_dialog(current: &str) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("テキストファイルを開く")
            .add_filter("Text files", &["txt"])
            .add_filter("すべてのファイル", &["*"])
            .set_directory(parent_of(current))
            .pick_file()
    }

    /// Open-trainer-state file picker (§35).
    pub fn open_trainer_state_dialog(current: &str) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("トレーナー状態を開く")
            .add_filter("Trainer state JSON", &["json"])
            .add_filter("すべてのファイル", &["*"])
            .set_directory(parent_of(current))
            .pick_file()
    }

    /// Save-trainer-state file picker (§35).
    pub fn save_trainer_state_dialog(current: &str) -> Option<PathBuf> {
        let cur = PathBuf::from(current);
        let name = cur.file_name().and_then(|n| n.to_str())
            .unwrap_or("trainer.syntrail-trainer.json");
        rfd::FileDialog::new()
            .set_title("トレーナー状態を保存")
            .add_filter("Trainer state JSON", &["json"])
            .set_file_name(name)
            .set_directory(parent_of(current))
            .save_file()
    }

    /// Dirty-discard confirmation dialog (§35).
    ///
    /// Returns true if the user chose to discard (Yes) or there was nothing dirty.
    pub fn confirm_discard_dialog() -> bool {
        rfd::MessageDialog::new()
            .set_title("未保存の変更")
            .set_description("保存されていない変更があります。続けますか？")
            .set_buttons(rfd::MessageButtons::YesNo)
            .show() == rfd::MessageDialogResult::Yes
    }

    /// Result of the 3-button Save/Discard/Cancel dialog (§41).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ConfirmResult {
        Save,
        Discard,
        Cancel,
    }

    /// Save / Discard / Cancel confirmation dialog for unsaved changes (§41).
    ///
    /// Shows "Yes = 保存して続ける / No = 破棄して続ける / Cancel = キャンセル".
    pub fn confirm_save_discard_cancel_dialog() -> ConfirmResult {
        match rfd::MessageDialog::new()
            .set_title("未保存の変更")
            .set_description(
                "保存されていない変更があります。\n\
                 「はい」で保存して続ける、「いいえ」で破棄して続ける、\
                 「キャンセル」で操作を中断します。"
            )
            .set_buttons(rfd::MessageButtons::YesNoCancel)
            .show()
        {
            rfd::MessageDialogResult::Yes    => ConfirmResult::Save,
            rfd::MessageDialogResult::No     => ConfirmResult::Discard,
            rfd::MessageDialogResult::Cancel => ConfirmResult::Cancel,
            _                                => ConfirmResult::Cancel,
        }
    }
}
