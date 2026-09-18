/// File dialog wrappers using rfd (§83).
#[cfg(feature = "gui")]
pub use gui::*;

#[cfg(feature = "gui")]
mod gui {
    use std::path::PathBuf;

    pub fn open_model_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("Open Model")
            .add_filter("Model files", &["json", "db", "sqlite", "stm"])
            .add_filter("All files", &["*"])
            .pick_file()
    }

    pub fn save_model_dialog(default_name: &str) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("Save Model")
            .set_file_name(default_name)
            .add_filter("Model JSON", &["json"])
            .save_file()
    }

    pub fn open_dataset_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("Open Dataset")
            .add_filter("Text files", &["txt"])
            .add_filter("All files", &["*"])
            .pick_file()
    }

    pub fn open_trainer_state_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("Open Trainer State")
            .add_filter("Trainer state", &["json"])
            .add_filter("All files", &["*"])
            .pick_file()
    }

    pub fn save_trainer_state_dialog(default_name: &str) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("Save Trainer State")
            .set_file_name(default_name)
            .add_filter("Trainer state", &["json"])
            .save_file()
    }
}
