/// Desktop Shell — OS/Desktop共通操作 (§78).
///
/// `drop` and `file_ops` are always compiled (no gui dependency).
/// `dialogs`, `fonts`, and `platform` require the `gui` feature (rfd/muda/egui).
pub mod drop;
pub mod file_ops;
#[cfg(feature = "gui")]
pub mod dialogs;
#[cfg(feature = "gui")]
pub mod fonts;
#[cfg(feature = "gui")]
pub mod platform;

pub use file_ops::{FileCommand, FileKind};
