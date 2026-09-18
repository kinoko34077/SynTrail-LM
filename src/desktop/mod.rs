/// Desktop Shell — OS/Desktop共通操作 (§78).
///
/// Chat GUI / Trainer / 将来のTeacher等で共通するFile操作・Dialog・D&D・Font設定を集約。
/// App固有ロジック(conversation, training worker等)は各App側へ残す。
pub mod dialogs;
pub mod drop;
pub mod file_ops;
pub mod fonts;

pub use file_ops::{FileCommand, FileKind};
