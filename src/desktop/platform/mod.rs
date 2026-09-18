/// Platform-specific desktop integration (§88–90).
#[cfg(all(target_os = "windows", feature = "gui"))]
pub mod windows;
