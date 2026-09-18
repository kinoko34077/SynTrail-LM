/// Dataset loading: file I/O, encoding detection, line-ending normalisation.
///
/// TR-DATA-01: Open dialog + D&D (handled in GUI layer).
/// TR-DATA-02: UTF-8 first, then Shift_JIS; encoding error = hard error (no lossy decode).
/// TR-DATA-03: CRLF/CR → LF; empty lines kept; no other normalisation.
use std::path::{Path, PathBuf};

/// FNV-1a 64-bit — stable, no external crate required (§11).
pub fn fnv1a(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

pub struct Dataset {
    pub path: PathBuf,
    /// Normalised text (CRLF/CR → LF). This is the single source of truth.
    pub normalized: String,
    /// FNV-1a hash of `normalized` bytes for change detection.
    pub fingerprint: u64,
}

impl Dataset {
    /// Load a `.txt` file. Returns `Err` on I/O failure, encoding failure, or empty file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read(path)
            .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;

        // Try UTF-8 first (covers ASCII too).
        let text = if let Ok(s) = std::str::from_utf8(&raw) {
            s.to_owned()
        } else {
            // Try Shift_JIS (strict — no replacement chars).
            match encoding_rs::SHIFT_JIS.decode_without_bom_handling_and_without_replacement(&raw) {
                Some(s) => s.into_owned(),
                None => return Err(format!(
                    "{}: not valid UTF-8 or Shift_JIS (check encoding)",
                    path.display()
                )),
            }
        };

        // Normalise line endings (TR-DATA-03).
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");

        if normalized.is_empty() {
            return Err(format!("{}: dataset file is empty", path.display()));
        }

        let fingerprint = fnv1a(normalized.as_bytes());
        Ok(Dataset { path: path.to_path_buf(), normalized, fingerprint })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_bytes(data: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(data).unwrap();
        f
    }

    #[test]
    fn tr_t01_crlf_normalized() {
        let f = write_bytes(b"a\r\nb\r\nc");
        let ds = Dataset::load(f.path()).unwrap();
        assert_eq!(ds.normalized, "a\nb\nc", "TR-T01: CRLF not normalised");
    }

    #[test]
    fn tr_t01_cr_normalized() {
        let f = write_bytes(b"a\rb\rc");
        let ds = Dataset::load(f.path()).unwrap();
        assert_eq!(ds.normalized, "a\nb\nc", "TR-T01: CR not normalised");
    }

    #[test]
    fn tr_t02_empty_lines_kept() {
        let f = write_bytes(b"a\n\nb\n\nc");
        let ds = Dataset::load(f.path()).unwrap();
        // Empty lines must be preserved (TR-T02)
        assert!(ds.normalized.contains("\n\n"), "TR-T02: empty lines were removed");
    }

    #[test]
    fn fingerprint_stable() {
        let f = write_bytes(b"hello world");
        let ds1 = Dataset::load(f.path()).unwrap();
        let ds2 = Dataset::load(f.path()).unwrap();
        assert_eq!(ds1.fingerprint, ds2.fingerprint);
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_bytes(b"");
        assert!(Dataset::load(f.path()).is_err());
    }
}
