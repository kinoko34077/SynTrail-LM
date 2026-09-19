/// Supervised learning dataset — §27/§28/§29.
///
/// Canonical import format: JSONL with `{"prompt":"...","response":"..."}`.
/// Aliases: instruction/output (Alpaca-style), messages[role=user/assistant].
use std::path::{Path, PathBuf};
use serde::Deserialize;

/// A single prompt→response training pair (§28).
#[derive(Debug, Clone)]
pub struct SupervisedSample {
    pub prompt: String,
    pub response: String,
}

/// A collection of supervised samples loaded from a JSONL file (§29).
#[derive(Debug, Clone)]
pub struct SupervisedDataset {
    pub samples: Vec<SupervisedSample>,
    pub source_path: PathBuf,
}

impl SupervisedDataset {
    /// Load from a canonical JSONL file (§29).
    /// Each non-empty line must be valid JSON with prompt+response (or aliases).
    /// Malformed lines are reported with line number (§SFT-07).
    /// Samples with empty prompt or response are skipped (§SFT-08).
    pub fn load_jsonl(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut samples = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            let sample = parse_record(trimmed, i + 1)?;
            if sample.prompt.is_empty() || sample.response.is_empty() { continue; }
            samples.push(sample);
        }
        Ok(SupervisedDataset { samples, source_path: path.to_owned() })
    }

    /// §34: FNV-1a fingerprint over all (prompt, response) pairs for resume consistency.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 14695981039346656037;
        for s in &self.samples {
            for b in s.prompt.as_bytes().iter().chain(b"\x00".iter()).chain(s.response.as_bytes()) {
                h = h.wrapping_mul(1099511628211);
                h ^= *b as u64;
            }
        }
        h
    }
}

// ── Internal deserialization ─────────────────────────────────────────────

#[derive(Deserialize)]
struct RawRecord {
    prompt: Option<String>,
    response: Option<String>,
    instruction: Option<String>,
    output: Option<String>,
    messages: Option<Vec<RawMessage>>,
}

#[derive(Deserialize)]
struct RawMessage {
    role: Option<String>,
    content: Option<String>,
}

fn parse_record(line: &str, line_no: usize) -> Result<SupervisedSample, String> {
    let raw: RawRecord = serde_json::from_str(line)
        .map_err(|e| format!("Line {line_no}: malformed JSON: {e}"))?;

    let prompt = raw.prompt
        .or(raw.instruction)
        .or_else(|| {
            raw.messages.as_ref().and_then(|ms| {
                ms.iter().find(|m| m.role.as_deref() == Some("user"))
                    .and_then(|m| m.content.clone())
            })
        })
        .unwrap_or_default();

    let response = raw.response
        .or(raw.output)
        .or_else(|| {
            raw.messages.as_ref().and_then(|ms| {
                ms.iter().find(|m| m.role.as_deref() == Some("assistant"))
                    .and_then(|m| m.content.clone())
            })
        })
        .unwrap_or_default();

    Ok(SupervisedSample { prompt, response })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Result<SupervisedSample, String> { parse_record(line, 1) }

    #[test]
    fn sft_canonical_format() {
        let s = parse(r#"{"prompt":"hello","response":"world"}"#).unwrap();
        assert_eq!(s.prompt, "hello"); assert_eq!(s.response, "world");
    }

    #[test]
    fn sft_instruction_output_alias() {
        let s = parse(r#"{"instruction":"hi","output":"bye"}"#).unwrap();
        assert_eq!(s.prompt, "hi"); assert_eq!(s.response, "bye");
    }

    #[test]
    fn sft_messages_alias() {
        let s = parse(r#"{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}"#).unwrap();
        assert_eq!(s.prompt, "q"); assert_eq!(s.response, "a");
    }

    #[test]
    fn sft_malformed_json_error() {
        assert!(parse_record("{bad", 7).is_err());
        let err = parse_record("{bad", 7).unwrap_err();
        assert!(err.contains("Line 7"), "error should include line number: {err}");
    }

    #[test]
    fn sft_jsonl_roundtrip() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, r#"{{"prompt":"p1","response":"r1"}}"#).unwrap();
        writeln!(f, r#"{{"prompt":"p2","response":"r2"}}"#).unwrap();
        writeln!(f).unwrap(); // blank line — should be skipped
        let ds = SupervisedDataset::load_jsonl(f.path()).unwrap();
        assert_eq!(ds.samples.len(), 2);
        assert_eq!(ds.samples[0].prompt, "p1");
        assert_eq!(ds.samples[1].response, "r2");
    }
}
