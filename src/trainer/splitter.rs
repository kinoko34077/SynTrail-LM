/// Block splitter — slices normalised source into trainable Blocks.
///
/// Invariant (TR-SPLIT-01): `blocks.map(|b| b.text).join("") == normalized_source`.
/// The cursor into the source is exposed so TrainerState can save/restore it.
use super::adaptive::BlockLevel;

/// A slice of the normalised source, ready to pass to `ModelState::expose()`.
#[derive(Debug, Clone)]
pub struct Block {
    pub text: String,
    /// Byte offset of the start of this block in the normalised source.
    pub start: usize,
    /// Byte offset *after* the last byte of this block (exclusive).
    pub end: usize,
}

/// Produces Blocks from a normalised text string.
/// Calling `next_block()` advances a cursor; re-creating from a saved cursor
/// restores the exact position.
pub struct BlockSplitter {
    source: String,
    cursor: usize,
    level: BlockLevel,
    /// §15: 0 = legacy fixed behavior; non-zero = deterministic jitter seed.
    split_seed: u64,
}

impl BlockSplitter {
    pub fn new(source: String, level: BlockLevel) -> Self {
        Self { source, cursor: 0, level, split_seed: 0 }
    }

    /// Restore from a saved cursor (used on Resume).
    pub fn with_cursor(source: String, level: BlockLevel, cursor: usize) -> Self {
        Self { source, cursor, level, split_seed: 0 }
    }

    /// §15: Restore from a saved cursor with a deterministic jitter seed.
    pub fn with_seed(source: String, level: BlockLevel, cursor: usize, split_seed: u64) -> Self {
        Self { source, cursor, level, split_seed }
    }

    pub fn is_done(&self) -> bool { self.cursor >= self.source.len() }
    pub fn cursor(&self) -> usize { self.cursor }
    pub fn set_level(&mut self, level: BlockLevel) { self.level = level; }
    pub fn source(&self) -> &str { &self.source }

    /// Take the next Block, advancing the cursor. Returns `None` when exhausted.
    pub fn next_block(&mut self) -> Option<Block> {
        if self.is_done() { return None; }

        let start = self.cursor;
        let remaining = &self.source[start..];
        let target = jitter_target_lines(self.split_seed, start, self.level);
        let end_off = find_block_end(remaining, target, self.level.max_chars());
        let end = start + end_off;

        let text = self.source[start..end].to_owned();
        self.cursor = end;
        Some(Block { text, start, end })
    }

    /// Peek at the block that *would* start at `cursor` without advancing.
    pub fn peek_block(&self) -> Option<Block> {
        if self.is_done() { return None; }
        let start = self.cursor;
        let remaining = &self.source[start..];
        let target = jitter_target_lines(self.split_seed, start, self.level);
        let end_off = find_block_end(remaining, target, self.level.max_chars());
        let end = start + end_off;
        Some(Block { text: self.source[start..end].to_owned(), start, end })
    }
}

/// §15: Deterministic block target-line count from seed + block_start + level.
/// Returns `level.target_lines()` when seed is 0 (legacy fixed behavior).
pub fn jitter_target_lines(seed: u64, block_start: usize, level: BlockLevel) -> usize {
    if seed == 0 { return level.target_lines(); }
    let (min, max) = level.jitter_range();
    let level_id = match level { BlockLevel::S => 0u64, BlockLevel::M => 1, BlockLevel::L => 2, BlockLevel::XL => 3 };
    let h = seed
        .wrapping_add(block_start as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(level_id);
    min + (h as usize) % (max - min + 1)
}

/// Compute how many bytes (from the *start* of `text`) belong to one block.
fn find_block_end(text: &str, target_lines: usize, max_chars: usize) -> usize {
    let mut line_count = 0usize;
    let mut char_count = 0usize;
    let mut last_newline_after: usize = 0; // byte offset just after the last \n

    for (byte_off, ch) in text.char_indices() {
        if char_count >= max_chars {
            // Must split here — prefer the last line boundary we passed.
            if last_newline_after > 0 {
                return last_newline_after;
            }
            // No line boundary yet: search backwards for a sentence punct.
            return find_sentence_boundary(text, byte_off);
        }
        char_count += 1;
        if ch == '\n' {
            line_count += 1;
            last_newline_after = byte_off + 1; // byte position after the '\n'
            if line_count >= target_lines {
                return last_newline_after;
            }
        }
    }

    // Reached end of string before hitting limits.
    text.len()
}

/// Search backwards from `near` (byte offset) for 。！？.!? to find a
/// natural sentence boundary (TR-SPLIT-04).  Falls back to `near` itself.
fn find_sentence_boundary(text: &str, near: usize) -> usize {
    let window = &text[..near];
    for (i, ch) in window.char_indices().rev() {
        if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?') {
            return i + ch.len_utf8();
        }
    }
    near // hard cut at char boundary (already on a char boundary by char_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn concat_blocks(source: &str, level: BlockLevel) -> String {
        let mut sp = BlockSplitter::new(source.to_owned(), level);
        let mut out = String::new();
        while let Some(b) = sp.next_block() {
            out.push_str(&b.text);
        }
        out
    }

    #[test]
    fn tr_t03_block_contains_newline() {
        let src = "line1\nline2\nline3\nline4\nline5";
        let mut sp = BlockSplitter::new(src.to_owned(), BlockLevel::S);
        let block = sp.next_block().unwrap();
        // S = 4 target lines → block should contain '\n'
        assert!(block.text.contains('\n'), "TR-T03: block must contain \\n");
    }

    #[test]
    fn tr_t04_concat_equals_source() {
        let source = "a\nb\n\nc\nd\ne\nf\ng\nh\ni\nj\nk";
        assert_eq!(concat_blocks(source, BlockLevel::S), source, "TR-T04: concat != source");
    }

    #[test]
    fn tr_t04_concat_equals_source_cjk() {
        let source = "銀河鉄道の夜\n一　午后の授業\n「ではみなさんは、\nそういうふうに川というものは\n\nできているのですよ」";
        assert_eq!(concat_blocks(source, BlockLevel::S), source, "TR-T04 CJK");
    }

    #[test]
    fn tr_t05_long_single_line_no_loss() {
        // A single line longer than max_chars for S (800)
        let long_line = "あ".repeat(1000);
        let src = long_line.as_str();
        assert_eq!(concat_blocks(src, BlockLevel::S), src, "TR-T05: chars lost in long-line split");
    }

    #[test]
    fn block_start_end_are_consistent() {
        let source = "ab\ncd\nef\ngh\nij\nkl";
        let mut sp = BlockSplitter::new(source.to_owned(), BlockLevel::S);
        let mut prev_end = 0;
        while let Some(b) = sp.next_block() {
            assert_eq!(b.start, prev_end, "block start != previous end");
            prev_end = b.end;
        }
        assert_eq!(prev_end, source.len());
    }

    #[test]
    fn cursor_restore() {
        let source = "a\nb\nc\nd\ne\nf";
        let mut sp = BlockSplitter::new(source.to_owned(), BlockLevel::S);
        let b1 = sp.next_block().unwrap();
        let cursor_after_b1 = sp.cursor();

        // Restore from cursor and get same second block
        let mut sp2 = BlockSplitter::with_cursor(source.to_owned(), BlockLevel::S, cursor_after_b1);
        if let Some(b2a) = sp.next_block() {
            let b2b = sp2.next_block().unwrap();
            assert_eq!(b2a.text, b2b.text, "cursor restore: block text differs");
        }
        let _ = b1;
    }

    // ── §16: Boundary Jitter acceptance tests ────────────────────────────

    fn concat_with_seed(source: &str, level: BlockLevel, seed: u64) -> String {
        let mut sp = BlockSplitter::with_seed(source.to_owned(), level, 0, seed);
        let mut out = String::new();
        while let Some(b) = sp.next_block() { out.push_str(&b.text); }
        out
    }

    #[test]
    fn tr_split_j01_concat_equals_source() {
        let src = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt";
        for seed in [0u64, 1, 42, 99999, u64::MAX / 2] {
            assert_eq!(concat_with_seed(src, BlockLevel::S, seed), src,
                "TR-SPLIT-J01: concat != source (seed={seed})");
        }
    }

    #[test]
    fn tr_split_j04_same_seed_cursor_level_same_block() {
        let src = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj";
        let seed = 12345u64;
        let b1a = BlockSplitter::with_seed(src.to_owned(), BlockLevel::S, 0, seed)
            .peek_block().unwrap().text;
        let b1b = BlockSplitter::with_seed(src.to_owned(), BlockLevel::S, 0, seed)
            .peek_block().unwrap().text;
        assert_eq!(b1a, b1b, "TR-SPLIT-J04: same seed+cursor+level → different block");
    }

    #[test]
    fn tr_split_j05_resume_mid_block_matches() {
        let src = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl";
        let seed = 777u64;
        let mut sp = BlockSplitter::with_seed(src.to_owned(), BlockLevel::S, 0, seed);
        let b1 = sp.next_block().unwrap();
        let cursor = sp.cursor();
        let b2a = sp.next_block().unwrap().text;
        let b2b = BlockSplitter::with_seed(src.to_owned(), BlockLevel::S, cursor, seed)
            .next_block().unwrap().text;
        assert_eq!(b2a, b2b, "TR-SPLIT-J05: resume mid-block differs");
        let _ = b1;
    }

    #[test]
    fn tr_split_j06_different_seeds_change_boundaries() {
        let src = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np";
        let mut seen = std::collections::HashSet::new();
        for seed in [1u64, 2, 3, 100, 999, 12345] {
            let b = BlockSplitter::with_seed(src.to_owned(), BlockLevel::S, 0, seed)
                .next_block().unwrap().end;
            seen.insert(b);
        }
        assert!(seen.len() > 1, "TR-SPLIT-J06: all seeds produce identical boundary");
    }

    #[test]
    fn tr_split_j07_max_chars_maintained() {
        // A block of 1000-char lines — jitter must not exceed max_chars.
        let long_line = "x".repeat(1000);
        let src: Vec<&str> = (0..20).map(|_| long_line.as_str()).collect::<Vec<_>>();
        let src_joined = src.join("\n");
        for seed in [0u64, 42, 12345] {
            let mut sp = BlockSplitter::with_seed(src_joined.clone(), BlockLevel::S, 0, seed);
            while let Some(b) = sp.next_block() {
                assert!(b.text.chars().count() <= BlockLevel::S.max_chars() + long_line.len() + 1,
                    "TR-SPLIT-J07: block exceeds max_chars (seed={seed})");
            }
        }
    }

    #[test]
    fn tr_split_j_legacy_zero_seed_fixed_boundaries() {
        let src = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj";
        let legacy = BlockSplitter::with_seed(src.to_owned(), BlockLevel::S, 0, 0)
            .peek_block().unwrap().end;
        let fixed  = BlockSplitter::with_cursor(src.to_owned(), BlockLevel::S, 0)
            .peek_block().unwrap().end;
        assert_eq!(legacy, fixed, "TR-SPLIT-J: seed=0 must match legacy with_cursor");
    }
}
