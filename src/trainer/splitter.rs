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
}

impl BlockSplitter {
    pub fn new(source: String, level: BlockLevel) -> Self {
        Self { source, cursor: 0, level }
    }

    /// Restore from a saved cursor (used on Resume).
    pub fn with_cursor(source: String, level: BlockLevel, cursor: usize) -> Self {
        Self { source, cursor, level }
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
        let end_off = find_block_end(remaining, self.level.target_lines(), self.level.max_chars());
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
        let end_off = find_block_end(remaining, self.level.target_lines(), self.level.max_chars());
        let end = start + end_off;
        Some(Block { text: self.source[start..end].to_owned(), start, end })
    }
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
}
