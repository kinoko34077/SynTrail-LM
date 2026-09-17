/// A Primitive is the indivisible base unit: one Unicode scalar value.
/// Primitives are never deleted once registered.
use std::collections::HashMap;

/// Stable numeric ID assigned to each Primitive on first encounter.
/// IDs are 1-based; 0 is reserved as "unassigned / null".
pub type PrimitiveId = u32;

/// Registry that maps Unicode scalar values to their stable IDs.
/// All mutation goes through `register`; the registry only grows.
#[derive(Debug, Default)]
pub struct PrimitiveRegistry {
    /// scalar → id
    scalar_to_id: HashMap<char, PrimitiveId>,
    /// id → scalar  (indexed by id, slot 0 unused)
    id_to_scalar: Vec<char>,
}

impl PrimitiveRegistry {
    pub fn new() -> Self {
        Self {
            scalar_to_id: HashMap::new(),
            // slot 0 is the "null" placeholder
            id_to_scalar: vec!['\0'],
        }
    }

    /// Return the ID for `c`, registering it if this is the first encounter.
    pub fn register(&mut self, c: char) -> PrimitiveId {
        if let Some(&id) = self.scalar_to_id.get(&c) {
            return id;
        }
        let id = self.id_to_scalar.len() as PrimitiveId;
        self.scalar_to_id.insert(c, id);
        self.id_to_scalar.push(c);
        id
    }

    /// Look up by ID.  Returns `None` for id == 0 or unknown ids.
    pub fn scalar(&self, id: PrimitiveId) -> Option<char> {
        if id == 0 {
            return None;
        }
        self.id_to_scalar.get(id as usize).copied()
    }

    /// Look up by scalar.
    pub fn id(&self, c: char) -> Option<PrimitiveId> {
        self.scalar_to_id.get(&c).copied()
    }

    /// Total number of registered primitives (not counting the null slot).
    pub fn len(&self) -> usize {
        self.id_to_scalar.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Encode a string into a sequence of Primitive IDs, registering any
    /// new scalars encountered along the way.
    pub fn encode(&mut self, text: &str) -> Vec<PrimitiveId> {
        text.chars().map(|c| self.register(c)).collect()
    }

    /// Decode a sequence of Primitive IDs back to a String.
    /// Returns `None` if any ID is unregistered.
    pub fn decode(&self, ids: &[PrimitiveId]) -> Option<String> {
        ids.iter()
            .map(|&id| self.scalar(id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(text: &str) {
        let mut reg = PrimitiveRegistry::new();
        let ids = reg.encode(text);
        let recovered = reg.decode(&ids).expect("decode failed");
        assert_eq!(text, recovered, "roundtrip failed for: {text:?}");
    }

    #[test]
    fn test_ascii() {
        roundtrip("Hello, world!");
    }

    #[test]
    fn test_hiragana() {
        roundtrip("こんにちは世界");
    }

    #[test]
    fn test_katakana() {
        roundtrip("コンニチハセカイ");
    }

    #[test]
    fn test_kanji() {
        roundtrip("日本語処理");
    }

    #[test]
    fn test_non_bmp() {
        // U+1F600 GRINNING FACE (non-BMP, requires surrogate pair in UTF-16)
        roundtrip("😀🎉🦀");
    }

    #[test]
    fn test_mixed() {
        roundtrip("Hello, 世界! 🌏 αβγ €£¥");
    }

    #[test]
    fn test_empty() {
        roundtrip("");
    }

    #[test]
    fn test_id_stability() {
        let mut reg = PrimitiveRegistry::new();
        let id_a = reg.register('A');
        let id_b = reg.register('B');
        // re-registering must return the same id
        assert_eq!(reg.register('A'), id_a);
        assert_eq!(reg.register('B'), id_b);
        // ids must be distinct
        assert_ne!(id_a, id_b);
        // id 0 is reserved
        assert_ne!(id_a, 0);
        assert_ne!(id_b, 0);
    }

    #[test]
    fn test_registry_grows_only() {
        let mut reg = PrimitiveRegistry::new();
        reg.encode("あいう");
        let len_before = reg.len();
        // re-encoding same text must not grow the registry
        reg.encode("あいう");
        assert_eq!(reg.len(), len_before);
        // new char grows it by exactly 1
        reg.encode("え");
        assert_eq!(reg.len(), len_before + 1);
    }

    #[test]
    fn test_null_id() {
        let reg = PrimitiveRegistry::new();
        assert!(reg.scalar(0).is_none());
    }
}
