use crate::primitives::PrimitiveId;

/// Opaque ID for a Chunk (distinct namespace from PrimitiveId).
pub type ChunkId = u32;

/// A compact tagged reference that can point to either a Primitive or a Chunk.
/// Bit 31 == 0 → Primitive; bit 31 == 1 → Chunk.
/// Raw index fits in bits 0..30, giving ~2 billion of each kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnitId(u32);

const CHUNK_FLAG: u32 = 1 << 31;

impl UnitId {
    pub fn primitive(id: PrimitiveId) -> Self {
        debug_assert!(id & CHUNK_FLAG == 0, "PrimitiveId too large");
        UnitId(id)
    }

    pub fn chunk(id: ChunkId) -> Self {
        debug_assert!(id & CHUNK_FLAG == 0, "ChunkId too large");
        UnitId(id | CHUNK_FLAG)
    }

    pub fn is_primitive(self) -> bool { self.0 & CHUNK_FLAG == 0 }
    pub fn is_chunk(self) -> bool { self.0 & CHUNK_FLAG != 0 }

    /// Raw numeric index (type bit stripped).
    pub fn raw(self) -> u32 { self.0 & !CHUNK_FLAG }

    pub fn as_primitive(self) -> Option<PrimitiveId> {
        self.is_primitive().then_some(self.0)
    }

    pub fn as_chunk(self) -> Option<ChunkId> {
        self.is_chunk().then_some(self.raw())
    }
}
