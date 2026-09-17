use std::collections::HashMap;
use crate::units::{ChunkId, UnitId};
use crate::tier::Tier;

/// Decay factor for strength update: new = DECAY * old + reward  (§B §5)
pub const STRENGTH_DECAY: f64 = 0.99;
/// Reward for a successful prediction/use.
pub const STRENGTH_REWARD: f64 = 1.0;

/// A Chunk is a binary composition of two Units.
/// Once created, left/right are immutable; other fields evolve.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: ChunkId,
    pub left: UnitId,
    pub right: UnitId,
    pub tier: Tier,
    pub success_count: u32,
    pub total_count: u32,
    pub strength: f64,
    /// Logical clock tick of last use (set by caller).
    pub last_used: u64,
    /// Number of Primitives when fully expanded (cached).
    pub expanded_length: u32,
}

impl Chunk {
    /// Record a successful use: increment counts, update strength, evaluate tier.
    pub fn record_success(&mut self, tick: u64) {
        self.success_count += 1;
        self.total_count += 1;
        self.strength = STRENGTH_DECAY * self.strength + STRENGTH_REWARD;
        self.last_used = tick;
        self.tier = self.tier.maybe_promote(self.success_count, self.total_count);
    }

    /// Record a failed use: increment total only, update strength with 0 reward,
    /// evaluate tier demotion.
    pub fn record_failure(&mut self, tick: u64) {
        self.total_count += 1;
        self.strength = STRENGTH_DECAY * self.strength;
        self.last_used = tick;
        self.tier = self.tier.maybe_demote(self.success_count, self.total_count);
    }

    pub fn accuracy(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.total_count as f64
        }
    }
}

/// Central store for all Chunks.
/// Guarantees: each (left, right) pair maps to exactly one ChunkId.
#[derive(Debug, Default)]
pub struct ChunkRegistry {
    /// (left, right) → id  — deduplication key
    pair_to_id: HashMap<(UnitId, UnitId), ChunkId>,
    /// id → Chunk
    chunks: Vec<Chunk>,
}

impl ChunkRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the existing id for (left, right), or create a new Chunk.
    /// `expanded_length` must be provided by the caller
    /// (= left_expanded_len + right_expanded_len).
    pub fn get_or_create(
        &mut self,
        left: UnitId,
        right: UnitId,
        expanded_length: u32,
    ) -> ChunkId {
        if let Some(&id) = self.pair_to_id.get(&(left, right)) {
            return id;
        }
        let id = self.chunks.len() as ChunkId;
        let chunk = Chunk {
            id,
            left,
            right,
            tier: Tier::T0,
            success_count: 0,
            total_count: 0,
            strength: 0.0,
            last_used: 0,
            expanded_length,
        };
        self.pair_to_id.insert((left, right), id);
        self.chunks.push(chunk);
        id
    }

    pub fn get(&self, id: ChunkId) -> Option<&Chunk> {
        self.chunks.get(id as usize)
    }

    pub fn get_mut(&mut self, id: ChunkId) -> Option<&mut Chunk> {
        self.chunks.get_mut(id as usize)
    }

    pub fn find_by_pair(&self, left: UnitId, right: UnitId) -> Option<ChunkId> {
        self.pair_to_id.get(&(left, right)).copied()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Iterate all chunks for serialisation.
    pub fn iter_all(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::UnitId;

    fn prim(id: u32) -> UnitId { UnitId::primitive(id) }

    #[test]
    fn test_create_chunk() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        assert_eq!(id, 0);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_deduplication() {
        let mut reg = ChunkRegistry::new();
        let id1 = reg.get_or_create(prim(1), prim(2), 2);
        let id2 = reg.get_or_create(prim(1), prim(2), 2);
        assert_eq!(id1, id2);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_distinct_pairs() {
        let mut reg = ChunkRegistry::new();
        let id_ab = reg.get_or_create(prim(1), prim(2), 2);
        let id_ba = reg.get_or_create(prim(2), prim(1), 2);
        assert_ne!(id_ab, id_ba);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_initial_tier_is_t0() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        assert_eq!(reg.get(id).unwrap().tier, Tier::T0);
    }

    #[test]
    fn test_record_success_promotes() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        let chunk = reg.get_mut(id).unwrap();
        for tick in 0..16 {
            chunk.record_success(tick);
        }
        assert_eq!(chunk.tier, Tier::T1);
        assert_eq!(chunk.success_count, 16);
        assert_eq!(chunk.total_count, 16);
    }

    #[test]
    fn test_strength_increases_on_success() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        let chunk = reg.get_mut(id).unwrap();
        chunk.record_success(0);
        assert!(chunk.strength > 0.0);
        let s1 = chunk.strength;
        chunk.record_success(1);
        assert!(chunk.strength > s1);
    }

    #[test]
    fn test_strength_decays_on_failure() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        let chunk = reg.get_mut(id).unwrap();
        // build up some strength first
        for tick in 0..10 {
            chunk.record_success(tick);
        }
        let s_before = chunk.strength;
        chunk.record_failure(10);
        assert!(chunk.strength < s_before);
    }

    #[test]
    fn test_full_promotion_path() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(3), prim(4), 2);
        let chunk = reg.get_mut(id).unwrap();
        // T0 → T1: need 16 successes at ≥90%
        for tick in 0..16 { chunk.record_success(tick); }
        assert_eq!(chunk.tier, Tier::T1);
        // T1 → T2: need 64 successes total at ≥98% (already 16; need 48 more)
        for tick in 16..64 { chunk.record_success(tick); }
        assert_eq!(chunk.tier, Tier::T2);
    }

    #[test]
    fn test_expanded_length_stored() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 5);
        assert_eq!(reg.get(id).unwrap().expanded_length, 5);
    }

    #[test]
    fn test_find_by_pair() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(7), prim(8), 2);
        assert_eq!(reg.find_by_pair(prim(7), prim(8)), Some(id));
        assert_eq!(reg.find_by_pair(prim(8), prim(7)), None);
    }
}
