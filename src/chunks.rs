use std::collections::HashMap;
use crate::units::{ChunkId, UnitId};
use crate::tier::Tier;

/// Decay factor for usage_strength update: new = DECAY * old + reward  (§B §5)
pub const STRENGTH_DECAY: f64 = 0.99;
/// Reward for an exposure event.
pub const STRENGTH_REWARD: f64 = 1.0;

/// v0.4 Phase 5: Activity state (§10, §11).
///
/// HOT  — active memory; participates in segmentation and recall.
/// SLEEP — structural registry only; excluded from segmentation until reactivated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    Hot,
    Sleep,
}

impl Default for Residency {
    fn default() -> Self { Residency::Hot }
}

/// A Chunk is a binary composition of two Units.
/// Once created, left/right are the Immutable Chunk Core (§11).
/// All other fields form the Active Overlay and can be zeroed on SLEEP.
///
/// v0.2: separated usage (exposure) from evaluation (feedback).
/// v0.4: residency field controls whether chunk is active or dormant.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: ChunkId,
    /// Immutable Core: structural identity.
    pub left: UnitId,
    /// Immutable Core: structural identity.
    pub right: UnitId,
    pub tier: Tier,
    /// Times this chunk appeared in a segmented sequence (usage / exposure).
    pub use_count: u32,
    /// Decaying sum of usage events: new = DECAY * old + reward
    pub usage_strength: f64,
    /// Logical clock tick of last use (set by caller).
    pub last_used: u64,
    /// Number of Primitives when fully expanded (cached).
    pub expanded_length: u32,
    /// Accumulated feedback signal: V_new = decay * V_old + r
    pub feedback_value: f64,
    /// Number of feedback events applied.
    pub feedback_count: u32,
    /// v0.4: HOT = active; SLEEP = structural only, excluded from segmentation.
    pub residency: Residency,
}

impl Chunk {
    /// Record one usage (exposure) of this chunk.
    /// Updates use_count, usage_strength, and tier.
    pub fn record_usage(&mut self, tick: u64) {
        self.use_count += 1;
        self.usage_strength = STRENGTH_DECAY * self.usage_strength + STRENGTH_REWARD;
        self.last_used = tick;
        self.tier = self.tier.maybe_promote(self.use_count, self.use_count);
    }

    /// Apply one feedback credit r to this chunk.
    /// V_new = decay * V_old + r
    pub fn apply_feedback(&mut self, r: f64, decay: f64) {
        self.feedback_value = decay * self.feedback_value + r;
        self.feedback_count += 1;
    }

    /// Binary confidence: 1.0 if chunk has been used at all, else 0.0.
    pub fn confidence(&self) -> f64 {
        if self.use_count > 0 { 1.0 } else { 0.0 }
    }

    pub fn is_hot(&self) -> bool { self.residency == Residency::Hot }
    pub fn is_sleep(&self) -> bool { self.residency == Residency::Sleep }

    /// Move to SLEEP — excluded from active segmentation (§10).
    /// Core structure (left, right) is preserved.
    pub fn demote_to_sleep(&mut self) {
        self.residency = Residency::Sleep;
    }

    /// Restore to HOT — re-enters active segmentation (§12).
    pub fn promote_to_hot(&mut self) {
        self.residency = Residency::Hot;
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
            use_count: 0,
            usage_strength: 0.0,
            last_used: 0,
            expanded_length,
            feedback_value: 0.0,
            feedback_count: 0,
            residency: Residency::Hot,
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

    pub fn hot_count(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_hot()).count()
    }

    pub fn sleep_count(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_sleep()).count()
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
    fn test_record_usage_promotes() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        let chunk = reg.get_mut(id).unwrap();
        for tick in 0..16 {
            chunk.record_usage(tick);
        }
        assert_eq!(chunk.tier, Tier::T1);
        assert_eq!(chunk.use_count, 16);
    }

    #[test]
    fn test_usage_strength_increases() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        let chunk = reg.get_mut(id).unwrap();
        chunk.record_usage(0);
        assert!(chunk.usage_strength > 0.0);
        let s1 = chunk.usage_strength;
        chunk.record_usage(1);
        assert!(chunk.usage_strength > s1);
    }

    #[test]
    fn test_full_promotion_path() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(3), prim(4), 2);
        let chunk = reg.get_mut(id).unwrap();
        // T0 → T1: need 16 uses
        for tick in 0..16 { chunk.record_usage(tick); }
        assert_eq!(chunk.tier, Tier::T1);
        // T1 → T2: need 64 uses total
        for tick in 16..64 { chunk.record_usage(tick); }
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

    #[test]
    fn test_apply_feedback_accumulates() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        let chunk = reg.get_mut(id).unwrap();
        chunk.apply_feedback(1.0, 0.99);
        assert!((chunk.feedback_value - 1.0).abs() < 1e-9);
        chunk.apply_feedback(-0.5, 0.99);
        let expected = 0.99 * 1.0 + (-0.5);
        assert!((chunk.feedback_value - expected).abs() < 1e-9);
        assert_eq!(chunk.feedback_count, 2);
    }

    #[test]
    fn test_confidence_binary() {
        let mut reg = ChunkRegistry::new();
        let id = reg.get_or_create(prim(1), prim(2), 2);
        let chunk = reg.get_mut(id).unwrap();
        assert_eq!(chunk.confidence(), 0.0, "no uses yet");
        chunk.record_usage(0);
        assert_eq!(chunk.confidence(), 1.0, "after first use");
    }
}
