/// Structural Lineage — Chunk formation tree.
///
/// Records how each Chunk was formed from two parent UnitIds.
/// This is STRUCTURAL lineage (composition tree), NOT Representation Lineage.
///
/// Distinction (per spec §9 / §10):
///   Structural Lineage: Chunk ABC → left=AB, right=C (§9)
///   Representation Lineage: Content X had representations R0→R1→R2 over time (§10, representation.rs)
///
/// Derivation edge: child_chunk → (left, right).
/// Used by Fallback (Phase 4) to decompose chunks back to their components
/// when a higher-granularity route fails.
use std::collections::HashMap;

use crate::units::{ChunkId, UnitId};

/// Records how each Chunk was derived from its component UnitIds.
#[derive(Debug, Default, Clone)]
pub struct LineageStore {
    /// child ChunkId → (left parent UnitId, right parent UnitId)
    derived_from: HashMap<ChunkId, (UnitId, UnitId)>,
}

impl LineageStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `child` was created by merging `left` and `right`.
    /// Silently no-ops if the child is already registered (idempotent).
    pub fn record(&mut self, child: ChunkId, left: UnitId, right: UnitId) {
        self.derived_from.entry(child).or_insert((left, right));
    }

    /// Return the immediate parent pair for a Chunk, if recorded.
    pub fn parents_of(&self, child: ChunkId) -> Option<(UnitId, UnitId)> {
        self.derived_from.get(&child).copied()
    }

    /// Decompose `unit` by one level of lineage.
    ///
    /// - Primitive → `[unit]` (already atomic)
    /// - Chunk with recorded parents → `[left, right]`
    /// - Chunk without recorded parents (should not happen in normal operation)
    ///   → `[unit]` (safe fallback; Fallback will treat it as opaque)
    pub fn decompose_one(&self, unit: UnitId) -> Vec<UnitId> {
        if let Some(cid) = unit.as_chunk() {
            if let Some((l, r)) = self.derived_from.get(&cid) {
                return vec![*l, *r];
            }
        }
        vec![unit]
    }

    /// Fully decompose `unit` down to Primitives by repeatedly following lineage.
    pub fn decompose_to_primitives(&self, unit: UnitId) -> Vec<UnitId> {
        let mut stack = vec![unit];
        let mut out = Vec::new();
        while let Some(u) = stack.pop() {
            match u.as_chunk().and_then(|cid| self.derived_from.get(&cid)) {
                Some(&(l, r)) => {
                    // push right first so left is processed first (stack is LIFO)
                    stack.push(r);
                    stack.push(l);
                }
                None => out.push(u),
            }
        }
        out
    }

    pub fn entry_count(&self) -> usize {
        self.derived_from.len()
    }

    // ── Bulk access for persistence ───────────────────────────────────────

    pub fn all_entries(&self) -> impl Iterator<Item = (ChunkId, UnitId, UnitId)> + '_ {
        self.derived_from.iter().map(|(&child, &(l, r))| (child, l, r))
    }

    pub fn from_bulk(entries: Vec<(ChunkId, UnitId, UnitId)>) -> Self {
        let derived_from = entries.into_iter().map(|(c, l, r)| (c, (l, r))).collect();
        Self { derived_from }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(id: u32) -> UnitId { UnitId::primitive(id) }
    fn c(id: u32) -> UnitId { UnitId::chunk(id) }

    #[test]
    fn record_and_query() {
        let mut ls = LineageStore::new();
        ls.record(0, p(1), p(2));
        assert_eq!(ls.parents_of(0), Some((p(1), p(2))));
        assert_eq!(ls.parents_of(1), None);
    }

    #[test]
    fn decompose_one_primitive() {
        let ls = LineageStore::new();
        assert_eq!(ls.decompose_one(p(5)), vec![p(5)]);
    }

    #[test]
    fn decompose_one_chunk() {
        let mut ls = LineageStore::new();
        ls.record(0, p(1), p(2));
        assert_eq!(ls.decompose_one(c(0)), vec![p(1), p(2)]);
    }

    #[test]
    fn decompose_one_unknown_chunk_returns_itself() {
        let ls = LineageStore::new();
        assert_eq!(ls.decompose_one(c(99)), vec![c(99)]);
    }

    #[test]
    fn decompose_to_primitives_nested() {
        // C(2) = C(0) + P(3), C(0) = P(1) + P(2)
        let mut ls = LineageStore::new();
        ls.record(0, p(1), p(2));
        ls.record(2, c(0), p(3));
        let result = ls.decompose_to_primitives(c(2));
        assert_eq!(result, vec![p(1), p(2), p(3)]);
    }

    #[test]
    fn decompose_to_primitives_already_primitive() {
        let ls = LineageStore::new();
        assert_eq!(ls.decompose_to_primitives(p(7)), vec![p(7)]);
    }

    #[test]
    fn record_is_idempotent() {
        let mut ls = LineageStore::new();
        ls.record(0, p(1), p(2));
        ls.record(0, p(9), p(9));  // second call ignored
        assert_eq!(ls.parents_of(0), Some((p(1), p(2))));
        assert_eq!(ls.entry_count(), 1);
    }

    #[test]
    fn from_bulk_roundtrip() {
        let mut orig = LineageStore::new();
        orig.record(0, p(1), p(2));
        orig.record(1, c(0), p(3));
        let entries: Vec<_> = orig.all_entries().collect();
        let restored = LineageStore::from_bulk(entries);
        assert_eq!(restored.parents_of(0), orig.parents_of(0));
        assert_eq!(restored.parents_of(1), orig.parents_of(1));
        assert_eq!(restored.entry_count(), orig.entry_count());
    }
}
