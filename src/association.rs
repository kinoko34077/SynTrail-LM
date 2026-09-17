/// Phase 6-9: Association Store — sparse Top-K co-occurrence memory (§7-§11).
///
/// Association captures "what tends to co-occur with X?" — distinct from Route
/// which captures "given X, what comes NEXT?". Associations are bidirectional
/// and window-based; Routes are sequential and directed.
///
/// Each UnitId holds at most `top_k` associations sorted by strength DESC.
/// Weakest edges are evicted when the budget is exceeded (§10).
use std::collections::HashMap;

use crate::chunks::ChunkRegistry;
use crate::units::UnitId;

pub const ASSOCIATION_DECAY: f64 = 0.99;
pub const ASSOCIATION_REWARD: f64 = 1.0;
pub const DEFAULT_TOP_K: usize = 32;

/// One directed association edge: source recalls target with `strength`.
#[derive(Debug, Clone)]
pub struct AssociationEdge {
    pub source: UnitId,
    pub target: UnitId,
    /// Decaying usage sum: S_new = decay * S_old + reward
    pub strength: f64,
    pub use_count: u32,
    pub last_used: u64,
}

/// Sparse association store: each Content keeps at most `top_k` targets.
///
/// Invariant: for every source, `edges[source].len() <= top_k`
/// and the vec is sorted by `strength` DESC.
#[derive(Debug, Clone)]
pub struct AssociationStore {
    /// source → sorted (desc by strength) association edges
    edges: HashMap<UnitId, Vec<AssociationEdge>>,
    pub top_k: usize,
    pub decay: f64,
}

impl Default for AssociationStore {
    fn default() -> Self {
        Self::new(DEFAULT_TOP_K, ASSOCIATION_DECAY)
    }
}

impl AssociationStore {
    pub fn new(top_k: usize, decay: f64) -> Self {
        Self { edges: HashMap::new(), top_k, decay }
    }

    /// Record one co-occurrence of (source, target).
    /// Updates strength, re-sorts, evicts weakest if over budget.
    pub fn observe(&mut self, source: UnitId, target: UnitId, tick: u64) {
        if source == target { return; }
        let decay = self.decay;
        let top_k = self.top_k;
        let list = self.edges.entry(source).or_default();

        if let Some(edge) = list.iter_mut().find(|e| e.target == target) {
            edge.strength = decay * edge.strength + ASSOCIATION_REWARD;
            edge.use_count += 1;
            edge.last_used = tick;
        } else {
            list.push(AssociationEdge {
                source,
                target,
                strength: ASSOCIATION_REWARD,
                use_count: 1,
                last_used: tick,
            });
        }

        // Sort DESC by strength
        list.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));

        // Evict weakest if over budget (§10)
        if list.len() > top_k {
            list.truncate(top_k);
        }
    }

    /// Direct Top-K associations for `source` (RC-01, RC-02, RC-04).
    pub fn top_for(&self, source: UnitId) -> &[AssociationEdge] {
        self.edges.get(&source).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Return up to `limit` associations for `source`.
    /// Falls back to children if source has none (RC-06, §8.2).
    pub fn recall(
        &self,
        source: UnitId,
        chunks: &ChunkRegistry,
        limit: usize,
    ) -> Vec<(UnitId, f64)> {
        self.recall_inner(source, chunks, limit, 0)
    }

    fn recall_inner(
        &self,
        source: UnitId,
        chunks: &ChunkRegistry,
        limit: usize,
        depth: usize,
    ) -> Vec<(UnitId, f64)> {
        // Guard against unbounded recursion on deep chunk trees
        if depth > 8 { return vec![]; }

        let direct = self.top_for(source);
        if !direct.is_empty() {
            return direct.iter().take(limit).map(|e| (e.target, e.strength)).collect();
        }

        // RC-06: fallback to children's associations
        if let Some(cid) = source.as_chunk() {
            if let Some(chunk) = chunks.get(cid) {
                let mut results: Vec<(UnitId, f64)> = Vec::new();
                for child in [chunk.left, chunk.right] {
                    for (target, strength) in self.recall_inner(child, chunks, limit, depth + 1) {
                        if let Some(existing) = results.iter_mut().find(|(t, _)| *t == target) {
                            if strength > existing.1 { existing.1 = strength; }
                        } else {
                            results.push((target, strength));
                        }
                    }
                }
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(limit);
                return results;
            }
        }

        vec![]
    }

    /// Total number of stored associations.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(Vec::len).sum()
    }

    /// Iterate all edges for serialisation.
    pub fn iter_all(&self) -> impl Iterator<Item = &AssociationEdge> {
        self.edges.values().flatten()
    }

    /// Rebuild from a flat list (deserialisation).
    pub fn from_edges(edges: Vec<AssociationEdge>, top_k: usize, decay: f64) -> Self {
        let mut store = Self::new(top_k, decay);
        for edge in edges {
            let list = store.edges.entry(edge.source).or_default();
            list.push(edge);
        }
        for list in store.edges.values_mut() {
            list.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
            list.truncate(top_k);
        }
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::UnitId;
    use crate::chunks::ChunkRegistry;

    fn p(id: u32) -> UnitId { UnitId::primitive(id) }

    #[test]
    fn rc01_multiple_associations_per_content() {
        let mut store = AssociationStore::new(8, 0.99);
        store.observe(p(1), p(2), 0);
        store.observe(p(1), p(3), 1);
        store.observe(p(1), p(4), 2);
        assert_eq!(store.top_for(p(1)).len(), 3, "RC-01: must hold multiple associations");
    }

    #[test]
    fn rc02_sorted_by_strength() {
        let mut store = AssociationStore::new(8, 0.99);
        store.observe(p(1), p(2), 0);
        for _ in 0..5 { store.observe(p(1), p(3), 1); }
        let top = store.top_for(p(1));
        assert_eq!(top[0].target, p(3), "RC-02: highest strength should come first");
    }

    #[test]
    fn rc03_top_k_budget_enforced() {
        let mut store = AssociationStore::new(3, 0.99);
        for i in 2..10 { store.observe(p(1), p(i), i as u64); }
        assert!(store.top_for(p(1)).len() <= 3, "RC-03: must not exceed top_k");
    }

    #[test]
    fn rc04_no_full_scan_needed() {
        let mut store = AssociationStore::new(8, 0.99);
        // Populate many sources
        for i in 0..100u32 { store.observe(p(i), p(i + 1), i as u64); }
        // Lookup is O(1) per source — just verify it works and gives correct result
        let top = store.top_for(p(5));
        assert!(!top.is_empty(), "RC-04: direct lookup should find p(6)");
        assert_eq!(top[0].target, p(6));
    }

    #[test]
    fn rc05_chunk_has_own_associations() {
        let mut store = AssociationStore::new(8, 0.99);
        let chunk_unit = UnitId::chunk(0);
        store.observe(chunk_unit, p(99), 0);
        assert_eq!(store.top_for(chunk_unit).len(), 1, "RC-05: chunk should hold its own associations");
        assert_eq!(store.top_for(chunk_unit)[0].target, p(99));
    }

    #[test]
    fn rc06_hierarchical_fallback() {
        let mut chunks = ChunkRegistry::new();
        let cid = chunks.get_or_create(p(1), p(2), 2);
        let chunk_unit = UnitId::chunk(cid);

        let mut store = AssociationStore::new(8, 0.99);
        // Chunk itself has no associations, but its children do
        store.observe(p(1), p(99), 0);
        store.observe(p(2), p(88), 0);

        let results = store.recall(chunk_unit, &chunks, 4);
        assert!(!results.is_empty(), "RC-06: should fallback to children's associations");
        let targets: Vec<UnitId> = results.iter().map(|(t, _)| *t).collect();
        assert!(targets.contains(&p(99)) || targets.contains(&p(88)));
    }

    #[test]
    fn test_bidirectional_strength_update() {
        let mut store = AssociationStore::new(8, 0.99);
        store.observe(p(1), p(2), 0);
        store.observe(p(1), p(2), 1);
        let edge = &store.top_for(p(1))[0];
        let expected = 0.99 * 1.0 + 1.0;
        assert!((edge.strength - expected).abs() < 1e-9);
        assert_eq!(edge.use_count, 2);
    }

    #[test]
    fn test_self_association_ignored() {
        let mut store = AssociationStore::new(8, 0.99);
        store.observe(p(1), p(1), 0);
        assert!(store.top_for(p(1)).is_empty(), "self-association should be ignored");
    }
}
