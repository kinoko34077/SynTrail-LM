/// Phase 5: Segmentation with path competition.
/// Phase 6: Priority-queue + local re-scoring → O(L log L) instead of O(L²).
///
/// Score formula (§C §7, v0.2):
///   score = confidence + strength_norm - processing_cost
///   confidence = if use_count > 0 { 1.0 } else { 0.0 }
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::chunks::{Chunk, ChunkRegistry};
use crate::primitives::PrimitiveRegistry;
use crate::units::{ChunkId, UnitId};

const STRENGTH_NORM_CAP: f64 = 100.0;
const PROCESSING_COST: f64 = 0.1;

/// Compute the competition score for a candidate Chunk.
pub fn chunk_score(chunk: &Chunk) -> f64 {
    let confidence = chunk.confidence();
    let strength_norm = (chunk.usage_strength / STRENGTH_NORM_CAP).min(1.0);
    confidence + strength_norm - PROCESSING_COST
}

/// Max-heap entry for a merge candidate.
/// Validity is checked lazily: if `slots[left]` or `slots[right]` have been
/// replaced or deleted, the entry is stale and skipped on pop.
#[derive(PartialEq)]
struct MergeEntry {
    score: f64,
    left: usize,   // index into slots[]
    right: usize,  // index into slots[]
    chunk_id: ChunkId,
    expected_left: UnitId,
    expected_right: UnitId,
}

impl Eq for MergeEntry {}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

/// Segment a sequence of Primitives into Units using registered Chunks.
///
/// Phase 6: O(L log L) via priority queue + doubly-linked slot array.
/// Each merge only re-evaluates the two new neighbors of the merged slot.
pub fn segment(
    primitives: &[u32],
    chunks: &ChunkRegistry,
    min_score: f64,
) -> Vec<UnitId> {
    let n = primitives.len();
    if n == 0 { return Vec::new(); }

    // Slot array: slots[i] = current UnitId at that position (None = merged away).
    let mut slots: Vec<Option<UnitId>> = primitives
        .iter()
        .map(|&p| Some(UnitId::primitive(p)))
        .collect();

    // Doubly-linked list over active slots.
    // next[i] = next active slot index (n = sentinel "end").
    // prev[i] = prev active slot index (n = sentinel "start").
    let mut next: Vec<usize> = (1..=n).collect(); // next[n-1] = n
    let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { n } else { i - 1 }).collect();

    // Helper: push a candidate for slots[left] + slots[right] onto the heap.
    let mut heap: BinaryHeap<MergeEntry> = BinaryHeap::new();
    let mut push_candidate = |heap: &mut BinaryHeap<MergeEntry>,
                               left: usize, right: usize,
                               lu: UnitId, ru: UnitId| {
        if right >= n { return; }
        if let Some(cid) = chunks.find_by_pair(lu, ru) {
            if let Some(chunk) = chunks.get(cid) {
                if !chunk.is_hot() { return; }
                let score = chunk_score(chunk);
                if score > min_score {
                    heap.push(MergeEntry { score, left, right, chunk_id: cid,
                                           expected_left: lu, expected_right: ru });
                }
            }
        }
    };

    // Seed the heap with all adjacent pairs.
    let mut i = 0;
    while next[i] < n {
        let j = next[i];
        let lu = slots[i].unwrap();
        let ru = slots[j].unwrap();
        push_candidate(&mut heap, i, j, lu, ru);
        i = j;
    }

    // Drain heap: each pop either merges or is lazily discarded.
    while let Some(entry) = heap.pop() {
        let MergeEntry { left, right, chunk_id, expected_left, expected_right, .. } = entry;
        // Validate: slots must still hold the expected UnitIds.
        if slots[left] != Some(expected_left) || slots[right] != Some(expected_right) {
            continue; // stale entry
        }
        // Perform the merge: replace slot[left] with the chunk, delete slot[right].
        let merged = UnitId::chunk(chunk_id);
        slots[left] = Some(merged);
        slots[right] = None;
        // Relink: left → next[right], and next[right].prev = left.
        let new_right = next[right];
        next[left] = new_right;
        if new_right < n { prev[new_right] = left; }

        // Re-evaluate the two new neighbors.
        if prev[left] < n {
            let p = prev[left];
            let pu = slots[p].unwrap();
            push_candidate(&mut heap, p, left, pu, merged);
        }
        if new_right < n {
            let ru = slots[new_right].unwrap();
            push_candidate(&mut heap, left, new_right, merged, ru);
        }
    }

    // Collect surviving slots in order via the linked list.
    let mut result = Vec::new();
    let mut cur = 0;
    loop {
        if let Some(u) = slots[cur] {
            result.push(u);
        }
        let nx = next[cur];
        if nx >= n { break; }
        cur = nx;
    }
    result
}

/// Expand a UnitId sequence back to a sequence of PrimitiveIds.
pub fn expand(
    units: &[UnitId],
    chunks: &ChunkRegistry,
    primitives: &PrimitiveRegistry,
) -> Vec<u32> {
    let mut result = Vec::new();
    expand_into(units, chunks, primitives, &mut result);
    result
}

fn expand_into(
    units: &[UnitId],
    chunks: &ChunkRegistry,
    primitives: &PrimitiveRegistry,
    out: &mut Vec<u32>,
) {
    for &unit in units {
        if let Some(prim_id) = unit.as_primitive() {
            out.push(prim_id);
        } else if let Some(chunk_id) = unit.as_chunk() {
            if let Some(chunk) = chunks.get(chunk_id) {
                expand_into(&[chunk.left, chunk.right], chunks, primitives, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunks::ChunkRegistry;
    use crate::primitives::PrimitiveRegistry;
    use crate::units::UnitId;

    fn make_promoted_chunk(
        chunks: &mut ChunkRegistry,
        left: UnitId,
        right: UnitId,
    ) -> u32 {
        let id = chunks.get_or_create(left, right, 2);
        let chunk = chunks.get_mut(id).unwrap();
        for tick in 0..64 {
            chunk.record_usage(tick);
        }
        id
    }

    #[test]
    fn test_no_chunks_returns_primitives() {
        let chunks = ChunkRegistry::new();
        let prims = vec![1u32, 2, 3];
        let result = segment(&prims, &chunks, 0.0);
        assert_eq!(result, prims.iter().map(|&p| UnitId::primitive(p)).collect::<Vec<_>>());
    }

    #[test]
    fn test_single_merge() {
        let mut chunks = ChunkRegistry::new();
        let cid = make_promoted_chunk(&mut chunks, UnitId::primitive(1), UnitId::primitive(2));
        let prims = vec![1u32, 2, 3];
        let result = segment(&prims, &chunks, 0.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], UnitId::chunk(cid));
        assert_eq!(result[1], UnitId::primitive(3));
    }

    #[test]
    fn test_recursive_merge() {
        let mut chunks = ChunkRegistry::new();
        let ab = make_promoted_chunk(&mut chunks, UnitId::primitive(1), UnitId::primitive(2));
        let abc = make_promoted_chunk(&mut chunks, UnitId::chunk(ab), UnitId::primitive(3));
        let prims = vec![1u32, 2, 3];
        let result = segment(&prims, &chunks, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], UnitId::chunk(abc));
    }

    #[test]
    fn test_unused_chunk_not_merged() {
        let mut chunks = ChunkRegistry::new();
        // Freshly created chunk with no uses → confidence=0.0, score=-0.1
        let _id = chunks.get_or_create(UnitId::primitive(1), UnitId::primitive(2), 2);
        let prims = vec![1u32, 2];
        // min_score=0.0 means we skip anything ≤ 0.0; score=-0.1 is below threshold
        let result = segment(&prims, &chunks, 0.0);
        assert_eq!(result.len(), 2, "unused chunk should not be merged");
    }

    #[test]
    fn test_expand_roundtrip() {
        let mut prim_reg = PrimitiveRegistry::new();
        let ids = prim_reg.encode("abc");
        let mut chunks = ChunkRegistry::new();
        let units: Vec<UnitId> = ids.iter().map(|&p| UnitId::primitive(p)).collect();
        let expanded = expand(&units, &chunks, &prim_reg);
        assert_eq!(expanded, ids);
        let _ab = make_promoted_chunk(&mut chunks, UnitId::primitive(ids[0]), UnitId::primitive(ids[1]));
        let segmented = segment(&ids, &chunks, 0.0);
        let re_expanded = expand(&segmented, &chunks, &prim_reg);
        assert_eq!(re_expanded, ids);
    }
}
