/// Phase 5: Segmentation with path competition.
///
/// Given a flat sequence of UnitIds, scan left-to-right and greedily merge
/// adjacent pairs into Chunks where the chunk's score beats keeping them
/// separate.  Multiple passes may be run to allow recursive merging.
///
/// Score formula (§C §7): score = confidence + strength_norm - processing_cost
use crate::chunks::{Chunk, ChunkRegistry};
use crate::primitives::PrimitiveRegistry;
use crate::units::UnitId;

/// Normalisation ceiling for chunk strength (tunable).
const STRENGTH_NORM_CAP: f64 = 100.0;
/// Cost deducted from score for each decision (one use = one decision saved).
const PROCESSING_COST: f64 = 0.1;

/// Compute the competition score for a candidate Chunk.
/// Higher score → prefer merging.
pub fn chunk_score(chunk: &Chunk) -> f64 {
    let confidence = chunk.accuracy();
    let strength_norm = (chunk.strength / STRENGTH_NORM_CAP).min(1.0);
    confidence + strength_norm - PROCESSING_COST
}

/// Segment a sequence of Primitives into Units using registered Chunks.
///
/// `min_tier` controls which chunks are considered: only chunks whose
/// `tier >= min_tier` are candidates.  Use `Tier::T1` for normal operation;
/// `Tier::T0` to include exploratory chunks during training.
pub fn segment(
    primitives: &[u32], // PrimitiveId slice
    chunks: &ChunkRegistry,
    min_score: f64,
) -> Vec<UnitId> {
    // Start: every primitive is its own unit
    let mut units: Vec<UnitId> = primitives
        .iter()
        .map(|&p| UnitId::primitive(p))
        .collect();

    // Repeatedly scan for the highest-scoring adjacent pair to merge.
    // Stop when no beneficial merge remains.
    loop {
        let mut best_pos: Option<usize> = None;
        let mut best_score = min_score; // must beat threshold

        for i in 0..units.len().saturating_sub(1) {
            if let Some(chunk_id) = chunks.find_by_pair(units[i], units[i + 1]) {
                let chunk = chunks.get(chunk_id).unwrap();
                let score = chunk_score(chunk);
                if score > best_score {
                    best_score = score;
                    best_pos = Some(i);
                }
            }
        }

        match best_pos {
            None => break,
            Some(i) => {
                let chunk_id = chunks.find_by_pair(units[i], units[i + 1]).unwrap();
                // Replace units[i] and units[i+1] with the chunk unit
                units[i] = UnitId::chunk(chunk_id);
                units.remove(i + 1);
            }
        }
    }

    units
}

/// Expand a UnitId sequence back to a sequence of PrimitiveIds.
/// Used for verification (round-trip) and generation output.
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
        // Force T2: 64 successes at 100%
        for tick in 0..64 {
            chunk.record_success(tick);
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
        // units[0] should be the chunk, units[1] the remaining prim
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], UnitId::chunk(cid));
        assert_eq!(result[1], UnitId::primitive(3));
    }

    #[test]
    fn test_recursive_merge() {
        let mut chunks = ChunkRegistry::new();
        // ab chunk
        let ab = make_promoted_chunk(&mut chunks, UnitId::primitive(1), UnitId::primitive(2));
        // (ab)c chunk
        let abc = make_promoted_chunk(&mut chunks, UnitId::chunk(ab), UnitId::primitive(3));
        let prims = vec![1u32, 2, 3];
        let result = segment(&prims, &chunks, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], UnitId::chunk(abc));
    }

    #[test]
    fn test_low_score_chunk_not_merged() {
        let mut chunks = ChunkRegistry::new();
        // Create a chunk but record only 1 success out of 100 (very low score)
        let id = chunks.get_or_create(UnitId::primitive(1), UnitId::primitive(2), 2);
        {
            let chunk = chunks.get_mut(id).unwrap();
            chunk.record_success(0);
            for tick in 1..100 {
                chunk.record_failure(tick);
            }
        }
        let prims = vec![1u32, 2];
        // Use a high threshold so the low-score chunk is rejected
        let result = segment(&prims, &chunks, 0.5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_expand_roundtrip() {
        let mut prim_reg = PrimitiveRegistry::new();
        let ids = prim_reg.encode("abc");
        let mut chunks = ChunkRegistry::new();
        // No chunks — expand should return original primitives
        let units: Vec<UnitId> = ids.iter().map(|&p| UnitId::primitive(p)).collect();
        let expanded = expand(&units, &chunks, &prim_reg);
        assert_eq!(expanded, ids);
        // Now create chunk(a,b) and expand
        let _ab = make_promoted_chunk(&mut chunks, UnitId::primitive(ids[0]), UnitId::primitive(ids[1]));
        let segmented = segment(&ids, &chunks, 0.0);
        let re_expanded = expand(&segmented, &chunks, &prim_reg);
        assert_eq!(re_expanded, ids);
    }
}
