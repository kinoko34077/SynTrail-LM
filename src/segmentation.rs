/// Phase 5: Segmentation with path competition.
///
/// Score formula (§C §7, v0.2):
///   score = confidence + strength_norm - processing_cost
///   confidence = if use_count > 0 { 1.0 } else { 0.0 }
use crate::chunks::{Chunk, ChunkRegistry};
use crate::primitives::PrimitiveRegistry;
use crate::units::UnitId;

const STRENGTH_NORM_CAP: f64 = 100.0;
const PROCESSING_COST: f64 = 0.1;

/// Compute the competition score for a candidate Chunk.
pub fn chunk_score(chunk: &Chunk) -> f64 {
    let confidence = chunk.confidence();
    let strength_norm = (chunk.usage_strength / STRENGTH_NORM_CAP).min(1.0);
    confidence + strength_norm - PROCESSING_COST
}

/// Segment a sequence of Primitives into Units using registered Chunks.
pub fn segment(
    primitives: &[u32],
    chunks: &ChunkRegistry,
    min_score: f64,
) -> Vec<UnitId> {
    let mut units: Vec<UnitId> = primitives
        .iter()
        .map(|&p| UnitId::primitive(p))
        .collect();

    loop {
        let mut best_pos: Option<usize> = None;
        let mut best_score = min_score;

        for i in 0..units.len().saturating_sub(1) {
            if let Some(chunk_id) = chunks.find_by_pair(units[i], units[i + 1]) {
                let chunk = chunks.get(chunk_id).unwrap();
                // SLEEP chunks are in the structural registry but excluded from segmentation (§10)
                if !chunk.is_hot() { continue; }
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
                units[i] = UnitId::chunk(chunk_id);
                units.remove(i + 1);
            }
        }
    }

    units
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
