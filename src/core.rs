/// Phase 14: Physical Core / Overlay separation.
///
/// The **Core** consists of Primitives and HOT Chunks only.  Core operations
/// (segmentation, next-unit prediction) run entirely within the Core and never
/// touch SLEEP chunks.  This was already partially true after Phase 8
/// (hot_pair_to_id / find_by_pair_hot) — Phase 14 makes the boundary explicit
/// with a dedicated `CoreView` type and labelled API methods.
///
/// The **Overlay** is everything else: SLEEP chunks, Association edges,
/// Transform edges, Identity/View tables.  Overlay operations (recall,
/// association lookup, cross-view resolution) may touch any part of the model.
///
/// Concretely:
///   CoreView borrows ChunkRegistry + PredictionStore read-only and exposes:
///     - segment_core()  — HOT-only segmentation (delegates to segmentation::segment)
///     - predict_core()  — top-1 prediction for a context unit
///   ModelState gains core_view() to vend the borrow, plus is_core_unit().
use crate::chunks::ChunkRegistry;
use crate::prediction::PredictionStore;
use crate::segmentation::segment;
use crate::units::UnitId;

/// Read-only view over the Core (Primitives + HOT Chunks + Predictions).
///
/// Obtained via `ModelState::core_view()`.  Holds shared borrows so it is
/// cheap to create and intentionally prevents mutable overlay operations while
/// the Core is being inspected.
pub struct CoreView<'a> {
    pub(crate) chunks: &'a ChunkRegistry,
    pub(crate) predictions: &'a PredictionStore,
}

impl<'a> CoreView<'a> {
    /// Segment a Primitive-id sequence using HOT Chunks only.
    ///
    /// SLEEP chunks are physically absent from `hot_pair_to_id` (Phase 8),
    /// so `segment()` naturally uses only HOT chunks here.
    pub fn segment_core(&self, prims: &[u32]) -> Vec<UnitId> {
        segment(prims, self.chunks, 0.0)
    }

    /// Top-1 prediction for `context` using the Core prediction store.
    ///
    /// Returns `None` if no edge exists for `context`.
    pub fn predict_core(&self, context: UnitId) -> Option<(UnitId, f64)> {
        self.predictions.top1_with_score(context)
    }

    /// Return `true` if `unit` is a HOT chunk or a primitive (i.e., lives in
    /// the Core).
    pub fn is_core_unit(&self, unit: UnitId) -> bool {
        if unit.is_primitive() {
            return true;
        }
        // A chunk is in Core iff it is HOT (find_by_pair_hot would succeed).
        // The cheapest check: ask the registry for its residency.
        use crate::chunks::Residency;
        self.chunks
            .get(unit.raw())
            .map(|c| c.residency == Residency::Hot)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunks::ChunkRegistry;
    use crate::prediction::PredictionStore;
    use crate::units::UnitId;

    fn p(id: u32) -> UnitId { UnitId::primitive(id) }

    #[test]
    fn co01_segment_core_primitives_only() {
        let chunks = ChunkRegistry::new();
        let preds  = PredictionStore::new();
        let view = CoreView { chunks: &chunks, predictions: &preds };
        let prims = vec![1u32, 2, 3];
        let units = view.segment_core(&prims);
        assert_eq!(units, vec![p(1), p(2), p(3)]);
    }

    fn make_hot_chunk(chunks: &mut ChunkRegistry, left: UnitId, right: UnitId) -> u32 {
        let cid = chunks.get_or_create(left, right, 2);
        // Give the chunk enough usage so its score > 0.0.
        chunks.get_mut(cid).unwrap().record_usage(1);
        cid
    }

    #[test]
    fn co02_segment_core_uses_hot_chunks() {
        let mut chunks = ChunkRegistry::new();
        let preds  = PredictionStore::new();
        make_hot_chunk(&mut chunks, p(1), p(2));
        let view = CoreView { chunks: &chunks, predictions: &preds };
        let units = view.segment_core(&[1, 2, 3]);
        // AB should be merged into one chunk unit.
        assert_eq!(units.len(), 2, "AB merged + primitive(3)");
        assert!(units[0].is_chunk());
        assert_eq!(units[1], p(3));
    }

    #[test]
    fn co03_segment_core_ignores_sleep_chunks() {
        let mut chunks = ChunkRegistry::new();
        let preds  = PredictionStore::new();
        let cid = make_hot_chunk(&mut chunks, p(1), p(2));
        // Demote to SLEEP — should no longer appear in Core segmentation.
        chunks.demote(cid);
        let view = CoreView { chunks: &chunks, predictions: &preds };
        let units = view.segment_core(&[1, 2, 3]);
        assert_eq!(units, vec![p(1), p(2), p(3)], "SLEEP chunk absent from Core");
    }

    #[test]
    fn co04_predict_core_hit() {
        let chunks = ChunkRegistry::new();
        let mut preds = PredictionStore::new();
        preds.observe(p(1), p(2));
        let view = CoreView { chunks: &chunks, predictions: &preds };
        assert_eq!(view.predict_core(p(1)).map(|(u, _)| u), Some(p(2)));
    }

    #[test]
    fn co05_predict_core_miss() {
        let chunks = ChunkRegistry::new();
        let preds  = PredictionStore::new();
        let view = CoreView { chunks: &chunks, predictions: &preds };
        assert!(view.predict_core(p(99)).is_none());
    }

    #[test]
    fn co06_is_core_unit_primitive() {
        let chunks = ChunkRegistry::new();
        let preds  = PredictionStore::new();
        let view = CoreView { chunks: &chunks, predictions: &preds };
        assert!(view.is_core_unit(p(5)));
    }

    #[test]
    fn co07_is_core_unit_hot_chunk() {
        let mut chunks = ChunkRegistry::new();
        let preds  = PredictionStore::new();
        let cid = chunks.get_or_create(p(1), p(2), 2);
        let view = CoreView { chunks: &chunks, predictions: &preds };
        assert!(view.is_core_unit(UnitId::chunk(cid)));
    }

    #[test]
    fn co08_is_core_unit_sleep_chunk_false() {
        let mut chunks = ChunkRegistry::new();
        let preds  = PredictionStore::new();
        let cid = chunks.get_or_create(p(1), p(2), 2);
        chunks.demote(cid);
        let view = CoreView { chunks: &chunks, predictions: &preds };
        assert!(!view.is_core_unit(UnitId::chunk(cid)));
    }
}
