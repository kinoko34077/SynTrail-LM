/// Phase 9: JSON persistence for ModelState (§D §14).
///
/// ModelState is serialised to a flat snapshot struct and written as
/// pretty-printed JSON.  Load reconstructs the state exactly.
use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::chunks::{Chunk, ChunkRegistry};
use crate::model::{Metrics, ModelState};
use crate::prediction::{PredictionEdge, PredictionStore};
use crate::primitives::PrimitiveRegistry;
use crate::tier::Tier;
use crate::units::UnitId;

// ── Serialisable mirror types ──────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy)]
struct UnitIdDto {
    is_chunk: bool,
    raw: u32,
}

impl From<UnitId> for UnitIdDto {
    fn from(u: UnitId) -> Self {
        Self {
            is_chunk: u.is_chunk(),
            raw: u.raw(),
        }
    }
}

impl From<UnitIdDto> for UnitId {
    fn from(d: UnitIdDto) -> Self {
        if d.is_chunk {
            UnitId::chunk(d.raw)
        } else {
            UnitId::primitive(d.raw)
        }
    }
}

#[derive(Serialize, Deserialize)]
struct TierDto(u8);

impl From<Tier> for TierDto {
    fn from(t: Tier) -> Self {
        TierDto(t as u8)
    }
}

impl From<TierDto> for Tier {
    fn from(d: TierDto) -> Self {
        match d.0 {
            1 => Tier::T1,
            2 => Tier::T2,
            _ => Tier::T0,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ChunkDto {
    id: u32,
    left: UnitIdDto,
    right: UnitIdDto,
    tier: TierDto,
    success_count: u32,
    total_count: u32,
    strength: f64,
    last_used: u64,
    expanded_length: u32,
}

impl From<&Chunk> for ChunkDto {
    fn from(c: &Chunk) -> Self {
        Self {
            id: c.id,
            left: c.left.into(),
            right: c.right.into(),
            tier: c.tier.into(),
            success_count: c.success_count,
            total_count: c.total_count,
            strength: c.strength,
            last_used: c.last_used,
            expanded_length: c.expanded_length,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PredictionEdgeDto {
    context: UnitIdDto,
    next_unit: UnitIdDto,
    success_count: u32,
    total_count: u32,
    strength: f64,
}

impl From<&PredictionEdge> for PredictionEdgeDto {
    fn from(e: &PredictionEdge) -> Self {
        Self {
            context: e.context.into(),
            next_unit: e.next_unit.into(),
            success_count: e.success_count,
            total_count: e.total_count,
            strength: e.strength,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct MetricsDto {
    total_decisions: u64,
    total_characters: u64,
}

impl From<Metrics> for MetricsDto {
    fn from(m: Metrics) -> Self {
        Self {
            total_decisions: m.total_decisions,
            total_characters: m.total_characters,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct MergeCandidateDto {
    left: UnitIdDto,
    right: UnitIdDto,
    count: u32,
}

/// Top-level snapshot written to disk.
#[derive(Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub version: String,
    tick: u64,
    /// Primitives in registration order: [(id, scalar_u32), ...]
    primitives: Vec<(u32, u32)>,
    chunks: Vec<ChunkDto>,
    prediction_edges: Vec<PredictionEdgeDto>,
    merge_candidates: Vec<MergeCandidateDto>,
    metrics: MetricsDto,
}

// ── ModelState → snapshot ──────────────────────────────────────────────────

/// Expose internal iterator helpers via accessor traits to keep fields private.
/// We use public accessor methods added to each module instead.

pub fn save(model: &ModelState, path: &Path) -> std::io::Result<()> {
    let snapshot = to_snapshot(model);
    let json = serde_json::to_string_pretty(&snapshot)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

pub fn load(path: &Path) -> std::io::Result<ModelState> {
    let json = std::fs::read_to_string(path)?;
    let snapshot: ModelSnapshot = serde_json::from_str(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(from_snapshot(snapshot))
}

pub fn to_snapshot(model: &ModelState) -> ModelSnapshot {
    // Primitives: iterate ids 1..=len
    let primitives: Vec<(u32, u32)> = (1..=(model.primitives.len() as u32))
        .filter_map(|id| model.primitives.scalar(id).map(|c| (id, c as u32)))
        .collect();

    let chunks: Vec<ChunkDto> = model.chunks.iter_all().map(ChunkDto::from).collect();

    let prediction_edges: Vec<PredictionEdgeDto> = model
        .predictions
        .iter_all()
        .map(PredictionEdgeDto::from)
        .collect();

    let merge_candidates: Vec<MergeCandidateDto> = model
        .merge_candidates_iter()
        .map(|((left, right), count)| MergeCandidateDto {
            left: (*left).into(),
            right: (*right).into(),
            count: *count,
        })
        .collect();

    ModelSnapshot {
        version: "0.1".to_owned(),
        tick: model.tick,
        primitives,
        chunks,
        prediction_edges,
        merge_candidates,
        metrics: model.metrics.into(),
    }
}

pub fn from_snapshot(snap: ModelSnapshot) -> ModelState {
    // Rebuild PrimitiveRegistry
    let mut primitives = PrimitiveRegistry::new();
    for (id, scalar_u32) in &snap.primitives {
        // char::from_u32 is safe for valid Unicode scalars stored by us
        if let Some(c) = char::from_u32(*scalar_u32) {
            let assigned = primitives.register(c);
            debug_assert_eq!(assigned, *id, "primitive id mismatch during load");
        }
    }

    // Rebuild ChunkRegistry
    let mut chunks = ChunkRegistry::new();
    for dto in snap.chunks {
        let left: UnitId = dto.left.into();
        let right: UnitId = dto.right.into();
        let id = chunks.get_or_create(left, right, dto.expanded_length);
        debug_assert_eq!(id, dto.id);
        let chunk = chunks.get_mut(id).unwrap();
        chunk.tier = dto.tier.into();
        chunk.success_count = dto.success_count;
        chunk.total_count = dto.total_count;
        chunk.strength = dto.strength;
        chunk.last_used = dto.last_used;
    }

    // Rebuild PredictionStore
    let mut predictions = PredictionStore::new();
    for dto in snap.prediction_edges {
        let edge = predictions.get_or_create(dto.context.into(), dto.next_unit.into());
        edge.success_count = dto.success_count;
        edge.total_count = dto.total_count;
        edge.strength = dto.strength;
    }

    // Rebuild merge candidates
    let mut merge_candidates: HashMap<(UnitId, UnitId), u32> = HashMap::new();
    for dto in snap.merge_candidates {
        merge_candidates.insert((dto.left.into(), dto.right.into()), dto.count);
    }

    ModelState::from_parts(
        primitives,
        chunks,
        predictions,
        snap.tick,
        merge_candidates,
        Metrics {
            total_decisions: snap.metrics.total_decisions,
            total_characters: snap.metrics.total_characters,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn trained_model() -> ModelState {
        let mut m = ModelState::new();
        for _ in 0..20 {
            m.train("hello world");
        }
        m
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let model = trained_model();
        let file = NamedTempFile::new().unwrap();
        save(&model, file.path()).unwrap();
        let loaded = load(file.path()).unwrap();

        assert_eq!(loaded.primitive_count(), model.primitive_count());
        assert_eq!(loaded.chunk_count(), model.chunk_count());
        assert_eq!(loaded.edge_count(), model.edge_count());
        assert_eq!(
            loaded.metrics.total_characters,
            model.metrics.total_characters
        );
    }

    #[test]
    fn test_generate_nonempty_after_load() {
        let model = trained_model();
        let file = NamedTempFile::new().unwrap();
        save(&model, file.path()).unwrap();
        let loaded = load(file.path()).unwrap();

        // Both generate non-empty output from the same seed
        let out1 = model.generate("hel", 5);
        let out2 = loaded.generate("hel", 5);
        assert!(!out1.is_empty());
        assert!(!out2.is_empty());
        // Both contain the seed
        assert!(out1.contains("hel"));
        assert!(out2.contains("hel"));
    }

    #[test]
    fn test_empty_model_roundtrip() {
        let model = ModelState::new();
        let file = NamedTempFile::new().unwrap();
        save(&model, file.path()).unwrap();
        let loaded = load(file.path()).unwrap();
        assert_eq!(loaded.primitive_count(), 0);
        assert_eq!(loaded.chunk_count(), 0);
    }
}
