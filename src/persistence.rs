/// v0.3 JSON persistence for ModelState.
///
/// v0.3: adds association edges and avoidance field on prediction edges.
/// v0.1/v0.2 snapshots can be loaded: missing fields default to 0/empty.
use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::association::{AssociationEdge, AssociationStore};
use crate::chunks::{Chunk, ChunkRegistry, Residency};
use crate::identity::IdentityStore;
use crate::lineage::LineageStore;
use crate::model::{Metrics, ModelState};
use crate::relation::RelationStore;
use crate::prediction::{PredictionEdge, PredictionStore};
use crate::primitives::PrimitiveRegistry;
use crate::tier::Tier;
use crate::trace::TraceId;
use crate::units::UnitId;

// ── Serialisable mirror types ──────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy)]
struct UnitIdDto {
    is_chunk: bool,
    raw: u32,
}

impl From<UnitId> for UnitIdDto {
    fn from(u: UnitId) -> Self {
        Self { is_chunk: u.is_chunk(), raw: u.raw() }
    }
}

impl From<UnitIdDto> for UnitId {
    fn from(d: UnitIdDto) -> Self {
        if d.is_chunk { UnitId::chunk(d.raw) } else { UnitId::primitive(d.raw) }
    }
}

#[derive(Serialize, Deserialize)]
struct TierDto(u8);

impl From<Tier> for TierDto {
    fn from(t: Tier) -> Self { TierDto(t as u8) }
}

impl From<TierDto> for Tier {
    fn from(d: TierDto) -> Self {
        match d.0 { 1 => Tier::T1, 2 => Tier::T2, _ => Tier::T0 }
    }
}

#[derive(Serialize, Deserialize)]
struct ChunkDto {
    id: u32,
    left: UnitIdDto,
    right: UnitIdDto,
    tier: TierDto,
    /// v0.2: use_count. v0.1 files: "success_count" alias loads here via backward_compat.
    #[serde(alias = "success_count")]
    use_count: u32,
    /// v0.2: usage_strength. v0.1 alias: "strength".
    #[serde(alias = "strength")]
    usage_strength: f64,
    last_used: u64,
    expanded_length: u32,
    #[serde(default)]
    feedback_value: f64,
    #[serde(default)]
    feedback_count: u32,
    /// v0.4: 0=Hot (default), 1=Sleep
    #[serde(default)]
    residency: u8,
}

impl From<&Chunk> for ChunkDto {
    fn from(c: &Chunk) -> Self {
        Self {
            id: c.id,
            left: c.left.into(),
            right: c.right.into(),
            tier: c.tier.into(),
            use_count: c.use_count,
            usage_strength: c.usage_strength,
            last_used: c.last_used,
            expanded_length: c.expanded_length,
            feedback_value: c.feedback_value,
            feedback_count: c.feedback_count,
            residency: match c.residency { Residency::Hot => 0, Residency::Sleep => 1 },
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PredictionEdgeDto {
    context: UnitIdDto,
    next_unit: UnitIdDto,
    /// v0.2: use_count. v0.1 alias: "success_count".
    #[serde(alias = "success_count")]
    use_count: u32,
    /// v0.2: usage_strength. v0.1 alias: "strength".
    #[serde(alias = "strength")]
    usage_strength: f64,
    #[serde(default)]
    feedback_value: f64,
    #[serde(default)]
    feedback_count: u32,
    /// v0.3: contextual avoidance accumulator (§19).
    #[serde(default)]
    avoidance: f64,
    /// Phase 9: tick of last update for lazy decay.
    #[serde(default)]
    last_used_tick: u64,
}

impl From<&PredictionEdge> for PredictionEdgeDto {
    fn from(e: &PredictionEdge) -> Self {
        Self {
            context: e.context.into(),
            next_unit: e.next_unit.into(),
            use_count: e.use_count,
            usage_strength: e.usage_strength,
            feedback_value: e.feedback_value,
            feedback_count: e.feedback_count,
            avoidance: e.avoidance,
            last_used_tick: e.last_used_tick,
        }
    }
}

/// v0.3: association edge serialisation.
#[derive(Serialize, Deserialize)]
struct AssociationEdgeDto {
    source: UnitIdDto,
    target: UnitIdDto,
    strength: f64,
    use_count: u32,
    last_used: u64,
}

impl From<&AssociationEdge> for AssociationEdgeDto {
    fn from(e: &AssociationEdge) -> Self {
        Self {
            source: e.source.into(),
            target: e.target.into(),
            strength: e.strength,
            use_count: e.use_count,
            last_used: e.last_used,
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
        Self { total_decisions: m.total_decisions, total_characters: m.total_characters }
    }
}

#[derive(Serialize, Deserialize)]
struct MergeCandidateDto {
    left: UnitIdDto,
    right: UnitIdDto,
    count: u32,
}

/// Phase 2: serialisable View entry — (unit sequence, identity id).
#[derive(Serialize, Deserialize)]
struct ViewDto {
    units: Vec<UnitIdDto>,
    identity: u32,
}

/// Top-level snapshot written to disk.
#[derive(Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub version: String,
    tick: u64,
    primitives: Vec<(u32, u32)>,
    chunks: Vec<ChunkDto>,
    prediction_edges: Vec<PredictionEdgeDto>,
    merge_candidates: Vec<MergeCandidateDto>,
    metrics: MetricsDto,
    #[serde(default)]
    next_trace_id: TraceId,
    /// v0.3: association edges (§7-§11).
    #[serde(default)]
    association_edges: Vec<AssociationEdgeDto>,
    /// v0.3: Top-K budget stored so recall stays consistent after load.
    #[serde(default = "default_top_k")]
    association_top_k: usize,
    /// v0.3: decay stored alongside edges.
    #[serde(default = "default_decay")]
    association_decay: f64,
    /// Phase 2: canonical Primitive sequences indexed by IdentityId.
    #[serde(default)]
    identities: Vec<Vec<u32>>,
    /// Phase 2: Views (Chunk trees) with their Identity bindings.
    #[serde(default)]
    views: Vec<ViewDto>,
    /// Phase 3: Lineage — (child_chunk_id, left_unit, right_unit) triples.
    #[serde(default)]
    lineage_entries: Vec<(u32, UnitIdDto, UnitIdDto)>,
    /// Phase 11: Union-Find parent array for Cross-View Identity merges.
    #[serde(default)]
    identity_parent: Vec<u32>,
}

fn default_top_k() -> usize { 32 }
fn default_decay() -> f64 { 0.99 }

// ── ModelState → snapshot ──────────────────────────────────────────────────

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
    let primitives: Vec<(u32, u32)> = (1..=(model.primitives.len() as u32))
        .filter_map(|id| model.primitives.scalar(id).map(|c| (id, c as u32)))
        .collect();

    let chunks: Vec<ChunkDto> = model.chunks.iter_all().map(ChunkDto::from).collect();
    let prediction_edges: Vec<PredictionEdgeDto> =
        model.predictions.iter_all().map(PredictionEdgeDto::from).collect();
    let merge_candidates: Vec<MergeCandidateDto> = model
        .merge_candidates_iter()
        .map(|((left, right), count)| MergeCandidateDto {
            left: (*left).into(),
            right: (*right).into(),
            count: *count,
        })
        .collect();
    let association_edges: Vec<AssociationEdgeDto> =
        model.associations.iter_all().map(AssociationEdgeDto::from).collect();

    // Phase 2: identity / view store
    let identities: Vec<Vec<u32>> = model.identities.all_identities().to_vec();
    let views: Vec<ViewDto> = model.identities.all_views()
        .map(|(units, identity)| ViewDto {
            units: units.iter().copied().map(UnitIdDto::from).collect(),
            identity,
        })
        .collect();

    let lineage_entries: Vec<(u32, UnitIdDto, UnitIdDto)> = model.lineage.all_entries()
        .map(|(child, l, r)| (child, UnitIdDto::from(l), UnitIdDto::from(r)))
        .collect();

    let identity_parent: Vec<u32> = model.identities.all_parents().to_vec();

    ModelSnapshot {
        version: "0.5".to_owned(),
        tick: model.tick,
        primitives,
        chunks,
        prediction_edges,
        merge_candidates,
        metrics: model.metrics.into(),
        next_trace_id: model.next_trace_id,
        association_edges,
        association_top_k: model.associations.top_k,
        association_decay: model.associations.decay,
        identities,
        views,
        lineage_entries,
        identity_parent,
    }
}

pub fn from_snapshot(snap: ModelSnapshot) -> ModelState {
    let mut primitives = PrimitiveRegistry::new();
    for (id, scalar_u32) in &snap.primitives {
        if let Some(c) = char::from_u32(*scalar_u32) {
            let assigned = primitives.register(c);
            debug_assert_eq!(assigned, *id);
        }
    }

    let mut chunks = ChunkRegistry::new();
    for dto in snap.chunks {
        let left: UnitId = dto.left.into();
        let right: UnitId = dto.right.into();
        let id = chunks.get_or_create(left, right, dto.expanded_length);
        debug_assert_eq!(id, dto.id);
        let chunk = chunks.get_mut(id).unwrap();
        chunk.tier = dto.tier.into();
        chunk.use_count = dto.use_count;
        chunk.usage_strength = dto.usage_strength;
        chunk.last_used = dto.last_used;
        chunk.feedback_value = dto.feedback_value;
        chunk.feedback_count = dto.feedback_count;
        // Phase 8: use registry demote() so hot_pair_to_id stays consistent.
        if dto.residency == 1 {
            chunks.demote(id);
        }
    }

    let mut predictions = PredictionStore::new();
    for dto in snap.prediction_edges {
        let edge = predictions.get_or_create(dto.context.into(), dto.next_unit.into());
        edge.use_count = dto.use_count;
        edge.usage_strength = dto.usage_strength;
        edge.feedback_value = dto.feedback_value;
        edge.feedback_count = dto.feedback_count;
        edge.avoidance = dto.avoidance;
        edge.last_used_tick = dto.last_used_tick;
    }

    let mut merge_candidates: HashMap<(UnitId, UnitId), u32> = HashMap::new();
    for dto in snap.merge_candidates {
        merge_candidates.insert((dto.left.into(), dto.right.into()), dto.count);
    }

    // v0.3: restore associations
    let raw_assoc: Vec<AssociationEdge> = snap.association_edges.into_iter().map(|d| {
        AssociationEdge {
            source: d.source.into(),
            target: d.target.into(),
            strength: d.strength,
            use_count: d.use_count,
            last_used: d.last_used,
        }
    }).collect();
    let associations = AssociationStore::from_edges(raw_assoc, snap.association_top_k, snap.association_decay);

    // Phase 2: restore identity / view store
    let identity_seqs: Vec<Vec<u32>> = snap.identities;
    let view_pairs: Vec<(Vec<UnitId>, u32)> = snap.views.into_iter()
        .map(|d| {
            let units: Vec<UnitId> = d.units.into_iter().map(UnitId::from).collect();
            (units, d.identity)
        })
        .collect();
    let parent_opt = if snap.identity_parent.is_empty() { None } else { Some(snap.identity_parent) };
    let identities = IdentityStore::from_bulk(identity_seqs, view_pairs, parent_opt);

    let lineage_bulk: Vec<(u32, UnitId, UnitId)> = snap.lineage_entries.into_iter()
        .map(|(c, l, r)| (c, UnitId::from(l), UnitId::from(r)))
        .collect();
    let lineage = LineageStore::from_bulk(lineage_bulk);

    // Phase 7: RelationStore is derived from other stores; rebuilt on use after load.
    let relations = RelationStore::new();

    ModelState::from_parts(
        primitives,
        chunks,
        predictions,
        associations,
        identities,
        lineage,
        relations,
        snap.tick,
        merge_candidates,
        Metrics {
            total_decisions: snap.metrics.total_decisions,
            total_characters: snap.metrics.total_characters,
        },
        snap.next_trace_id,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn trained_model() -> ModelState {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
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
        assert_eq!(loaded.metrics.total_characters, model.metrics.total_characters);
    }

    #[test]
    fn test_generate_nonempty_after_load() {
        let model = trained_model();
        let file = NamedTempFile::new().unwrap();
        save(&model, file.path()).unwrap();
        let loaded = load(file.path()).unwrap();
        let out1 = model.generate("hel", 5);
        let out2 = loaded.generate("hel", 5);
        assert!(!out1.is_empty());
        assert!(!out2.is_empty());
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

    #[test]
    fn test_version_field() {
        let model = ModelState::new();
        let snap = to_snapshot(&model);
        assert_eq!(snap.version, "0.5");
    }

    #[test]
    fn test_feedback_fields_preserved() {
        let mut model = trained_model();
        // Apply some feedback
        let tid = model.alloc_trace_id();
        let (_, trace) = model.generate_with_trace("hel", "hel", 5, tid);
        if trace.decision_count > 0 {
            let credits = vec![1.0; trace.decision_count];
            model.apply_feedback_to_trace(&trace, &credits, 0.99);
        }
        let file = NamedTempFile::new().unwrap();
        save(&model, file.path()).unwrap();
        let loaded = load(file.path()).unwrap();
        // Total feedback_count in edges should be preserved
        let fb_before: u32 = model.predictions.iter_all().map(|e| e.feedback_count).sum();
        let fb_after: u32 = loaded.predictions.iter_all().map(|e| e.feedback_count).sum();
        assert_eq!(fb_before, fb_after);
    }
}
