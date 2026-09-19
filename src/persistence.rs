/// JSON and binary persistence for ModelState.
///
/// Phase 16: binary format via `bincode` for compact, fast serialisation.
/// The on-disk format is chosen by the file extension:
///   `.json`          → serde_json (human-readable, forward-compatible)
///   anything else    → bincode (binary, Phase 16+)
///
/// Both formats use the same `ModelSnapshot` struct so conversion is trivial.
///
/// v0.3: adds association edges and avoidance field on prediction edges.
/// v0.1/v0.2 snapshots can be loaded: missing fields default to 0/empty.
use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::association::{AssociationEdge, AssociationStore};
use crate::db::Database;
use crate::chunks::{Chunk, ChunkRegistry, Residency};
use crate::identity::IdentityStore;
use crate::lineage::LineageStore;
use crate::model::{Metrics, ModelState};
use crate::relation::RelationStore;
use crate::prediction::{PredictionEdge, PredictionStore};
use crate::primitives::PrimitiveRegistry;
use crate::representation::{RepresentationEntry, RepresentationStore};
use crate::tier::Tier;
use crate::trace::TraceId;
use crate::transform::{TransformKind, TransformStore};
use crate::units::UnitId;

// ── Serialisable mirror types ──────────────────────────────────────────────

/// §16: Packed UnitId serialization: single varint u32 = (raw << 1) | is_chunk.
/// Saves one byte per UnitId vs the v2 struct layout (bool + u32).
#[derive(Clone, Copy)]
struct UnitIdDto {
    is_chunk: bool,
    raw: u32,
}

impl Serialize for UnitIdDto {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            // JSON: keep {is_chunk, raw} for forward/backward compat
            use serde::ser::SerializeStruct;
            let mut st = s.serialize_struct("UnitIdDto", 2)?;
            st.serialize_field("is_chunk", &self.is_chunk)?;
            st.serialize_field("raw", &self.raw)?;
            st.end()
        } else {
            // bincode v3: packed u32 = (raw << 1) | is_chunk (§16)
            let packed: u32 = (self.raw << 1) | (self.is_chunk as u32);
            packed.serialize(s)
        }
    }
}

impl<'de> Deserialize<'de> for UnitIdDto {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            // JSON: {is_chunk: bool, raw: u32}
            #[derive(Deserialize)]
            struct JsonForm { is_chunk: bool, raw: u32 }
            let j = JsonForm::deserialize(d)?;
            Ok(Self { is_chunk: j.is_chunk, raw: j.raw })
        } else {
            // bincode v3: packed u32
            let packed = u32::deserialize(d)?;
            Ok(Self { is_chunk: (packed & 1) != 0, raw: packed >> 1 })
        }
    }
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

/// Phase 17: strength fields in DTOs use f32 for compact serialisation.
/// Runtime model still uses f64; the cast is lossless for values in [0, 1e6].
/// §17: id field removed — position in snapshot array is the implicit id.
#[derive(Serialize, Deserialize)]
struct ChunkDto {
    left: UnitIdDto,
    right: UnitIdDto,
    tier: TierDto,
    /// v0.2: use_count. v0.1 files: "success_count" alias loads here via backward_compat.
    #[serde(alias = "success_count")]
    use_count: u32,
    /// Phase 17: stored as f32. v0.1 alias: "strength".
    #[serde(alias = "strength")]
    usage_strength: f32,
    last_used: u64,
    expanded_length: u32,
    #[serde(default)]
    feedback_value: f32,
    #[serde(default)]
    feedback_count: u32,
    /// v0.4: 0=Hot (default), 1=Sleep
    #[serde(default)]
    residency: u8,
    /// §28: ticks-since-last-use delta (0 = use raw last_used for legacy compat).
    /// Encoded as (model.tick - last_used) + 1; 0 means "not set or last_used==0".
    #[serde(default)]
    last_used_delta: u32,
}

impl From<&Chunk> for ChunkDto {
    fn from(c: &Chunk) -> Self {
        Self {
            left: c.left.into(),
            right: c.right.into(),
            tier: c.tier.into(),
            use_count: c.use_count,
            usage_strength: c.usage_strength as f32,
            last_used: c.last_used,
            expanded_length: c.expanded_length,
            feedback_value: c.feedback_value as f32,
            feedback_count: c.feedback_count,
            residency: match c.residency { Residency::Hot => 0, Residency::Sleep => 1 },
            last_used_delta: 0, // set by to_snapshot(); 0 = use raw last_used
        }
    }
}

/// Phase 17: strength fields stored as f32.
/// Kept for loading legacy flat prediction_edges field.
#[derive(Serialize, Deserialize)]
struct PredictionEdgeDto {
    context: UnitIdDto,
    next_unit: UnitIdDto,
    /// v0.2: use_count. v0.1 alias: "success_count".
    #[serde(alias = "success_count")]
    use_count: u32,
    /// Phase 17: stored as f32. v0.1 alias: "strength".
    #[serde(alias = "strength")]
    usage_strength: f32,
    #[serde(default)]
    feedback_value: f32,
    #[serde(default)]
    feedback_count: u32,
    /// v0.3: contextual avoidance accumulator (§19). Phase 17: f32.
    #[serde(default)]
    avoidance: f32,
    /// Phase 9: tick of last update for lazy decay.
    #[serde(default)]
    last_used_tick: u64,
    /// Phase C: external route evidence — only updated by Experience, not Replay.
    #[serde(default)]
    external_route_evidence: f32,
}

/// §25: Per-edge payload within a source group (context omitted — implied by group key).
#[derive(Serialize, Deserialize)]
struct PredEdgeEntryDto {
    next_unit: UnitIdDto,
    use_count: u32,
    usage_strength: f32,
    #[serde(default)]
    feedback_value: f32,
    #[serde(default)]
    feedback_count: u32,
    #[serde(default)]
    avoidance: f32,
    #[serde(default)]
    last_used_tick: u64,
    /// §28: ticks-since-last-use delta (0 = use raw last_used_tick for legacy compat).
    #[serde(default)]
    last_used_tick_delta: u32,
    #[serde(default)]
    external_route_evidence: f32,
}

/// §26: Per-edge payload within a source group (source omitted — implied by group key).
#[derive(Serialize, Deserialize)]
struct AssocEdgeEntryDto {
    target: UnitIdDto,
    strength: f64,
    use_count: u32,
    last_used: u64,
    /// §28: ticks-since-last-use delta (0 = use raw last_used for legacy compat).
    #[serde(default)]
    last_used_delta: u32,
}

/// Phase B: Representation Lineage entry DTO.
/// §17: rep_id removed — position in snapshot array is the implicit id.
#[derive(Serialize, Deserialize)]
struct RepresentationEntryDto {
    identity_id: u32,
    units: Vec<UnitIdDto>,
    acquired_at: u64,
    practice_count: u32,
    confidence: f32,
    #[serde(default)]
    predecessor_rep_id: Option<u32>,
}

impl From<&RepresentationEntry> for RepresentationEntryDto {
    fn from(e: &RepresentationEntry) -> Self {
        Self {
            identity_id: e.identity_id,
            units: e.units.iter().copied().map(UnitIdDto::from).collect(),
            acquired_at: e.acquired_at,
            practice_count: e.practice_count,
            confidence: e.confidence,
            predecessor_rep_id: e.predecessor_rep_id,
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

/// §19: Transform persistence DTO.
/// kind_tag: 0=EquivalentView, 1=Mapping, 2=Inverse, 3=Composed.
/// kind_arg1/arg2: TransformId references for Inverse/Composed.
#[derive(Serialize, Deserialize)]
struct TransformEntryDto {
    source: u32,
    target: u32,
    kind_tag: u8,
    kind_arg1: u32,
    kind_arg2: u32,
    value: f64,
    evidence: f64,
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
    /// Phase B: Representation Lineage entries.
    #[serde(default)]
    representation_entries: Vec<RepresentationEntryDto>,
    /// §19: Transform entries (directed Identity relations).
    #[serde(default)]
    transforms: Vec<TransformEntryDto>,
    /// §22: Factorization right-reuse map: (right_unit, [left_units...]).
    /// Restoring this avoids factorization pressure starting from zero after a load.
    #[serde(default)]
    merge_right_reuse: Vec<(UnitIdDto, Vec<UnitIdDto>)>,
    /// §32: Links this model snapshot to the trainer state saved in the same operation.
    #[serde(default)]
    pub checkpoint_generation: u64,
    /// §25: Source-grouped prediction edges (context appears once per group).
    /// New saves populate this and leave prediction_edges=[].
    #[serde(default)]
    prediction_edge_groups: Vec<(UnitIdDto, Vec<PredEdgeEntryDto>)>,
    /// §26: Source-grouped association edges (source appears once per group).
    /// New saves populate this and leave association_edges=[].
    #[serde(default)]
    association_edge_groups: Vec<(UnitIdDto, Vec<AssocEdgeEntryDto>)>,
}

fn default_top_k() -> usize { 32 }
fn default_decay() -> f64 { 0.99 }

// ── ModelState → snapshot ──────────────────────────────────────────────────

/// Save to JSON (human-readable).
pub fn save(model: &ModelState, path: &Path) -> std::io::Result<()> {
    save_with_generation(model, path, 0)
}

/// §32: Save to JSON and embed a checkpoint_generation for trainer-state pairing.
/// §9: Streams directly to a BufWriter — avoids building the full JSON string in memory.
pub fn save_with_generation(model: &ModelState, path: &Path, generation: u64) -> std::io::Result<()> {
    let mut snapshot = to_snapshot(model);
    snapshot.checkpoint_generation = generation;
    let tmp = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    }
    std::fs::rename(&tmp, path)
}

/// §4 (Storage P0): Read checkpoint_generation from any supported model format.
///
/// Previously JSON-only, causing STM generation reads to silently return 0
/// and skip the trainer-state pair verification on resume.
///
/// Format dispatch:
/// - `.stm` → STM container (§12: header + zstd-bincode; legacy raw bincode also handled)
/// - `.db` / `.sqlite` → latest snapshot JSON in the DB
/// - anything else → JSON
pub fn load_checkpoint_generation(path: &Path) -> std::io::Result<u64> {
    let ext = path.extension().and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("stm") => {
            let bytes = std::fs::read(path)?;
            let snapshot = stm_read_container(&bytes)?;
            Ok(snapshot.checkpoint_generation)
        }
        Some("db") | Some("sqlite") => {
            let db = Database::open(&path.to_string_lossy())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            // §8: prefer blob_data; fall back to json_blob for legacy rows.
            let snapshot: ModelSnapshot = if let Some(blob) = db.load_latest_snapshot_blob()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            {
                bincode::deserialize(&blob)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            } else {
                let json = db.load_latest_snapshot_json()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no snapshot in DB"))?;
                serde_json::from_str(&json)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            };
            Ok(snapshot.checkpoint_generation)
        }
        _ => {
            // JSON path (original behaviour).
            let json = std::fs::read_to_string(path)?;
            let snapshot: ModelSnapshot = serde_json::from_str(&json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            Ok(snapshot.checkpoint_generation)
        }
    }
}

/// Load from JSON.
pub fn load(path: &Path) -> std::io::Result<ModelState> {
    let json = std::fs::read_to_string(path)?;
    let snapshot: ModelSnapshot = serde_json::from_str(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(from_snapshot(snapshot))
}

// ── STM container format (§12 / §27) ─────────────────────────────────────
//
// Layout (10-byte header + payload):
//   [0..4]  magic:       b"STM1"
//   [4]     version:     1 = fixed-int bincode; 2 = varint (§27); 3 = packed UnitId + implicit IDs (§16/§17)
//   [5]     flags:       bit 0 = zstd compressed; remaining bits reserved
//   [6..10] payload_len: u32 little-endian (byte length of payload)
//   [10..]  payload:     bincode(ModelSnapshot), optionally zstd-compressed
//
// Legacy files (no STM1 header) are detected by checking the first 4 bytes
// and are deserialized as raw bincode for backward compatibility.
const STM_MAGIC: &[u8; 4] = b"STM1";
// Version 3 = varint bincode + packed UnitId + implicit chunk/rep IDs (§16/§17).
const STM_VERSION: u8 = 3;
const STM_FLAG_ZSTD: u8 = 0b0000_0001;

fn bincode_serialize_varint(snapshot: &ModelSnapshot) -> Result<Vec<u8>, std::io::Error> {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .serialize(snapshot)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn bincode_deserialize_varint(payload: &[u8]) -> Result<ModelSnapshot, std::io::Error> {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .deserialize(payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ── STM v2 backward-compat deserialization (§16/§17) ─────────────────────────
// V2 uses struct UnitId { bool, u32 }, has ChunkDto.id, RepresentationEntryDto.rep_id.

mod v2_compat {
    use serde::Deserialize;

    #[derive(Deserialize, Clone, Copy)]
    pub struct UnitIdDto { pub is_chunk: bool, pub raw: u32 }

    #[derive(Deserialize)]
    pub struct ChunkDto {
        pub id: u32,
        pub left: UnitIdDto,
        pub right: UnitIdDto,
        pub tier: super::TierDto,
        #[serde(alias = "success_count")]
        pub use_count: u32,
        #[serde(alias = "strength")]
        pub usage_strength: f32,
        pub last_used: u64,
        pub expanded_length: u32,
        #[serde(default)]
        pub feedback_value: f32,
        #[serde(default)]
        pub feedback_count: u32,
        #[serde(default)]
        pub residency: u8,
        #[serde(default)]
        pub last_used_delta: u32,
    }

    #[derive(Deserialize)]
    pub struct PredictionEdgeDto {
        pub context: UnitIdDto,
        pub next_unit: UnitIdDto,
        #[serde(alias = "success_count")]
        pub use_count: u32,
        #[serde(alias = "strength")]
        pub usage_strength: f32,
        #[serde(default)]
        pub feedback_value: f32,
        #[serde(default)]
        pub feedback_count: u32,
        #[serde(default)]
        pub avoidance: f32,
        #[serde(default)]
        pub last_used_tick: u64,
        #[serde(default)]
        pub external_route_evidence: f32,
    }

    #[derive(Deserialize)]
    pub struct PredEdgeEntryDto {
        pub next_unit: UnitIdDto,
        pub use_count: u32,
        pub usage_strength: f32,
        #[serde(default)]
        pub feedback_value: f32,
        #[serde(default)]
        pub feedback_count: u32,
        #[serde(default)]
        pub avoidance: f32,
        #[serde(default)]
        pub last_used_tick: u64,
        #[serde(default)]
        pub last_used_tick_delta: u32,
        #[serde(default)]
        pub external_route_evidence: f32,
    }

    #[derive(Deserialize)]
    pub struct AssocEdgeEntryDto {
        pub target: UnitIdDto,
        pub strength: f64,
        pub use_count: u32,
        pub last_used: u64,
        #[serde(default)]
        pub last_used_delta: u32,
    }

    #[derive(Deserialize)]
    pub struct AssociationEdgeDto {
        pub source: UnitIdDto,
        pub target: UnitIdDto,
        pub strength: f64,
        pub use_count: u32,
        pub last_used: u64,
    }

    #[derive(Deserialize)]
    pub struct RepresentationEntryDto {
        pub rep_id: u32,
        pub identity_id: u32,
        pub units: Vec<UnitIdDto>,
        pub acquired_at: u64,
        pub practice_count: u32,
        pub confidence: f32,
        #[serde(default)]
        pub predecessor_rep_id: Option<u32>,
    }

    #[derive(Deserialize)]
    pub struct ViewDto {
        pub units: Vec<UnitIdDto>,
        pub identity: u32,
    }

    #[derive(Deserialize)]
    pub struct MergeCandidateDto {
        pub left: UnitIdDto,
        pub right: UnitIdDto,
        pub count: u32,
    }

    #[derive(Deserialize)]
    pub struct ModelSnapshot {
        pub version: String,
        pub tick: u64,
        pub primitives: Vec<(u32, u32)>,
        pub chunks: Vec<ChunkDto>,
        pub prediction_edges: Vec<PredictionEdgeDto>,
        pub merge_candidates: Vec<MergeCandidateDto>,
        pub metrics: super::MetricsDto,
        #[serde(default)]
        pub next_trace_id: crate::trace::TraceId,
        #[serde(default)]
        pub association_edges: Vec<AssociationEdgeDto>,
        #[serde(default = "super::default_top_k")]
        pub association_top_k: usize,
        #[serde(default = "super::default_decay")]
        pub association_decay: f64,
        #[serde(default)]
        pub identities: Vec<Vec<u32>>,
        #[serde(default)]
        pub views: Vec<ViewDto>,
        #[serde(default)]
        pub lineage_entries: Vec<(u32, UnitIdDto, UnitIdDto)>,
        #[serde(default)]
        pub identity_parent: Vec<u32>,
        #[serde(default)]
        pub representation_entries: Vec<RepresentationEntryDto>,
        #[serde(default)]
        pub transforms: Vec<super::TransformEntryDto>,
        #[serde(default)]
        pub merge_right_reuse: Vec<(UnitIdDto, Vec<UnitIdDto>)>,
        #[serde(default)]
        pub checkpoint_generation: u64,
        #[serde(default)]
        pub prediction_edge_groups: Vec<(UnitIdDto, Vec<PredEdgeEntryDto>)>,
        #[serde(default)]
        pub association_edge_groups: Vec<(UnitIdDto, Vec<AssocEdgeEntryDto>)>,
    }
}

fn bincode_deserialize_varint_v2(payload: &[u8]) -> Result<v2_compat::ModelSnapshot, std::io::Error> {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .deserialize(payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn from_v2_snapshot(v2: v2_compat::ModelSnapshot) -> ModelSnapshot {
    let cv = |u: v2_compat::UnitIdDto| -> UnitIdDto {
        UnitIdDto { is_chunk: u.is_chunk, raw: u.raw }
    };
    ModelSnapshot {
        version: v2.version,
        tick: v2.tick,
        primitives: v2.primitives,
        chunks: v2.chunks.into_iter().map(|c| ChunkDto {
            left: cv(c.left), right: cv(c.right), tier: c.tier,
            use_count: c.use_count, usage_strength: c.usage_strength,
            last_used: c.last_used, last_used_delta: c.last_used_delta,
            expanded_length: c.expanded_length,
            feedback_value: c.feedback_value, feedback_count: c.feedback_count,
            residency: c.residency,
        }).collect(),
        prediction_edges: v2.prediction_edges.into_iter().map(|e| PredictionEdgeDto {
            context: cv(e.context), next_unit: cv(e.next_unit),
            use_count: e.use_count, usage_strength: e.usage_strength,
            feedback_value: e.feedback_value, feedback_count: e.feedback_count,
            avoidance: e.avoidance, last_used_tick: e.last_used_tick,
            external_route_evidence: e.external_route_evidence,
        }).collect(),
        merge_candidates: v2.merge_candidates.into_iter().map(|m| MergeCandidateDto {
            left: cv(m.left), right: cv(m.right), count: m.count,
        }).collect(),
        metrics: v2.metrics,
        next_trace_id: v2.next_trace_id,
        association_edges: v2.association_edges.into_iter().map(|a| AssociationEdgeDto {
            source: cv(a.source), target: cv(a.target),
            strength: a.strength, use_count: a.use_count, last_used: a.last_used,
        }).collect(),
        association_top_k: v2.association_top_k,
        association_decay: v2.association_decay,
        identities: v2.identities,
        views: v2.views.into_iter().map(|v| ViewDto {
            units: v.units.into_iter().map(cv).collect(), identity: v.identity,
        }).collect(),
        lineage_entries: v2.lineage_entries.into_iter().map(|(id, l, r)| (id, cv(l), cv(r))).collect(),
        identity_parent: v2.identity_parent,
        representation_entries: v2.representation_entries.into_iter().map(|e| RepresentationEntryDto {
            identity_id: e.identity_id,
            units: e.units.into_iter().map(cv).collect(),
            acquired_at: e.acquired_at, practice_count: e.practice_count,
            confidence: e.confidence, predecessor_rep_id: e.predecessor_rep_id,
        }).collect(),
        transforms: v2.transforms,
        merge_right_reuse: v2.merge_right_reuse.into_iter()
            .map(|(r, ls)| (cv(r), ls.into_iter().map(cv).collect())).collect(),
        checkpoint_generation: v2.checkpoint_generation,
        prediction_edge_groups: v2.prediction_edge_groups.into_iter()
            .map(|(ctx, edges)| (cv(ctx), edges.into_iter().map(|e| PredEdgeEntryDto {
                next_unit: cv(e.next_unit), use_count: e.use_count,
                usage_strength: e.usage_strength, feedback_value: e.feedback_value,
                feedback_count: e.feedback_count, avoidance: e.avoidance,
                last_used_tick: e.last_used_tick, last_used_tick_delta: e.last_used_tick_delta,
                external_route_evidence: e.external_route_evidence,
            }).collect())).collect(),
        association_edge_groups: v2.association_edge_groups.into_iter()
            .map(|(src, edges)| (cv(src), edges.into_iter().map(|e| AssocEdgeEntryDto {
                target: cv(e.target), strength: e.strength,
                use_count: e.use_count, last_used: e.last_used, last_used_delta: e.last_used_delta,
            }).collect())).collect(),
    }
}

fn stm_write_container(
    writer: &mut impl std::io::Write,
    snapshot: &ModelSnapshot,
    compress: bool,
) -> std::io::Result<()> {
    // §27: version 2 — varint bincode
    let payload = bincode_serialize_varint(snapshot)?;
    let (flags, encoded) = if compress {
        let compressed = zstd::bulk::compress(&payload, 3)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        (STM_FLAG_ZSTD, compressed)
    } else {
        (0u8, payload)
    };
    let payload_len = encoded.len() as u32;
    writer.write_all(STM_MAGIC)?;
    writer.write_all(&[STM_VERSION, flags])?;
    writer.write_all(&payload_len.to_le_bytes())?;
    writer.write_all(&encoded)
}

fn stm_read_container(bytes: &[u8]) -> std::io::Result<ModelSnapshot> {
    if bytes.len() >= 4 && &bytes[..4] == STM_MAGIC {
        if bytes.len() < 10 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "STM header truncated"));
        }
        let version = bytes[4];
        let flags = bytes[5];
        let payload_len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        if bytes.len() < 10 + payload_len {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "STM payload truncated"));
        }
        let encoded = &bytes[10..10 + payload_len];
        let payload = if flags & STM_FLAG_ZSTD != 0 {
            zstd::bulk::decompress(encoded, 256 * 1024 * 1024)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        } else {
            encoded.to_vec()
        };
        // version 3 = packed UnitId + implicit IDs (current); version 2 = varint struct UnitId
        if version >= 3 {
            bincode_deserialize_varint(&payload)
        } else if version == 2 {
            let v2 = bincode_deserialize_varint_v2(&payload)?;
            Ok(from_v2_snapshot(v2))
        } else {
            bincode::deserialize(&payload)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }
    } else {
        // Legacy: raw bincode without header
        bincode::deserialize(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Phase 16: save to binary format (bincode); generation=0 (legacy/unversioned).
pub fn save_binary(model: &ModelState, path: &Path) -> std::io::Result<()> {
    save_binary_with_generation(model, path, 0)
}

/// §3/§12: Save to STM container format with optional Zstd compression.
///
/// Writes the STM1 header + bincode payload (Zstd-compressed by default).
/// Backward compatible: load_binary detects the header and falls back to
/// raw bincode for files written before §12.
pub fn save_binary_with_generation(model: &ModelState, path: &Path, generation: u64) -> std::io::Result<()> {
    let mut snapshot = to_snapshot(model);
    snapshot.checkpoint_generation = generation;
    let tmp = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp)?;
        let mut writer = std::io::BufWriter::new(file);
        stm_write_container(&mut writer, &snapshot, true)?;
    }
    std::fs::rename(&tmp, path)
}

/// Phase 16 / §12: Load from STM container format (or legacy raw bincode).
pub fn load_binary(path: &Path) -> std::io::Result<ModelState> {
    let bytes = std::fs::read(path)?;
    let snapshot = stm_read_container(&bytes)?;
    Ok(from_snapshot(snapshot))
}

/// Convenience: dispatch to JSON or binary based on the file extension.
///
/// `.json` → JSON, everything else → binary.
pub fn save_auto(model: &ModelState, path: &Path) -> std::io::Result<()> {
    if path.extension().and_then(|e| e.to_str()) == Some("json") {
        save(model, path)
    } else {
        save_binary(model, path)
    }
}

/// Convenience: load from JSON or binary based on file extension.
pub fn load_auto(path: &Path) -> std::io::Result<ModelState> {
    if path.extension().and_then(|e| e.to_str()) == Some("json") {
        load(path)
    } else {
        load_binary(path)
    }
}

/// §28: Encode an absolute tick as a delta from model_tick.
/// Returns 0 when `actual == 0` (never-used sentinel); otherwise (model_tick - actual) + 1.
/// Saturates at u32::MAX - 1 so the +1 never wraps.
#[inline]
fn encode_tick_delta(actual: u64, model_tick: u64) -> u32 {
    if actual == 0 { return 0; }
    model_tick.saturating_sub(actual).min(u32::MAX as u64 - 1) as u32 + 1
}

/// §28: Decode a delta back to an absolute tick.
/// `delta > 0`: `snap_tick - (delta - 1)`. `delta == 0`: use legacy `raw` value.
#[inline]
fn decode_tick_delta(delta: u32, snap_tick: u64, raw: u64) -> u64 {
    if delta > 0 { snap_tick.saturating_sub(delta as u64 - 1) } else { raw }
}

pub fn to_snapshot(model: &ModelState) -> ModelSnapshot {
    let primitives: Vec<(u32, u32)> = (1..=(model.primitives.len() as u32))
        .filter_map(|id| model.primitives.scalar(id).map(|c| (id, c as u32)))
        .collect();

    let model_tick = model.tick;
    let chunks: Vec<ChunkDto> = model.chunks.iter_all().map(|c| {
        let delta = encode_tick_delta(c.last_used, model_tick);
        ChunkDto {
            left: c.left.into(),
            right: c.right.into(),
            tier: c.tier.into(),
            use_count: c.use_count,
            usage_strength: c.usage_strength as f32,
            last_used: if delta > 0 { 0 } else { c.last_used },
            last_used_delta: delta,
            expanded_length: c.expanded_length,
            feedback_value: c.feedback_value as f32,
            feedback_count: c.feedback_count,
            residency: match c.residency { Residency::Hot => 0, Residency::Sleep => 1 },
        }
    }).collect();

    // §25: source-grouped prediction edges — context appears once per group.
    let prediction_edges: Vec<PredictionEdgeDto> = Vec::new(); // deprecated; new saves use groups
    let mut pred_groups: std::collections::HashMap<UnitId, Vec<PredEdgeEntryDto>> =
        std::collections::HashMap::new();
    for e in model.predictions.iter_all() {
        let delta = encode_tick_delta(e.last_used_tick, model_tick);
        pred_groups.entry(e.context).or_default().push(PredEdgeEntryDto {
            next_unit: e.next_unit.into(),
            use_count: e.use_count,
            usage_strength: e.usage_strength as f32,
            feedback_value: e.feedback_value as f32,
            feedback_count: e.feedback_count,
            avoidance: e.avoidance as f32,
            last_used_tick: if delta > 0 { 0 } else { e.last_used_tick },
            last_used_tick_delta: delta,
            external_route_evidence: e.external_route_evidence as f32,
        });
    }
    let mut prediction_edge_groups: Vec<(UnitIdDto, Vec<PredEdgeEntryDto>)> =
        pred_groups.into_iter().map(|(ctx, edges)| (UnitIdDto::from(ctx), edges)).collect();
    prediction_edge_groups.sort_by_key(|(ctx, _)| (ctx.is_chunk, ctx.raw));

    let merge_candidates: Vec<MergeCandidateDto> = model
        .merge_candidates_iter()
        .map(|((left, right), count)| MergeCandidateDto {
            left: (*left).into(),
            right: (*right).into(),
            count: *count,
        })
        .collect();

    // §26: source-grouped association edges — source appears once per group.
    let association_edges: Vec<AssociationEdgeDto> = Vec::new(); // deprecated; new saves use groups
    let mut assoc_groups: std::collections::HashMap<UnitId, Vec<AssocEdgeEntryDto>> =
        std::collections::HashMap::new();
    for e in model.associations.iter_all() {
        let delta = encode_tick_delta(e.last_used, model_tick);
        assoc_groups.entry(e.source).or_default().push(AssocEdgeEntryDto {
            target: e.target.into(),
            strength: e.strength,
            use_count: e.use_count,
            last_used: if delta > 0 { 0 } else { e.last_used },
            last_used_delta: delta,
        });
    }
    let mut association_edge_groups: Vec<(UnitIdDto, Vec<AssocEdgeEntryDto>)> =
        assoc_groups.into_iter().map(|(src, edges)| (UnitIdDto::from(src), edges)).collect();
    association_edge_groups.sort_by_key(|(src, _)| (src.is_chunk, src.raw));

    // Phase 2: identity / view store
    let identities: Vec<Vec<u32>> = model.identities.all_identities().to_vec();
    let views: Vec<ViewDto> = model.identities.all_views()
        .map(|(units, identity)| ViewDto {
            units: units.iter().copied().map(UnitIdDto::from).collect(),
            identity,
        })
        .collect();

    // §19: lineage_entries omitted from new saves — reconstructed from chunks on load.
    let lineage_entries: Vec<(u32, UnitIdDto, UnitIdDto)> = Vec::new();

    let identity_parent: Vec<u32> = model.identities.all_parents().to_vec();
    let representation_entries: Vec<RepresentationEntryDto> =
        model.representations.iter_all().map(RepresentationEntryDto::from).collect();

    let transforms: Vec<TransformEntryDto> = model.transforms.iter_all()
        .map(|t| {
            let (kind_tag, kind_arg1, kind_arg2) = match &t.kind {
                TransformKind::EquivalentView => (0u8, 0u32, 0u32),
                TransformKind::Mapping       => (1u8, 0u32, 0u32),
                TransformKind::Inverse(a)    => (2u8, *a,   0u32),
                TransformKind::Composed(a,b) => (3u8, *a,   *b),
            };
            TransformEntryDto {
                source: t.source,
                target: t.target,
                kind_tag,
                kind_arg1,
                kind_arg2,
                value: t.value,
                evidence: t.evidence,
            }
        })
        .collect();

    // §22: serialize merge_right_reuse so factorization pressure survives reload.
    let merge_right_reuse: Vec<(UnitIdDto, Vec<UnitIdDto>)> = model
        .merge_right_reuse
        .iter()
        .map(|(right, lefts)| {
            (UnitIdDto::from(*right), lefts.iter().map(|l| UnitIdDto::from(*l)).collect())
        })
        .collect();

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
        representation_entries,
        transforms,
        merge_right_reuse,
        checkpoint_generation: 0,
        prediction_edge_groups,
        association_edge_groups,
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
        let chunk = chunks.get_mut(id).unwrap();
        chunk.tier = dto.tier.into();
        chunk.use_count = dto.use_count;
        chunk.usage_strength = dto.usage_strength as f64; // Phase 17: f32 → f64
        chunk.last_used = decode_tick_delta(dto.last_used_delta, snap.tick, dto.last_used);
        chunk.feedback_value = dto.feedback_value as f64;
        chunk.feedback_count = dto.feedback_count;
        // Phase 8: use registry demote() so hot_pair_to_id stays consistent.
        if dto.residency == 1 {
            chunks.demote(id);
        }
    }

    // §25: use grouped format when present; fall back to flat for legacy files.
    let mut predictions = PredictionStore::new();
    if !snap.prediction_edge_groups.is_empty() {
        for (ctx_dto, entries) in snap.prediction_edge_groups {
            let ctx: UnitId = ctx_dto.into();
            for e in entries {
                let edge = predictions.get_or_create(ctx, e.next_unit.into());
                edge.use_count = e.use_count;
                edge.usage_strength = e.usage_strength as f64;
                edge.feedback_value = e.feedback_value as f64;
                edge.feedback_count = e.feedback_count;
                edge.avoidance = e.avoidance as f64;
                edge.last_used_tick = decode_tick_delta(e.last_used_tick_delta, snap.tick, e.last_used_tick);
                edge.external_route_evidence = e.external_route_evidence as f64;
            }
        }
    } else {
        for dto in snap.prediction_edges {
            let edge = predictions.get_or_create(dto.context.into(), dto.next_unit.into());
            edge.use_count = dto.use_count;
            edge.usage_strength = dto.usage_strength as f64;
            edge.feedback_value = dto.feedback_value as f64;
            edge.feedback_count = dto.feedback_count;
            edge.avoidance = dto.avoidance as f64;
            edge.last_used_tick = dto.last_used_tick;
            edge.external_route_evidence = dto.external_route_evidence as f64;
        }
    }

    let mut merge_candidates: HashMap<(UnitId, UnitId), u32> = HashMap::new();
    for dto in snap.merge_candidates {
        merge_candidates.insert((dto.left.into(), dto.right.into()), dto.count);
    }

    // §26: use grouped format when present; fall back to flat for legacy files.
    let raw_assoc: Vec<AssociationEdge> = if !snap.association_edge_groups.is_empty() {
        snap.association_edge_groups.into_iter().flat_map(|(src_dto, entries)| {
            let src: UnitId = src_dto.into();
            entries.into_iter().map(move |e| AssociationEdge {
                source: src,
                target: e.target.into(),
                strength: e.strength,
                use_count: e.use_count,
                last_used: decode_tick_delta(e.last_used_delta, snap.tick, e.last_used),
            })
        }).collect()
    } else {
        snap.association_edges.into_iter().map(|d| AssociationEdge {
            source: d.source.into(),
            target: d.target.into(),
            strength: d.strength,
            use_count: d.use_count,
            last_used: d.last_used,
        }).collect()
    };
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

    // §19: Lineage is fully derivable from chunks (each chunk stores left+right).
    // New saves write lineage_entries=[]; legacy files may still carry it — use
    // the stored data when present so old saves load correctly.
    let lineage = if !snap.lineage_entries.is_empty() {
        let bulk: Vec<(u32, UnitId, UnitId)> = snap.lineage_entries.into_iter()
            .map(|(c, l, r)| (c, UnitId::from(l), UnitId::from(r)))
            .collect();
        LineageStore::from_bulk(bulk)
    } else {
        let bulk: Vec<(u32, UnitId, UnitId)> = chunks.iter_all()
            .map(|ch| (ch.id, ch.left, ch.right))
            .collect();
        LineageStore::from_bulk(bulk)
    };

    // §26: RelationStore is derived — create empty; rebuild_relations() below fills it.
    let relations = RelationStore::new();

    // Phase B: restore Representation Lineage.
    // Legacy snapshots have no entries — no synthetic history is created (REP-07).
    let representation_bulk: Vec<RepresentationEntry> = snap.representation_entries
        .into_iter()
        .enumerate()
        .map(|(idx, d)| {
            let units: Vec<UnitId> = d.units.into_iter().map(UnitId::from).collect();
            RepresentationEntry {
                rep_id: idx as u32, // §17: implicit id from position in snapshot array
                identity_id: d.identity_id,
                units,
                acquired_at: d.acquired_at,
                practice_count: d.practice_count,
                confidence: d.confidence,
                predecessor_rep_id: d.predecessor_rep_id,
            }
        })
        .collect();
    let representations = RepresentationStore::from_bulk(representation_bulk);

    // §19: rebuild TransformStore from DTOs.
    let mut transforms = TransformStore::new();
    for dto in snap.transforms {
        let kind = match dto.kind_tag {
            0 => TransformKind::EquivalentView,
            1 => TransformKind::Mapping,
            2 => TransformKind::Inverse(dto.kind_arg1),
            3 => TransformKind::Composed(dto.kind_arg1, dto.kind_arg2),
            _ => TransformKind::Mapping,
        };
        transforms.restore_entry(dto.source, dto.target, kind, dto.value, dto.evidence);
    }

    // §22: restore merge_right_reuse so factorization pressure is correct after load.
    use std::collections::HashSet;
    let merge_right_reuse: std::collections::HashMap<crate::units::UnitId, HashSet<crate::units::UnitId>> =
        snap.merge_right_reuse
            .into_iter()
            .map(|(right_dto, lefts_dto)| {
                let right = crate::units::UnitId::from(right_dto);
                let lefts: HashSet<_> = lefts_dto.into_iter().map(crate::units::UnitId::from).collect();
                (right, lefts)
            })
            .collect();

    let mut model = ModelState::from_parts(
        primitives,
        chunks,
        predictions,
        associations,
        identities,
        lineage,
        relations,
        representations,
        transforms,
        snap.tick,
        merge_candidates,
        merge_right_reuse,
        Metrics {
            total_decisions: snap.metrics.total_decisions,
            total_characters: snap.metrics.total_characters,
        },
        snap.next_trace_id,
    );
    // §26: rebuild RelationStore from canonical stores (Predictions, Associations, Lineage).
    model.rebuild_relations();
    // §7: rebuild unit→identity direct index from primitives + chunks.
    model.rebuild_unit_identities();
    model
}

// ── §1/§2 Storage Profiler ─────────────────────────────────────────────────

/// Per-section storage measurement.
#[derive(Debug, Default)]
pub struct SectionStats {
    pub name: &'static str,
    pub count: usize,
    /// Serialized size in bytes (bincode).
    pub bincode_bytes: usize,
    /// Serialized size in bytes (JSON).
    pub json_bytes: usize,
}

/// Full storage profile for a model file.
#[derive(Debug)]
pub struct StorageProfile {
    pub sections: Vec<SectionStats>,
    pub json_total_bytes: usize,
    pub bincode_total_bytes: usize,
    pub zstd_json_bytes: usize,
    pub zstd_bincode_bytes: usize,
    pub json_save_ms: u64,
    pub json_load_ms: u64,
    pub bincode_save_ms: u64,
    pub bincode_load_ms: u64,
    pub file_size_on_disk: u64,
}

/// §1/§2: Profile the storage of a model and print a report to stdout.
///
/// Measures per-section sizes (bincode), JSON/bincode totals, Zstd ratios,
/// and save/load timing.  No files are written to disk during profiling.
pub fn profile_storage(model: &ModelState) -> StorageProfile {
    use std::time::Instant;

    let snap = to_snapshot(model);

    // ── Per-section bincode + JSON sizes ─────────────────────────────────
    macro_rules! section {
        ($name:expr, $field:expr) => {{
            let bc = bincode::serialize(&$field).map(|v| v.len()).unwrap_or(0);
            let js = serde_json::to_string(&$field).map(|s| s.len()).unwrap_or(0);
            SectionStats { name: $name, count: $field.len(), bincode_bytes: bc, json_bytes: js }
        }};
    }
    macro_rules! section_scalar {
        ($name:expr, $field:expr) => {{
            let bc = bincode::serialize(&$field).map(|v| v.len()).unwrap_or(0);
            let js = serde_json::to_string(&$field).map(|s| s.len()).unwrap_or(0);
            SectionStats { name: $name, count: 1, bincode_bytes: bc, json_bytes: js }
        }};
    }

    let sections = vec![
        section!("primitives",              snap.primitives),
        section!("chunks",                  snap.chunks),
        section!("prediction_edge_groups",  snap.prediction_edge_groups),
        section!("association_edge_groups", snap.association_edge_groups),
        section!("merge_candidates",        snap.merge_candidates),
        section!("identities",              snap.identities),
        section!("identity_parent",         snap.identity_parent),
        section!("views",                   snap.views),
        section!("representation_entries",  snap.representation_entries),
        section!("transforms",              snap.transforms),
        section!("merge_right_reuse",       snap.merge_right_reuse),
        section_scalar!("metrics",       snap.metrics),
    ];

    // ── Total JSON / bincode ───────────────────────────────────────────────
    let snap2 = to_snapshot(model);
    let json_bytes = serde_json::to_string(&snap2).map(|s| s.len()).unwrap_or(0);
    let snap3 = to_snapshot(model);
    let bincode_bytes = bincode::serialize(&snap3).map(|v| v.len()).unwrap_or(0);

    // ── Zstd compressed sizes ──────────────────────────────────────────────
    let snap_json_str = serde_json::to_string(&to_snapshot(model)).unwrap_or_default();
    let zstd_json_bytes = zstd::encode_all(snap_json_str.as_bytes(), 3)
        .map(|v| v.len()).unwrap_or(0);
    let snap_bc = bincode::serialize(&to_snapshot(model)).unwrap_or_default();
    let zstd_bincode_bytes = zstd::encode_all(snap_bc.as_slice(), 3)
        .map(|v| v.len()).unwrap_or(0);

    // ── Save / load timing (in-memory, no disk) ────────────────────────────
    let t0 = Instant::now();
    let _jb = serde_json::to_string(&to_snapshot(model)).unwrap_or_default();
    let json_save_ms = t0.elapsed().as_millis() as u64;

    let t0 = Instant::now();
    let _: Result<ModelState, _> = serde_json::from_str(&_jb)
        .map(|s: ModelSnapshot| from_snapshot(s));
    let json_load_ms = t0.elapsed().as_millis() as u64;

    let t0 = Instant::now();
    let bc_bytes = bincode::serialize(&to_snapshot(model)).unwrap_or_default();
    let bincode_save_ms = t0.elapsed().as_millis() as u64;

    let t0 = Instant::now();
    let _: Result<ModelState, _> = bincode::deserialize::<ModelSnapshot>(&bc_bytes)
        .map(from_snapshot);
    let bincode_load_ms = t0.elapsed().as_millis() as u64;

    StorageProfile {
        sections,
        json_total_bytes: json_bytes,
        bincode_total_bytes: bincode_bytes,
        zstd_json_bytes,
        zstd_bincode_bytes,
        json_save_ms,
        json_load_ms,
        bincode_save_ms,
        bincode_load_ms,
        file_size_on_disk: 0, // filled in by caller from actual file
    }
}

/// Print the storage profile in tabular form.
pub fn print_storage_profile(profile: &StorageProfile) {
    println!("{:<28} {:>8} {:>12} {:>12}", "Section", "Count", "Bincode", "JSON");
    println!("{}", "-".repeat(64));
    for s in &profile.sections {
        println!("{:<28} {:>8} {:>12} {:>12}",
            s.name, s.count, fmt_bytes(s.bincode_bytes), fmt_bytes(s.json_bytes));
    }
    println!("{}", "-".repeat(64));
    println!("{:<28} {:>8} {:>12} {:>12}", "TOTAL", "",
        fmt_bytes(profile.bincode_total_bytes), fmt_bytes(profile.json_total_bytes));
    println!();
    println!("Compressed (Zstd-3):");
    println!("  JSON  → Zstd : {} ({:.1}×)",
        fmt_bytes(profile.zstd_json_bytes),
        ratio(profile.json_total_bytes, profile.zstd_json_bytes));
    println!("  STM   → Zstd : {} ({:.1}×)",
        fmt_bytes(profile.zstd_bincode_bytes),
        ratio(profile.bincode_total_bytes, profile.zstd_bincode_bytes));
    println!();
    println!("Serialize timing (in-memory):");
    println!("  JSON   save: {}ms   load: {}ms", profile.json_save_ms, profile.json_load_ms);
    println!("  Bincode save: {}ms  load: {}ms", profile.bincode_save_ms, profile.bincode_load_ms);
    if profile.file_size_on_disk > 0 {
        println!();
        println!("File on disk: {}", fmt_bytes(profile.file_size_on_disk as usize));
    }
}

fn fmt_bytes(n: usize) -> String {
    if n >= 1_048_576 { format!("{:.2} MB", n as f64 / 1_048_576.0) }
    else if n >= 1024 { format!("{:.1} KB", n as f64 / 1024.0) }
    else { format!("{} B", n) }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
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
        let mut model = trained_model();
        let file = NamedTempFile::new().unwrap();
        save(&model, file.path()).unwrap();
        let mut loaded = load(file.path()).unwrap();
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

    // ── Phase 16: binary persistence ────────────────────────────────────

    #[test]
    fn test_binary_save_load_roundtrip() {
        let model = trained_model();
        let file = NamedTempFile::new().unwrap();
        save_binary(&model, file.path()).unwrap();
        let loaded = load_binary(file.path()).unwrap();
        assert_eq!(loaded.primitive_count(), model.primitive_count());
        assert_eq!(loaded.chunk_count(), model.chunk_count());
        assert_eq!(loaded.edge_count(), model.edge_count());
        assert_eq!(loaded.metrics.total_characters, model.metrics.total_characters);
    }

    #[test]
    fn test_binary_smaller_than_json() {
        let model = trained_model();
        let json_file  = NamedTempFile::new().unwrap();
        let bin_file   = NamedTempFile::new().unwrap();
        save(&model, json_file.path()).unwrap();
        save_binary(&model, bin_file.path()).unwrap();
        let json_size = std::fs::metadata(json_file.path()).unwrap().len();
        let bin_size  = std::fs::metadata(bin_file.path()).unwrap().len();
        assert!(bin_size < json_size, "binary ({bin_size}) should be smaller than JSON ({json_size})");
    }

    #[test]
    fn test_save_auto_json_extension() {
        use tempfile::Builder;
        let model = trained_model();
        let file = Builder::new().suffix(".json").tempfile().unwrap();
        save_auto(&model, file.path()).unwrap();
        let loaded = load_auto(file.path()).unwrap();
        assert_eq!(loaded.primitive_count(), model.primitive_count());
    }

    #[test]
    fn test_save_auto_bin_extension() {
        use tempfile::Builder;
        let model = trained_model();
        let file = Builder::new().suffix(".syntrail").tempfile().unwrap();
        save_auto(&model, file.path()).unwrap();
        let loaded = load_auto(file.path()).unwrap();
        assert_eq!(loaded.primitive_count(), model.primitive_count());
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
