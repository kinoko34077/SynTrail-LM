/// Phase 13: Micro-ISA — typed instruction set for SynTrail-LM operations.
///
/// All model operations are expressible as a sequence of `Instruction` values.
/// The `Vm` struct holds the execution context and dispatches each opcode.
///
/// Opcodes defined here:
///   ID.RESOLVE   — resolve a UnitId to its canonical IdentityId
///   REL.GET      — fetch the top-1 prediction for a context unit
///   REL.BUMP     — record an observation (context → next_unit)
///   REL.DECAY    — apply tick-based lazy decay to a single prediction edge
///   REL.INV      — look up or derive inverse of a Transform
///   REL.COMPOSE  — look up or derive composition of two Transforms
///   CHUNK.FIND   — find a HOT chunk by (left, right) pair
///   CHUNK.MAKE   — get-or-create a chunk by (left, right) pair
///   CHUNK.SLEEP  — demote a chunk to SLEEP residency
///   CHUNK.WAKE   — reactivate a chunk to HOT residency
///   REP.FALLBACK — predict_with_fallback: try direct, one-level, primitives
///   SELECT       — pick the best unit from a scored candidate list
///   COMPILE      — segment a primitive sequence into a unit sequence
///   EMIT         — generate the next unit given a context unit
use crate::chunks::ChunkRegistry;
use crate::identity::{IdentityId, IdentityStore};
use crate::lineage::LineageStore;
use crate::prediction::PredictionStore;
use crate::segmentation::segment;
use crate::transform::{TransformId, TransformStore};
use crate::units::UnitId;

// ── Instruction set ──────────────────────────────────────────────────────────

/// A single Micro-ISA instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// ID.RESOLVE — resolve `unit` to its canonical IdentityId via the
    /// identity store.  Result stored in `out`.
    IdResolve { unit: UnitId, out: RegId },

    /// REL.GET — top-1 prediction for `context`, writing the winning UnitId
    /// and its score into `out_unit` / `out_score`.
    RelGet { context: UnitId, out_unit: RegId, out_score: RegId },

    /// REL.BUMP — record one observation: `context` → `next_unit` at `tick`.
    RelBump { context: UnitId, next_unit: UnitId, tick: u64 },

    /// REL.DECAY — advance the lazy decay for a specific prediction edge to
    /// `current_tick` (read-only; does not mutate stored strength).
    RelDecay { context: UnitId, next_unit: UnitId, tick: u64, out_score: RegId },

    /// REL.INV — derive or retrieve the inverse of Transform `tid`.
    RelInv { tid: TransformId, out: RegId },

    /// REL.COMPOSE — compose Transforms `f` then `g`.  Writes the resulting
    /// TransformId (or NONE sentinel) to `out`.
    RelCompose { f: TransformId, g: TransformId, out: RegId },

    /// CHUNK.FIND — look up a HOT chunk by pair, writing ChunkId or NONE to `out`.
    ChunkFind { left: UnitId, right: UnitId, out: RegId },

    /// CHUNK.MAKE — get-or-create a chunk (adds to HOT index).
    ChunkMake { left: UnitId, right: UnitId, expanded_length: u32, out: RegId },

    /// CHUNK.SLEEP — demote chunk `id` to SLEEP residency.
    ChunkSleep { id: u32 },

    /// CHUNK.WAKE — reactivate chunk `id` to HOT residency.
    ChunkWake { id: u32 },

    /// REP.FALLBACK — attempt predict_with_fallback for `context`:
    /// direct prediction → one-level decomposition → full primitive chain.
    /// Writes winning UnitId (or NONE) to `out_unit`, score to `out_score`.
    RepFallback { context: UnitId, out_unit: RegId, out_score: RegId },

    /// SELECT — choose the best candidate from a scored list.
    /// Takes a Vec of (UnitId, f64) already computed by the caller,
    /// writes the winner to `out`.
    Select { candidates: Vec<(UnitId, f64)>, out: RegId },

    /// COMPILE — segment a primitive-id sequence `prims` into units.
    /// `prims` contains raw PrimitiveIds (u32), not UnitIds.
    /// Writes the resulting `Vec<UnitId>` to `out`.
    Compile { prims: Vec<u32>, out: RegId },

    /// EMIT — generate one next unit given `context`, using REP.FALLBACK
    /// internally.  Writes the result (or NONE) to `out`.
    Emit { context: UnitId, out: RegId },
}

/// Register identifier — a small index into the Vm's register file.
pub type RegId = u8;

/// Sentinel value written to an integer register when no result exists.
pub const NONE_U32: u32 = u32::MAX;
/// Sentinel value written to a score register when no result exists.
pub const NONE_F64: f64 = f64::NAN;

// ── Register file ────────────────────────────────────────────────────────────

/// A single register can hold a u32 (id), an f64 (score), a UnitId, or a
/// Vec<UnitId> (compiled unit sequence).
#[derive(Debug, Clone)]
pub enum RegValue {
    None,
    U32(u32),
    F64(f64),
    Unit(UnitId),
    Units(Vec<UnitId>),
}

/// Fixed-size register file (256 slots).
#[derive(Debug, Clone)]
pub struct Registers([RegValue; 256]);

impl Default for Registers {
    fn default() -> Self {
        // SAFETY: RegValue::None is Copy-able via clone; const init is fine here.
        Self(std::array::from_fn(|_| RegValue::None))
    }
}

impl Registers {
    pub fn write_u32(&mut self, r: RegId, v: u32)           { self.0[r as usize] = RegValue::U32(v); }
    pub fn write_f64(&mut self, r: RegId, v: f64)           { self.0[r as usize] = RegValue::F64(v); }
    pub fn write_unit(&mut self, r: RegId, v: UnitId)       { self.0[r as usize] = RegValue::Unit(v); }
    pub fn write_units(&mut self, r: RegId, v: Vec<UnitId>) { self.0[r as usize] = RegValue::Units(v); }
    pub fn write_none(&mut self, r: RegId)                  { self.0[r as usize] = RegValue::None; }

    pub fn read_u32(&self, r: RegId) -> Option<u32>           { if let RegValue::U32(v)    = &self.0[r as usize] { Some(*v)    } else { None } }
    pub fn read_f64(&self, r: RegId) -> Option<f64>           { if let RegValue::F64(v)    = &self.0[r as usize] { Some(*v)    } else { None } }
    pub fn read_unit(&self, r: RegId) -> Option<UnitId>       { if let RegValue::Unit(v)   = &self.0[r as usize] { Some(*v)    } else { None } }
    pub fn read_units(&self, r: RegId) -> Option<&[UnitId]>   { if let RegValue::Units(v)  = &self.0[r as usize] { Some(v)     } else { None } }
    pub fn is_none(&self, r: RegId) -> bool                   { matches!(&self.0[r as usize], RegValue::None) }
}

// ── Virtual machine ──────────────────────────────────────────────────────────

/// Execution context for Micro-ISA programs.
///
/// The VM borrows the relevant model stores mutably so that REL.BUMP /
/// CHUNK.MAKE / CHUNK.SLEEP etc. can update state in-place.
pub struct Vm<'a> {
    pub regs: Registers,
    pub chunks: &'a mut ChunkRegistry,
    pub predictions: &'a mut PredictionStore,
    pub identities: &'a mut IdentityStore,
    pub lineage: &'a LineageStore,
    pub transforms: &'a mut TransformStore,
}

impl<'a> Vm<'a> {
    pub fn new(
        chunks: &'a mut ChunkRegistry,
        predictions: &'a mut PredictionStore,
        identities: &'a mut IdentityStore,
        lineage: &'a LineageStore,
        transforms: &'a mut TransformStore,
    ) -> Self {
        Self { regs: Registers::default(), chunks, predictions, identities, lineage, transforms }
    }

    /// Execute a single instruction.
    pub fn exec(&mut self, instr: &Instruction) {
        match instr {
            Instruction::IdResolve { unit, out } => {
                // Map UnitId → canonical IdentityId.  Primitives map 1-to-1
                // (their PrimitiveId equals an IdentityId if it was interned);
                // for now we return the raw primitive id or NONE_U32 for chunks.
                let id: IdentityId = if unit.is_primitive() {
                    self.identities.canonical(unit.raw())
                } else {
                    NONE_U32
                };
                self.regs.write_u32(*out, id);
            }

            Instruction::RelGet { context, out_unit, out_score } => {
                if let Some((u, s)) = self.predictions.top1_with_score(*context) {
                    self.regs.write_unit(*out_unit, u);
                    self.regs.write_f64(*out_score, s);
                } else {
                    self.regs.write_none(*out_unit);
                    self.regs.write_f64(*out_score, NONE_F64);
                }
            }

            Instruction::RelBump { context, next_unit, tick } => {
                self.predictions.observe_at(*context, *next_unit, *tick);
            }

            Instruction::RelDecay { context, next_unit, tick, out_score } => {
                let score = self.predictions
                    .edge_lazy_strength(*context, *next_unit, *tick)
                    .unwrap_or(NONE_F64);
                self.regs.write_f64(*out_score, score);
            }

            Instruction::RelInv { tid, out } => {
                match self.transforms.inverse(*tid) {
                    Some(inv) => self.regs.write_u32(*out, inv),
                    None      => self.regs.write_u32(*out, NONE_U32),
                }
            }

            Instruction::RelCompose { f, g, out } => {
                match self.transforms.compose(*f, *g) {
                    Some(c) => self.regs.write_u32(*out, c),
                    None    => self.regs.write_u32(*out, NONE_U32),
                }
            }

            Instruction::ChunkFind { left, right, out } => {
                match self.chunks.find_by_pair_hot(*left, *right) {
                    Some(id) => self.regs.write_u32(*out, id),
                    None     => self.regs.write_u32(*out, NONE_U32),
                }
            }

            Instruction::ChunkMake { left, right, expanded_length, out } => {
                let id = self.chunks.get_or_create(*left, *right, *expanded_length);
                self.regs.write_u32(*out, id);
            }

            Instruction::ChunkSleep { id } => {
                self.chunks.demote(*id);
            }

            Instruction::ChunkWake { id } => {
                self.chunks.reactivate(*id);
            }

            Instruction::RepFallback { context, out_unit, out_score } => {
                // Direct prediction.
                if let Some((u, s)) = self.predictions.top1_with_score(*context) {
                    self.regs.write_unit(*out_unit, u);
                    self.regs.write_f64(*out_score, s);
                    return;
                }
                // One-level decomposition via lineage.
                let one_level = self.lineage.decompose_one(*context);
                if one_level.len() > 1 || one_level.first() != Some(context) {
                    for &u in one_level.iter().rev() {
                        if let Some((r, s)) = self.predictions.top1_with_score(u) {
                            self.regs.write_unit(*out_unit, r);
                            self.regs.write_f64(*out_score, s);
                            return;
                        }
                    }
                }
                // Full primitive decomposition.
                let prims = self.lineage.decompose_to_primitives(*context);
                for &u in prims.iter().rev() {
                    if u != *context {
                        if let Some((r, s)) = self.predictions.top1_with_score(u) {
                            self.regs.write_unit(*out_unit, r);
                            self.regs.write_f64(*out_score, s);
                            return;
                        }
                    }
                }
                self.regs.write_none(*out_unit);
                self.regs.write_f64(*out_score, NONE_F64);
            }

            Instruction::Select { candidates, out } => {
                let best = candidates
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                match best {
                    Some((u, _)) => self.regs.write_unit(*out, *u),
                    None         => self.regs.write_none(*out),
                }
            }

            Instruction::Compile { prims, out } => {
                let units = segment(prims, self.chunks, 0.0);
                self.regs.write_units(*out, units);
            }

            Instruction::Emit { context, out } => {
                // Equivalent to a REP.FALLBACK dispatched inline.
                self.exec(&Instruction::RepFallback {
                    context: *context,
                    out_unit: *out,
                    out_score: 255,
                });
            }
        }
    }

    /// Execute a program (slice of instructions) in order.
    pub fn run(&mut self, program: &[Instruction]) {
        for instr in program {
            self.exec(instr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::association::AssociationStore;
    use crate::lineage::LineageStore;

    fn p(id: u32) -> UnitId { UnitId::primitive(id) }
    fn c(id: u32) -> UnitId { UnitId::chunk(id) }

    fn fresh_vm_parts() -> (ChunkRegistry, PredictionStore, IdentityStore, LineageStore, TransformStore) {
        (
            ChunkRegistry::new(),
            PredictionStore::new(),
            IdentityStore::new(),
            LineageStore::new(),
            TransformStore::new(),
        )
    }

    #[test]
    fn isa01_rel_bump_and_get() {
        let (mut chunks, mut preds, mut ids, lin, mut ts) = fresh_vm_parts();
        let mut vm = Vm::new(&mut chunks, &mut preds, &mut ids, &lin, &mut ts);
        vm.exec(&Instruction::RelBump { context: p(1), next_unit: p(2), tick: 1 });
        vm.exec(&Instruction::RelGet  { context: p(1), out_unit: 0, out_score: 1 });
        assert_eq!(vm.regs.read_unit(0), Some(p(2)));
        assert!(vm.regs.read_f64(1).map(|s| s > 0.0).unwrap_or(false));
    }

    #[test]
    fn isa02_chunk_make_and_find() {
        let (mut chunks, mut preds, mut ids, lin, mut ts) = fresh_vm_parts();
        let mut vm = Vm::new(&mut chunks, &mut preds, &mut ids, &lin, &mut ts);
        vm.exec(&Instruction::ChunkMake { left: p(1), right: p(2), expanded_length: 2, out: 0 });
        let id = vm.regs.read_u32(0).unwrap();
        assert_ne!(id, NONE_U32);
        vm.exec(&Instruction::ChunkFind { left: p(1), right: p(2), out: 1 });
        assert_eq!(vm.regs.read_u32(1), Some(id));
    }

    #[test]
    fn isa03_chunk_sleep_wake() {
        let (mut chunks, mut preds, mut ids, lin, mut ts) = fresh_vm_parts();
        let mut vm = Vm::new(&mut chunks, &mut preds, &mut ids, &lin, &mut ts);
        vm.exec(&Instruction::ChunkMake { left: p(1), right: p(2), expanded_length: 2, out: 0 });
        let id = vm.regs.read_u32(0).unwrap();
        vm.exec(&Instruction::ChunkSleep { id });
        // After sleep, find_by_pair_hot should return None.
        vm.exec(&Instruction::ChunkFind { left: p(1), right: p(2), out: 1 });
        assert_eq!(vm.regs.read_u32(1), Some(NONE_U32));
        vm.exec(&Instruction::ChunkWake { id });
        vm.exec(&Instruction::ChunkFind { left: p(1), right: p(2), out: 2 });
        assert_eq!(vm.regs.read_u32(2), Some(id));
    }

    #[test]
    fn isa04_select_picks_best() {
        let (mut chunks, mut preds, mut ids, lin, mut ts) = fresh_vm_parts();
        let mut vm = Vm::new(&mut chunks, &mut preds, &mut ids, &lin, &mut ts);
        let candidates = vec![(p(1), 0.3), (p(2), 0.9), (p(3), 0.1)];
        vm.exec(&Instruction::Select { candidates, out: 0 });
        assert_eq!(vm.regs.read_unit(0), Some(p(2)));
    }

    #[test]
    fn isa05_select_empty_gives_none() {
        let (mut chunks, mut preds, mut ids, lin, mut ts) = fresh_vm_parts();
        let mut vm = Vm::new(&mut chunks, &mut preds, &mut ids, &lin, &mut ts);
        vm.exec(&Instruction::Select { candidates: vec![], out: 0 });
        assert!(vm.regs.is_none(0));
    }

    #[test]
    fn isa06_emit_via_fallback() {
        let (mut chunks, mut preds, mut ids, lin, mut ts) = fresh_vm_parts();
        let mut vm = Vm::new(&mut chunks, &mut preds, &mut ids, &lin, &mut ts);
        for _ in 0..5 { vm.exec(&Instruction::RelBump { context: p(1), next_unit: p(2), tick: 1 }); }
        vm.exec(&Instruction::Emit { context: p(1), out: 0 });
        assert_eq!(vm.regs.read_unit(0), Some(p(2)));
    }

    #[test]
    fn isa07_rel_inv_compose() {
        let (mut chunks, mut preds, mut ids, lin, mut ts) = fresh_vm_parts();
        ids.intern_identity(&[1]);
        ids.intern_identity(&[2]);
        ids.intern_identity(&[3]);
        let f = ts.register(0, 1, &mut ids);
        let g = ts.register(1, 2, &mut ids);
        let mut vm = Vm::new(&mut chunks, &mut preds, &mut ids, &lin, &mut ts);
        vm.exec(&Instruction::RelInv     { tid: f, out: 0 });
        vm.exec(&Instruction::RelCompose { f,  g,  out: 1 });
        let inv_id = vm.regs.read_u32(0).unwrap();
        let comp_id = vm.regs.read_u32(1).unwrap();
        assert_ne!(inv_id, NONE_U32);
        assert_ne!(comp_id, NONE_U32);
        let inv = vm.transforms.get(inv_id).unwrap();
        assert_eq!((inv.source, inv.target), (1, 0));
        let comp = vm.transforms.get(comp_id).unwrap();
        assert_eq!((comp.source, comp.target), (0, 2));
    }

    #[test]
    fn isa08_run_program() {
        let (mut chunks, mut preds, mut ids, lin, mut ts) = fresh_vm_parts();
        let mut vm = Vm::new(&mut chunks, &mut preds, &mut ids, &lin, &mut ts);
        let prog = vec![
            Instruction::RelBump { context: p(10), next_unit: p(11), tick: 1 },
            Instruction::RelBump { context: p(10), next_unit: p(11), tick: 2 },
            Instruction::RelGet  { context: p(10), out_unit: 0, out_score: 1 },
        ];
        vm.run(&prog);
        assert_eq!(vm.regs.read_unit(0), Some(p(11)));
    }
}
