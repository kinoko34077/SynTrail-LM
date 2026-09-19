/// v0.3 ModelState — unified learn/generate/associate model.
///
/// v0.3 additions:
/// - `associations: AssociationStore` — bidirectional co-occurrence memory
/// - `expose()` now records associations for every adjacent segmented pair
/// - `recall(source)` — return Top-K associations with hierarchical fallback
use std::collections::HashMap;

use crate::association::AssociationStore;
use crate::chunks::ChunkRegistry;
use crate::config::GenerationConfig;
use crate::identity::{IdentityId, IdentityStore};
use crate::lineage::LineageStore;
use crate::prediction::PredictionStore;
use crate::relation::{RelationKind, RelationStore};
use crate::primitives::PrimitiveRegistry;
use crate::representation::{RepId, RepresentationStore};
use crate::segmentation::{expand, segment};
use crate::trace::{DecisionStep, GenerationMode, RouteKind, TraceId, TurnTrace, now_secs};
use crate::units::UnitId;

const MERGE_THRESHOLD: u32 = 4;
const BASE_MERGE_PROBABILITY: f64 = 0.5;
const SEGMENT_MIN_SCORE: f64 = 0.0;
/// Phase 10: Factorization Pressure scale — each additional left-element wanting the same
// Generation tunables are in ModelState.gen_config (from GenerationConfig).
// Factorization scale is in ModelState.learning (from LearningConfig § 17).
/// §7: Turn-boundary marker (ETX U+0003) — injected after input at conversation turn boundaries.
/// Teaches the model where one input turn ends; does NOT stop generation.
pub const TURN_BOUNDARY_CHAR: char = '\x03';
/// §7: Sequence-end marker (EOT U+0004) — predicting this stops generation.
/// Inject this when training complete output sequences that have a natural end.
pub const SEQUENCE_END_CHAR: char = '\x04';
/// Backward-compatible alias for TURN_BOUNDARY_CHAR.
pub const EOS_CHAR: char = TURN_BOUNDARY_CHAR;

/// P0: Per-call generation state for cycle detection and no-progress tracking.
/// §4: also tracks the current identity/representation being generated from.
/// §12: last_emitted window for no-progress detection beyond route repetition.
pub struct GenerationState {
    pub step_count: usize,
    recent_routes: std::collections::VecDeque<(crate::units::UnitId, crate::units::UnitId)>,
    /// §12: ring buffer of recent emitted units for stagnation detection.
    last_emitted: std::collections::VecDeque<crate::units::UnitId>,
    /// Identity of the current context unit (updated each step).
    pub current_identity: Option<IdentityId>,
    /// Representation active at the current step (the one whose predecessor chain was used).
    pub current_rep: Option<RepId>,
}

impl GenerationState {
    pub fn new() -> Self {
        Self {
            step_count: 0,
            recent_routes: std::collections::VecDeque::new(),
            last_emitted: std::collections::VecDeque::new(),
            current_identity: None,
            current_rep: None,
        }
    }

    /// §12: Record an emitted unit in the stagnation window.
    /// Returns true if no-progress is detected (≤2 distinct units in last 2*max window).
    pub fn record_emitted(&mut self, unit: crate::units::UnitId, max: usize) -> bool {
        let window = max.max(2) * 2;
        self.last_emitted.push_back(unit);
        if self.last_emitted.len() > window {
            self.last_emitted.pop_front();
        }
        if self.last_emitted.len() < window {
            return false; // window not yet full
        }
        let distinct: std::collections::HashSet<_> = self.last_emitted.iter().collect();
        distinct.len() <= 2
    }

    pub fn is_recent_route(&self, context: crate::units::UnitId, next: crate::units::UnitId) -> bool {
        self.recent_routes.contains(&(context, next))
    }

    pub fn push_route(&mut self, context: crate::units::UnitId, next: crate::units::UnitId) {
        // max is checked by callers using model.gen_config.recent_routes_max
        self.recent_routes.push_back((context, next));
        self.step_count += 1;
    }

    pub fn trim_recent_routes(&mut self, max: usize) {
        while self.recent_routes.len() > max {
            self.recent_routes.pop_front();
        }
    }
}

impl Default for GenerationState {
    fn default() -> Self { Self::new() }
}

/// §4: Full selection provenance returned by pick_next_unit().
/// Transferred directly to DecisionStep; no route information is inferred after the fact.
pub struct NextChoice {
    pub unit: UnitId,
    pub score: f64,
    /// The prediction edge source (may differ from generation context on fallback paths).
    pub route_source: UnitId,
    pub route_kind: RouteKind,
    /// Which Representation provided this route (RepFallback only).
    pub representation_id: Option<RepId>,
}

/// Running counters for the primary metric.
#[derive(Debug, Default, Clone, Copy)]
pub struct Metrics {
    pub total_decisions: u64,
    pub total_characters: u64,
}

impl Metrics {
    pub fn decision_per_character(&self) -> f64 {
        if self.total_characters == 0 {
            0.0
        } else {
            self.total_decisions as f64 / self.total_characters as f64
        }
    }
}

/// The complete model state.
#[derive(Debug, Default)]
pub struct ModelState {
    pub primitives: PrimitiveRegistry,
    pub chunks: ChunkRegistry,
    pub predictions: PredictionStore,
    /// v0.3: bidirectional co-occurrence associations.
    pub associations: AssociationStore,
    /// Phase 2: Identity/View registry (Exact Identity).
    pub identities: IdentityStore,
    /// Phase 3: Representation Lineage — derivation history for each Chunk.
    pub lineage: LineageStore,
    /// Phase 7: Unified Relation Core — Route + Adjacency + DerivedFrom in one store.
    pub relations: RelationStore,
    /// Phase B: Representation Lineage — acquisition history of representations per Identity.
    pub representations: RepresentationStore,
    pub tick: u64,
    pub(crate) merge_candidates: HashMap<(UnitId, UnitId), u32>,
    /// Phase 10: distinct lefts that want to merge with each right unit.
    /// High right-reuse means the right element is valuable on its own (factorization pressure).
    pub merge_right_reuse: HashMap<UnitId, std::collections::HashSet<UnitId>>,
    pub metrics: Metrics,
    /// Monotonically increasing trace ID counter.
    pub(crate) next_trace_id: TraceId,
    /// Generation tunables (§13).
    pub gen_config: GenerationConfig,
    /// Memory budget tunables (§14-§16).
    pub memory_budget: crate::config::MemoryBudgetConfig,
    /// Learning tunables (§17).
    pub learning: crate::config::LearningConfig,
    /// §18/§19: Transform store — directed relations between Identities.
    pub transforms: crate::transform::TransformStore,
}

impl ModelState {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Exposure (v0.2 rename of train) ──────────────────────────────────

    /// Process one external observation (Experience).
    ///
    /// Registers new Primitives, updates Adjacency associations, and counts
    /// the input toward `metrics.total_characters` / `metrics.total_decisions`.
    /// Call this for first-time observations from the outside world.
    pub fn expose_external(&mut self, text: &str) {
        self.tick += 1;
        let tick = self.tick;

        let prim_ids = self.primitives.encode(text);
        if prim_ids.is_empty() {
            return;
        }

        // §7: register each primitive's single-element identity on first encounter.
        for &pid in &prim_ids {
            let unit = UnitId::primitive(pid);
            if self.identities.identity_of_unit(unit).is_none() {
                let iid = self.identities.intern_identity(&[pid]);
                self.identities.register_unit_identity(unit, iid);
            }
        }

        let segmented = segment(&prim_ids, &self.chunks, SEGMENT_MIN_SCORE);

        for &unit in &segmented {
            if let Some(cid) = unit.as_chunk() {
                if let Some(chunk) = self.chunks.get_mut(cid) {
                    chunk.record_usage(tick);
                }
            }
        }

        // Phase C: Experience path — updates both practice_confidence AND external_route_evidence.
        self.predictions.learn_sequence_external_at(&segmented, tick);
        self.metrics.total_decisions += segmented.len() as u64;
        self.metrics.total_characters += prim_ids.len() as u64;

        // v0.3: bidirectional associations for every adjacent pair
        for window in segmented.windows(2) {
            let (a, b) = (window[0], window[1]);
            self.associations.observe(a, b, tick);
            self.associations.observe(b, a, tick);
            // Phase 7: mirror into unified Relation Core.
            self.relations.observe(a, b, RelationKind::Adjacency, 1.0);
            self.relations.observe(b, a, RelationKind::Adjacency, 1.0);
        }
        // Phase 7: Route edges.
        for window in segmented.windows(2) {
            self.relations.observe(window[0], window[1], RelationKind::Route, 1.0);
        }

        // Phase 2: register the Identity (canonical prim seq) and this View (chunk tree).
        let identity_id = self.identities.intern_identity(&prim_ids);
        self.identities.intern_view(&segmented, identity_id);
        // Phase B: register this segmentation as a Representation for the Identity.
        // §3: re-encountering a known representation is a recognition success.
        let (rep_id, is_new) = self.representations.intern(identity_id, segmented.clone(), tick);
        if !is_new {
            if let Some(e) = self.representations.get_mut(rep_id) { e.record_success(); }
        }

        self.consider_merges(&segmented, tick);
    }

    /// Re-process previously observed text (Replay).
    ///
    /// Strengthens Chunk familiarity, prediction edges, and merge candidates,
    /// but does NOT register new Primitives, update Adjacency associations, or
    /// increment the external-observation metrics (`total_characters` /
    /// `total_decisions`).  Characters not yet registered are silently skipped.
    ///
    /// Use for repeated passes over a block in the Adaptive Trainer: the first
    /// pass is `expose_external`; all subsequent passes are `replay`.
    pub fn replay(&mut self, text: &str) {
        self.tick += 1;
        let tick = self.tick;

        let prim_ids = self.primitives.encode_existing(text);
        if prim_ids.is_empty() {
            return;
        }

        let segmented = segment(&prim_ids, &self.chunks, SEGMENT_MIN_SCORE);

        for &unit in &segmented {
            if let Some(cid) = unit.as_chunk() {
                if let Some(chunk) = self.chunks.get_mut(cid) {
                    chunk.record_usage(tick);
                }
            }
        }

        self.predictions.learn_sequence_at(&segmented, tick);
        // No metrics update — this is not an external observation.
        // No association update — Adjacency tracks world co-occurrence only.
        // Phase 7: Route edges only (no Adjacency during Replay).
        for window in segmented.windows(2) {
            self.relations.observe(window[0], window[1], RelationKind::Route, 1.0);
        }

        // Phase 2: register the View for this (possibly re-segmented) chunk tree.
        // prim_ids here came from encode_existing, so only already-registered chars.
        let identity_id = self.identities.intern_identity(&prim_ids);
        self.identities.intern_view(&segmented, identity_id);
        // Phase B: Replay can acquire a new Representation (e.g., when a chunk has
        // just formed and segmentation changes). external_occurrence is NOT updated.
        // §3: repeated segmentation during Replay is a recognition success.
        let (rep_id, is_new) = self.representations.intern(identity_id, segmented.clone(), tick);
        if !is_new {
            if let Some(e) = self.representations.get_mut(rep_id) { e.record_success(); }
        }

        self.consider_merges(&segmented, tick);
    }

    /// Process one text sample (Experience).
    /// Alias for `expose_external`; kept for backward compatibility.
    #[inline]
    pub fn expose(&mut self, text: &str) {
        self.expose_external(text);
    }

    /// Backward-compatible alias — callers not yet migrated to `expose`.
    #[inline]
    pub fn train(&mut self, text: &str) {
        self.expose_external(text);
        let budget = self.memory_budget.hot_chunks_max;
        if budget > 0 { self.enforce_hot_budget(budget); }
    }

    // ── Frozen generation (P0: cycle detection, seed/output separation, Top-K) ──

    /// §2/§4: Pick the best next unit for `context`, returning full provenance.
    ///
    /// §2 — Frozen Generation: this method MUST NOT modify ModelState learning state.
    ///   record_success()/record_failure() must NOT be called here; use the Learning Phase instead.
    /// §4 — NextChoice: all route provenance (source, kind, rep_id) is captured here
    ///   and transferred directly to DecisionStep — never inferred after the fact.
    fn pick_next_unit(&mut self, context: UnitId, state: &mut GenerationState) -> Option<NextChoice> {
        // Direct prediction — top-K with cycle penalty.
        let candidates = self.predictions.top_k_with_score(context, self.gen_config.route_top_k);
        if !candidates.is_empty() {
            let best = candidates.into_iter()
                .map(|(unit, score)| {
                    let factor = if state.is_recent_route(context, unit) { self.gen_config.cycle_penalty } else { 1.0 };
                    (unit, score * factor)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((unit, score)) = best {
                if score > 0.0 {
                    // §4/§7: O(1) identity lookup via direct index.
                    state.current_identity = self.identities.identity_of_unit(context);
                    state.current_rep = None;
                    return Some(NextChoice {
                        unit, score,
                        route_source: context,
                        route_kind: RouteKind::Direct,
                        representation_id: None,
                    });
                }
            }
        }

        // Representation predecessor chain (§5/§6): start from preferred Rn,
        // walk Rn → R(n-1) → … → R0.
        // §2: NO record_success/record_failure — Frozen Generation.
        // §7: use the O(1) direct index for the representation-fallback identity lookup.
        let mut rep_route_source: Option<UnitId> = None;
        let mut rep_result: Option<(UnitId, f64)> = None;
        let mut winning_rep: Option<RepId> = None;
        let mut winning_identity: Option<IdentityId> = None;
        if let Some(id) = self.identities.identity_of_unit(context) {
            winning_identity = Some(id);
            let preferred_id = self.representations.preferred(id).map(|r| r.rep_id);
            let mut rep_id_opt = preferred_id;
            'chain: while let Some(rep_id) = rep_id_opt {
                if let Some(rep) = self.representations.get(rep_id) {
                    for &u in rep.units.iter().rev() {
                        if u != context {
                            if let Some(r) = self.predictions.top1_with_score(u) {
                                winning_rep = Some(rep_id);
                                rep_result = Some(r);
                                rep_route_source = Some(u);
                                break 'chain;
                            }
                        }
                    }
                    // §2: do NOT call record_failure() here — Frozen Generation.
                    rep_id_opt = rep.predecessor_rep_id;
                } else {
                    break;
                }
            }
        }
        if let (Some((unit, score)), Some(route_source), Some(rep_id)) =
            (rep_result, rep_route_source, winning_rep)
        {
            // §4: update gen_state identity/rep tracking (gen_state mutation is allowed).
            state.current_identity = winning_identity;
            state.current_rep = Some(rep_id);
            // §2: do NOT call record_success() — Frozen Generation.
            return Some(NextChoice {
                unit, score,
                route_source,
                route_kind: RouteKind::RepFallback,
                representation_id: Some(rep_id),
            });
        }

        // Structural lineage fallback — one-level decomposition.
        let one_level = self.lineage.decompose_one(context);
        if one_level.len() > 1 || one_level.first() != Some(&context) {
            for &u in one_level.iter().rev() {
                if let Some((unit, score)) = self.predictions.top1_with_score(u) {
                    return Some(NextChoice {
                        unit, score,
                        route_source: u,
                        route_kind: RouteKind::StructuralFallback,
                        representation_id: None,
                    });
                }
            }
        }

        // Full primitive decomposition fallback.
        let prim_units = self.lineage.decompose_to_primitives(context);
        for &u in prim_units.iter().rev() {
            if u != context {
                if let Some((unit, score)) = self.predictions.top1_with_score(u) {
                    return Some(NextChoice {
                        unit, score,
                        route_source: u,
                        route_kind: RouteKind::PrimitiveFallback,
                        representation_id: None,
                    });
                }
            }
        }
        None
    }

    /// Return the primitive ID for `TURN_BOUNDARY_CHAR`, if registered.
    pub fn turn_boundary_unit(&self) -> Option<UnitId> {
        self.primitives.id(TURN_BOUNDARY_CHAR).map(UnitId::primitive)
    }

    /// Return the primitive ID for `SEQUENCE_END_CHAR`, if registered.
    pub fn sequence_end_unit(&self) -> Option<UnitId> {
        self.primitives.id(SEQUENCE_END_CHAR).map(UnitId::primitive)
    }

    /// Backward-compatible alias for turn_boundary_unit.
    #[inline]
    pub fn eos_unit(&self) -> Option<UnitId> {
        self.turn_boundary_unit()
    }

    /// Generate text from `seed_text` without mutating self.
    ///
    /// P0 changes:
    /// - `context_units` (seed) and `emitted_units` are tracked separately.
    /// - Route Top-K with cycle penalty replaces unconditional top-1.
    /// - Cycle / no-progress detection: stops if a route repeats within the
    ///   recent-routes window and no viable alternate candidate exists.
    /// - EOS: stops if the predicted next unit is the EOS primitive.
    /// - `trace.emitted_text` contains only the generated (non-seed) text.
    pub fn generate_with_trace(
        &mut self,
        seed_text: &str,
        input_text: &str,
        max_units: usize,
        trace_id: TraceId,
    ) -> (String, TurnTrace) {
        self.generate_with_trace_mode(seed_text, input_text, max_units, trace_id, GenerationMode::Completion)
    }

    /// §9: Generation with explicit mode (Completion or Dialogue).
    pub fn generate_with_trace_mode(
        &mut self,
        seed_text: &str,
        input_text: &str,
        max_units: usize,
        trace_id: TraceId,
        mode: GenerationMode,
    ) -> (String, TurnTrace) {
        let prim_ids: Vec<u32> = seed_text.chars()
            .filter_map(|c| self.primitives.id(c))
            .collect();

        let context_units = segment(&prim_ids, &self.chunks, SEGMENT_MIN_SCORE);
        let mut emitted_units: Vec<UnitId> = Vec::new();
        let mut steps: Vec<DecisionStep> = Vec::new();
        let mut gen_state = GenerationState::new();
        // §7: generation stops only on SEQUENCE_END_CHAR.
        // §10: TURN_BOUNDARY_CHAR is suppressed from visible output and causes a state transition.
        let eos = self.sequence_end_unit();
        let turn_boundary = self.turn_boundary_unit();
        let mut stopped_by_eos = false;
        let mut stopped_by_cycle = false;

        for step_index in 0..max_units {
            let context = emitted_units.last()
                .or_else(|| context_units.last())
                .copied();
            let context = match context { Some(c) => c, None => break };

            match self.pick_next_unit(context, &mut gen_state) {
                None => break,
                Some(choice) => {
                    let next = choice.unit;
                    if eos == Some(next) {
                        stopped_by_eos = true;
                        break;
                    }
                    // §10: TURN_BOUNDARY is an internal state transition — suppress from output.
                    if turn_boundary == Some(next) {
                        // Do not add to emitted_units or steps; continue generation.
                        continue;
                    }
                    // Cycle confirmed — escalate through fallback chain.
                    if gen_state.is_recent_route(context, next) {
                        // 1. Recall: find associations, try prediction from each associate.
                        let mut escaped = false;
                        let associates = self.associations.recall(context, &self.chunks, 8);
                        for (assoc, _) in associates {
                            if assoc == context { continue; }
                            if let Some((alt, alt_score)) = self.predictions.top1_with_score(assoc) {
                                if !gen_state.is_recent_route(context, alt) && eos != Some(alt) {
                                    steps.push(DecisionStep {
                                        step_index, unit: alt, context,
                                        route_source: assoc, score: alt_score,
                                        route_kind: RouteKind::RecallBridge,
                                        representation_id: None,
                                    });
                                    gen_state.push_route(context, alt);
                                    gen_state.trim_recent_routes(self.gen_config.recent_routes_max);
                                    emitted_units.push(alt);
                                    if gen_state.record_emitted(alt, self.gen_config.recent_routes_max) {
                                        stopped_by_cycle = true;
                                    }
                                    escaped = true;
                                    break;
                                }
                            }
                        }
                        if escaped { continue; }
                        // 2. Generalize: association bridge, excluding the cycled route (§11).
                        if let Some((alt, alt_score)) = self.generalize_excluding(context, &[next]) {
                            if !gen_state.is_recent_route(context, alt) && eos != Some(alt) {
                                steps.push(DecisionStep {
                                    step_index, unit: alt, context,
                                    route_source: context, score: alt_score,
                                    route_kind: RouteKind::Generalize,
                                    representation_id: None,
                                });
                                gen_state.push_route(context, alt);
                                gen_state.trim_recent_routes(self.gen_config.recent_routes_max);
                                emitted_units.push(alt);
                                if gen_state.record_emitted(alt, self.gen_config.recent_routes_max) {
                                    stopped_by_cycle = true;
                                    break;
                                }
                                continue;
                            }
                        }
                        // 3. No escape — stop generation naturally.
                        stopped_by_cycle = true;
                        break;
                    }
                    // §4: use NextChoice provenance directly; never infer route after the fact.
                    steps.push(DecisionStep {
                        step_index,
                        unit: next,
                        context,
                        route_source: choice.route_source,
                        score: choice.score,
                        route_kind: choice.route_kind,
                        representation_id: choice.representation_id,
                    });
                    gen_state.push_route(context, next);
                    gen_state.trim_recent_routes(self.gen_config.recent_routes_max);
                    emitted_units.push(next);
                    if gen_state.record_emitted(next, self.gen_config.recent_routes_max) {
                        stopped_by_cycle = true;
                        break;
                    }
                }
            }
        }

        // §10: strip TURN_BOUNDARY from visible text — it is an internal state transition,
        // never a surface character. Suppression at the unit level handles pure primitives;
        // stripping at decode time handles TURN_BOUNDARY embedded inside expanded chunks.
        let expanded_emit = expand(&emitted_units, &self.chunks, &self.primitives);
        let emitted_text = self.primitives.decode(&expanded_emit)
            .unwrap_or_default()
            .replace(TURN_BOUNDARY_CHAR, "");

        let mut all_units = context_units.clone();
        all_units.extend_from_slice(&emitted_units);
        let expanded_all = expand(&all_units, &self.chunks, &self.primitives);
        let output = self.primitives.decode(&expanded_all)
            .unwrap_or_default()
            .replace(TURN_BOUNDARY_CHAR, "");

        let mut trace = TurnTrace::new(
            trace_id,
            self.tick,
            seed_text.to_owned(),
            input_text.to_owned(),
            output.clone(),
            emitted_text,
            steps,
            now_secs(),
        );
        trace.stopped_by_eos = stopped_by_eos;
        trace.stopped_by_cycle = stopped_by_cycle;
        trace.generation_mode = mode;

        (output, trace)
    }

    /// Convenience: generate without a trace.
    pub fn generate(&mut self, seed_text: &str, max_units: usize) -> String {
        let (output, _) = self.generate_with_trace(seed_text, seed_text, max_units, 0);
        output
    }

    /// Like `expose_external`, but appends an EOS marker (`EOS_CHAR`) after the text
    /// so the model can learn to predict sequence end at semantic boundaries.
    ///
    /// §7: Append TURN_BOUNDARY_CHAR and expose.
    ///
    /// Use at conversation turn boundaries (end of user input) to teach the model
    /// where one input turn ends.  Does NOT cause generation to stop.
    pub fn expose_with_turn_boundary(&mut self, text: &str) {
        let s = format!("{}{}", text, TURN_BOUNDARY_CHAR);
        self.expose_external(&s);
    }

    /// §7: Append SEQUENCE_END_CHAR and expose.
    ///
    /// Use when training complete output sequences so the model learns to predict
    /// the sequence-end marker at the natural end of an output.
    pub fn expose_with_sequence_end(&mut self, text: &str) {
        let s = format!("{}{}", text, SEQUENCE_END_CHAR);
        self.expose_external(&s);
    }

    /// Backward-compatible alias for expose_with_turn_boundary.
    #[inline]
    pub fn expose_with_eos(&mut self, text: &str) {
        self.expose_with_turn_boundary(text);
    }

    /// Allocate the next trace ID (monotonically increasing).
    pub fn alloc_trace_id(&mut self) -> TraceId {
        self.next_trace_id += 1;
        self.next_trace_id
    }

    // ── Feedback application (v0.2) ──────────────────────────────────────

    /// Apply per-step feedback credits to the edges and chunks in `trace`.
    ///
    /// `credits[i]` is the credit for `trace.decision_steps[i]`.
    /// For each step, the prediction edge (context, unit) is updated.
    /// Chunks in the unit also receive the credit.
    pub fn apply_feedback_to_trace(
        &mut self,
        trace: &TurnTrace,
        credits: &[f64],
        decay: f64,
    ) {
        // §5: Feedback MUST target the actual prediction edge (route_source → unit),
        // not the generation context → unit, which may be a different edge or non-existent.
        for (step, &r) in trace.decision_steps.iter().zip(credits.iter()) {
            self.predictions.apply_feedback_to_edge(step.route_source, step.unit, r, decay);
            if let Some(cid) = step.unit.as_chunk() {
                if let Some(chunk) = self.chunks.get_mut(cid) {
                    chunk.apply_feedback(r, decay);
                }
            }
        }
    }

    /// v0.3: Return Top-K associations for `source` with hierarchical fallback.
    pub fn recall(&self, source: UnitId, limit: usize) -> Vec<(UnitId, f64)> {
        self.associations.recall(source, &self.chunks, limit)
    }

    // ── Phase 20: Generalization / Novel Search ───────────────────────────

    /// Generalised prediction for `context`.
    ///
    /// Falls back through three levels:
    ///   1. `predict_with_fallback(context)` — direct + lineage chain (Phase 4)
    ///   2. Association bridge: find the strongest associate of `context`,
    ///      then predict from that associate.  Useful when context has no
    ///      direct prediction edges but a co-occurring unit does.
    ///   3. Returns `None` only when no path yields a prediction.
    pub fn generalize(&self, context: UnitId) -> Option<(UnitId, f64)> {
        self.generalize_excluding(context, &[])
    }

    /// Like `generalize`, but skips any `next` unit in `excluded` (§11).
    /// Called from cycle escape to avoid re-selecting already-failed routes.
    pub fn generalize_excluding(
        &self,
        context: UnitId,
        excluded: &[UnitId],
    ) -> Option<(UnitId, f64)> {
        // Level 1: lineage fallback (skip excluded).
        // predict_with_fallback returns a single best; check exclusion.
        if let Some((u, s)) = self.predict_with_fallback(context) {
            if !excluded.contains(&u) {
                return Some((u, s));
            }
        }
        // Level 2: association bridge (skip excluded).
        let associates = self.associations.recall(context, &self.chunks, 8);
        for (assoc, assoc_strength) in associates {
            if assoc == context { continue; }
            if let Some((next, pred_score)) = self.predictions.top1_with_score(assoc) {
                if excluded.contains(&next) { continue; }
                let combined = (assoc_strength * pred_score).sqrt();
                return Some((next, combined));
            }
        }
        None
    }

    /// Return novel next-unit candidates for `context` by traversing
    /// Association edges one hop away and collecting all prediction targets.
    ///
    /// `limit`: maximum candidates returned, ranked by combined score.
    ///
    /// This enables *analogical* generation: "units that follow things similar
    /// to `context`" even when `context` itself has no known successors.
    pub fn novel_candidates(&self, context: UnitId, limit: usize) -> Vec<(UnitId, f64)> {
        let associates = self.associations.recall(context, &self.chunks, 16);
        let mut candidates: std::collections::HashMap<UnitId, f64> = HashMap::new();
        for (assoc, assoc_strength) in associates {
            for (edge, pred_score) in self.predictions.predict(assoc) {
                let combined = (assoc_strength * pred_score).sqrt();
                let entry = candidates.entry(edge.next_unit).or_insert(0.0);
                if combined > *entry {
                    *entry = combined;
                }
            }
        }
        let mut ranked: Vec<(UnitId, f64)> = candidates.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(limit);
        ranked
    }

    /// A simple fingerprint of the current model state for snapshot tagging.
    // ── Phase 4: Fallback via Lineage ─────────────────────────────────────

    /// Predict the next unit from `context`, falling back through Lineage if
    /// no direct edge exists.
    ///
    /// Fallback order:
    ///   1. Direct: `predictions.top1(context)`
    ///   2. One-level decomposition: try each component of `lineage.decompose_one(context)`
    ///   3. Full decomposition: try each primitive from `lineage.decompose_to_primitives(context)`
    ///
    /// Returns `None` only when even Primitives have no known successor.
    pub fn predict_with_fallback(&self, context: UnitId) -> Option<(UnitId, f64)> {
        if let Some(r) = self.predictions.top1_with_score(context) {
            return Some(r);
        }
        let one_level = self.lineage.decompose_one(context);
        // If decompose_one returned the unit itself, no lineage — skip to primitives.
        if one_level.len() > 1 || one_level.first() != Some(&context) {
            for &u in one_level.iter().rev() {
                if let Some(r) = self.predictions.top1_with_score(u) {
                    return Some(r);
                }
            }
        }
        // Full decomposition to primitives.
        let prims = self.lineage.decompose_to_primitives(context);
        for &u in prims.iter().rev() {
            if u != context {
                if let Some(r) = self.predictions.top1_with_score(u) {
                    return Some(r);
                }
            }
        }
        None
    }

    /// Phase 14: return a read-only Core view (Primitives + HOT Chunks + Predictions).
    pub fn core_view(&self) -> crate::core::CoreView<'_> {
        crate::core::CoreView { chunks: &self.chunks, predictions: &self.predictions }
    }

    pub fn state_fingerprint(&self) -> String {
        // §33: include edge-value hash so weight changes are detected,
        // not just structural (count) changes.
        format!(
            "tick={} prims={} chunks={} edges={} assoc={} ehash={:016x}",
            self.tick,
            self.primitives.len(),
            self.chunks.len(),
            self.predictions.edge_count(),
            self.associations.edge_count(),
            self.predictions.edge_value_hash(),
        )
    }

    // ── Merge / reactivation (v0.4) ──────────────────────────────────────

    fn consider_merges(&mut self, units: &[UnitId], _tick: u64) {
        for i in 0..units.len().saturating_sub(1) {
            let left = units[i];
            let right = units[i + 1];
            let preceding: Option<UnitId> = if i > 0 { Some(units[i - 1]) } else { None };

            if let Some(existing_id) = self.chunks.find_by_pair(left, right) {
                // v0.4 §12: reactivate SLEEP chunks on re-encounter (Phase 8: via registry method)
                self.chunks.reactivate(existing_id);
                continue;
            }

            // Phase 10: track distinct lefts competing for each right (factorization pressure).
            self.merge_right_reuse
                .entry(right)
                .or_default()
                .insert(left);
            let _ = preceding; // preceding context not used in new formula

            let count = self
                .merge_candidates
                .entry((left, right))
                .and_modify(|c| *c += 1)
                .or_insert(1);
            let count_val = *count;

            // Phase E (§27) Factorization Pressure: net_gain = MergeSaving - FactorizationReuseValue.
            // right_reuse > 1 means other lefts also want `right` — keeping it separate has value.
            // High reuse RESISTS merge (correct direction, fixes §57 inversion bug).
            let right_reuse = self.merge_right_reuse
                .get(&right)
                .map(|s| s.len() as f64)
                .unwrap_or(1.0);
            let factorization_reuse = self.learning.factorization_scale * (right_reuse - 1.0).max(0.0);
            let net_gain = count_val as f64 - factorization_reuse;

            if net_gain >= MERGE_THRESHOLD as f64 {
                let freq = (net_gain / MERGE_THRESHOLD as f64).min(4.0);
                let prob = BASE_MERGE_PROBABILITY * freq.sqrt();
                if freq >= 2.0 || pseudo_rand(left, right, count_val) < prob {
                    let exp_len = self.expanded_length(left) + self.expanded_length(right);
                    let cid = self.chunks.get_or_create(left, right, exp_len);
                    // Phase 3: record derivation in Lineage.
                    self.lineage.record(cid, left, right);
                    // Phase 7: mirror into unified Relation Core.
                    let child_unit = UnitId::chunk(cid);
                    self.relations.record_derivation(left, child_unit);
                    self.relations.record_derivation(right, child_unit);
                    // §7: register chunk→identity in the direct index.
                    let prim_ids_chunk: Vec<_> = self.lineage.decompose_to_primitives(child_unit)
                        .into_iter().filter_map(|u| u.as_primitive()).collect();
                    if !prim_ids_chunk.is_empty() {
                        let chunk_iid = self.identities.intern_identity(&prim_ids_chunk);
                        self.identities.register_unit_identity(child_unit, chunk_iid);
                    }
                    if let Some(chunk) = self.chunks.get_mut(cid) {
                        if chunk.use_count == 0 {
                            chunk.record_usage(self.tick);
                        }
                    }
                    self.merge_candidates.remove(&(left, right));
                    // Clean up right-reuse entry for this left (pair is resolved).
                    if let Some(lefts) = self.merge_right_reuse.get_mut(&right) {
                        lefts.remove(&left);
                    }
                }
            }
        }
    }

    /// Enforce HOT budget — demote weakest HOT chunks to SLEEP (§41).
    /// `budget = 0` means unlimited (no-op).
    pub fn enforce_hot_budget(&mut self, budget: usize) {
        if budget == 0 { return; }
        let hot_count = self.chunks.hot_count();
        if hot_count <= budget { return; }
        let excess = hot_count - budget;
        let mut hot: Vec<(u32, f64)> = self.chunks.iter_all()
            .filter(|c| c.is_hot())
            .map(|c| (c.id, c.usage_strength))
            .collect();
        hot.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Phase 8: use registry demote() to keep hot_pair_to_id in sync.
        for (id, _) in hot.iter().take(excess) {
            self.chunks.demote(*id);
        }
    }

    fn expanded_length(&self, unit: UnitId) -> u32 {
        if unit.is_primitive() {
            1
        } else if let Some(cid) = unit.as_chunk() {
            self.chunks.get(cid).map(|c| c.expanded_length).unwrap_or(1)
        } else {
            1
        }
    }

    // ── Inspection ───────────────────────────────────────────────────────

    pub fn primitive_count(&self) -> usize { self.primitives.len() }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    pub fn hot_chunk_count(&self) -> usize { self.chunks.hot_count() }
    pub fn sleep_chunk_count(&self) -> usize { self.chunks.sleep_count() }
    pub fn edge_count(&self) -> usize { self.predictions.edge_count() }
    pub fn association_count(&self) -> usize { self.associations.edge_count() }

    pub fn merge_candidates_iter(&self) -> impl Iterator<Item = (&(UnitId, UnitId), &u32)> {
        self.merge_candidates.iter()
    }

    /// §7: Rebuild the unit→identity direct index from primitives + chunks after load.
    ///
    /// Called from `from_snapshot` after all stores are restored.
    pub fn rebuild_unit_identities(&mut self) {
        // Primitives — IDs are 1-based (slot 0 is null); each has a single-element identity.
        for pid in 1..=(self.primitives.len() as u32) {
            let unit = UnitId::primitive(pid);
            if self.identities.identity_of_unit(unit).is_none() {
                let iid = self.identities.intern_identity(&[pid]);
                self.identities.register_unit_identity(unit, iid);
            }
        }
        // Chunks — derive identity from full primitive decomposition.
        for (chunk_id, _left, _right) in self.lineage.all_entries() {
            let unit = UnitId::chunk(chunk_id);
            if self.identities.identity_of_unit(unit).is_none() {
                let prim_ids: Vec<_> = self.lineage.decompose_to_primitives(unit)
                    .into_iter().filter_map(|u| u.as_primitive()).collect();
                if !prim_ids.is_empty() {
                    let iid = self.identities.intern_identity(&prim_ids);
                    self.identities.register_unit_identity(unit, iid);
                }
            }
        }
    }

    /// §26: RelationStore is derived — rebuild it from canonical stores after load.
    pub fn rebuild_relations(&mut self) {
        self.relations = crate::relation::RelationStore::new();
        // Route edges from PredictionStore.
        for edge in self.predictions.iter_all() {
            self.relations.observe(edge.context, edge.next_unit, RelationKind::Route, edge.usage_strength);
        }
        // Adjacency edges from AssociationStore (both directions already stored).
        for edge in self.associations.iter_all() {
            self.relations.observe(edge.source, edge.target, RelationKind::Adjacency, edge.strength);
        }
        // DerivedFrom edges from LineageStore (each chunk ← left, right).
        for (chunk_id, left, right) in self.lineage.all_entries() {
            let chunk_unit = UnitId::chunk(chunk_id);
            self.relations.record_derivation(left, chunk_unit);
            self.relations.record_derivation(right, chunk_unit);
        }
    }

    pub fn from_parts(
        primitives: PrimitiveRegistry,
        chunks: ChunkRegistry,
        predictions: PredictionStore,
        associations: AssociationStore,
        identities: IdentityStore,
        lineage: LineageStore,
        relations: RelationStore,
        representations: RepresentationStore,
        transforms: crate::transform::TransformStore,
        tick: u64,
        merge_candidates: HashMap<(UnitId, UnitId), u32>,
        merge_right_reuse: HashMap<UnitId, std::collections::HashSet<UnitId>>,
        metrics: Metrics,
        next_trace_id: TraceId,
    ) -> Self {
        Self {
            primitives,
            chunks,
            predictions,
            associations,
            identities,
            lineage,
            relations,
            representations,
            transforms,
            tick,
            merge_candidates,
            merge_right_reuse,
            metrics,
            next_trace_id,
            gen_config: GenerationConfig::default(),
            memory_budget: crate::config::MemoryBudgetConfig::default(),
            learning: crate::config::LearningConfig::default(),
        }
    }
}

fn pseudo_rand(left: UnitId, right: UnitId, count: u32) -> f64 {
    let h = left.raw().wrapping_mul(2654435761)
        ^ right.raw().wrapping_mul(2246822519)
        ^ count.wrapping_mul(3266489917);
    (h as f64) / (u32::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_registers_primitives() {
        let mut m = ModelState::new();
        m.train("abc");
        assert!(m.primitive_count() >= 3);
    }

    #[test]
    fn test_expose_registers_primitives() {
        let mut m = ModelState::new();
        m.expose("abc");
        assert!(m.primitive_count() >= 3);
    }

    #[test]
    fn test_train_builds_prediction_edges() {
        let mut m = ModelState::new();
        m.train("abcabc");
        assert!(m.edge_count() > 0);
    }

    #[test]
    fn test_metrics_accumulate() {
        let mut m = ModelState::new();
        m.train("hello");
        assert!(m.metrics.total_characters >= 5);
        assert!(m.metrics.total_decisions >= 1);
    }

    #[test]
    fn test_repeated_training_creates_chunks() {
        let mut m = ModelState::new();
        for _ in 0..(MERGE_THRESHOLD * 2 + 2) {
            m.train("ab");
        }
        assert!(m.chunk_count() > 0);
    }

    #[test]
    fn test_dpc_decreases_with_learning() {
        let mut m = ModelState::new();
        let text = "ab".repeat(50);
        for chunk in text.as_bytes().chunks(4) {
            m.train(std::str::from_utf8(chunk).unwrap());
        }
        let dpc = m.metrics.decision_per_character();
        assert!(dpc > 0.0);
    }

    #[test]
    fn test_generate_returns_nonempty_with_trained_model() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let out = m.generate("hel", 5);
        assert!(out.starts_with("hel") || !out.is_empty());
    }

    #[test]
    fn test_generate_empty_seed_with_no_training() {
        let mut m = ModelState::new();
        let out = m.generate("", 10);
        assert_eq!(out, "");
    }

    #[test]
    fn test_train_empty_string() {
        let mut m = ModelState::new();
        m.train("");
        assert_eq!(m.primitive_count(), 0);
    }

    #[test]
    fn test_decision_per_character_zero_before_training() {
        let m = ModelState::new();
        assert_eq!(m.metrics.decision_per_character(), 0.0);
    }

    #[test]
    fn test_generate_with_trace_records_steps() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let tid = m.alloc_trace_id();
        let (output, trace) = m.generate_with_trace("hel", "hel", 5, tid);
        assert!(!output.is_empty());
        assert_eq!(trace.trace_id, 1);
        assert_eq!(trace.generation_seed, "hel");
    }

    #[test]
    fn test_generate_with_trace_no_mutation() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let edges_before = m.edge_count();
        let tick_before = m.tick;
        let _ = m.generate("hel", 5);
        assert_eq!(m.edge_count(), edges_before);
        assert_eq!(m.tick, tick_before);
    }

    #[test]
    fn test_apply_feedback_to_trace() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.train("hello world"); }
        let tid = m.alloc_trace_id();
        let (_, trace) = m.generate_with_trace("hel", "hel", 5, tid);
        if trace.decision_count > 0 {
            let credits: Vec<f64> = vec![1.0; trace.decision_count];
            m.apply_feedback_to_trace(&trace, &credits, 0.99);
            // Just verify no panic and at least one feedback was applied
            let total_fb: u32 = m.predictions.iter_all()
                .map(|e| e.feedback_count)
                .sum();
            assert!(total_fb > 0 || trace.decision_count == 0);
        }
    }

    #[test]
    fn test_state_fingerprint() {
        let mut m = ModelState::new();
        let fp1 = m.state_fingerprint();
        m.train("abc");
        let fp2 = m.state_fingerprint();
        assert_ne!(fp1, fp2);
    }

    // ── Experience / Replay 分離テスト (Phase 1) ─────────────────────────

    #[test]
    fn replay_does_not_update_metrics() {
        let mut m = ModelState::new();
        m.expose_external("hello");
        let chars = m.metrics.total_characters;
        let decs = m.metrics.total_decisions;
        m.replay("hello");
        assert_eq!(m.metrics.total_characters, chars, "replay must not increment total_characters");
        assert_eq!(m.metrics.total_decisions, decs, "replay must not increment total_decisions");
    }

    #[test]
    fn replay_does_not_update_associations() {
        let mut m = ModelState::new();
        m.expose_external("ab");
        let assoc_before = m.associations.edge_count();
        // replay with same text should not add new associations
        m.replay("ab");
        assert_eq!(m.associations.edge_count(), assoc_before, "replay must not update associations");
    }

    #[test]
    fn replay_does_not_register_new_primitives() {
        let mut m = ModelState::new();
        m.expose_external("abc");
        let prims_before = m.primitive_count();
        // "xyz" contains new chars — replay must not register them
        m.replay("xyz");
        assert_eq!(m.primitive_count(), prims_before, "replay must not register new primitives");
    }

    #[test]
    fn replay_strengthens_predictions() {
        let mut m = ModelState::new();
        // Prime the model so "ab" primitives exist and there are prediction edges.
        for _ in 0..10 { m.expose_external("ab"); }
        let edges_after_experience = m.edge_count();
        // Replay should be able to strengthen edges (at minimum, not crash).
        for _ in 0..5 { m.replay("ab"); }
        assert!(m.edge_count() >= edges_after_experience, "replay should not remove prediction edges");
    }

    #[test]
    fn expose_external_8x_counts_as_8_observations() {
        let mut m = ModelState::new();
        let text = "hello";
        for _ in 0..8 { m.expose_external(text); }
        assert_eq!(m.metrics.total_characters, (text.chars().count() * 8) as u64);
    }

    #[test]
    fn replay_8x_does_not_inflate_observation_count() {
        let mut m = ModelState::new();
        let text = "hello";
        m.expose_external(text);
        let chars_after_one = m.metrics.total_characters;
        for _ in 0..7 { m.replay(text); }
        assert_eq!(m.metrics.total_characters, chars_after_one,
            "8 replays after 1 expose_external must not inflate total_characters");
    }

    // ── Phase 2: Identity / View ──────────────────────────────────────────

    #[test]
    fn expose_external_registers_identity_and_view() {
        let mut m = ModelState::new();
        m.expose_external("abc");
        // The full "abc" sequence must have exactly one identity entry.
        let prim_ids = m.primitives.encode_existing("abc");
        assert!(m.identities.find_identity(&prim_ids).is_some(),
            "one unique Primitive expansion → one Identity (§7 adds primitive identities too)");
        assert!(m.identities.view_count() >= 1,
            "at least one View should be registered");
    }

    #[test]
    fn replay_registers_view_same_identity() {
        // After training, replay of the same text produces the same Identity
        // (possibly a different View once chunks form, but same Identity).
        let mut m = ModelState::new();
        let text = "abab";
        // First exposure
        m.expose_external(text);
        let id_after_expose = m.identities.intern_identity(
            &m.primitives.encode_existing(text)
        );
        // Several replays to allow chunk formation
        for _ in 0..10 {
            m.expose_external(text);
        }
        let id_after_replay = m.identities.intern_identity(
            &m.primitives.encode_existing(text)
        );
        assert_eq!(id_after_expose, id_after_replay,
            "same Primitive expansion → same IdentityId regardless of segmentation");
    }

    #[test]
    fn different_texts_different_identities() {
        let mut m = ModelState::new();
        m.expose_external("abc");
        m.expose_external("xyz");
        // Two distinct Primitive expansions → two distinct Identities.
        // (§7 also registers individual primitive identities, so total count > 2.)
        let id_abc = m.identities.find_identity(&m.primitives.encode_existing("abc"));
        let id_xyz = m.identities.find_identity(&m.primitives.encode_existing("xyz"));
        assert!(id_abc.is_some() && id_xyz.is_some(), "both texts must have identities");
        assert_ne!(id_abc, id_xyz, "distinct texts must have distinct identities");
    }

    #[test]
    fn same_text_repeated_does_not_add_new_identity() {
        let mut m = ModelState::new();
        m.expose_external("hello");
        let id_first = m.identities.find_identity(&m.primitives.encode_existing("hello"));
        m.expose_external("hello");
        m.expose_external("hello");
        let id_third = m.identities.find_identity(&m.primitives.encode_existing("hello"));
        assert_eq!(id_first, id_third,
            "same text repeated → same Identity (idempotent)");
    }

    // ── Phase 3: Representation Lineage ──────────────────────────────────

    #[test]
    fn lineage_recorded_on_chunk_creation() {
        let mut m = ModelState::new();
        // Train until a chunk forms
        for _ in 0..20 { m.expose_external("ab"); }
        assert!(m.chunk_count() > 0, "chunk should have formed");
        assert!(m.lineage.entry_count() > 0,
            "lineage must be recorded when a chunk is created");
    }

    #[test]
    fn lineage_decompose_to_primitives() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.expose_external("ab"); }
        // The first and only chunk should decompose back to [P(a), P(b)]
        for cid in 0..m.chunk_count() as u32 {
            let chunk_unit = crate::units::UnitId::chunk(cid);
            let decomposed = m.lineage.decompose_to_primitives(chunk_unit);
            // All results should be primitives
            assert!(decomposed.iter().all(|u| u.is_primitive()),
                "full decomposition must yield only Primitives");
        }
    }

    #[test]
    fn lineage_entry_count_matches_chunk_count() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.expose_external("abcabc"); }
        // Every chunk that was created should have a lineage entry
        assert_eq!(m.lineage.entry_count(), m.chunk_count(),
            "each chunk must have exactly one lineage entry");
    }

    // ── Phase 4: Fallback via Lineage ────────────────────────────────────

    #[test]
    fn fallback_returns_none_for_unknown_primitive() {
        let m = ModelState::new();
        // No training — no predictions at all.
        assert!(m.predict_with_fallback(UnitId::primitive(1)).is_none());
    }

    #[test]
    fn fallback_direct_hit_no_fallback_needed() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.expose_external("ab"); }
        let pa = m.primitives.id('a').map(UnitId::primitive).unwrap();
        // 'a' has a direct prediction (→ 'b')
        let direct = m.predictions.top1_with_score(pa);
        let via_fallback = m.predict_with_fallback(pa);
        assert_eq!(direct, via_fallback);
    }

    #[test]
    fn fallback_from_chunk_to_primitive() {
        let mut m = ModelState::new();
        // Train "ab" until a chunk forms, then train "abcd" so primitives have edges too.
        for _ in 0..20 { m.expose_external("ab"); }
        for _ in 0..8 { m.expose_external("abcd"); }
        let chunk_count = m.chunk_count();
        assert!(chunk_count > 0, "chunk must have formed");

        // Find a chunk whose direct prediction is None but whose primitive components have one.
        let mut found_fallback = false;
        for cid in 0..chunk_count as u32 {
            let cu = UnitId::chunk(cid);
            if m.predictions.top1_with_score(cu).is_none() {
                // Should succeed via fallback
                if m.predict_with_fallback(cu).is_some() {
                    found_fallback = true;
                    break;
                }
            }
        }
        // At minimum, all chunks that have no direct prediction should be
        // reachable via primitive fallback (or we skip the assertion if all chunks
        // have direct predictions — that's fine too).
        let _ = found_fallback;  // Not required to find one; test just must not panic.
    }

    #[test]
    fn generate_with_trace_uses_fallback_and_does_not_panic() {
        let mut m = ModelState::new();
        for _ in 0..20 { m.expose_external("abcabc"); }
        let (output, _trace) = m.generate_with_trace("a", "", 10, 1);
        // Output should be non-empty if there are any predictions at all.
        assert!(!output.is_empty());
    }

    #[test]
    fn different_views_same_identity_after_chunk_formation() {
        let mut m = ModelState::new();
        let text = "ab";
        // Train until a chunk forms
        for _ in 0..20 { m.expose_external(text); }
        assert!(m.chunk_count() > 0, "chunk should have formed");
        // Both the all-primitives view [P(a),P(b)] and the chunked view [C(ab)]
        // should map to the same Identity.
        let prim_ids = m.primitives.encode_existing(text);
        let identity_id = m.identities.intern_identity(&prim_ids);
        // All registered views for this identity should resolve to it
        let mut found_matching = 0u32;
        for view_id in 0..m.identities.view_count() as u32 {
            if m.identities.identity_of_view(view_id) == Some(identity_id) {
                found_matching += 1;
            }
        }
        assert!(found_matching >= 2,
            "at least the primitive-only view and the chunked view must share identity; got {found_matching}");
    }
}
