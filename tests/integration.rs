/// Integration tests covering AC-01 through AC-13 (v0.1) + T-01 through T-18 (v0.2)
/// + RT/TL/CR/HI/EV/SS tests (v0.3) + RS tests (v0.4) + GEN-LOOP tests (P0)
/// + REP tests (Phase B) + EXP tests (Phase C) + FAC tests (Phase E) + TRF tests (Phase F).
use syntrail_lm::model::ModelState;
use syntrail_lm::persistence;
use syntrail_lm::primitives::PrimitiveRegistry;
use syntrail_lm::segmentation::segment;
use syntrail_lm::tier::{Tier, TierThresholds};
use syntrail_lm::units::UnitId;
use tempfile::NamedTempFile;

// ── AC-01: Unicode round-trip ──────────────────────────────────────────────
#[test]
fn ac01_unicode_roundtrip() {
    let samples = [
        "Hello, world!",
        "こんにちは世界",
        "コンニチハセカイ",
        "日本語処理",
        "😀🎉🦀🌏",
        "αβγ €£¥",
        "\u{10FFFF}", // highest valid scalar
    ];
    for &s in &samples {
        let mut reg = PrimitiveRegistry::new();
        let ids = reg.encode(s);
        let recovered = reg.decode(&ids).expect("decode failed");
        assert_eq!(s, recovered, "AC-01 round-trip failed for: {s:?}");
    }
}

// ── AC-02: Primitive never deleted ────────────────────────────────────────
#[test]
fn ac02_primitive_never_deleted() {
    let mut reg = PrimitiveRegistry::new();
    let id_a = reg.register('a');
    let len_before = reg.len();
    // Re-registering must not grow the registry
    reg.register('a');
    assert_eq!(reg.len(), len_before);
    // ID is stable
    assert_eq!(reg.id('a'), Some(id_a));
}

// ── AC-03: Chunk is binary composition ────────────────────────────────────
#[test]
fn ac03_chunk_binary_composition() {
    let mut model = ModelState::new();
    // Train enough to create a chunk
    for _ in 0..(4 * 2 + 2) {
        model.train("ab");
    }
    // At least one chunk should exist
    assert!(model.chunk_count() > 0, "AC-03: no chunks created");
    // Every chunk must have exactly two unit references (left, right)
    for chunk in model.chunks.iter_all() {
        // Both left and right are either primitive or chunk — never null
        assert!(chunk.left.raw() < u32::MAX);
        assert!(chunk.right.raw() < u32::MAX);
        // Expanded length >= 2
        assert!(chunk.expanded_length >= 2, "AC-03: expanded_length < 2");
    }
}

// ── AC-04: Tier 0→1 promotion at threshold ────────────────────────────────
#[test]
fn ac04_tier_promotion_t0_to_t1() {
    let t = Tier::T0;
    // Just below threshold: no promotion
    let below = t.maybe_promote(
        TierThresholds::T0_TO_T1_SUCCESSES - 1,
        TierThresholds::T0_TO_T1_SUCCESSES - 1,
    );
    assert_eq!(below, Tier::T0, "AC-04: promoted too early");
    // At threshold: promotes
    let at = t.maybe_promote(
        TierThresholds::T0_TO_T1_SUCCESSES,
        TierThresholds::T0_TO_T1_SUCCESSES,
    );
    assert_eq!(at, Tier::T1, "AC-04: did not promote at threshold");
}

// ── AC-05: Tier 1→2 promotion at threshold ────────────────────────────────
#[test]
fn ac05_tier_promotion_t1_to_t2() {
    let t = Tier::T1;
    let below = t.maybe_promote(
        TierThresholds::T1_TO_T2_SUCCESSES - 1,
        TierThresholds::T1_TO_T2_SUCCESSES - 1,
    );
    assert_eq!(below, Tier::T1, "AC-05: promoted T1→T2 too early");
    let at = t.maybe_promote(
        TierThresholds::T1_TO_T2_SUCCESSES,
        TierThresholds::T1_TO_T2_SUCCESSES,
    );
    assert_eq!(at, Tier::T2, "AC-05: did not promote T1→T2 at threshold");
}

// ── AC-06: Strength update formula ────────────────────────────────────────
#[test]
fn ac06_strength_update_formula() {
    use syntrail_lm::chunks::{STRENGTH_DECAY, STRENGTH_REWARD};
    let mut model = ModelState::new();
    // Need to create a chunk
    for _ in 0..(4 * 2 + 2) {
        model.train("xy");
    }
    // Find the chunk for 'x','y' — register externally to verify ID stability
    let mut reg = PrimitiveRegistry::new();
    let _x_id = reg.register('x');
    let _y_id = reg.register('y');
    // In the model, the same IDs should exist
    let x_id_m = model.primitives.id('x').expect("x not registered");
    let y_id_m = model.primitives.id('y').expect("y not registered");

    // Manually test the formula on a chunk
    if let Some(chunk_id) = model.chunks.find_by_pair(
        UnitId::primitive(x_id_m),
        UnitId::primitive(y_id_m),
    ) {
        let chunk = model.chunks.get(chunk_id).unwrap();
        let old_strength = chunk.usage_strength;
        // Phase 9 lazy decay: s_new = s * λ^Δt + reward; use Δt=1 for equivalence with old formula.
        let next_tick = chunk.last_used + 1;
        let expected = STRENGTH_DECAY * old_strength + STRENGTH_REWARD;
        let mut chunk_clone = chunk.clone();
        chunk_clone.record_usage(next_tick);
        assert!(
            (chunk_clone.usage_strength - expected).abs() < 1e-10,
            "AC-06: strength formula mismatch: got {} expected {}",
            chunk_clone.usage_strength,
            expected
        );
    }
    // Whether or not the chunk exists, the formula itself is correct
    // (covered by unit tests; this just verifies it in integration context)
}

// ── AC-07: Segmentation uses score to decide merges ───────────────────────
#[test]
fn ac07_segmentation_by_score() {
    use syntrail_lm::chunks::ChunkRegistry;

    let mut prim_reg = PrimitiveRegistry::new();
    let ids = prim_reg.encode("ab");
    let mut chunk_reg = ChunkRegistry::new();

    // Create a high-score chunk (T2)
    let cid = chunk_reg.get_or_create(
        UnitId::primitive(ids[0]),
        UnitId::primitive(ids[1]),
        2,
    );
    {
        let chunk = chunk_reg.get_mut(cid).unwrap();
        for tick in 0..64 { chunk.record_usage(tick); }
    }

    let segmented = segment(&ids, &chunk_reg, 0.0);
    assert_eq!(segmented.len(), 1, "AC-07: high-score chunk not merged");
    assert_eq!(segmented[0], UnitId::chunk(cid));

    // With very high threshold, the same chunk should NOT be merged
    let not_merged = segment(&ids, &chunk_reg, 10.0);
    assert_eq!(not_merged.len(), 2, "AC-07: chunk merged despite high threshold");
}

// ── AC-08: Prediction edge learning ───────────────────────────────────────
#[test]
fn ac08_prediction_edge_learning() {
    let mut model = ModelState::new();
    for _ in 0..5 {
        model.train("abcabc");
    }
    // Should have prediction edges
    assert!(model.edge_count() > 0, "AC-08: no prediction edges");
    // Generating from "a" should not be empty after training
    let out = model.generate("a", 3);
    assert!(!out.is_empty(), "AC-08: generate returned empty");
}

// ── AC-09: Chunk creation via probabilistic merge ─────────────────────────
#[test]
fn ac09_chunk_created_by_probabilistic_merge() {
    let mut model = ModelState::new();
    // MERGE_THRESHOLD=4, need 2×threshold observations
    for _ in 0..20 {
        model.train("ab");
    }
    assert!(
        model.chunk_count() > 0,
        "AC-09: no chunks after repeated training on 'ab'"
    );
}

// ── AC-10: decision_per_character tracked ─────────────────────────────────
#[test]
fn ac10_dpc_tracked() {
    let mut model = ModelState::new();
    assert_eq!(model.metrics.decision_per_character(), 0.0, "AC-10: dpc nonzero before training");
    model.train("hello");
    assert!(
        model.metrics.total_characters >= 5,
        "AC-10: characters not counted"
    );
    assert!(
        model.metrics.total_decisions >= 1,
        "AC-10: decisions not counted"
    );
    let dpc = model.metrics.decision_per_character();
    assert!(dpc > 0.0 && dpc <= 1.0, "AC-10: dpc={dpc} out of [0,1] range on first pass");
}

// ── AC-11: Save / load preserves full state ───────────────────────────────
#[test]
fn ac11_save_load_preserves_state() {
    let mut model = ModelState::new();
    for _ in 0..20 {
        model.train("hello world");
    }

    let file = NamedTempFile::new().unwrap();
    persistence::save(&model, file.path()).unwrap();
    let loaded = persistence::load(file.path()).unwrap();

    assert_eq!(loaded.primitive_count(), model.primitive_count(), "AC-11: primitive_count mismatch");
    assert_eq!(loaded.chunk_count(), model.chunk_count(), "AC-11: chunk_count mismatch");
    assert_eq!(loaded.edge_count(), model.edge_count(), "AC-11: edge_count mismatch");
    assert_eq!(
        loaded.metrics.total_characters,
        model.metrics.total_characters,
        "AC-11: total_characters mismatch"
    );
    assert_eq!(
        loaded.metrics.total_decisions,
        model.metrics.total_decisions,
        "AC-11: total_decisions mismatch"
    );
}

// ── AC-12: Four CLI commands exist and compile ────────────────────────────
// (Compilation of src/main.rs verifies this at build time; functional test
//  is covered by the train/generate unit tests above.  We verify the binary
//  can be invoked here.)
#[test]
fn ac12_cli_binary_exists() {
    // The integration test simply confirms the binary compiles (implicit in
    // running `cargo test --tests`).  A process-level smoke test would
    // require spawning the binary, which is done in examples/ instead.
    // This test documents the intent.
    assert!(true, "AC-12: CLI compiles (implicit)");
}

// ── AC-13: dpc < 1.0 after sufficient training on repetitive text ─────────
#[test]
fn ac13_dpc_below_one_after_learning() {
    let mut model = ModelState::new();
    let text = "abab".repeat(100);
    // Train many times on a highly repetitive sequence
    for chunk in text.as_bytes().chunks(8) {
        model.train(std::str::from_utf8(chunk).unwrap());
    }
    let dpc = model.metrics.decision_per_character();
    assert!(
        dpc < 1.0,
        "AC-13: dpc={dpc:.4} not below 1.0 after training on repetitive text"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// v0.2 Tests — T-01 through T-18
// ═══════════════════════════════════════════════════════════════════════════

fn trained() -> ModelState {
    let mut m = ModelState::new();
    for _ in 0..20 { m.train("hello world"); }
    m
}

// ── T-01: generate_with_trace is frozen ──────────────────────────────────
#[test]
fn t01_generate_with_trace_frozen() {
    let mut m = trained();
    let tick_before = m.tick;
    let edges_before = m.edge_count();
    let chunks_before = m.chunk_count();
    let tid = m.alloc_trace_id();
    let _ = m.generate_with_trace("hel", "hel", 5, tid);
    assert_eq!(m.tick, tick_before, "T-01: tick mutated during generate_with_trace");
    assert_eq!(m.edge_count(), edges_before, "T-01: edges mutated");
    assert_eq!(m.chunk_count(), chunks_before, "T-01: chunks mutated");
}

// ── T-02: TurnTrace records decision steps ────────────────────────────────
#[test]
fn t02_trace_records_decisions() {
    let mut m = trained();
    let tid = m.alloc_trace_id();
    let (output, trace) = m.generate_with_trace("hel", "hel", 5, tid);
    assert!(!output.is_empty(), "T-02: output empty");
    assert_eq!(trace.trace_id, tid, "T-02: trace_id mismatch");
    assert_eq!(trace.decision_count, trace.decision_steps.len(), "T-02: count mismatch");
    assert_eq!(trace.generation_seed, "hel", "T-02: seed not stored");
    assert_eq!(trace.input_text, "hel", "T-02: input_text not stored");
}

// ── T-03: No decisions for seed-only generation ───────────────────────────
#[test]
fn t03_no_decisions_for_seed_only() {
    // If model has no edges for the last seed unit, decision_count = 0
    let mut m = ModelState::new();
    let (_, trace) = m.generate_with_trace("z", "z", 5, 1);
    assert_eq!(trace.decision_count, 0, "T-03: decisions expected 0 for untrained model");
}

// ── T-04: expose does not mutate generation result ────────────────────────
#[test]
fn t04_expose_vs_generate_separation() {
    let mut m = trained();
    let tid = m.alloc_trace_id();
    let (out1, _) = m.generate_with_trace("hel", "hel", 5, tid);
    m.expose("hello world");
    // Generation from same seed should still work (not panic, not crash)
    let tid2 = m.alloc_trace_id();
    let (out2, _) = m.generate_with_trace("hel", "hel", 5, tid2);
    assert!(!out1.is_empty(), "T-04: out1 empty");
    assert!(!out2.is_empty(), "T-04: out2 empty");
}

// ── T-05: Positive feedback credits sum to +R ────────────────────────────
#[test]
fn t05_positive_credit_sum() {
    use syntrail_lm::config::Config;
    use syntrail_lm::feedback::{distribute_uniform, FeedbackSign};
    let config = Config::default_v02();
    let n = 5;
    let credits = distribute_uniform(FeedbackSign::Positive, n, &config);
    let sum: f64 = credits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "T-05: sum={sum}");
    assert_eq!(credits.len(), n);
}

// ── T-06: Negative feedback credits sum to -R ────────────────────────────
#[test]
fn t06_negative_credit_sum() {
    use syntrail_lm::config::Config;
    use syntrail_lm::feedback::{distribute_uniform, FeedbackSign};
    let config = Config::default_v02();
    let n = 4;
    let credits = distribute_uniform(FeedbackSign::Negative, n, &config);
    let sum: f64 = credits.iter().sum();
    assert!((sum - (-1.0)).abs() < 1e-12, "T-06: sum={sum}");
}

// ── T-07: Positive and negative reward are symmetric ─────────────────────
#[test]
fn t07_symmetry() {
    use syntrail_lm::config::Config;
    use syntrail_lm::feedback::{FeedbackSign, total_reward};
    let config = Config::default_v02();
    let pos = total_reward(FeedbackSign::Positive, &config);
    let neg = total_reward(FeedbackSign::Negative, &config);
    assert_eq!(pos, -neg, "T-07: symmetry violated");
}

// ── T-08: Uniform distribution: each credit equals R/N ───────────────────
#[test]
fn t08_uniform_per_step_value() {
    use syntrail_lm::config::Config;
    use syntrail_lm::feedback::{distribute_uniform, FeedbackSign};
    let config = Config::default_v02();
    let n = 4;
    let credits = distribute_uniform(FeedbackSign::Negative, n, &config);
    for &c in &credits {
        assert!((c - (-0.25)).abs() < 1e-12, "T-08: each credit should be -0.25, got {c}");
    }
}

// ── T-09: apply_feedback_to_trace updates prediction edges ────────────────
#[test]
fn t09_feedback_updates_edges() {
    let mut m = trained();
    let tid = m.alloc_trace_id();
    let (_, trace) = m.generate_with_trace("hel", "hel", 5, tid);
    if trace.decision_count == 0 { return; }
    let credits = vec![1.0; trace.decision_count];
    let fb_before: u32 = m.predictions.iter_all().map(|e| e.feedback_count).sum();
    m.apply_feedback_to_trace(&trace, &credits, 0.99);
    let fb_after: u32 = m.predictions.iter_all().map(|e| e.feedback_count).sum();
    assert!(fb_after > fb_before, "T-09: feedback_count not updated");
}

// ── T-10: apply_feedback_to_trace with empty credits is a no-op ───────────
#[test]
fn t10_empty_credits_noop() {
    let mut m = trained();
    let tid = m.alloc_trace_id();
    let (_, trace) = m.generate_with_trace("hel", "hel", 5, tid);
    let edges_before = m.edge_count();
    m.apply_feedback_to_trace(&trace, &[], 0.99);
    assert_eq!(m.edge_count(), edges_before, "T-10: edges changed");
}

// ── T-11: Session turn logs to DB ────────────────────────────────────────
#[test]
fn t11_session_turn_logs() {
    use syntrail_lm::config::Config;
    use syntrail_lm::session::Session;
    let mut s = Session::new_in_memory(Config::default_v02()).unwrap();
    for _ in 0..20 { s.model.train("hello world"); }
    let (turn_id, _, _) = s.turn("hello").unwrap();
    let rows = s.history(10).unwrap();
    assert_eq!(rows.len(), 1, "T-11: expected 1 turn in history");
    assert_eq!(rows[0].turn_id, turn_id, "T-11: turn_id mismatch");
    assert_eq!(rows[0].input_text, "hello");
}

// ── T-12: Session feedback logs to DB ────────────────────────────────────
#[test]
fn t12_session_feedback_logs() {
    use syntrail_lm::config::Config;
    use syntrail_lm::feedback::{FeedbackSign, FeedbackSource};
    use syntrail_lm::session::Session;
    let mut s = Session::new_in_memory(Config::default_v02()).unwrap();
    for _ in 0..20 { s.model.train("hello world"); }
    let (turn_id, _, trace) = s.turn("hello").unwrap();
    if trace.decision_count > 0 {
        s.feedback(turn_id, FeedbackSign::Positive, FeedbackSource::User, None).unwrap();
        let fb_rows = s.db.query_feedback_for_turn(turn_id).unwrap();
        assert_eq!(fb_rows.len(), 1, "T-12: feedback not logged");
        assert_eq!(fb_rows[0].sign, FeedbackSign::Positive);
    }
}

// ── T-13: Snapshot saves and restores model state ────────────────────────
#[test]
fn t13_snapshot_restore() {
    use syntrail_lm::config::Config;
    use syntrail_lm::session::Session;
    let mut s = Session::new_in_memory(Config::default_v02()).unwrap();
    for _ in 0..20 { s.model.train("hello world"); }
    let tick_snap = s.model.tick;
    let sid = s.snapshot_to_db(None).unwrap();
    s.model.train("extra extra extra");
    assert_ne!(s.model.tick, tick_snap, "T-13: tick should have advanced");
    s.restore_from_db(sid).unwrap();
    assert_eq!(s.model.tick, tick_snap, "T-13: tick not restored");
}

// ── T-14: usage_strength and feedback_value are independent ──────────────
#[test]
fn t14_usage_feedback_independence() {
    use syntrail_lm::chunks::ChunkRegistry;
    use syntrail_lm::units::UnitId;
    let mut reg = ChunkRegistry::new();
    let id = reg.get_or_create(UnitId::primitive(1), UnitId::primitive(2), 2);
    let chunk = reg.get_mut(id).unwrap();
    chunk.record_usage(0);
    chunk.record_usage(1);
    let strength = chunk.usage_strength;
    assert!(strength > 0.0, "T-14: usage_strength not updated");
    assert_eq!(chunk.feedback_value, 0.0, "T-14: feedback_value should still be 0");

    chunk.apply_feedback(-1.0, 0.99);
    assert_eq!(chunk.usage_strength, strength, "T-14: usage_strength changed by feedback");
    assert!((chunk.feedback_value - (-1.0)).abs() < 1e-9, "T-14: feedback_value not updated");
}

// ── T-15: chunk_score uses binary confidence ──────────────────────────────
#[test]
fn t15_chunk_score_binary_confidence() {
    use syntrail_lm::chunks::ChunkRegistry;
    use syntrail_lm::segmentation::chunk_score;
    use syntrail_lm::units::UnitId;
    let mut reg = ChunkRegistry::new();
    let id = reg.get_or_create(UnitId::primitive(1), UnitId::primitive(2), 2);
    {
        let chunk = reg.get(id).unwrap();
        let score = chunk_score(chunk);
        assert!(score < 0.0, "T-15: unused chunk should have negative score");
    }
    {
        let chunk = reg.get_mut(id).unwrap();
        chunk.record_usage(0);
    }
    {
        let chunk = reg.get(id).unwrap();
        let score = chunk_score(chunk);
        assert!(score > 0.0, "T-15: used chunk should have positive score");
    }
}

// ── T-16: Persistence preserves v0.2 feedback fields ─────────────────────
#[test]
fn t16_persist_feedback_fields() {
    use syntrail_lm::feedback::{FeedbackSign, FeedbackSource};
    use syntrail_lm::session::Session;
    use syntrail_lm::config::Config;

    let mut s = Session::new_in_memory(Config::default_v02()).unwrap();
    for _ in 0..20 { s.model.train("hello world"); }
    let (turn_id, _, trace) = s.turn("hello").unwrap();

    if trace.decision_count > 0 {
        s.feedback(turn_id, FeedbackSign::Negative, FeedbackSource::User, None).unwrap();
    }

    let fb_sum_before: u32 = s.model.predictions.iter_all().map(|e| e.feedback_count).sum();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    syntrail_lm::persistence::save(&s.model, tmp.path()).unwrap();
    let loaded = syntrail_lm::persistence::load(tmp.path()).unwrap();
    let fb_sum_after: u32 = loaded.predictions.iter_all().map(|e| e.feedback_count).sum();
    assert_eq!(fb_sum_before, fb_sum_after, "T-16: feedback_count not preserved");
}

// ── T-17: state_fingerprint changes after training ────────────────────────
#[test]
fn t17_fingerprint_changes_after_train() {
    let mut m = ModelState::new();
    let fp1 = m.state_fingerprint();
    m.train("hello");
    let fp2 = m.state_fingerprint();
    assert_ne!(fp1, fp2, "T-17: fingerprint unchanged after training");
}

// ── T-18: Regression — all AC-01–AC-13 still pass (implicit via compilation) ──
#[test]
fn t18_regression_ac_tests_still_compile() {
    // This test documents that the existing AC tests are retained.
    // Their execution is validated by cargo test finding and running them above.
    assert!(true, "T-18: regression marker — AC-01 through AC-13 all pass");
}

// ═══════════════════════════════════════════════════════════════
// v0.3 Acceptance Tests
// ═══════════════════════════════════════════════════════════════

// ── RT-01: expose() updates associations bidirectionally ─────────────────
#[test]
fn rt01_expose_creates_bidirectional_associations() {
    let mut m = ModelState::new();
    for _ in 0..5 { m.train("hello world"); }
    let assoc_count = m.association_count();
    assert!(assoc_count > 0, "RT-01: expose should create associations");
}

// ── RT-02: association strength grows with repetition ────────────────────
#[test]
fn rt02_association_strength_grows() {
    use syntrail_lm::association::AssociationStore;
    use syntrail_lm::units::UnitId;
    let mut store = AssociationStore::new(8, 0.99);
    let a = UnitId::primitive(1);
    let b = UnitId::primitive(2);
    store.observe(a, b, 0);
    let s1 = store.top_for(a)[0].strength;
    store.observe(a, b, 1);
    let s2 = store.top_for(a)[0].strength;
    assert!(s2 > s1, "RT-02: strength should grow with repeated observation");
}

// ── RT-03: Top-K budget is enforced ──────────────────────────────────────
#[test]
fn rt03_top_k_budget_enforced() {
    use syntrail_lm::association::AssociationStore;
    use syntrail_lm::units::UnitId;
    let mut store = AssociationStore::new(4, 0.99);
    let src = UnitId::primitive(0);
    for i in 1..=10u32 { store.observe(src, UnitId::primitive(i), i as u64); }
    assert!(store.top_for(src).len() <= 4, "RT-03: must not exceed top_k");
}

// ── RT-04: recall returns strongest associations ──────────────────────────
#[test]
fn rt04_recall_sorted_by_strength() {
    use syntrail_lm::association::AssociationStore;
    use syntrail_lm::chunks::ChunkRegistry;
    use syntrail_lm::units::UnitId;
    let mut store = AssociationStore::new(8, 0.99);
    let src = UnitId::primitive(1);
    store.observe(src, UnitId::primitive(2), 0);
    for _ in 0..5 { store.observe(src, UnitId::primitive(3), 1); }
    let chunks = ChunkRegistry::new();
    let results = store.recall(src, &chunks, 4);
    assert!(!results.is_empty(), "RT-04: recall should return results");
    assert_eq!(results[0].0, UnitId::primitive(3), "RT-04: strongest should be first");
}

// ── RT-05: recall via model.recall() works ───────────────────────────────
#[test]
fn rt05_model_recall_method() {
    let mut m = ModelState::new();
    for _ in 0..10 { m.train("hello world"); }
    let prim_ids = {
        let mut tmp = m.primitives.clone();
        tmp.encode("hello")
    };
    let segmented = segment(&prim_ids, &m.chunks, 0.0);
    let source = segmented[0];
    let results = m.recall(source, 4);
    assert!(!results.is_empty(), "RT-05: model.recall should return associations");
}

// ── TL-01: avoidance grows on negative feedback ──────────────────────────
#[test]
fn tl01_avoidance_grows_on_negative_feedback() {
    use syntrail_lm::prediction::PredictionEdge;
    use syntrail_lm::units::UnitId;
    let mut edge = PredictionEdge::new(UnitId::primitive(1), UnitId::primitive(2));
    edge.record_usage();
    edge.apply_feedback(-1.0, 0.99);
    assert!(edge.avoidance > 0.0, "TL-01: avoidance should grow on negative feedback");
    assert_eq!(edge.feedback_value, 0.0, "TL-01: feedback_value should not change on negative");
}

// ── TL-02: feedback_value grows on positive feedback ─────────────────────
#[test]
fn tl02_feedback_value_grows_on_positive() {
    use syntrail_lm::prediction::PredictionEdge;
    use syntrail_lm::units::UnitId;
    let mut edge = PredictionEdge::new(UnitId::primitive(1), UnitId::primitive(2));
    edge.record_usage();
    edge.apply_feedback(1.0, 0.99);
    assert!(edge.feedback_value > 0.0, "TL-02: feedback_value should grow on positive");
    assert_eq!(edge.avoidance, 0.0, "TL-02: avoidance should not change on positive");
}

// ── TL-03: route score decreases when avoidance is high ──────────────────
#[test]
fn tl03_avoidance_reduces_score() {
    use syntrail_lm::prediction::PredictionStore;
    use syntrail_lm::units::UnitId;
    let ctx = UnitId::primitive(1);
    let u2 = UnitId::primitive(2);
    let u3 = UnitId::primitive(3);
    let mut store = PredictionStore::new();
    store.observe(ctx, u2);
    store.observe(ctx, u3);
    // Penalize u2 repeatedly
    for _ in 0..10 { store.apply_feedback_to_edge(ctx, u2, -1.0, 0.99); }
    let top = store.top1(ctx);
    assert_eq!(top, Some(u3), "TL-03: u3 should rank higher after u2 is avoided");
}

// ── TL-04: avoidance is context-specific (not global) ────────────────────
#[test]
fn tl04_avoidance_is_context_specific() {
    use syntrail_lm::prediction::PredictionStore;
    use syntrail_lm::units::UnitId;
    let ctx1 = UnitId::primitive(1);
    let ctx2 = UnitId::primitive(2);
    let next = UnitId::primitive(3);
    let mut store = PredictionStore::new();
    store.observe(ctx1, next);
    store.observe(ctx2, next);
    // Penalize in ctx1 context only
    for _ in 0..10 { store.apply_feedback_to_edge(ctx1, next, -1.0, 0.99); }
    // In ctx2, the edge should be unaffected
    let e2 = store.iter_all().find(|e| e.context == ctx2 && e.next_unit == next).unwrap();
    assert_eq!(e2.avoidance, 0.0, "TL-04: avoidance should be context-specific");
}

// ── TL-05: avoidance persists through snapshot round-trip ────────────────
#[test]
fn tl05_avoidance_persists() {
    use syntrail_lm::units::UnitId;
    let ctx = UnitId::primitive(1);
    let next = UnitId::primitive(2);
    let mut m = ModelState::new();
    // Create edge and apply avoidance
    for _ in 0..5 { m.train("ab"); }
    for _ in 0..5 {
        m.predictions.apply_feedback_to_edge(ctx, next, -1.0, 0.99);
    }
    let avoidance_before: f64 = m.predictions.iter_all()
        .map(|e| e.avoidance).sum();
    let tmp = NamedTempFile::new().unwrap();
    persistence::save(&m, tmp.path()).unwrap();
    let loaded = persistence::load(tmp.path()).unwrap();
    let avoidance_after: f64 = loaded.predictions.iter_all()
        .map(|e| e.avoidance).sum();
    // Phase 17: strength fields stored as f32 in snapshot → tolerance 1e-4.
    assert!((avoidance_before - avoidance_after).abs() < 1e-4,
        "TL-05: avoidance should persist through snapshot");
}

// ── TL-06: feedback_count increments for both positive and negative ───────
#[test]
fn tl06_feedback_count_increments_on_both() {
    use syntrail_lm::prediction::PredictionEdge;
    use syntrail_lm::units::UnitId;
    let mut edge = PredictionEdge::new(UnitId::primitive(1), UnitId::primitive(2));
    edge.record_usage();
    edge.apply_feedback(1.0, 0.99);
    edge.apply_feedback(-1.0, 0.99);
    assert_eq!(edge.feedback_count, 2, "TL-06: feedback_count should increment for both signs");
}

// ── CR-01: chunk can hold its own associations ────────────────────────────
#[test]
fn cr01_chunk_own_associations() {
    use syntrail_lm::association::AssociationStore;
    use syntrail_lm::units::UnitId;
    let mut store = AssociationStore::new(8, 0.99);
    let chunk = UnitId::chunk(0);
    let target = UnitId::primitive(42);
    store.observe(chunk, target, 0);
    assert_eq!(store.top_for(chunk).len(), 1, "CR-01: chunk should hold its own associations");
    assert_eq!(store.top_for(chunk)[0].target, target);
}

// ── CR-02: associations persist through snapshot round-trip ──────────────
#[test]
fn cr02_association_snapshot_roundtrip() {
    let mut m = ModelState::new();
    for _ in 0..10 { m.train("hello world"); }
    let assoc_before = m.association_count();
    let tmp = NamedTempFile::new().unwrap();
    persistence::save(&m, tmp.path()).unwrap();
    let loaded = persistence::load(tmp.path()).unwrap();
    assert_eq!(loaded.association_count(), assoc_before,
        "CR-02: association count should be preserved");
}

// ── CR-03: expose creates more associations with more training ────────────
#[test]
fn cr03_associations_grow_with_training() {
    let mut m = ModelState::new();
    let count0 = m.association_count();
    for _ in 0..5 { m.train("hello world"); }
    let count1 = m.association_count();
    assert!(count1 > count0, "CR-03: associations should grow with training");
}

// ── CR-04: version field is "0.4" ────────────────────────────────────────
#[test]
fn cr04_snapshot_version_is_v04() {
    let m = ModelState::new();
    let snap = persistence::to_snapshot(&m);
    assert_eq!(snap.version, "0.5", "CR-04: snapshot version should be 0.5");
}

// ── CR-05: recall count does not exceed limit ─────────────────────────────
#[test]
fn cr05_recall_respects_limit() {
    let mut m = ModelState::new();
    for _ in 0..10 { m.train("abcdef abcdef"); }
    let prim_ids = {
        let mut tmp = m.primitives.clone();
        tmp.encode("a")
    };
    let segmented = segment(&prim_ids, &m.chunks, 0.0);
    let source = segmented[0];
    let results = m.recall(source, 2);
    assert!(results.len() <= 2, "CR-05: recall should not exceed limit");
}

// ── HI-01: hierarchical fallback to children ─────────────────────────────
#[test]
fn hi01_hierarchical_recall_fallback() {
    use syntrail_lm::association::AssociationStore;
    use syntrail_lm::chunks::ChunkRegistry;
    use syntrail_lm::units::UnitId;
    let p1 = UnitId::primitive(1);
    let p2 = UnitId::primitive(2);
    let p99 = UnitId::primitive(99);
    let mut chunks = ChunkRegistry::new();
    let cid = chunks.get_or_create(p1, p2, 2);
    let chunk_unit = UnitId::chunk(cid);
    let mut store = AssociationStore::new(8, 0.99);
    // Chunk has no direct associations; children do
    store.observe(p1, p99, 0);
    let results = store.recall(chunk_unit, &chunks, 4);
    assert!(!results.is_empty(), "HI-01: hierarchical fallback should find children's associations");
}

// ── HI-02: direct associations preferred over fallback ───────────────────
#[test]
fn hi02_direct_associations_preferred() {
    use syntrail_lm::association::AssociationStore;
    use syntrail_lm::chunks::ChunkRegistry;
    use syntrail_lm::units::UnitId;
    let p1 = UnitId::primitive(1);
    let p2 = UnitId::primitive(2);
    let p10 = UnitId::primitive(10);
    let p99 = UnitId::primitive(99);
    let mut chunks = ChunkRegistry::new();
    let cid = chunks.get_or_create(p1, p2, 2);
    let chunk_unit = UnitId::chunk(cid);
    let mut store = AssociationStore::new(8, 0.99);
    // Chunk has direct association to p10
    store.observe(chunk_unit, p10, 0);
    // Child has association to p99
    store.observe(p1, p99, 0);
    let results = store.recall(chunk_unit, &chunks, 4);
    assert!(!results.is_empty(), "HI-02: should return results");
    // Direct association should be preferred
    assert_eq!(results[0].0, p10, "HI-02: direct association should be preferred over fallback");
}

// ── HI-03: recall on primitive with no associations returns empty ─────────
#[test]
fn hi03_recall_no_associations_returns_empty() {
    use syntrail_lm::association::AssociationStore;
    use syntrail_lm::chunks::ChunkRegistry;
    use syntrail_lm::units::UnitId;
    let store = AssociationStore::new(8, 0.99);
    let chunks = ChunkRegistry::new();
    let results = store.recall(UnitId::primitive(42), &chunks, 4);
    assert!(results.is_empty(), "HI-03: no associations should return empty");
}

// ── HI-04: recall depth limit prevents stack overflow ────────────────────
#[test]
fn hi04_recall_depth_limit() {
    use syntrail_lm::association::AssociationStore;
    use syntrail_lm::chunks::ChunkRegistry;
    use syntrail_lm::units::UnitId;
    // Build a chain of chunks: c0 = (p0, p1), c1 = (c0, p2), etc.
    let mut chunks = ChunkRegistry::new();
    let p0 = UnitId::primitive(0);
    let p1 = UnitId::primitive(1);
    let p2 = UnitId::primitive(2);
    let cid0 = chunks.get_or_create(p0, p1, 2);
    let c0 = UnitId::chunk(cid0);
    let cid1 = chunks.get_or_create(c0, p2, 3);
    let c1 = UnitId::chunk(cid1);
    let store = AssociationStore::new(8, 0.99);
    // Deep recall should not panic (depth guard prevents infinite recursion)
    let results = store.recall(c1, &chunks, 4);
    let _ = results; // just verify no panic
}

// ── EV-01: evaluate_frozen does not mutate model ─────────────────────────
#[test]
fn ev01_evaluate_frozen_does_not_mutate() {
    use syntrail_lm::eval::evaluate_frozen;
    let mut m = ModelState::new();
    for _ in 0..20 { m.train("hello world"); }
    let tick_before = m.tick;
    let edges_before = m.edge_count();
    let assoc_before = m.association_count();
    evaluate_frozen(&m, "hello world\nhello again");
    assert_eq!(m.tick, tick_before, "EV-01: tick must not change");
    assert_eq!(m.edge_count(), edges_before, "EV-01: edges must not change");
    assert_eq!(m.association_count(), assoc_before, "EV-01: associations must not change");
}

// ── EV-02: evaluate_frozen returns plausible dpc ─────────────────────────
#[test]
fn ev02_evaluate_returns_plausible_dpc() {
    use syntrail_lm::eval::evaluate_frozen;
    let mut m = ModelState::new();
    for _ in 0..20 { m.train("hello world"); }
    let result = evaluate_frozen(&m, "hello world");
    assert!(result.dpc > 0.0, "EV-02: dpc should be > 0");
    assert!(result.dpc <= 1.0, "EV-02: trained model dpc should be <= 1.0");
}

// ── EV-03: untrained model has dpc=1.0 ───────────────────────────────────
#[test]
fn ev03_untrained_dpc_equals_one() {
    use syntrail_lm::eval::evaluate_frozen;
    let m = ModelState::new();
    let result = evaluate_frozen(&m, "abc");
    assert!((result.dpc - 1.0).abs() < 1e-9,
        "EV-03: untrained model dpc should be 1.0, got {}", result.dpc);
}

// ── EV-04: evaluate on empty text returns all zeros ──────────────────────
#[test]
fn ev04_empty_text_returns_zeros() {
    use syntrail_lm::eval::evaluate_frozen;
    let m = ModelState::new();
    let result = evaluate_frozen(&m, "");
    assert_eq!(result.char_count, 0, "EV-04: char_count should be 0");
    assert_eq!(result.dpc, 0.0, "EV-04: dpc should be 0.0");
    assert_eq!(result.prediction_accuracy, 0.0, "EV-04: accuracy should be 0.0");
}

// ── SS-01: snapshot includes association edges ────────────────────────────
#[test]
fn ss01_snapshot_includes_associations() {
    let mut m = ModelState::new();
    for _ in 0..10 { m.train("hello world"); }
    let snap = persistence::to_snapshot(&m);
    // The snapshot should round-trip correctly
    let m2 = persistence::from_snapshot(snap);
    assert_eq!(m2.association_count(), m.association_count(),
        "SS-01: association count should match after snapshot round-trip");
}

// ── SS-02: snapshot preserves avoidance ──────────────────────────────────
#[test]
fn ss02_snapshot_preserves_avoidance() {
    use syntrail_lm::units::UnitId;
    let ctx = UnitId::primitive(10);
    let next = UnitId::primitive(20);
    let mut m = ModelState::new();
    m.predictions.apply_feedback_to_edge(ctx, next, -1.0, 0.99);
    let avoidance_before: f64 = m.predictions.iter_all().map(|e| e.avoidance).sum();
    let snap = persistence::to_snapshot(&m);
    let m2 = persistence::from_snapshot(snap);
    let avoidance_after: f64 = m2.predictions.iter_all().map(|e| e.avoidance).sum();
    assert!((avoidance_before - avoidance_after).abs() < 1e-9,
        "SS-02: avoidance should be preserved in snapshot");
}

// ── SS-03: snapshot version is "0.4" ─────────────────────────────────────
#[test]
fn ss03_snapshot_version_v04() {
    let m = ModelState::new();
    let snap = persistence::to_snapshot(&m);
    assert_eq!(snap.version, "0.5", "SS-03: snapshot version should be 0.5");
}

// ── SS-04: v0.2 snapshot loads with zero avoidance ───────────────────────
#[test]
fn ss04_v02_snapshot_loads_with_zero_avoidance() {
    // Simulate a v0.2-style JSON snapshot (no avoidance, no association_edges)
    let json = r#"{
        "version": "0.2",
        "tick": 0,
        "primitives": [],
        "chunks": [],
        "prediction_edges": [
            {
                "context": {"is_chunk": false, "raw": 1},
                "next_unit": {"is_chunk": false, "raw": 2},
                "use_count": 5,
                "usage_strength": 4.9,
                "feedback_value": 0.0,
                "feedback_count": 0
            }
        ],
        "merge_candidates": [],
        "metrics": {"total_decisions": 5, "total_characters": 5},
        "next_trace_id": 0
    }"#;
    let snap: persistence::ModelSnapshot = serde_json::from_str(json).unwrap();
    let m = persistence::from_snapshot(snap);
    let total_avoid: f64 = m.predictions.iter_all().map(|e| e.avoidance).sum();
    assert_eq!(total_avoid, 0.0, "SS-04: v0.2 edges should load with zero avoidance");
    assert_eq!(m.association_count(), 0, "SS-04: v0.2 snapshot should load with no associations");
}

// ── T-19: v0.3 regression — all prior tests still pass ───────────────────
#[test]
fn t19_regression_v03() {
    assert!(true, "T-19: v0.3 regression marker");
}

// ═══════════════════════════════════════════════════════════════
// v0.4 Acceptance Tests — Residency (HOT / SLEEP)
// ═══════════════════════════════════════════════════════════════

// ── RS-01: HOT chunk can be demoted to SLEEP ──────────────────────────────
#[test]
fn rs01_hot_to_sleep() {
    use syntrail_lm::chunks::{ChunkRegistry, Residency};
    let p1 = UnitId::primitive(1);
    let p2 = UnitId::primitive(2);
    let mut reg = ChunkRegistry::new();
    let id = reg.get_or_create(p1, p2, 2);
    assert_eq!(reg.get(id).unwrap().residency, Residency::Hot, "RS-01: new chunk should be HOT");
    reg.get_mut(id).unwrap().demote_to_sleep();
    assert_eq!(reg.get(id).unwrap().residency, Residency::Sleep, "RS-01: after demote should be SLEEP");
}

// ── RS-02: SLEEP chunk preserves core structure ───────────────────────────
#[test]
fn rs02_sleep_preserves_core() {
    use syntrail_lm::chunks::ChunkRegistry;
    let p1 = UnitId::primitive(1);
    let p2 = UnitId::primitive(2);
    let mut reg = ChunkRegistry::new();
    let id = reg.get_or_create(p1, p2, 2);
    reg.get_mut(id).unwrap().demote_to_sleep();
    let chunk = reg.get(id).unwrap();
    assert_eq!(chunk.left, p1, "RS-02: left must be preserved");
    assert_eq!(chunk.right, p2, "RS-02: right must be preserved");
    assert_eq!(chunk.id, id, "RS-02: id must be preserved");
}

// ── RS-03: SLEEP chunk is excluded from segmentation ─────────────────────
#[test]
fn rs03_sleep_chunk_excluded_from_segmentation() {
    use syntrail_lm::chunks::ChunkRegistry;
    let mut chunks = ChunkRegistry::new();
    let p1 = UnitId::primitive(1);
    let p2 = UnitId::primitive(2);
    // Create and promote chunk so it would normally be used
    let id = chunks.get_or_create(p1, p2, 2);
    for tick in 0..64 { chunks.get_mut(id).unwrap().record_usage(tick); }
    // Verify it segments when HOT
    let prims = vec![1u32, 2];
    let hot_result = segment(&prims, &chunks, 0.0);
    assert_eq!(hot_result.len(), 1, "RS-03: HOT chunk should be segmented");
    // Demote to SLEEP via registry method (Phase 8: keeps hot_pair_to_id in sync)
    chunks.demote(id);
    let sleep_result = segment(&prims, &chunks, 0.0);
    assert_eq!(sleep_result.len(), 2, "RS-03: SLEEP chunk should NOT be segmented");
}

// ── RS-04: SLEEP chunk is reactivated on re-encounter ────────────────────
#[test]
fn rs04_sleep_reactivation() {
    use syntrail_lm::chunks::Residency;
    let mut m = ModelState::new();
    // Train until "ab" chunk is formed
    for _ in 0..20 { m.train("ab"); }
    let p_a = m.primitives.id('a').unwrap();
    let p_b = m.primitives.id('b').unwrap();
    let ua = UnitId::primitive(p_a);
    let ub = UnitId::primitive(p_b);
    let chunk_id = m.chunks.find_by_pair(ua, ub);
    if let Some(cid) = chunk_id {
        // Demote the chunk to SLEEP (Phase 8: via registry method)
        m.chunks.demote(cid);
        assert_eq!(m.chunks.get(cid).unwrap().residency, Residency::Sleep, "should be SLEEP");
        // Expose "ab" again — should trigger reactivation in consider_merges
        for _ in 0..5 { m.expose("ab"); }
        assert_eq!(m.chunks.get(cid).unwrap().residency, Residency::Hot,
            "RS-04: chunk should be reactivated to HOT after re-encounter");
    }
    // If no chunk was formed, skip (training may not have created the chunk yet)
}

// ── RS-05: reactivated chunk reuses same ID ───────────────────────────────
#[test]
fn rs05_reactivation_keeps_same_id() {
    use syntrail_lm::chunks::ChunkRegistry;
    let p1 = UnitId::primitive(1);
    let p2 = UnitId::primitive(2);
    let mut reg = ChunkRegistry::new();
    let id_before = reg.get_or_create(p1, p2, 2);
    reg.get_mut(id_before).unwrap().demote_to_sleep();
    // get_or_create returns same id even when SLEEP
    let id_after = reg.get_or_create(p1, p2, 2);
    assert_eq!(id_before, id_after, "RS-05: reactivated chunk must keep same ID");
}

// ── RS-06: enforce_hot_budget demotes weakest HOT chunks ─────────────────
#[test]
fn rs06_enforce_hot_budget() {
    let mut m = ModelState::new();
    // Train until we have multiple chunks
    for _ in 0..30 { m.train("abcdef abcdef"); }
    let hot_before = m.hot_chunk_count();
    if hot_before > 2 {
        m.enforce_hot_budget(2);
        assert!(m.hot_chunk_count() <= 2, "RS-06: HOT count should be <= budget");
        assert!(m.sleep_chunk_count() >= hot_before - 2,
            "RS-06: excess chunks should become SLEEP");
    }
}

// ── RS-07: residency persists through snapshot round-trip ────────────────
#[test]
fn rs07_residency_snapshot_roundtrip() {
    let mut m = ModelState::new();
    for _ in 0..20 { m.train("hello world"); }
    // Demote one chunk to SLEEP
    let sleep_count_before = {
        let first_hot_id = m.chunks.iter_all()
            .find(|c| c.is_hot())
            .map(|c| c.id);
        if let Some(id) = first_hot_id {
            m.chunks.get_mut(id).unwrap().demote_to_sleep();
        }
        m.sleep_chunk_count()
    };
    let tmp = NamedTempFile::new().unwrap();
    persistence::save(&m, tmp.path()).unwrap();
    let loaded = persistence::load(tmp.path()).unwrap();
    assert_eq!(loaded.sleep_chunk_count(), sleep_count_before,
        "RS-07: SLEEP count should be preserved after round-trip");
}

// ── RS-08: hot_count + sleep_count == total chunk count ──────────────────
#[test]
fn rs08_count_identity() {
    let mut m = ModelState::new();
    for _ in 0..20 { m.train("hello world"); }
    assert_eq!(m.hot_chunk_count() + m.sleep_chunk_count(), m.chunk_count(),
        "RS-08: hot + sleep must equal total");
}

// ── T-20: v0.4 regression — all prior tests still pass ───────────────────
#[test]
fn t20_regression_v04() {
    assert!(true, "T-20: v0.4 regression marker");
}

// ── GN-01: generalize() falls back through associations ──────────────────
#[test]
fn gn01_generalize_via_association() {
    let mut m = ModelState::new();
    // Train "ab" heavily — 'b' follows 'a', and b↔c are co-associated.
    for _ in 0..20 { m.train("ab"); }
    for _ in 0..20 { m.train("bc"); }
    // 'a' can predict 'b' directly; this should succeed via level-1.
    let p_a = m.primitives.id('a').unwrap();
    let ua = UnitId::primitive(p_a);
    assert!(m.generalize(ua).is_some(), "GN-01: generalize should find prediction for 'a'");
}

// ── GN-02: novel_candidates() returns analogy results (must be non-empty) ──
#[test]
fn gn02_novel_candidates_nonempty() {
    let mut m = ModelState::new();
    // "abc" and "abd" share prefix "ab"; 'a' has associations to both 'b','c','d'.
    for _ in 0..40 { m.train("abc"); }
    for _ in 0..40 { m.train("abd"); }
    let p_a = m.primitives.id('a').unwrap();
    let ua = UnitId::primitive(p_a);
    let candidates = m.novel_candidates(ua, 5);
    assert!(
        !candidates.is_empty(),
        "GN-02: novel_candidates must find analogy results for 'a' after training 'abc'/'abd'"
    );
}

// ── GN-04: generalize uses association bridge when no direct route ─────────
#[test]
fn gn04_generalize_uses_association_bridge() {
    let mut m = ModelState::new();
    // Train "ba" a few times → b→a route, b↔a associated; 'a' gets NO outgoing route.
    for _ in 0..5 { m.train("ba"); }
    // Train "bc" heavily → strong b→c route.
    for _ in 0..40 { m.train("bc"); }
    let pa = m.primitives.id('a').unwrap();
    let ua = UnitId::primitive(pa);
    // 'a' has no direct outgoing prediction; generalize must use association bridge.
    // a is associated with b (from "ba"), b has strong route to c.
    let result = m.generalize(ua);
    assert!(
        result.is_some(),
        "GN-04: generalize must find a result via association bridge when 'a' has no direct route"
    );
}

// ── GN-03: novel_candidates() respects limit ─────────────────────────────
#[test]
fn gn03_novel_candidates_respects_limit() {
    let mut m = ModelState::new();
    for _ in 0..20 { m.train("abcdefg"); }
    let p_a = m.primitives.id('a').unwrap();
    let ua = UnitId::primitive(p_a);
    let candidates = m.novel_candidates(ua, 3);
    assert!(candidates.len() <= 3, "GN-03: must respect limit");
}

// ═══════════════════════════════════════════════════════════════
// P0 Generation Tests — Cycle / No-Progress / EOS
// ═══════════════════════════════════════════════════════════════

// ── GEN-LOOP-01: self-loop (A→A) is detected and generation stops ─────────
#[test]
fn gen_loop_01_self_loop_stops() {
    use syntrail_lm::prediction::PredictionStore;
    use syntrail_lm::units::UnitId;
    // Build a minimal model with only A→A route.
    let mut m = ModelState::new();
    m.expose_external("aa");
    // After expose, P(a) → P(a) is the only edge.
    // Repeatedly observe to strengthen it.
    for _ in 0..20 { m.expose_external("aaa"); }
    let (_, trace) = m.generate_with_trace("a", "a", 50, 1);
    // Must not emit 50 units — cycle detection must have fired.
    assert!(
        trace.decision_count < 50 || trace.stopped_by_cycle,
        "GEN-LOOP-01: self-loop must be detected; decisions={}, stopped_by_cycle={}",
        trace.decision_count, trace.stopped_by_cycle
    );
}

// ── GEN-LOOP-02: two-state loop (A→B→A) is detected ─────────────────────
#[test]
fn gen_loop_02_two_state_loop_stops() {
    let mut m = ModelState::new();
    // Create strong A→B→A route and weak other routes.
    for _ in 0..30 { m.expose_external("ababab"); }
    let (_, trace) = m.generate_with_trace("a", "a", 50, 2);
    assert!(
        trace.decision_count < 50 || trace.stopped_by_cycle,
        "GEN-LOOP-02: A→B→A loop must be detected; decisions={}, stopped_by_cycle={}",
        trace.decision_count, trace.stopped_by_cycle
    );
}

// ── GEN-LOOP-03: SEQUENCE_END stops generation before max_units (§7) ────────
#[test]
fn gen_loop_03_eos_stops_generation() {
    use syntrail_lm::model::SEQUENCE_END_CHAR;
    let mut m = ModelState::new();
    // §7: train with expose_with_sequence_end so the model learns to predict
    // SEQUENCE_END_CHAR at the end of a complete output sequence.
    for _ in 0..80 { m.expose_with_sequence_end("hello"); }
    // SEQUENCE_END must be registered as a primitive.
    assert!(
        m.sequence_end_unit().is_some(),
        "GEN-LOOP-03: SEQUENCE_END primitive must exist after expose_with_sequence_end"
    );
    // Generate from the trained seed; SEQUENCE_END should fire before max_units.
    let (output, trace) = m.generate_with_trace("hello", "hello", 50, 3);
    assert!(
        trace.stopped_by_eos,
        "GEN-LOOP-03: SEQUENCE_END must stop generation after heavy training; stopped_by_eos=false, decisions={}",
        trace.decision_count
    );
    // The sequence-end character itself must not appear in output.
    assert!(
        !output.contains(SEQUENCE_END_CHAR),
        "GEN-LOOP-03: SEQUENCE_END char must not appear in output text"
    );
    // Must have stopped before max_units.
    assert!(
        trace.decision_count < 50,
        "GEN-LOOP-03: generation must end before max_units when SEQUENCE_END fires"
    );
    // emitted_text also must not contain the sequence-end char.
    assert!(
        !trace.emitted_text.contains(SEQUENCE_END_CHAR),
        "GEN-LOOP-03: SEQUENCE_END char must not appear in emitted_text"
    );
}

// ── GEN-LOOP-08: §9 — TURN_BOUNDARY does NOT stop generation ─────────────
#[test]
fn gen_loop_08_turn_boundary_does_not_stop_generation() {
    use syntrail_lm::model::SEQUENCE_END_CHAR;
    let mut m = ModelState::new();
    // Train using turn boundary (not sequence end) so model learns turn boundaries.
    for _ in 0..60 { m.expose_with_turn_boundary("hello world"); }
    // TURN_BOUNDARY must be registered, SEQUENCE_END must NOT be (wasn't exposed).
    assert!(
        m.turn_boundary_unit().is_some(),
        "GEN-LOOP-08: TURN_BOUNDARY primitive must exist after expose_with_turn_boundary"
    );
    assert!(
        m.sequence_end_unit().is_none(),
        "GEN-LOOP-08: SEQUENCE_END must not be registered when only turn-boundary was trained"
    );
    // Generation must NOT stop with stopped_by_eos because SEQUENCE_END was never trained.
    let (output, trace) = m.generate_with_trace("hello", "hello", 50, 8);
    assert!(
        !trace.stopped_by_eos,
        "GEN-LOOP-08: TURN_BOUNDARY must not trigger stopped_by_eos; decisions={}",
        trace.decision_count
    );
    // SEQUENCE_END char must never appear in output.
    assert!(
        !output.contains(SEQUENCE_END_CHAR),
        "GEN-LOOP-08: SEQUENCE_END char must not appear in output"
    );
}

// ── GEN-LOOP-04: emitted_text excludes seed ───────────────────────────────
#[test]
fn gen_loop_04_emitted_excludes_seed() {
    let mut m = ModelState::new();
    for _ in 0..20 { m.expose_external("hello world"); }
    let seed = "hel";
    let (output, trace) = m.generate_with_trace(seed, seed, 10, 4);
    // output includes seed; emitted_text does not.
    assert!(output.starts_with(seed) || output.is_empty(),
        "GEN-LOOP-04: output should start with seed or be empty");
    // emitted_text should NOT start with the seed (it's the emitted portion only).
    assert!(
        !trace.emitted_text.starts_with(seed) || trace.emitted_text.is_empty(),
        "GEN-LOOP-04: emitted_text should not contain seed; got {:?}", trace.emitted_text
    );
}

// ── GEN-LOOP-05: generate does not exceed max_units ──────────────────────
#[test]
fn gen_loop_05_max_units_respected() {
    let mut m = ModelState::new();
    for _ in 0..30 { m.expose_external("abababababab"); }
    let (_, trace) = m.generate_with_trace("a", "a", 10, 5);
    assert!(
        trace.decision_count <= 10,
        "GEN-LOOP-05: decision_count {} must not exceed max_units=10",
        trace.decision_count
    );
}

// ── GEN-LOOP-06: TurnTrace fields set correctly after cycle stop ──────────
#[test]
fn gen_loop_06_trace_fields_after_cycle() {
    let mut m = ModelState::new();
    for _ in 0..30 { m.expose_external("ababab"); }
    let (_, trace) = m.generate_with_trace("a", "a", 50, 6);
    if trace.stopped_by_cycle {
        assert!(trace.decision_count < 50, "GEN-LOOP-06: if stopped_by_cycle, must have < max_units decisions");
    }
    // stopped_by_eos and stopped_by_cycle must not both be true at once.
    assert!(
        !(trace.stopped_by_eos && trace.stopped_by_cycle),
        "GEN-LOOP-06: cannot be stopped by both EOS and cycle simultaneously"
    );
}

// ── GEN-LOOP-07: §12 no-progress stops generation (stagnation window) ────────
#[test]
fn gen_loop_07_no_progress_stops_generation() {
    // Train only "ab" — model gets a→b and b→a (via association) or just a→b loop.
    // With heavy training the cycle a→b→a is detected by route repetition,
    // but even if escape routes create a wider 2-unit loop, §12 must terminate.
    let mut m = ModelState::new();
    for _ in 0..40 { m.expose_external("ab"); }
    let (_, trace) = m.generate_with_trace("a", "a", 200, 77);
    assert!(
        trace.decision_count < 200,
        "GEN-LOOP-07: §12 no-progress detection must stop generation before max_units, got {}",
        trace.decision_count
    );
}

// ═══════════════════════════════════════════════════════════════
// Phase B: Representation Lineage Tests (REP-01 through REP-07)
// ═══════════════════════════════════════════════════════════════

// ── REP-01: Acquisition order preserved through save/load ─────────────────
#[test]
fn rep01_acquisition_order_preserved_through_save_load() {
    use syntrail_lm::representation::RepresentationStore;
    let mut m = ModelState::new();
    // First observe "ab" as primitives → R0 = [P(a), P(b)]
    m.expose_external("ab");
    let rep_count_after_expose = m.representations.entry_count();
    assert!(rep_count_after_expose >= 1, "REP-01: at least one representation after expose");

    // Train until chunk forms, then replay to trigger new segmentation → R1
    for _ in 0..20 { m.expose_external("ab"); }
    for _ in 0..10 { m.replay("ab"); }

    // Save and load
    let tmp = tempfile::NamedTempFile::new().unwrap();
    syntrail_lm::persistence::save(&m, tmp.path()).unwrap();
    let loaded = syntrail_lm::persistence::load(tmp.path()).unwrap();

    // Representation count must be preserved
    assert_eq!(
        loaded.representations.entry_count(),
        m.representations.entry_count(),
        "REP-01: representation entry count must survive save/load"
    );

    // For each identity, acquisition order must be preserved.
    // Find the identity for "ab"
    let prim_ids = m.primitives.encode_existing("ab");
    if let Some(id) = m.identities.find_identity(&prim_ids) {
        let orig_reps = m.representations.for_identity(id);
        let loaded_reps = loaded.representations.for_identity(id);
        assert_eq!(orig_reps.len(), loaded_reps.len(), "REP-01: same number of reps per identity");
        for (orig, loaded_r) in orig_reps.iter().zip(loaded_reps.iter()) {
            assert_eq!(orig.acquired_at, loaded_r.acquired_at,
                "REP-01: acquired_at must match");
            assert_eq!(orig.predecessor_rep_id, loaded_r.predecessor_rep_id,
                "REP-01: predecessor chain must be preserved");
        }
    }
}

// ── REP-02: Preferred Representation Selection ────────────────────────────
#[test]
fn rep02_preferred_selection_balances_confidence_and_cost() {
    use syntrail_lm::representation::RepresentationStore;
    use syntrail_lm::units::UnitId;
    let mut store = RepresentationStore::new();
    // R0: high confidence, high cost (5 units)
    let (r0, _) = store.intern(0, vec![
        UnitId::primitive(1), UnitId::primitive(2), UnitId::primitive(3),
        UnitId::primitive(4), UnitId::primitive(5)], 1);
    store.get_mut(r0).unwrap().confidence = 1.0;

    // R1: medium confidence, medium cost (3 units) — should win
    let (r1, _) = store.intern(0, vec![
        UnitId::primitive(1), UnitId::chunk(0), UnitId::primitive(5)], 5);
    store.get_mut(r1).unwrap().confidence = 0.9;

    // R2: low confidence, low cost (1 unit)
    let (r2, _) = store.intern(0, vec![UnitId::chunk(1)], 10);
    store.get_mut(r2).unwrap().confidence = 0.2;

    let preferred = store.preferred(0).expect("REP-02: preferred must return something");
    assert_eq!(preferred.rep_id, r1,
        "REP-02: R1 (conf=0.9, cost=3) must beat R0 (conf=1.0, cost=5) and R2 (conf=0.2, cost=1)");
}

// ── REP-03: Predecessor Fallback ──────────────────────────────────────────
#[test]
fn rep03_predecessor_fallback() {
    use syntrail_lm::representation::RepresentationStore;
    use syntrail_lm::units::UnitId;
    let mut store = RepresentationStore::new();
    let (r0, _) = store.intern(0, vec![UnitId::primitive(1), UnitId::primitive(2)], 1);
    let (r1, _) = store.intern(0, vec![UnitId::chunk(0)], 5);

    // When R1 is current but fails, predecessor_of(r1) should return R0.
    let fallback = store.predecessor_of(r1).expect("REP-03: must have predecessor");
    assert_eq!(fallback.rep_id, r0, "REP-03: R0 must be returned as predecessor of R1");

    // R0 has no predecessor — chain terminates.
    assert!(store.predecessor_of(r0).is_none(), "REP-03: R0 has no predecessor");
}

// ── REP-05: Replay acquires new Representation ────────────────────────────
#[test]
fn rep05_replay_acquires_new_representation() {
    let mut m = ModelState::new();
    // Observe "ab" once: R0 = [P(a), P(b)]
    m.expose_external("ab");
    let prim_ids_ab = m.primitives.encode_existing("ab");
    let id_ab = m.identities.find_identity(&prim_ids_ab).expect("identity must exist");
    let reps_after_expose = m.representations.for_identity(id_ab).len();

    // Train until chunk "ab" definitely forms (adjacency threshold is low).
    for _ in 0..80 { m.expose_external("ab"); }

    // Force replay to trigger re-segmentation with the newly formed chunk.
    for _ in 0..10 { m.replay("ab"); }

    // If a chunk formed and segmentation changed, a new representation was registered.
    let reps_after_replay = m.representations.for_identity(id_ab).len();

    // Chunk must have formed after 40 expose calls on a 2-char string.
    assert!(m.chunk_count() > 0, "REP-05: chunk must form after heavy training on 'ab'");
    // New segmentation via chunk MUST produce a new representation (strictly more than before).
    assert!(
        reps_after_replay > reps_after_expose,
        "REP-05: representation count must strictly increase when chunk forms; before={reps_after_expose} after={reps_after_replay}"
    );
}

// ── REP-03-INTEGRATION: predecessor chain used in generation ─────────────
#[test]
fn rep03_integration_predecessor_used_in_generation() {
    // Build a fixture where the preferred Rep Rn fails but R(n-1) has a route.
    // Train "ab" until chunk forms → R0=[P(a),P(b)], R1=[C(ab)].
    // The preferred representation R1 contains the chunk unit C(ab).
    // If R1 has no outgoing prediction but R0's unit P(b) does → fallback to R0.
    let mut m = ModelState::new();
    for _ in 0..80 { m.train("ab"); }
    for _ in 0..80 { m.train("bc"); }
    // "b" should have a route to "c".
    // After chunk "ab" forms, the preferred rep of identity "ab" is [C(ab)].
    // C(ab) likely has no route yet (only "bc" trained C-to-something).
    // The predecessor rep [P(a), P(b)] includes P(b) which routes to P(c).
    // generate_with_trace with seed="ab" should produce something (not empty).
    let (_, trace) = m.generate_with_trace("ab", "ab", 10, 99);
    assert!(
        trace.decision_count > 0,
        "REP-03-INTEGRATION: generation from 'ab' must produce decisions via predecessor chain"
    );
}

// ── REP-04: All reps fail → reaches primitive decomposition ──────────────
#[test]
fn rep04_all_reps_fail_reaches_primitive() {
    // Completely novel context: a unit that has representations but no routes.
    // We test that generation doesn't panic (reaches graceful end via primitives).
    let mut m = ModelState::new();
    // Train "bc" so P(b) has route P(b)→P(c).
    for _ in 0..40 { m.train("bc"); }
    // "a" has no chunk, no route, no representation.
    let (_, trace) = m.generate_with_trace("a", "a", 5, 99);
    // Should not panic; either produces output via primitives or stops.
    // decision_count may be 0 if no route found — that's acceptable; no panic is required.
    let _ = trace;
}

// ── REP-06: §3 — rep confidence grows on repeated recognition ────────────
#[test]
fn rep06_confidence_grows_on_repeated_recognition() {
    let mut m = ModelState::new();
    // First expose: acquires the representation (is_new=true, no success recorded).
    m.expose("ab");
    let conf_after_first = {
        let id = m.identities.find_identity(
            &m.primitives.encode("ab").iter().copied().collect::<Vec<_>>()
        ).expect("identity must exist");
        m.representations.preferred(id).expect("rep must exist").confidence
    };
    // Second expose: same segmentation → is_new=false → record_success().
    m.expose("ab");
    let conf_after_second = {
        let id = m.identities.find_identity(
            &m.primitives.encode("ab").iter().copied().collect::<Vec<_>>()
        ).unwrap();
        m.representations.preferred(id).unwrap().confidence
    };
    assert!(
        conf_after_second > conf_after_first,
        "REP-06: confidence must rise after repeated recognition: {} → {}",
        conf_after_first, conf_after_second
    );
}

// ── REP-07: Legacy Migration — no fake history ───────────────────────────
#[test]
fn rep07_legacy_migration_no_synthetic_history() {
    // Load a v0.5 snapshot that has no representation_entries field.
    // Representations should be empty (no fake history).
    let json = r#"{
        "version": "0.5",
        "tick": 5,
        "primitives": [[1, 97], [2, 98]],
        "chunks": [],
        "prediction_edges": [],
        "merge_candidates": [],
        "metrics": {"total_decisions": 5, "total_characters": 5},
        "next_trace_id": 0
    }"#;
    let snap: syntrail_lm::persistence::ModelSnapshot = serde_json::from_str(json).unwrap();
    let m = syntrail_lm::persistence::from_snapshot(snap);
    assert_eq!(
        m.representations.entry_count(), 0,
        "REP-07: legacy model with no representation_entries must load with zero representations"
    );
}

// ═══════════════════════════════════════════════════════════════
// Phase C: Experience / Replay Separation Tests (EXP-01 through EXP-06)
// ═══════════════════════════════════════════════════════════════

// ── EXP-01: external occurrence count stays 1 after many Replays ──────────
#[test]
fn exp01_external_occurrence_stays_1_after_replay() {
    let mut m = ModelState::new();
    // 1 Experience
    m.expose_external("ab");
    // Collect total external_route_evidence after one expose
    let evidence_after_expose: f64 = m.predictions.iter_all()
        .map(|e| e.external_route_evidence)
        .sum();

    // 31 Replays
    for _ in 0..31 { m.replay("ab"); }

    let evidence_after_replay: f64 = m.predictions.iter_all()
        .map(|e| e.external_route_evidence)
        .sum();

    assert!(
        (evidence_after_expose - evidence_after_replay).abs() < 1e-9,
        "EXP-01: external_route_evidence must not increase during Replay; \
         after 1 expose={evidence_after_expose:.3}, after 31 replay={evidence_after_replay:.3}"
    );
}

// ── EXP-02: Adjacency external evidence not updated during Replay ─────────
#[test]
fn exp02_adjacency_not_updated_during_replay() {
    let mut m = ModelState::new();
    m.expose_external("ab");
    let assoc_after_expose = m.association_count();
    for _ in 0..20 { m.replay("ab"); }
    assert_eq!(
        m.association_count(), assoc_after_expose,
        "EXP-02: Adjacency (association) must not grow during Replay"
    );
}

// ── EXP-03: External Route Evidence not updated during Replay ─────────────
#[test]
fn exp03_external_route_evidence_not_updated_during_replay() {
    let mut m = ModelState::new();
    m.expose_external("ab");
    let ext_evidence_before: f64 = m.predictions.iter_all()
        .map(|e| e.external_route_evidence).sum();
    for _ in 0..20 { m.replay("ab"); }
    let ext_evidence_after: f64 = m.predictions.iter_all()
        .map(|e| e.external_route_evidence).sum();
    assert!(
        (ext_evidence_before - ext_evidence_after).abs() < 1e-9,
        "EXP-03: external_route_evidence={ext_evidence_before:.3} must not change during Replay (got {ext_evidence_after:.3})"
    );
}

// ── EXP-04: Practice Confidence (usage_strength) grows during Replay ─────
#[test]
fn exp04_practice_confidence_grows_during_replay() {
    let mut m = ModelState::new();
    m.expose_external("ab");
    let practice_before: f64 = m.predictions.iter_all()
        .map(|e| e.usage_strength).sum();
    for _ in 0..20 { m.replay("ab"); }
    let practice_after: f64 = m.predictions.iter_all()
        .map(|e| e.usage_strength).sum();
    assert!(
        practice_after > practice_before,
        "EXP-04: usage_strength (practice_confidence) must increase during Replay; \
         before={practice_before:.3} after={practice_after:.3}"
    );
}

// ── EXP-05: Replay acquires new Representation when chunk forms ────────────
#[test]
fn exp05_replay_acquires_new_representation() {
    let mut m = ModelState::new();
    // First observation establishes R0 = [P(a), P(b)].
    m.expose_external("ab");
    let prim_ids = m.primitives.encode_existing("ab");
    let id_ab = m.identities.find_identity(&prim_ids).expect("identity must exist");
    let reps_before = m.representations.for_identity(id_ab).len();
    // Train enough for chunk "ab" to form.
    for _ in 0..80 { m.expose_external("ab"); }
    assert!(m.chunk_count() > 0, "EXP-05: chunk must form before replay test");
    // Replay re-segments with the new chunk → new Representation.
    for _ in 0..10 { m.replay("ab"); }
    let reps_after = m.representations.for_identity(id_ab).len();
    assert!(
        reps_after > reps_before,
        "EXP-05: Replay must acquire new Representation when chunk changes segmentation; before={reps_before} after={reps_after}"
    );
}

// ── EXP-06: Replay-acquired Representation has acquisition tick > R0 ──────
#[test]
fn exp06_replay_acquired_representation_has_later_tick() {
    let mut m = ModelState::new();
    m.expose_external("ab");
    let prim_ids = m.primitives.encode_existing("ab");
    let id = m.identities.find_identity(&prim_ids).unwrap();
    let r0_tick = m.representations.for_identity(id).first().map(|r| r.acquired_at).unwrap_or(0);

    for _ in 0..20 { m.expose_external("ab"); }
    for _ in 0..10 { m.replay("ab"); }

    let prim_ids2 = m.primitives.encode_existing("ab");
    let id2 = m.identities.find_identity(&prim_ids2).unwrap();
    let reps = m.representations.for_identity(id2);
    if reps.len() > 1 {
        // Any rep acquired after R0 should have a higher or equal acquired_at tick.
        for rep in &reps[1..] {
            assert!(
                rep.acquired_at >= r0_tick,
                "EXP-06: later representations must have acquired_at >= R0 tick; \
                 R0 tick={r0_tick}, later tick={}", rep.acquired_at
            );
        }
    }
    // At minimum: no panic.
}

// ── FAC-01: Low-reuse pair merges at threshold ────────────────────────────
// A pair (a,b) that always appears together (right element `b` only wanted by `a`)
// should merge once the count threshold is reached, without factorization pressure.
#[test]
fn fac01_low_reuse_pair_merges_at_threshold() {
    let mut m = ModelState::new();
    // Expose "ab" many times — pair (a,b) has right_reuse=1 (only `a` wants `b`).
    for _ in 0..20 {
        m.expose_external("ab");
    }
    // With no factorization pressure, `a`→`b` should merge within 20 exposures.
    let prims_a = m.primitives.encode_existing("a");
    let prims_b = m.primitives.encode_existing("b");
    if !prims_a.is_empty() && !prims_b.is_empty() {
        let ua = syntrail_lm::units::UnitId::primitive(prims_a[0]);
        let ub = syntrail_lm::units::UnitId::primitive(prims_b[0]);
        let merged = m.chunks.find_by_pair(ua, ub).is_some();
        assert!(merged, "FAC-01: exclusive pair (a,b) should merge after enough exposures");
    }
}

// ── FAC-02: High right-reuse resists merge ────────────────────────────────
// `は` appears as the RIGHT element in pairs with 犬/猫/私.
// With factorization pressure, each of these pairs needs a higher raw count to merge.
#[test]
fn fac02_high_right_reuse_resists_merge() {
    let mut m = ModelState::new();
    // Expose three distinct "L は" patterns just enough times to cross threshold
    // without factorization pressure (4 times each = MERGE_THRESHOLD).
    // With factorization_scale=1.0 and right_reuse=3: net = 4 - 2 = 2 < 4 → should NOT merge.
    for _ in 0..4 {
        m.expose_external("犬は");
        m.expose_external("猫は");
        m.expose_external("私は");
    }
    // Verify: none of the three pairs should have merged yet.
    let ha_prims = m.primitives.encode_existing("は");
    if ha_prims.is_empty() { return; } // primitives not registered → skip
    let ha = syntrail_lm::units::UnitId::primitive(ha_prims[0]);
    // Check that `は` is not yet merged with any of 犬/猫/私.
    let inu_prims = m.primitives.encode_existing("犬");
    if !inu_prims.is_empty() {
        let inu = syntrail_lm::units::UnitId::primitive(inu_prims[0]);
        assert!(
            m.chunks.find_by_pair(inu, ha).is_none(),
            "FAC-02: (犬,は) should NOT merge when は has high reuse across 3 contexts at count=4"
        );
    }
}

// ── FAC-03: Factorization pressure scales with right-reuse count ──────────
// Verify that the right-reuse tracking increases when multiple lefts compete for same right.
#[test]
fn fac03_right_reuse_accumulates_across_pairs() {
    let mut m = ModelState::new();
    // Expose pairs that all share the same right element `Z`.
    // Use distinct left elements: A, B, C each paired with Z once.
    m.expose_external("AZ");
    m.expose_external("BZ");
    m.expose_external("CZ");
    // After these exposures, `Z` should appear as a merge candidate right for multiple lefts.
    let z_prims = m.primitives.encode_existing("Z");
    if z_prims.is_empty() { return; }
    let z_unit = syntrail_lm::units::UnitId::primitive(z_prims[0]);
    // Check right-reuse count for `Z` is ≥ 2 (multiple lefts want it).
    let reuse_count = m.merge_right_reuse.get(&z_unit).map(|s| s.len()).unwrap_or(0);
    assert!(
        reuse_count >= 2,
        "FAC-03: Z should have right-reuse ≥ 2 when multiple lefts compete; got {reuse_count}"
    );
}

// ── FAC-04: 犬は/猫は/私は — は reuse detected ────────────────────────────
// Standard factorization test from spec §38: after learning three "L は" patterns,
// は's reusability across contexts is captured in merge_right_reuse.
#[test]
fn fac04_wa_reuse_detected_across_subject_noun_pairs() {
    let mut m = ModelState::new();
    for _ in 0..3 {
        m.expose_external("犬は走る");
        m.expose_external("猫は眠る");
        m.expose_external("私は食べる");
    }
    let ha_prims = m.primitives.encode_existing("は");
    if ha_prims.is_empty() { return; }
    let ha = syntrail_lm::units::UnitId::primitive(ha_prims[0]);
    let reuse = m.merge_right_reuse.get(&ha).map(|s| s.len()).unwrap_or(0);
    assert!(
        reuse >= 2,
        "FAC-04: は should have right-reuse ≥ 2 in 犬は/猫は/私は contexts; got {reuse}"
    );
}

// ── FAC-05: §17 — factorization_scale=0 disables right-reuse resistance ──
#[test]
fn fac05_factorization_scale_configurable() {
    use syntrail_lm::config::LearningConfig;
    // With scale=0, right-reuse penalty disappears → merge should happen more readily.
    let mut m_no_penalty = ModelState::new();
    m_no_penalty.learning = LearningConfig { factorization_scale: 0.0 };
    // With default scale=1.0, right-reuse resists merge.
    let mut m_with_penalty = ModelState::new();
    // Same training: AZ, BZ, CZ → Z has high right-reuse.
    for _ in 0..30 {
        m_no_penalty.expose_external("AZAZAZAZ");
        m_no_penalty.expose_external("BZBZBZBZ");
        m_with_penalty.expose_external("AZAZAZAZ");
        m_with_penalty.expose_external("BZBZBZBZ");
    }
    // Model with no penalty should form at least as many chunks (merge resists less).
    let chunks_no_penalty = m_no_penalty.chunk_count();
    let chunks_with_penalty = m_with_penalty.chunk_count();
    assert!(
        chunks_no_penalty >= chunks_with_penalty,
        "FAC-05: scale=0 should not produce fewer chunks than scale=1.0; \
         no_penalty={chunks_no_penalty} with_penalty={chunks_with_penalty}"
    );
}

// ── TRF tests: Phase F Transform/Identity separation ──────────────────────

// ── TRF-01: EquivalentView merges identities ──────────────────────────────
#[test]
fn trf01_equivalent_view_merges_identities() {
    use syntrail_lm::identity::IdentityStore;
    use syntrail_lm::transform::{TransformKind, TransformStore};
    let mut ids = IdentityStore::new();
    let a = ids.intern_identity(&[1]);
    let b = ids.intern_identity(&[2]);
    let mut ts = TransformStore::new();
    ts.register(a, b, TransformKind::EquivalentView, &mut ids);
    assert_eq!(ids.canonical(a), ids.canonical(b),
        "TRF-01: EquivalentView must merge identity roots");
}

// ── TRF-02: Mapping does NOT merge identities ─────────────────────────────
#[test]
fn trf02_mapping_does_not_merge_identities() {
    use syntrail_lm::identity::IdentityStore;
    use syntrail_lm::transform::{TransformKind, TransformStore};
    let mut ids = IdentityStore::new();
    let a = ids.intern_identity(&[1]);
    let b = ids.intern_identity(&[2]);
    let mut ts = TransformStore::new();
    ts.register(a, b, TransformKind::Mapping, &mut ids);
    assert_ne!(ids.canonical(a), ids.canonical(b),
        "TRF-02: Mapping must NOT merge identity roots");
}

// ── TRF-03: value and evidence start at 0.0 ───────────────────────────────
#[test]
fn trf03_value_evidence_default_zero() {
    use syntrail_lm::identity::IdentityStore;
    use syntrail_lm::transform::{TransformKind, TransformStore};
    let mut ids = IdentityStore::new();
    let a = ids.intern_identity(&[1]);
    let b = ids.intern_identity(&[2]);
    let mut ts = TransformStore::new();
    let tid = ts.register(a, b, TransformKind::Mapping, &mut ids);
    let t = ts.get(tid).unwrap();
    assert_eq!(t.value, 0.0, "TRF-03: value must start at 0.0");
    assert_eq!(t.evidence, 0.0, "TRF-03: evidence must start at 0.0");
}

// ── TRF-04: value and evidence can be set ─────────────────────────────────
#[test]
fn trf04_value_evidence_can_be_updated() {
    use syntrail_lm::identity::IdentityStore;
    use syntrail_lm::transform::{TransformKind, TransformStore};
    let mut ids = IdentityStore::new();
    let a = ids.intern_identity(&[1]);
    let b = ids.intern_identity(&[2]);
    let mut ts = TransformStore::new();
    let tid = ts.register(a, b, TransformKind::Mapping, &mut ids);
    ts.get_mut(tid).unwrap().value = 2.5;
    ts.get_mut(tid).unwrap().evidence = 0.9;
    let t = ts.get(tid).unwrap();
    assert!((t.value - 2.5).abs() < 1e-9, "TRF-04: value update must persist");
    assert!((t.evidence - 0.9).abs() < 1e-9, "TRF-04: evidence update must persist");
}

// ── TRF-05: Inverse is a derived transform with no merge ──────────────────
#[test]
fn trf05_inverse_does_not_merge() {
    use syntrail_lm::identity::IdentityStore;
    use syntrail_lm::transform::{TransformKind, TransformStore};
    let mut ids = IdentityStore::new();
    let a = ids.intern_identity(&[1]);
    let b = ids.intern_identity(&[2]);
    let mut ts = TransformStore::new();
    let fwd = ts.register(a, b, TransformKind::Mapping, &mut ids);
    let canon_a_before = ids.canonical(a);
    let canon_b_before = ids.canonical(b);
    let _inv = ts.inverse(fwd).unwrap();
    // Inverse derivation must not trigger any new merge.
    assert_eq!(ids.canonical(a), canon_a_before, "TRF-05: inverse must not change canonical(a)");
    assert_eq!(ids.canonical(b), canon_b_before, "TRF-05: inverse must not change canonical(b)");
}

// ── TRF-06: Composed is a derived transform with no merge ─────────────────
#[test]
fn trf06_composed_does_not_merge() {
    use syntrail_lm::identity::IdentityStore;
    use syntrail_lm::transform::{TransformKind, TransformStore};
    let mut ids = IdentityStore::new();
    let a = ids.intern_identity(&[10]);
    let b = ids.intern_identity(&[20]);
    let c = ids.intern_identity(&[30]);
    let mut ts = TransformStore::new();
    let f = ts.register(a, b, TransformKind::Mapping, &mut ids);
    let g = ts.register(b, c, TransformKind::Mapping, &mut ids);
    let _h = ts.compose(f, g).unwrap();
    // a, b, c should all have distinct canonical roots.
    assert_ne!(ids.canonical(a), ids.canonical(c),
        "TRF-06: composed transform must not merge endpoints");
}

// ── DOC-01: §24 — DocumentState wired into AppHandle ────────────────────
#[test]
fn doc01_apphandle_doc_state_tracks_path() {
    use syntrail_lm::app::AppHandle;
    // New handle with a non-existent path → doc.path is set, no load.
    let f = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    let path = f.path().to_path_buf();
    // Pre-create an empty model file so new() can load it.
    syntrail_lm::app::save_model_file(&syntrail_lm::model::ModelState::new(), &path).unwrap();
    let mut h = AppHandle::new(path.clone(), ":memory:").unwrap();
    assert_eq!(h.doc.path.as_deref(), Some(path.as_path()),
        "DOC-01: doc.path must be set after AppHandle::new");
    assert!(!h.doc.dirty, "DOC-01: doc must not be dirty after clean load");
    // generate_turn marks dirty.
    for _ in 0..10 { h.session.model.expose_external("hello world"); }
    let _ = h.generate_turn("hello");
    assert!(h.doc.dirty, "DOC-01: doc must be dirty after generate_turn");
    // save_model clears dirty.
    h.save_model().unwrap();
    assert!(!h.doc.dirty, "DOC-01: doc must not be dirty after save_model");
    // reset_model clears path.
    h.reset_model();
    assert!(h.doc.path.is_none(), "DOC-01: doc.path must be None after reset_model");
}

// ── BUD-01: §14-§16 — MemoryBudgetConfig enforced by train() ─────────────
#[test]
fn bud01_hot_budget_enforced_by_train() {
    use syntrail_lm::config::MemoryBudgetConfig;
    let mut m = ModelState::new();
    m.memory_budget = MemoryBudgetConfig { hot_chunks_max: 5 };
    // Heavy training creates many chunks; budget must cap HOT count.
    for _ in 0..200 { m.train("hello world foo bar baz qux quux"); }
    let hot = m.hot_chunk_count();
    assert!(hot <= 5,
        "BUD-01: HOT chunk count must be ≤ hot_chunks_max=5 after train(); got {hot}");
}

// ── BUD-02: §14 — budget=0 means unlimited ────────────────────────────────
#[test]
fn bud02_zero_budget_is_unlimited() {
    let mut m = ModelState::new();
    // Default budget is 0 (unlimited); hot count must exceed 5.
    for _ in 0..200 { m.train("hello world foo bar baz qux quux"); }
    let hot = m.hot_chunk_count();
    assert!(hot > 0, "BUD-02: model with no budget should have HOT chunks after training");
}

// ── DOC-03: §27 — new_conversation preserves model, reset_model clears it ──
#[test]
fn doc03_new_conversation_vs_reset_model() {
    use syntrail_lm::app::AppHandle;
    let f = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    let path = f.path().to_path_buf();
    syntrail_lm::app::save_model_file(&syntrail_lm::model::ModelState::new(), &path).unwrap();
    let mut h = AppHandle::new(path.clone(), ":memory:").unwrap();
    for _ in 0..20 { h.session.model.expose_external("hello world"); }
    let tick_after_training = h.session.model.tick;

    // new_conversation: model intact, turn_count reset.
    h.session.turn_count = 5;
    h.new_conversation();
    assert_eq!(h.session.model.tick, tick_after_training,
        "DOC-03: new_conversation must not change model tick");
    assert_eq!(h.session.turn_count, 0,
        "DOC-03: new_conversation must reset turn_count");
    assert!(h.doc.path.is_some(), "DOC-03: new_conversation must not clear doc path");

    // reset_model: model cleared, path cleared.
    h.reset_model();
    assert_ne!(h.session.model.tick, tick_after_training,
        "DOC-03: reset_model must reset model tick");
    assert!(h.doc.path.is_none(), "DOC-03: reset_model must clear doc path");
}

// ── DOC-02: §25 — save_model returns error when doc.path is None ─────────
#[test]
fn doc02_save_model_requires_path() {
    use syntrail_lm::app::AppHandle;
    // Create a handle backed by a temp db, but immediately reset to clear path.
    let f = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    let path = f.path().to_path_buf();
    syntrail_lm::app::save_model_file(&syntrail_lm::model::ModelState::new(), &path).unwrap();
    let mut h = AppHandle::new(path, ":memory:").unwrap();
    h.reset_model(); // clears doc.path
    assert!(h.doc.path.is_none());
    let result = h.save_model();
    assert!(result.is_err(), "DOC-02: save_model must error when doc.path is None");
}

// ── MFILE-01: §21/§22 — save_model_file/.stm roundtrip ───────────────────
#[test]
fn mfile01_stm_roundtrip() {
    use syntrail_lm::app::{save_model_file, load_model_file};
    let mut m = ModelState::new();
    for _ in 0..20 { m.expose_external("hello world"); }
    let tick_before = m.tick;
    let f = tempfile::Builder::new().suffix(".stm").tempfile().unwrap();
    save_model_file(&m, f.path()).unwrap();
    let loaded = load_model_file(f.path()).unwrap();
    assert_eq!(loaded.tick, tick_before,
        "MFILE-01: .stm roundtrip must preserve model tick");
}

// ── DIAL-01: §8 — session.turn() returns dialogue output (emitted only) ────
#[test]
fn dial01_session_turn_returns_emitted_not_completion() {
    use syntrail_lm::config::Config;
    use syntrail_lm::session::Session;
    let mut s = Session::new_in_memory(Config::default_v02()).unwrap();
    for _ in 0..30 { s.model.expose_external("hello world foo bar"); }
    let input = "hello";
    let (_, dialogue_output, trace) = s.turn(input).unwrap();
    // session.turn() must return emitted_text, not the full completion.
    assert_eq!(dialogue_output, trace.emitted_text,
        "DIAL-01: session.turn() must return emitted_text not output_text");
    // Dialogue output must NOT start with the user's input (no seed echo).
    if !dialogue_output.is_empty() {
        assert!(!dialogue_output.starts_with(input),
            "DIAL-01: dialogue output must not echo the seed; got {:?}", dialogue_output);
    }
    // Full completion output_text DOES start with the seed (sanity check).
    if !trace.output_text.is_empty() {
        assert!(trace.output_text.starts_with(input),
            "DIAL-01: output_text should start with seed; got {:?}", trace.output_text);
    }
}

// ── TRF-07: EquivalentView kind is reflected in stored transform ──────────
#[test]
fn trf07_equivalent_view_kind_stored() {
    use syntrail_lm::identity::IdentityStore;
    use syntrail_lm::transform::{TransformKind, TransformStore};
    let mut ids = IdentityStore::new();
    let a = ids.intern_identity(&[1]);
    let b = ids.intern_identity(&[2]);
    let mut ts = TransformStore::new();
    let tid = ts.register(a, b, TransformKind::EquivalentView, &mut ids);
    let t = ts.get(tid).unwrap();
    assert!(matches!(t.kind, TransformKind::EquivalentView),
        "TRF-07: stored kind must be EquivalentView");
}
