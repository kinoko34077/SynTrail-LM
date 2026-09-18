# 現在の実装状態

最終確認: 2026-09-18  
main HEAD: Phase 2完了後更新予定  
crate version: 0.5.0

---

## 実装済

- Unicode Primitive (stable ID)
- u32 tagged UnitId
- Recursive binary Chunk (left/right)
- Merge candidate (count-based)
- Usage/Tier (T0/T1/T2)
- HOT/SLEEP 論理状態
- PredictionStore (Route)
- AssociationStore (Top-K, Adjacency相当)
- Feedback / Avoidance (Context-dependent)
- Frozen generation (with Trace)
- Frozen Eval (evaluate_frozen, evaluate_sample_frozen)
- History DB (SQLite)
- Snapshot (JSON / SQLite)
- Chat GUI (eframe/egui)
- Adaptive Trainer (UTF-8/Shift_JIS, S/M/L/XL block, 4→8→16→32 repeat, Pause/Resume)
- **Experience / Replay 分離** (expose_external vs replay, encode_existing) ← Phase 1
- **Identity / View 最小実装** (IdentityStore, Exact Identity, persistence) ← Phase 2
- **Representation Lineage** (LineageStore, derived_from edges) ← Phase 3
- **Fallback via Lineage** (predict_with_fallback, generate_with_trace) ← Phase 4
- **Route Source Index** (source_index in PredictionStore, O(N)→O(K)) ← Phase 5
- **Segmentation O(L log L)** (priority queue + doubly-linked slots) ← Phase 6
- **Unified Relation Core** (RelationStore: Route/Adjacency/DerivedFrom) ← Phase 7
- **HOT/SLEEP 物理分離** (hot_pair_to_id Recognition index) ← Phase 8
- **Lazy Decay** (s = s * λ^Δt + reward for Chunk and PredictionEdge) ← Phase 9
- **Factorization Pressure** (Context Diversity bonus in consider_merges) ← Phase 10
- **Cross-View Identity** (Union-Find merge_identities, canonical resolution, persistence) ← Phase 11
- **Transform / Inverse / Composition** (TransformStore, register→merge, inverse, compose) ← Phase 12
- **Micro-ISA** (Instruction enum, Vm dispatcher, register file — all 14 opcodes) ← Phase 13
- **Physical Core / Overlay 分離** (CoreView: segment_core, predict_core, is_core_unit; ModelState::core_view) ← Phase 14
- **Flat Arrays / Arena Index** (PredictionStore: Vec<PredictionEdge> + edge_index + source_index<usize>) ← Phase 15
- **Binary Persistence** (save_binary/load_binary via bincode; save_auto/load_auto dispatcher) ← Phase 16
- **Fixed Point / Packing** (DTO strength fields: f64 → f32 in ChunkDto/PredictionEdgeDto; runtime stays f64) ← Phase 17
- **Variable-bit ID / Region Encoding** (LEB128 + zigzag delta codec for UnitId sequences) ← Phase 18
- **SIMD / Assembly** — profile後のみ実装のためスキップ ← Phase 19
- **Generalization / Novel Search** (generalize() + novel_candidates() via association bridge) ← Phase 20

---

## 未実装

- Representation Lineage
- Fallback via Lineage
- Cross-view Identity
- Unified Relation Core
- Source-indexed RouteBank
- Factorization Pressure (Context Diversity考慮)
- Lazy Decay
- Physical Core/Overlay split
- Recall/Recognition 分離
- Micro-ISA
- Arena local IDs
- Variable-width IDs
- Flat packed arrays
- Binary journal/snapshot
- Fixed point arithmetic

---

## 現行の主要パフォーマンス課題

### Route (PredictionStore)

`predict()` は対象Contextごとに全Prediction Edgeを走査。N Edgeで `O(N)`。
→ **Route source-index化が高優先 (Phase 5)**

### Segmentation

繰返し全候補走査 → 1 merge → 再度全候補走査。長文でO(L²)寄り。
→ **Phase 6で priority queue + 局所再計算へ**

### AssociationStore

既に `source → Top-K` のため比較的良い。

---

## ロードマップ

### A. 意味を正す (Phase 0〜4)

| Phase | 内容 | 完了条件 |
|-------|------|---------|
| 0 | 仕様書正本化・Baseline計測 | 変更を数字で比較できる |
| ~~1~~ | ~~Experience / Replay 分離~~ | ✅ 完了 |
| ~~2~~ | ~~Identity / View 最小実装 (Exact Identity)~~ | ✅ 完了 |
| ~~3~~ | ~~Representation Lineage~~ | ✅ 完了 |
| ~~4~~ | ~~Fallback via Lineage~~ | ✅ 完了 |

### B. 現行ボトルネックを取る (Phase 5〜10)

| Phase | 内容 |
|-------|------|
| ~~5~~ | ~~Route Source Index化~~ | ✅ 完了 |
| ~~6~~ | ~~Segmentation 局所化 (O(L log L))~~ | ✅ 完了 |
| ~~7~~ | ~~Unified Relation Core~~ | ✅ 完了 |
| ~~8~~ | ~~Memory 階層 (HOT/SLEEP物理分離・Recognition/Recall分離)~~ | ✅ 完了 |
| ~~9~~ | ~~Lazy Decay / Dirty Consolidation~~ | ✅ 完了 |
| ~~10~~ | ~~Factorization Pressure~~ | ✅ 完了 |

### C. 新しいIdentity/Relation理論を完成させる (Phase 11〜13)

| Phase | 内容 |
|-------|------|
| ~~11~~ | ~~Cross-View Identity~~ | ✅ 完了 |
| ~~12~~ | ~~Transform / Inverse / Composition~~ | ✅ 完了 |
| ~~13~~ | ~~Micro-ISA 定義~~ | ✅ 完了 |

### D. 機械レベルへ落とす (Phase 14〜20)

| Phase | 内容 |
|-------|------|
| ~~14~~ | ~~Physical Core / Overlay 分離~~ | ✅ 完了 |
| ~~15~~ | ~~Flat Arrays / Arena Index~~ | ✅ 完了 |
| ~~16~~ | ~~Binary Persistence~~ | ✅ 完了 |
| ~~17~~ | ~~Fixed Point / Packing~~ | ✅ 完了 |
| ~~18~~ | ~~Variable-bit ID / Region Encoding~~ | ✅ 完了 |
| 19 | SIMD / Assembly (profile後のみ) | ⏭️ スキップ (profile要) |
| ~~20~~ | ~~Generalization / Novel Search~~ | ✅ 完了 |
