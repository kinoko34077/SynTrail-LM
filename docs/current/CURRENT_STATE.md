# 現在の実装状態

最終確認: 2026-09-18  
main HEAD: 3536088 (Phase F完了)  
crate version: 0.5.0  
integration tests: 100

---

## コアモデル — 実装済

- Unicode Primitive (stable ID)
- u32 tagged UnitId
- Recursive binary Chunk (left/right)
- Merge candidate (count-based)
- Usage/Tier (T0/T1/T2)
- HOT/SLEEP 論理状態
- PredictionStore (Route)
- AssociationStore (Top-K, Adjacency相当)
- Feedback / Avoidance (Context-dependent)
- Frozen generation (with Trace, EOS, cycle detection) ← Phase D (P0)
- Frozen Eval (evaluate_frozen, evaluate_sample_frozen)
- History DB (SQLite)
- Snapshot (JSON / SQLite)
- **Representation Lineage** (RepresentationStore, preference_score, predecessor_of) ← Phase B
- **Experience / Replay 分離** (expose_external vs replay; external_route_evidence分離) ← Phase C
- **Factorization Pressure** (merge_right_reuse; net_gain = count - FACTORIZATION_SCALE*(reuse-1)) ← Phase E (§27修正)
- **TransformKind** (EquivalentView/Mapping/Inverse/Composed; conditional merge_identities) ← Phase F (§17修正)
- **Route Source Index** (source_index in PredictionStore, O(N)→O(K)) ← Phase 5
- **Segmentation O(L log L)** (priority queue + doubly-linked slots) ← Phase 6
- **Unified Relation Core** (RelationStore: Route/Adjacency/DerivedFrom) ← Phase 7
- **HOT/SLEEP 物理分離** (hot_pair_to_id Recognition index) ← Phase 8
- **Lazy Decay** (s = s * λ^Δt + reward) ← Phase 9
- **Cross-View Identity** (Union-Find merge_identities, canonical, persistence) ← Phase 11
- **Transform / Inverse / Composition** (TransformStore) ← Phase 12
- **Micro-ISA** (14 opcodes, Vm, register file) ← Phase 13
- **Physical Core / Overlay 分離** (CoreView) ← Phase 14
- **Flat Arrays / Arena Index** (Vec<PredictionEdge> + edge_index + source_index<usize>) ← Phase 15
- **Binary Persistence** (save_binary/load_binary; save_auto/load_auto) ← Phase 16
- **Fixed Point / Packing** (f32 in DTOs) ← Phase 17
- **Variable-bit ID / Region Encoding** (LEB128 + zigzag delta codec) ← Phase 18
- **Generalization / Novel Search** (generalize() + novel_candidates()) ← Phase 20
- **Chat GUI** (eframe/egui)
- **Adaptive Trainer** (S/M/L/XL block, 4→8→16→32 repeat, Pause/Resume)

---

## コアモデル — 未実装 / 残課題

| 優先 | 内容 |
|------|------|
| P2 | Phase G: RelationStore canonical化 (方針A/B選択, §18) |
| P3 | Phase H: MemoryBudgetConfig, Global Active Budget (§19-§20) |
| P4 | Phase I: Core/Overlay physical split final |
| P4 | Phase J: Generalization real unseen tests (GEN-01..04) |
| P4 | Phase A残: docs/spec §64 8ファイル構成 + §65 template |
| P5 | save_auto/load_auto が通常save経路に未統合 |
| P5 | codec.rs が binary snapshot に未接続 |

---

## Desktop UI / GUI — 既知の問題（修正必須）

### Trainer (Critical)

| 優先 | 箇所 | 問題 |
|------|------|------|
| P0 | `worker_main` / `save_and_exit` | `let _ = save_model_file(...)` / `let _ = tr_state.save(...)` でsave errorを握り潰し、その後 `Saved` イベントを送信 → **データ破損リスク** (§110) |
| P0 | `save_and_exit()` | `tr_state.model_fingerprint` を更新せずにsave → Stopして再Resume時にfingerprint不一致 (§112) |
| P0 | `update()` D&D処理 | Running中でも `try_load_model()`/`try_load_dataset()` が呼ばれ、最終的に `check_ready()` → state が Ready に上書きされる (§108) |
| P1 | Resume ボタン | `is_paused` 状態でResume押下が `start_training(true)` (=Resume Saved Session) を呼ぶ。Paused状態の既存workerへ `TrainerCommand::Resume` を送る経路がない (§107) |
| P1 | Path TextEdit | model_path / dataset_path がTextEdit可能だが文字変更だけでは loaded_model と一致しない (§116) |

### Chat GUI (Important)

| 優先 | 箇所 | 問題 |
|------|------|------|
| P1 | `send_message()` | `self.input.trim().to_string()` → モデルに渡す文字列の先頭末尾空白を削除。空白もPrimitive学習対象なため分離が必要 (§100) |
| P1 | `generate_turn()` / Analytics | `last_output_len = output.len()` はUTF-8バイト数。表示名「Last Out Len」は文字数と誤解されやすい (§101) |
| P2 | Analytics | N turnごと更新のため、画面Turn数とAnalytics表示が一時的に不一致。「last updated at turn N」等が必要 (§102) |
| P2 | New操作 | history.sqliteの過去記録は残るがchat_history.clear()される。「New Model」と「New Conversation」の意味が不明確 (§103) |

### 共通 (Architecture)

| 優先 | 内容 |
|------|------|
| P1 | `setup_fonts()` がgui/trainerで重複 (§122) |
| P1 | File Dialog定義 (rfd::FileDialog) がgui/trainerで重複 (§98) |
| P1 | D&D処理がgui/trainerで独立実装 (§99) |
| P1 | `desktop/` 共通層未作成 (§78) |
| P2 | Windows Native Menu未実装 (§88-94) |
| P2 | Keyboard shortcuts (Ctrl+N/O/S/Shift+S) 未実装 (§92) |

---

## ロードマップ

### コアモデル Phases

| Phase | 内容 | 状態 |
|-------|------|------|
| D (P0) | Generation cycle detection, EOS, seed/output分離 | ✅ 完了 |
| B | Representation Lineage (RepresentationStore) | ✅ 完了 |
| C | Experience/Replay 分離 (external_route_evidence) | ✅ 完了 |
| E | Factorization Pressure 修正 (merge_right_reuse) | ✅ 完了 |
| F | TransformKind + conditional merge (§17修正) | ✅ 完了 |
| G | RelationStore canonical化 | 未着手 |
| H | Memory Budget | 未着手 |
| I | Core/Overlay physical split | 未着手 |
| J | Generalization real tests | 未着手 |

### Desktop UI Phases (§136)

| Phase | 内容 | 状態 |
|-------|------|------|
| UI-1 | README/CURRENT_STATE/spec現状整理 | ✅ 完了 |
| UI-2 | FileKind / FileDialogSpec / ModelFileService 共通層 | 未着手 |
| UI-3 | D&D共通化 (DropRouter) | 未着手 |
| UI-4 | Trainer state machine修正 (Save error, fingerprint, Running guard, Resume) | 未着手 |
| UI-5 | Chat GUI修正 (raw input, last_output_len) | 未着手 |
| UI-6 | Windows Native Menu Spike | 未着手 |
| UI-7 | Native Menu本実装 | 未着手 |
| UI-8 | Trainer menu/file統合 | 未着手 |
| UI-9 | README最終更新 + docs/spec更新 | 未着手 |
