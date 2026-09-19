# SynTrail-LM — Current Implementation State

Last updated: 2026-09-20 (post integrated-fix-spec, baseline d0ae892)

---

## Storage / Persistence

| Feature | Status |
|---------|--------|
| JSON persistence (model.json) | Implemented |
| STM binary container (Zstd, varint, packed UnitId) | Implemented — v4 |
| Streaming STM save/load (no large intermediate buffer) | Implemented |
| Lossless delta fields (u64 tick-delta) | Implemented |
| SQLite DB snapshot (blob_data bincode preferred) | Implemented |
| DB load unified API (blob > json fallback) | Implemented |
| Transactional save (temp write + rename) | Implemented |
| STM payload_len explicit overflow check | Implemented |
| UnitId packing SSOT (codec.rs) | Implemented |
| Lossy quantization (f64 → f32 etc.) | Not Implemented (P6 deferred) |
| Journal / incremental append | Not Implemented (deferred) |

---

## Model Core

| Feature | Status |
|---------|--------|
| Chunk merge (merge_right_reuse) | Implemented |
| HOT/SLEEP logical split + index | Implemented |
| HOT budget enforcement (enforce_hot_budget) | Implemented |
| Physical Core/Overlay storage split | Not Implemented (Partial — CoreView API exists) |
| Experience / Replay separation | Implemented |
| Representation Lineage | Implemented |
| Factorization pressure | Implemented |
| TransformKind (EquivalentView/Mapping/Inverse/Composed) | Implemented |
| Identity Union-Find merge | Implemented |
| Cross-view identity evidence learning | Not Implemented |
| Relation canonical core | Not Implemented |
| Stable IDs | Implemented |

---

## Generation

| Feature | Status |
|---------|--------|
| Frozen generation | Implemented |
| Dialogue prompt conditioning (1-step) | Implemented |
| External evidence route scoring | Implemented |
| Surface repetition detector | Implemented |
| Feedback ranking (positive/negative) | Implemented |
| Route provenance (cycle fallback) | Implemented |
| TURN_BOUNDARY / SEQUENCE_END markers | Implemented |

---

## Trainer

| Feature | Status |
|---------|--------|
| Adaptive 4/8/16/32 scheduler | Implemented |
| Block splitter with deterministic jitter | Implemented |
| Start position UI (line / %) | Implemented |
| HOT budget enforcement per block | Implemented |
| Pause / Resume / Resume Saved | Implemented |
| Save failure → ErrorPaused state | Implemented |
| Transactional trainer state save | Implemented |
| DocumentState / dirty tracking | Implemented |
| Model + Dataset combo D&D | Implemented |
| Supervised Dataset (.jsonl) | Implemented |
| Supervised training loop (§34) | Implemented |

---

## Codec

| Feature | Status |
|---------|--------|
| LEB128 variable-length encoding | Implemented |
| Delta encoding for UnitId sequences | Implemented |
| Packed UnitId (SSOT in codec.rs) | Implemented |
| Integration with persistence bincode serializer | Implemented |
