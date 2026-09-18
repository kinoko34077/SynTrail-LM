# SynTrail-LM 仕様書インデックス

最終更新: 2026-09-18

## 構成

| ファイル | 内容 |
|---------|------|
| [01_SEMANTIC_MODEL.md](01_SEMANTIC_MODEL.md) | 最上位目的 / Content / Primitive / Chunk / Identity / View / Representation / Transform / Relation |
| [02_MEMORY_MODEL.md](02_MEMORY_MODEL.md) | Adjacency / Recall / Recognition / Route / Lineage / Fallback / Shortcut |
| [03_LEARNING_MODEL.md](03_LEARNING_MODEL.md) | Merge/Factorization / Context Diversity / Experience/Replay / Feedback / Turn Learning |
| [04_EXECUTION_MODEL.md](04_EXECUTION_MODEL.md) | Memory Architecture / HOT/SLEEP / 疎計算原則 / Relation Budget / Lazy Decay / Consolidation |
| [05_PHYSICAL_MODEL.md](05_PHYSICAL_MODEL.md) | Stable ID / Arena / Variable-width ID / Flat Array / Packed Binary / Fixed Point / Micro-ISA |
| [06_EVALUATION.md](06_EVALUATION.md) | 評価指標一覧 |
| [09_DESKTOP_UI.md](09_DESKTOP_UI.md) | Desktop Shell / File Operations / Native Menu / Trainer State Machine |

## 関連ドキュメント

| ファイル | 内容 |
|---------|------|
| [../current/CURRENT_STATE.md](../current/CURRENT_STATE.md) | 現在の実装状態・未実装一覧・ロードマップ |
| [../adr/](../adr/) | Architecture Decision Records |

## 状態区分

| 状態 | 意味 |
|------|------|
| 確定・実装済 | 仕様として採用済み、現行コードに存在 |
| 確定・未実装 | 仕様として採用済み、コード未導入 |
| 調整対象 | 構造は採用、数値・係数は実験で変更可能 |
| 候補 | 有力だが方式未固定 |
| 廃止・置換 | 旧案。現行仕様として使用しない |
