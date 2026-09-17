# SynTrail-LM

動的チャンク学習・生成モデルの MVP 実装。  
Unicode スカラー値を最小単位として、頻出パターンを自動的により大きなチャンクに合体させながら
「1文字あたりの決定回数（dpc）」を下げていく自己組織化言語モデル。

---

## 中心仮説

> 頻繁に使われる処理シーケンスを大きなチャンクに合体させることで、
> 予測精度を保ちながら平均 dpc を 1.0 未満に下げられる。

---

## コア概念

| 用語 | 説明 |
|------|------|
| **Primitive** | Unicode スカラー値 1 個。不変・削除不可の最小単位。`u32` ID で管理。 |
| **Chunk** | 二分合成 `(left: UnitId, right: UnitId)`。再帰的・可変長の再利用可能処理単位。 |
| **Unit** | Primitive または Chunk の総称。`UnitId` は 32 bit タグ付き値（bit31=0: Primitive, 1: Chunk）。 |
| **Tier** | チャンクの検証状態。文字列長ではなくエビデンス量で決まる。T0 → T1 → T2 と昇格。 |
| **Strength** | 使用実績の減衰和。`new = 0.99 × old + reward`。忘却と強化を同時に表現。 |
| **dpc** | `total_decisions / total_characters`。主要評価指標。目標: 十分な学習後 < 1.0。 |

### Tier 遷移（暫定閾値）

| 遷移 | 条件 |
|------|------|
| T0 → T1 | success ≥ 16 かつ accuracy ≥ 90% |
| T1 → T2 | success ≥ 64 かつ accuracy ≥ 98% |
| T2 → T1 | accuracy < 90% |
| T1 → T0 | accuracy < 80% |

### スコア式（パス競合）

```
score = confidence + strength_norm - processing_cost
```

複数セグメンテーション候補が競合し、最高スコアの分割が採用される。

### 確率的マージ

```
merge_probability = base_probability × sqrt(frequency_factor)
```

同じ (left, right) ペアが `MERGE_THRESHOLD × 2` 回以上観測されると新規チャンクを作成。

---

## 学習・生成フロー（統一パス）

```
入力テキスト
  │
  ▼
PrimitiveRegistry.encode()  →  [PrimitiveId...]
  │
  ▼
segment()                   →  [UnitId...]   ← 既存チャンクをスコアで適用
  │
  ├─ record_success() on matched chunks
  ├─ predictions.learn_sequence()            ← 自己教師あり予測エッジ学習
  ├─ metrics 更新 (decisions / chars)
  └─ consider_merges()                       ← 新規チャンク候補の確率的作成

生成時: seed → segment → top1(context) を繰り返す
```

学習と生成は同一パス。専用の「学習モード」「生成モード」は存在しない。

---

## ビルドと実行

### ビルド

```bash
cargo build --release
```

### テスト（67 テスト・AC-01〜AC-13 含む）

```bash
cargo test
```

### CLI

#### train — テキストファイルで学習してモデルを保存

```bash
syntrail train --input <テキストファイル> [--model <モデルパス>]
```

- `--input`: 学習テキストファイル（UTF-8、改行区切り）
- `--model`: モデル保存先 JSON（デフォルト: `model.json`）
- 既存モデルがあれば読み込んで追加学習する

#### generate — テキスト生成

```bash
syntrail generate --seed <開始テキスト> [--max-units <N>] [--model <モデルパス>]
```

- `--seed`: 生成の起点テキスト
- `--max-units`: 最大生成 Unit 数（デフォルト: 50）

#### inspect — モデル統計の表示

```bash
syntrail inspect [--model <モデルパス>]
```

出力例:
```
Primitives    : 87
Chunks        : 142  (T0=38 T1=71 T2=33)
Pred. edges   : 1204
dpc           : 0.631482
```

#### evaluate — ファイル上での dpc 計測

```bash
syntrail evaluate --input <テキストファイル> [--model <モデルパス>]
```

---

## モジュール構成

```
src/
├── lib.rs            クレートルート
├── main.rs           CLI エントリポイント (train/generate/inspect/evaluate)
├── primitives.rs     PrimitiveRegistry — Unicode scalar ↔ u32 ID
├── units.rs          UnitId — タグ付き u32 (bit31: Primitive/Chunk)
├── tier.rs           Tier enum + 昇格/降格ロジック
├── chunks.rs         Chunk 構造体 + ChunkRegistry
├── prediction.rs     PredictionEdge + PredictionStore
├── segmentation.rs   segment() / expand() — スコアベースパス競合
├── model.rs          ModelState — 統合学習/生成ループ
└── persistence.rs    JSON 永続化 (save/load)

tests/
└── integration.rs    AC-01〜AC-13 受入テスト
```

---

## 評価指標

| 指標 | 意味 |
|------|------|
| `dpc` | 主指標。decisions / characters。< 1.0 が学習成功の目安。 |
| `primitive_count` | 語彙サイズ相当 |
| `chunk_count` | 形成されたパターン数（T0/T1/T2 内訳も確認） |
| `edge_count` | 予測エッジ数（生成多様性の目安） |

---

## スコープ外（MVP）

Transformer・GPU・JIT コンパイル・GUI・外部コーパス依存・マルチスレッド学習・
量子化・蒸留・アテンション機構・BPE/WordPiece との互換性

---

## ライセンス

Private / WIP
