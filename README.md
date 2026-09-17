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
| **Residency** | チャンクの活性状態。`Hot`（セグメント対象）または `Sleep`（構造レジストリのみ）。 |

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

### テスト（94 lib + 69 integration テスト）

```bash
cargo test
```

---

## CLI

### バッチ学習系（v0.1 から継続）

#### train — テキストファイルで学習してモデルを保存

```bash
syntrail train --input <テキストファイル> [--model <モデルパス>] [--hot-budget N]
```

- 既存モデルがあれば読み込んで追加学習する
- `--model` デフォルト: `model.json`
- `--hot-budget N`: 学習後に HOT チャンク数を N 以内に制限（弱いものを SLEEP に降格）。`0`（デフォルト）は無制限

#### generate — テキスト生成

```bash
syntrail generate --seed <開始テキスト> [--model <モデルパス>] [--max-units N]
```

#### inspect — モデル統計の表示

```bash
syntrail inspect [--model <モデルパス>]
```

出力例:
```
Primitives    : 87
Chunks        : 142 (HOT=130 SLEEP=12)
Pred. edges   : 1204
Assoc. edges  : 840
Chunk tiers   : T0=38 T1=71 T2=33
dpc           : 0.631482
```

#### evaluate — ファイル上での dpc 計測

```bash
syntrail evaluate --input <テキストファイル> [--model <モデルパス>]
```

---

### 会話系（v0.2 新規）

会話系コマンドは SQLite セッション DB（デフォルト: `syntrail.db`）を共有する。
`--db <パス>` でパスを変更できる。

#### chat — 1ターン会話して出力を表示

```bash
syntrail chat --input <テキスト> [--db <DBパス>]
```

- 生成（frozen）→ 入力テキストを expose → DB に記録、の順で実行
- stderr に `[turn_id=N decisions=M tick=K]` を出力

#### feedback — 過去のターンに ±1 フィードバックを付与

```bash
syntrail feedback --turn-id <N> --sign <+|-|1|-1> [--db <DBパス>]
```

- `--sign +` / `+1` / `1`: 正フィードバック（R = +1.0）
- `--sign -` / `-1`: 負フィードバック（R = −1.0）
- クレジットはターン内の決定数 N で均等分配（r_i = R / N）

#### snapshot — 現在のモデル状態を DB に保存

```bash
syntrail snapshot [--db <DBパス>] [--model <JSONパス>]
```

- `--model` を指定すると JSON ファイルにも同時保存

#### restore — スナップショットからモデル状態を復元

```bash
syntrail restore --snapshot-id <N> [--db <DBパス>] [--model <JSONパス>]
```

#### history — 会話履歴を表示

```bash
syntrail history [--limit N] [--db <DBパス>]
```

#### recall — 連想記憶から関連ユニットを検索（v0.3 新規）

```bash
syntrail recall --unit <テキスト> [--model <モデルパス>] [--limit N]
```

- `--unit` で指定したテキストの最初のユニットに対し、Top-K 関連ユニットを表示
- `--limit N`: 取得件数上限（デフォルト: 8）
- 直接の関連がない場合はチャンクを展開して階層的にフォールバック

---

## v0.4 主要変更（Residency: HOT/SLEEP）

### チャンクの HOT/SLEEP

| 状態 | 説明 |
|------|------|
| **Hot** | 通常の活性状態。`segment()` の対象。学習・生成に参加する。 |
| **Sleep** | 構造レジストリにのみ存在。`segment()` には現れないが、(left, right) ペアは保持される。 |

- `enforce_hot_budget(N)` で上位 N 個のチャンクだけを HOT に保てる
- SLEEP チャンクの (left, right) ペアが再出現したとき、同一 ID のまま HOT に自動復帰
- `inspect` 出力に `HOT=N SLEEP=M` を追加

---

## v0.3 主要変更（AssociationStore・Contextual Avoidance・Frozen Evaluation）

| 機能 | 説明 |
|------|------|
| **AssociationStore** | 共起 Top-K メモリ。`recall --unit` コマンドで参照。 |
| **Contextual Avoidance** | 負フィードバックで `avoidance` フィールドが蓄積。ルートスコア = `confidence + strength_norm + pos_value − avoidance`。 |
| **Frozen Evaluation** | `evaluate` コマンドはモデルを変更しない読み取り専用評価。 |

---

## v0.2 主要変更

| 概念 | v0.1 | v0.2 |
|------|------|------|
| 学習関数 | `train(text)` | `expose(text)`（`train` はエイリアス） |
| 生成関数 | `generate(&self)` — 元から immutable | `generate_with_trace(&self)` — TurnTrace を返す |
| 強度フィールド | `strength` | `usage_strength`（使用）+ `feedback_value`（評価） |
| カウントフィールド | `success_count / total_count` | `use_count`（使用回数のみ） |
| スコア confidence | `success / total` | `use_count > 0 → 1.0`（2値） |
| フィードバック | なし | ±1 符号付き、均一クレジット分配 |
| ログ | なし | SQLite（turns / trace_steps / feedback_events / snapshots） |

---

## モジュール構成

```
src/
├── lib.rs            クレートルート
├── main.rs           CLI（train/generate/inspect/evaluate/recall/chat/feedback/snapshot/restore/history）
├── config.rs         Config — 全チューナブル（hot_budget 含む）
├── trace.rs          TurnTrace / DecisionStep
├── feedback.rs       FeedbackSign / distribute_uniform / credit distribution
├── db.rs             SQLite Database wrapper (rusqlite)
├── session.rs        Session — ModelState + Database + Config
├── primitives.rs     PrimitiveRegistry — Unicode scalar ↔ u32 ID
├── units.rs          UnitId — タグ付き u32 (bit31: Primitive/Chunk)
├── tier.rs           Tier enum + 昇格/降格ロジック
├── chunks.rs         Chunk 構造体 + ChunkRegistry（Residency: Hot/Sleep）
├── prediction.rs     PredictionEdge（avoidance フィールド）+ PredictionStore
├── segmentation.rs   segment() / expand() — HOT チャンクのみ対象
├── association.rs    AssociationStore — 共起 Top-K メモリ（v0.3）
├── eval.rs           evaluate_frozen() — 読み取り専用評価（v0.3）
├── model.rs          ModelState — 統合学習/生成ループ（enforce_hot_budget 含む）
└── persistence.rs    JSON 永続化 v0.4 (save/load, v0.1–v0.3 後方互換)

tests/
└── integration.rs    AC-01〜AC-13（v0.1回帰）+ T-01〜T-19 + TL-01〜TL-06 + RS-01〜RS-08（69テスト）
```

---

## 評価指標

| 指標 | 意味 |
|------|------|
| `dpc` | 主指標。decisions / characters。< 1.0 が学習成功の目安。 |
| `primitive_count` | 語彙サイズ相当 |
| `chunk_count` | 形成されたパターン数（T0/T1/T2 内訳も確認） |
| `edge_count` | 予測エッジ数（生成多様性の目安） |
| `hot_chunk_count` / `sleep_chunk_count` | HOT/SLEEP チャンク数（v0.4 Residency） |
| `association_count` | 共起関連エッジ数（v0.3 AssociationStore） |

---

## スコープ外（MVP）

Transformer・GPU・JIT コンパイル・GUI・外部コーパス依存・マルチスレッド学習・
量子化・蒸留・アテンション機構・BPE/WordPiece との互換性

---

## ライセンス

Private / WIP
