# SynTrail-LM

動的チャンク学習・生成モデル。  
Unicode スカラー値を最小単位として、頻出パターンを自動的により大きなチャンクに合体させながら「1文字あたりの決定回数（dpc）」を下げていく自己組織化言語モデル。

---

## ビルド

```bash
# CLI のみ（依存が軽い）
cargo build --release

# GUI 込み（eframe/egui が追加される）
cargo build --release --features gui
```

成果物:

| バイナリ | パス |
|---------|------|
| CLI | `target/release/syntrail` |
| GUI | `target/release/syntrail-gui`（`--features gui` 時のみ） |

---

## ファイル構成（実行時）

| ファイル | 内容 | 備考 |
|---------|------|------|
| `model.json` | モデル状態（チャンク・予測エッジ・連想エッジ） | CLI `--model`、GUI の Model 欄で変更可 |
| `history.sqlite` | GUI の会話履歴 | モデルとは独立。削除してもモデルは変わらない |
| `syntrail.db` | CLI chat コマンドのセッション DB | CLI 専用 |

---

## GUI の使い方

### 起動

```bash
cargo run --release --features gui --bin syntrail-gui
```

または、ビルド済みバイナリなら:

```bash
./target/release/syntrail-gui
```

カレントディレクトリに `model.json` があれば自動で読み込む。なければ空のモデルで起動する。

### 画面構成

```
┌─ New / Load / Save ─ Model: model.json ─ Gen: 0  HOT: 0  dpc: 0.0000 ─┐
├────────────────────────────────────┬───────────────────────────────────┤
│                                    │ Analytics                         │
│  チャット履歴（スクロール）          │  Generation  0                    │
│                                    │  Primitive   0                    │
│  User:  入力テキスト                │  Chunk       0                    │
│  Model: 生成テキスト                │  HOT / SLEEP 0 / 0                │
│                                    │  …                                │
│                                    │  [Refresh / 5 turns] [Update Now] │
├──────────── ステータス ── [skip][○][×] ────────────────────────────────┤
│  [入力テキスト（複数行）]                                    [Send]      │
└────────────────────────────────────────────────────────────────────────┘
```

### 操作手順

**会話する**

1. 下の入力欄にテキストを入力する
2. `Send` ボタンを押す（または `Enter` キー）
3. `Shift+Enter` で入力欄内の改行

**フィードバックを付ける**

- 返答後にステータス行の右側に `○` `×` `skip` が出る
- `○` = ポジティブ（+1）
- `×` = ネガティブ（−1）
- `skip` または何も押さず次の入力 = フィードバックなしで次ターンへ

フィードバックは最新ターンの予測エッジと強度値に反映される。

**モデルの操作**

| ボタン | 動作 |
|--------|------|
| `New` | 空のモデルを新規作成（未保存状態） |
| `Load` | `Model:` 欄のパスから JSON を読み込む |
| `Save` | `Model:` 欄のパスへ JSON として保存する |

`Model:` 欄のパスは直接入力で変更できる。

**Analytics パネル**

右パネルに現在のモデル状態を数値一覧で表示する。

- `Refresh / N turns` のスライダーで N ターンごとに自動更新（デフォルト 5）
- `Update Now` で即時更新

**履歴の永続化**

- 会話履歴は `history.sqlite` に自動保存される
- アプリを再起動すると直近 60 ターン分が画面に復元される

---

## CLI の使い方

### train — テキストファイルで学習

```bash
syntrail train --input <ファイル> [--model <パス>] [--hot-budget N]
```

- 既存モデルがあれば追加学習する。なければ新規作成。
- `--model` デフォルト: `model.json`
- `--hot-budget N`: 学習後に HOT チャンクを N 個以内に絞る（弱いものを SLEEP に降格）。`0`（デフォルト）は無制限。

```bash
syntrail train --input corpus.txt
syntrail train --input corpus.txt --hot-budget 500
```

学習中は 1000 行ごとに進捗を stderr へ出力:

```
[1000/5000] primitives=87 chunks=142(HOT=142 SLEEP=0) edges=1204 dpc=0.8312
```

### generate — テキスト生成

```bash
syntrail generate --seed <テキスト> [--model <パス>] [--max-units N]
```

- `--seed` を起点にして予測エッジを辿って生成する
- `--max-units` デフォルト: 50

```bash
syntrail generate --seed "hello"
```

### inspect — モデル統計の表示

```bash
syntrail inspect [--model <パス>]
```

出力例:

```
Model file    : model.json
Primitives    : 87
Chunks        : 142 (HOT=130 SLEEP=12)
Pred. edges   : 1204
Assoc. edges  : 840
Tick          : 204
Total chars   : 58320
Total decisions: 42110
dpc           : 0.721824
Chunk tiers   : T0=38 T1=71 T2=33
```

### evaluate — dpc 計測（モデル変更なし）

```bash
syntrail evaluate --input <ファイル> [--model <パス>]
```

モデルを一切変更しない読み取り専用評価（Frozen Evaluation）。

```
Lines evaluated    : 500
Characters         : 24000
Decisions          : 18120
dpc                : 0.755000
Pred. accuracy     : 0.6312
Correct / total    : 11432/18120
```

### recall — 連想記憶の検索

```bash
syntrail recall --unit <テキスト> [--model <パス>] [--limit N]
```

指定テキストの最初のユニットに対して、AssociationStore から Top-K 関連ユニットを表示する。直接の関連がない場合はチャンクを展開して階層的にフォールバックする。`--limit` デフォルト: 8。

```bash
syntrail recall --unit "hello"
```

出力例:

```
Associations for "hello" (top 8):
  " world"  strength=3.8120
  "!"       strength=1.2041
  ...
```

### chat — 1 ターン会話

```bash
syntrail chat --input <テキスト> [--db <DBパス>] [--model <パス>]
```

生成（frozen） → 入力を expose → DB に記録、の順で実行。`--db` デフォルト: `syntrail.db`。

```bash
syntrail chat --input "hello"
```

stderr に `[turn_id=N decisions=M tick=K]` を出力する。

### feedback — 過去のターンにフィードバックを付与

```bash
syntrail feedback --turn-id <N> --sign <+|-|1|-1> [--db <DBパス>]
```

- `--sign +` / `1` / `+1`: ポジティブ（R = +1.0）
- `--sign -` / `-1`: ネガティブ（R = −1.0）

```bash
syntrail feedback --turn-id 3 --sign +
syntrail feedback --turn-id 5 --sign -1
```

### snapshot — モデル状態を DB に保存

```bash
syntrail snapshot [--db <DBパス>] [--model <JSONパス>]
```

`--model` を指定すると JSON ファイルにも同時保存する。

### restore — スナップショットから復元

```bash
syntrail restore --snapshot-id <N> [--db <DBパス>] [--model <JSONパス>]
```

### history — 会話履歴の表示

```bash
syntrail history [--limit N] [--db <DBパス>]
```

デフォルトで直近 20 ターンを表示する。

---

## Analytics 項目の説明

| 項目 | 説明 |
|------|------|
| Generation | 現セッションのターン数 |
| Tick | expose() 呼び出し累計（学習量の目安） |
| Primitive | 登録済みの Unicode スカラー数（語彙サイズ相当） |
| Chunk | 形成されたチャンク総数 |
| HOT | セグメント対象のアクティブチャンク数 |
| SLEEP | 構造レジストリのみに存在する休止チャンク数 |
| Association | 共起関連エッジ総数（AssociationStore） |
| Route (edges) | 予測エッジ総数（生成の多様性の目安） |
| T0 / T1 / T2 | チャンクのティア分布 |
| Avg Exp Len | チャンクの平均展開長（1 チャンク = 何文字相当か） |
| Total Chars | 累計入力文字数 |
| Total Decisions | 累計決定回数 |
| dpc | `decisions / characters`。学習が進むと下がる。目標: < 1.0 |
| Last Decisions | 直前ターンの決定回数 |
| Last Out Len | 直前ターンの出力文字数 |
| Positive FB | セッション内のポジティブフィードバック累計 |
| Negative FB | セッション内のネガティブフィードバック累計 |

---

## コア概念

| 用語 | 説明 |
|------|------|
| **Primitive** | Unicode スカラー値 1 個。削除不可の最小単位。 |
| **Chunk** | 二分合成 `(left, right)`。再帰的・可変長の再利用可能単位。 |
| **HOT** | セグメント対象のアクティブ状態。`--hot-budget` で上限を設定できる。 |
| **SLEEP** | セグメント対象外の休止状態。同じ `(left, right)` ペアが再出現すると自動復帰。 |
| **dpc** | `decisions / characters`。主要評価指標。 |
| **AssociationStore** | 共起 Top-K メモリ。`recall` コマンドで参照できる。 |
| **Frozen Evaluation** | `evaluate` コマンドはモデルを変更しない読み取り専用評価。 |

---

## モジュール構成

```
src/
├── lib.rs            クレートルート
├── main.rs           CLI
├── app.rs            Application API（CLI/GUI 共通）
├── bin/
│   └── syntrail_gui.rs  GUI バイナリ（--features gui）
├── config.rs         Config（hot_budget 含む全チューナブル）
├── trace.rs          TurnTrace / DecisionStep
├── feedback.rs       FeedbackSign / credit distribution
├── db.rs             SQLite Database
├── session.rs        Session（ModelState + Database + Config）
├── primitives.rs     PrimitiveRegistry
├── units.rs          UnitId（タグ付き u32）
├── tier.rs           Tier（T0/T1/T2）
├── chunks.rs         Chunk + ChunkRegistry（Residency: Hot/Sleep）
├── prediction.rs     PredictionEdge（avoidance）+ PredictionStore
├── segmentation.rs   segment() / expand()（HOT チャンクのみ）
├── association.rs    AssociationStore（共起 Top-K）
├── eval.rs           evaluate_frozen()
├── model.rs          ModelState（enforce_hot_budget 含む）
└── persistence.rs    JSON 永続化 v0.4（v0.1〜v0.3 後方互換）

tests/
└── integration.rs    69 テスト（AC / T / TL / RS 系列）
```

---

## テスト

```bash
cargo test
```

94 lib + 69 integration、警告ゼロ。

---

## スコープ外（MVP）

Transformer・GPU・グラフ可視化・複数会話管理・テーマ編集・モデル内部値の手動編集

---

## ライセンス

Private / WIP
