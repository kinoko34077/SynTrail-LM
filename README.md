# SynTrail-LM

動的チャンク学習・生成モデル。  
Unicode スカラー値を最小単位として、頻出パターンを自動的により大きなチャンクに合体させながら「1文字あたりの決定回数（dpc）」を下げていく自己組織化言語モデル。

---

## ビルド

```bash
# CLI のみ（依存が軽い）
cargo build --release

# GUI 込み（eframe/egui + muda が追加される）
cargo build --release --features gui
```

成果物:

| バイナリ | パス |
|---------|------|
| CLI | `target/release/syntrail` |
| GUI（Chat） | `target/release/syntrail-gui`（`--features gui` 時のみ） |
| Trainer | `target/release/syntrail-trainer`（`--features gui` 時のみ） |

---

## ファイル構成（実行時）

| ファイル | 内容 | 備考 |
|---------|------|------|
| `model.json` | モデル状態（チャンク・予測エッジ・連想エッジ） | CLI `--model`、GUI で変更可 |
| `history.sqlite` | GUI の会話履歴 | モデルとは独立。削除してもモデルは変わらない |
| `syntrail.db` | CLI chat コマンドのセッション DB | CLI 専用 |
| `<dataset>.syntrail-trainer.json` | Trainer の学習進捗 | Pause/Resume に使用 |

---

## GUI の使い方

### 起動

```bash
cargo run --release --features gui --bin syntrail-gui
```

カレントディレクトリに `model.json` があれば自動で読み込む。なければ空のモデルで起動する。

### Windows Native Menu / キーボードショートカット

| 操作 | 内容 |
|------|------|
| `Ctrl+N` | 空のモデルを新規作成 |
| `Ctrl+O` | ファイルを開く |
| `Ctrl+S` | 現在のパスへ保存 |
| `Ctrl+Shift+S` | 名前を付けて保存 |

メニューバーの `File` からも同じ操作が可能。ツールバーの New / Open… / Save / Save As… ボタンとすべて同一コマンド経路。

### 画面構成

```
┌─ File ─────────────────────────────────────────────────────────┐  ← Native Menu
├─ New / Open… / Save / Save As… ─ model.json ─ Gen/Chunk/dpc ─┤  ← Toolbar
├──────────────────────────────┬────────────────────────────────┤
│                              │ Analytics                      │
│  チャット履歴（スクロール）    │  Generation  0                 │
│                              │  Chunk       0 (HOT:0)        │
│  User:  入力テキスト          │  dpc         0.000000         │
│  Model: 生成テキスト          │  Last Out Chars  0            │
│                              │  …                            │
│                              │  [Refresh / 5] [Update Now]   │
├────── ステータス ── [○][×] ──────────────────────────────────┤
│  [入力テキスト（複数行）]                            [Send]    │
└──────────────────────────────────────────────────────────────┘
```

### 操作手順

**会話する**

1. 下の入力欄にテキストを入力する
2. `Send` ボタンまたは `Enter` キー
3. `Shift+Enter` で入力欄内の改行

**フィードバック**

- 返答後にステータス行の右側に `○` `×` が出る
- `○` = ポジティブ（+1）、`×` = ネガティブ（−1）
- 何も押さず次の入力 = フィードバックなし

**Analytics パネル**

- `Last Out Chars` = 直前出力の Unicode 文字数（バイト数ではない）
- `Refresh / N turns` のスライダーで N ターンごとに自動更新（デフォルト 5）

**履歴の永続化**

- 会話履歴は `history.sqlite` に自動保存される
- アプリ再起動で直近 60 ターン分を画面に復元

---

## CLI の使い方

### train — テキストファイルで学習

```bash
syntrail train --input <ファイル> [--model <パス>] [--hot-budget N]
```

- 既存モデルがあれば追加学習する。なければ新規作成。
- `--hot-budget N`: 学習後に HOT チャンクを N 個以内に絞る。`0`（デフォルト）は無制限。

### generate — テキスト生成

```bash
syntrail generate --seed <テキスト> [--model <パス>] [--max-units N]
```

### inspect — モデル統計の表示

```bash
syntrail inspect [--model <パス>]
```

### evaluate — dpc 計測（モデル変更なし）

```bash
syntrail evaluate --input <ファイル> [--model <パス>]
```

### recall — 連想記憶の検索

```bash
syntrail recall --unit <テキスト> [--model <パス>] [--limit N]
```

### chat — 1 ターン会話

```bash
syntrail chat --input <テキスト> [--db <DBパス>] [--model <パス>]
```

### feedback — 過去のターンにフィードバックを付与

```bash
syntrail feedback --turn-id <N> --sign <+|-|1|-1> [--db <DBパス>]
```

---

## SynTrail Trainer

長文テキストを複数行ブロック単位で Adaptive 反復学習させる専用 GUI。

### 起動

```bash
cargo run --release --features gui --bin syntrail-trainer
```

### 使い方

1. **Model** — `.json` / `.db` / `.sqlite` を Open またはD&D
2. **Dataset** — `.txt` ファイルを Open またはD&D（UTF-8 / Shift_JIS 自動判定）
3. **Start** で学習開始

Windows では `File` メニューと `Ctrl+O` / `Ctrl+S` / `Ctrl+Shift+S` が使用可能。

### Pause / Resume

| ボタン | 動作 |
|--------|------|
| **Pause** | 即時保存して一時停止。既存 worker は保持。 |
| **Resume** | 停止中の既存 worker を再開（`TrainerCommand::Resume` 送信）。 |
| **Resume Saved** | 保存済み TrainerState から新規 worker を起動（アプリ再起動後の再開）。 |
| **Stop** | 保存して worker を終了。 |

- Pause→Resume は同セッション内のみ。アプリを終了した場合は Resume Saved を使う。
- dataset または model の fingerprint が一致しない場合は Resume Saved は失敗する。

### Adaptive 反復（同一ブロック）

チェックポイント: 4 → 8 → 16 → 32 回。

| 評価タイミング | 判断基準 | 動作 |
|---|---|---|
| 4 回目 | 常に継続 | → 8 回へ |
| 8 回目 | dpc 改善率 ≥ 2% | → 16 回へ、それ以外終了 |
| 16 回目 | dpc 改善率 ≥ 1% | → 32 回へ、それ以外終了 |
| 32 回目 | 常に終了 | ブロック完了 |

### Block サイズ自動調整

直近 8 ブロックの pre-dpc と反復数中央値で判断（S / M / L / XL）。

### Experience / Replay 分離

- **First pass**: `expose_external()` — 新規テキストとして扱う
- **Subsequent passes**: `replay()` — 内部ルート強化のみ（外部ルートバイアスなし）

### D&D

- Running / Paused 中はモデル/データセットの差し替えをブロックする（状態破壊防止）
- 複数ファイルの同時ドロップはエラー表示

---

## Analytics 項目の説明

| 項目 | 説明 |
|------|------|
| Generation | 現セッションのターン数 |
| Tick | expose() 呼び出し累計 |
| Primitive | Unicode スカラー数（語彙サイズ相当） |
| Chunk | チャンク総数 |
| HOT | アクティブチャンク数（セグメント対象） |
| SLEEP | 休止チャンク数 |
| Association | 共起関連エッジ総数 |
| Route (edges) | 予測エッジ総数 |
| T0 / T1 / T2 | チャンクのティア分布 |
| Avg Exp Len | チャンクの平均展開長（文字数） |
| Total Chars | 累計入力文字数 |
| Total Decisions | 累計決定回数 |
| dpc | `decisions / characters`。学習が進むと下がる。目標: < 1.0 |
| Last Decisions | 直前ターンの決定回数 |
| Last Out Chars | 直前ターンの出力 Unicode 文字数 |
| Positive FB | ポジティブフィードバック累計 |
| Negative FB | ネガティブフィードバック累計 |

---

## コア概念

| 用語 | 説明 |
|------|------|
| **Primitive** | Unicode スカラー値 1 個。削除不可の最小単位。 |
| **Chunk** | 二分合成 `(left, right)`。再帰的・可変長の再利用可能単位。 |
| **HOT** | セグメント対象のアクティブ状態。 |
| **SLEEP** | セグメント対象外の休止状態。再出現で自動復帰。 |
| **dpc** | `decisions / characters`。主要評価指標。 |
| **Experience** | 新規テキストの expose（外部ルートバイアスあり）。 |
| **Replay** | 既学習テキストの再 expose（内部ルート強化のみ）。 |
| **AssociationStore** | 共起 Top-K メモリ。`recall` コマンドで参照。 |
| **Frozen Evaluation** | モデルを変更しない読み取り専用評価。 |
| **TransformKind** | EquivalentView / Mapping / Inverse / Composed。EquivalentView のみ Identity を統合。 |

---

## モジュール構成

```
src/
├── lib.rs                クレートルート
├── main.rs               CLI
├── app.rs                Application API（CLI/GUI 共通）
├── bin/
│   ├── syntrail_gui.rs   Chat GUI（--features gui）
│   └── syntrail_trainer.rs  Trainer GUI（--features gui）
├── desktop/              OS/Desktop 共通層（--features gui）
│   ├── file_ops.rs       FileKind / FileCommand / DocumentState
│   ├── dialogs.rs        rfd ファイルダイアログ
│   ├── drop.rs           DropRouter（D&D 分類）
│   ├── fonts.rs          CJK フォント設定
│   └── platform/
│       └── windows.rs    Windows Native Menu（muda）
├── trainer/              Trainer ロジック
│   ├── adaptive.rs       Adaptive Block Level
│   ├── dataset.rs        Dataset（UTF-8 / Shift_JIS 読み込み）
│   ├── scheduler.rs      TrainingScheduler（checkpoint 判定）
│   ├── splitter.rs       BlockSplitter
│   └── state.rs          TrainerState（Pause/Resume 状態）
├── config.rs             Config（hot_budget 含む全チューナブル）
├── model.rs              ModelState（merge_right_reuse, enforce_hot_budget）
├── transform.rs          TransformStore / TransformKind
├── representation.rs     RepresentationStore（Lineage）
├── identity.rs           IdentityStore（Union-Find）
├── prediction.rs         PredictionEdge + PredictionStore
├── association.rs        AssociationStore（共起 Top-K）
├── segmentation.rs       segment() / expand()
├── eval.rs               evaluate_frozen()
└── persistence.rs        JSON / SQLite 永続化

tests/
└── integration.rs        100 integration tests
```

---

## テスト

```bash
cargo test
```

100 integration tests、警告ゼロ（lib warnings 2 件は既存コードの未使用 API）。

---

## スコープ外（MVP）

Transformer・GPU・グラフ可視化・複数会話管理・テーマ編集・モデル内部値の手動編集

---

## ライセンス

Private / WIP
