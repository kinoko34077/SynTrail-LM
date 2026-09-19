# 09 Desktop UI 共通仕様

本ドキュメントは §77〜135 の恒久仕様部分を正本化したもの。

一時的な現行バグ一覧は [CURRENT_STATE.md](../current/CURRENT_STATE.md) を参照。

---

## Desktop Shell (§78)

Chat GUI / Trainer / 将来のTeacher等に共通するOS/Desktop操作を各binへ直接書かない。

```
src/desktop/
    mod.rs
    file_ops.rs      FileKind, FileCommand
    dialogs.rs       open_model_dialog, save_model_dialog, open_dataset_dialog,
                     confirm_discard_dialog, confirm_save_discard_cancel_dialog
    drop.rs          route_drop(), AcceptedKinds, DropResult → FileCommand
    fonts.rs         setup_fonts() (CJK font candidates)
    platform/
        windows.rs   NativeMenu (Chat), TrainerMenu (Trainer)
```

---

## FileKind (§82)

```rust
pub enum FileKind {
    ModelJson,       // .json
    ModelDatabase,   // .db / .sqlite
    ModelBinary,     // .stm 等 (将来)
    DatasetText,     // .txt
    TrainerState,    // .syntrail-trainer.json
    Unknown,
}
```

拡張子判定はcase-insensitive。例: .JSON / .Json / json → ModelJson。

---

## FileCommand / AppCommand (§81)

```rust
pub enum FileCommand {
    New,
    NewConversation,   // Chat GUIのみ — 会話履歴リセット、モデル保持
    Open,
    OpenDataset,       // Trainer専用 — テキストデータセット選択 (§39)
    Save,
    SaveAs,
    LoadPath(PathBuf), // D&D / 最近使ったファイル用
}
```

Native Menu / Keyboard Shortcut / Toolbar / D&D はすべて同じ `FileCommand` へ変換する。

---

## FileDialogSpec (§83)

ファイルダイアログの title / filters / default_name / initial_directory を一箇所へ集約。Chat GUI と Trainer で同じ「モデルを開く」操作には同じ Filter 定義を使う。

---

## ModelFileService (§84)

モデルに対する load / save / save_as / detect_format を一箇所へ集約。

```
GUI → ModelFileService → persistence::save / load
```

現在 `app.rs` の `save_model_file` / `load_model_file` が近い実装だが、
二重保存・binary形式・format detection を統一する。

---

## Save / Save As の意味 (§86)

**Save**: 現在の Document Path へ保存。Path が無い場合は Save As にフォールバック。

**Save As**: ユーザーが新 Path を選択。**保存成功後のみ** Current Path を更新する。保存失敗時は Current Path を変更しない。

---

## Dirty State (§87, §41)

**Chat GUI** は `DocumentState { path, dirty }` で管理。

**Trainer** は `model_dirty: bool` フィールドで管理 (§41):
- 停止時の最終 save が失敗した場合: `model_dirty = true`
- `TrainerEvent::Saved` 受信時: `model_dirty = false`
- `TrainerEvent::Completed` 受信時: `model_dirty = false`
- New / Open 前に dirty なら `confirm_save_discard_cancel_dialog()` を表示:
  - **Save**: 保存して続行
  - **Discard**: 保存せず続行
  - **Cancel**: 操作を中断

---

## Windows Native Menu (§88〜90, §39)

Chat GUI (NativeMenu):
```
File
  New Model       Ctrl+N
  New Conversation Ctrl+Shift+N
  Open...         Ctrl+O
  Save            Ctrl+S
  Save As...      Ctrl+Shift+S
  ────────────
  Exit
```

Trainer (TrainerMenu — §39):
```
File
  New Session     Ctrl+N
  Open Model...   Ctrl+O
  Open Dataset... Ctrl+Shift+O
  Save            Ctrl+S
  Save As...      Ctrl+Shift+S
  ────────────
  Exit
```

Native Menu event → `FileCommand` → App-specific handler。

Windows 固有コードは `desktop/platform/windows.rs` へ隔離。
非 Windows では egui toolbar ボタンが fallback として機能する (§38)。

---

## Keyboard Shortcuts (§92)

| ショートカット | コマンド |
|--------------|---------|
| Ctrl+N | FileCommand::New |
| Ctrl+O | FileCommand::Open → dialog |
| Ctrl+S | FileCommand::Save |
| Ctrl+Shift+S | FileCommand::SaveAs → dialog |

Native Menu と同一 Command 経路を使う（二重実装禁止）。

---

## D&D 共通化 (§95〜96)

```
DroppedPath → FileKind detection → App accepted kinds → FileCommand
```

| ファイル種別 | Chat | Trainer |
|------------|------|---------|
| ModelJson/ModelDatabase | OpenModel | OpenModel |
| DatasetText | — | OpenDataset |
| 複数同種 | エラー表示 | エラー表示 |
| Unsupported | status表示 | status表示 |

Running 中の Trainer への D&D は状態を変更しない。`Stop training before replacing model/dataset` を表示。

---

## Trainer State Machine (§109)

```
Idle
Ready
Running
Paused
Stopping
Completed
Error
```

合法遷移:

```
Ready → Start → Running
Running → Pause → Paused
Paused → Resume → Running       ← 既存Pause workerへ TrainerCommand::Resume
Running/Paused → Stop → Stopping → Ready/Completed
Ready + saved state → ResumeSaved → Running  ← 新workerで start_training(true)
```

UI enable/disable だけでなく command layer でも不正操作を拒否する。

---

## Trainer Save 意味論 (§110〜113)

- Save error を `let _ =` で握り潰してはならない
- 保存成功を確認してから Saved イベントを送る
- `save_and_exit()` でも必ず最新 fingerprint を反映してから save
- Running/Paused 中の Save As は `TrainerCommand::SaveAs(path)` → worker が保存

---

## Trainer Atomic Save (§111)

1. 保存対象状態確定
2. model fingerprint 確定 (`tr_state.model_fingerprint = model.state_fingerprint()`)
3. model 保存
4. trainer state 保存
5. 両方成功確認
6. Saved 通知

---

## Chat GUI Raw Input (§100)

```rust
// 禁止: モデルへ渡す文字列をtrimしない
let raw_input = self.input.clone();  // モデルへ渡す文字列
let trimmed = raw_input.trim();      // 空入力判定のみに使用
if trimmed.is_empty() { return; }
self.handle.generate_turn(&raw_input);  // 原文を渡す
```

---

## Chat GUI Last Output (§101)

```rust
// app.rs AppHandle
self.last_output_len = output.len();        // bytes (internal)
self.last_output_chars = output.chars().count();  // Unicode chars (表示用)
```

Analytics の表示名は `Last Out Chars` (Unicode文字数) とする。

---

## Trainer Path TextEdit (§116)

Model / Dataset Path は read-only display にする。Open 操作でのみ変更。

「表示 Path」と「実際に load 済み object」がズレないこと。

---

## Font Setup 共通化 (§122)

CJK font candidate リストと `setup_fonts()` を `desktop/fonts.rs` へ共通化。

---

## Acceptance Tests (§129〜131)

主要 acceptance criteria は CURRENT_STATE.md の Desktop UI 既知問題セクションを参照。テスト実装は tests/ または手動 smoke test で確認する。

---

## 実装完了条件 (§137)

- Chat GUI で Windows Native Menu が表示
- New/Open/Save/Save As が動作
- Keyboard shortcuts が同一 Command 経路
- Chat/Trainer の Model Open/Save が同一 Service
- D&D 分類が共通
- Trainer Pause → Resume (in-process) が動作
- Trainer Running 中 Open/D&D で状態破壊しない
- Stop 時 fingerprint 整合
- 保存 Error を握り潰さない
- Trainer Save As 動作
- `cargo fmt --check` / `cargo test` / `cargo build --release --features gui`
- Windows Native Menu 手動 Smoke Test
