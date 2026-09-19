/// SynTrail Trainer GUI — adaptive multi-line exposure training engine.
///
/// Build:  cargo build --features gui --bin syntrail-trainer
/// Run:    cargo run   --features gui --bin syntrail-trainer
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use syntrail_lm::app::{load_model_file, model_analytics, save_model_file, Analytics};
use syntrail_lm::desktop::file_ops::FileCommand;
use syntrail_lm::eval::evaluate_sample_frozen;
use syntrail_lm::model::ModelState;
use syntrail_lm::trainer::adaptive::{decide_level_change, BlockLevel};
use syntrail_lm::trainer::dataset::Dataset;
use syntrail_lm::trainer::scheduler::{CheckpointOutcome, TrainingScheduler};
use syntrail_lm::trainer::splitter::BlockSplitter;
use syntrail_lm::trainer::state::{TrainerState, TrainerStatus};

#[cfg(all(target_os = "windows", feature = "gui"))]
use syntrail_lm::desktop::platform::windows::NativeMenu;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 700.0])
            .with_title("SynTrail Trainer")
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "SynTrail Trainer",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            Ok(Box::new(TrainerApp::new()))
        }),
    )
}

fn setup_fonts(ctx: &egui::Context) {
    let candidates: &[&str] = &[
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ];
    if let Some(data) = candidates.iter().find_map(|p| std::fs::read(p).ok()) {
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert("cjk".to_owned(), egui::FontData::from_owned(data));
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(family).or_default().push("cjk".to_owned());
        }
        ctx.set_fonts(fonts);
    }
}

// ── Worker ────────────────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
enum TrainerCommand {
    Pause,
    Resume,
    Stop,
}

#[derive(Debug)]
struct BlockInfo {
    idx: usize,
    preview: String,
    pre_dpc: f64,
    pre_accuracy: f64,
    source_bytes_done: usize,
    total_bytes: usize,
}

#[derive(Debug)]
struct CheckpointInfo {
    checkpoint: u32,
    repeat: u32,
    dpc: f64,
    accuracy: f64,
}

#[derive(Debug)]
#[allow(dead_code, clippy::enum_variant_names)]
enum TrainerEvent {
    BlockStarted(BlockInfo),
    ExposureDone(u32),
    Checkpoint(CheckpointInfo),
    BlockDone { idx: usize, repeats: u32 },
    LevelChanged(BlockLevel),
    Saved,
    Paused,
    Stopped,
    Completed(usize),
    Error(String),
    Analytics(Box<Analytics>),
}

fn worker_main(
    mut model: ModelState,
    dataset: Dataset,
    mut tr_state: TrainerState,
    model_path: PathBuf,
    cmd_rx: Receiver<TrainerCommand>,
    ev_tx: Sender<TrainerEvent>,
) {
    let total_bytes = dataset.normalized.len();

    // Determine starting position.
    let resume_block_start = tr_state.current_block_start;
    let resume_repeat = tr_state.current_block_repeat;
    let resume_checkpoint_dpcs = tr_state.checkpoint_dpcs;
    let resume_pre_dpc = tr_state.current_block_pre_dpc;
    let is_resuming = resume_repeat > 0 && resume_block_start < total_bytes;

    let mut splitter = BlockSplitter::with_cursor(
        dataset.normalized.clone(),
        tr_state.block_level,
        if is_resuming { resume_block_start } else { tr_state.cursor },
    );

    macro_rules! check_cmd {
        () => {{
            use std::sync::mpsc::TryRecvError;
            loop {
                match cmd_rx.try_recv() {
                    Ok(TrainerCommand::Pause) => {
                        tr_state.status = TrainerStatus::Paused;
                        tr_state.model_fingerprint = model.state_fingerprint();
                        // Only enter Paused state if save succeeded.
                        if !do_save(&model, &model_path, &tr_state, &ev_tx) {
                            // save failed — stay Running; Error already sent by do_save
                            break;
                        }
                        let _ = ev_tx.send(TrainerEvent::Paused);
                        // Wait for Resume or Stop.
                        loop {
                            match cmd_rx.recv() {
                                Ok(TrainerCommand::Resume) => {
                                    tr_state.status = TrainerStatus::Running;
                                    break;
                                }
                                Ok(TrainerCommand::Stop) | Err(_) => {
                                    save_and_exit(&mut model, &model_path, &mut tr_state, &ev_tx);
                                    return;
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(TrainerCommand::Stop) => {
                        save_and_exit(&mut model, &model_path, &mut tr_state, &ev_tx);
                        return;
                    }
                    Ok(TrainerCommand::Resume) => {}  // already running
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => { save_and_exit(&mut model, &model_path, &mut tr_state, &ev_tx); return; }
                }
            }
        }};
    }

    // Capture initial count so the resume-path condition stays stable as
    // completed_block_count and block_idx are incremented in lockstep.
    let resume_block_idx = tr_state.completed_block_count;
    let mut block_idx = resume_block_idx;

    // ── Main training loop ────────────────────────────────────────────────
    loop {
        check_cmd!();

        // Get the current block.
        let block = match if is_resuming && block_idx == resume_block_idx {
            // Re-read the in-progress block from its start.
            let mut tmp = BlockSplitter::with_cursor(
                dataset.normalized.clone(),
                tr_state.block_level,
                resume_block_start,
            );
            tmp.next_block()
        } else {
            splitter.next_block()
        } {
            Some(b) => b,
            None => {
                tr_state.status = TrainerStatus::Completed;
                tr_state.model_fingerprint = model.state_fingerprint();
                do_save(&model, &model_path, &tr_state, &ev_tx);
                let _ = ev_tx.send(TrainerEvent::Completed(block_idx));
                return;
            }
        };

        // Update splitter cursor to after this block (if not resuming).
        if !(is_resuming && block_idx == tr_state.completed_block_count) {
            // Already advanced by next_block above.
        }

        // Pre-evaluation.
        let pre_eval = evaluate_sample_frozen(&model, &block.text);

        let _ = ev_tx.send(TrainerEvent::BlockStarted(BlockInfo {
            idx: block_idx,
            preview: block.text.chars().take(200).collect(),
            pre_dpc: pre_eval.dpc,
            pre_accuracy: pre_eval.prediction_accuracy,
            source_bytes_done: block.start,
            total_bytes,
        }));

        // Save block info to trainer state.
        tr_state.current_block_start = block.start;
        tr_state.current_block_end = block.end;
        tr_state.current_block_pre_dpc = pre_eval.dpc;

        // Build scheduler — restore if resuming.
        let mut sched = if is_resuming && block_idx == tr_state.completed_block_count && resume_repeat > 0 {
            TrainingScheduler::resume(resume_pre_dpc, resume_repeat, resume_checkpoint_dpcs)
        } else {
            TrainingScheduler::new(pre_eval.dpc)
        };

        // ── Per-block exposure loop ───────────────────────────────────────
        loop {
            check_cmd!();

            // First pass = Experience; subsequent passes = Replay (Phase 1).
            if sched.repeat_count == 0 {
                model.expose_external(&block.text);
            } else {
                model.replay(&block.text);
            }
            sched.record_exposure();
            tr_state.current_block_repeat = sched.repeat_count;

            let _ = ev_tx.send(TrainerEvent::ExposureDone(sched.repeat_count));

            if sched.at_checkpoint() {
                let eval = evaluate_sample_frozen(&model, &block.text);
                let outcome = sched.evaluate_checkpoint(eval.dpc);

                tr_state.checkpoint_dpcs = sched.checkpoint_dpcs();
                tr_state.current_checkpoint = sched.next_checkpoint();

                let _ = ev_tx.send(TrainerEvent::Checkpoint(CheckpointInfo {
                    checkpoint: sched.repeat_count,
                    repeat: sched.repeat_count,
                    dpc: eval.dpc,
                    accuracy: eval.prediction_accuracy,
                }));

                // Save at each checkpoint.
                tr_state.model_fingerprint = model.state_fingerprint();
                do_save(&model, &model_path, &tr_state, &ev_tx);
                let _ = ev_tx.send(TrainerEvent::Analytics(Box::new(model_analytics(&model))));

                if outcome == CheckpointOutcome::Finished {
                    break;
                }
            }
        }

        // Block complete.
        let repeats_used = sched.repeat_count;
        tr_state.push_block_result(pre_eval.dpc, repeats_used);
        tr_state.completed_block_count += 1;
        tr_state.current_block_repeat = 0;
        tr_state.cursor = block.end;

        // After the resumed block, advance the main splitter past it.
        if is_resuming && block_idx == resume_block_idx {
            splitter = BlockSplitter::with_cursor(
                dataset.normalized.clone(),
                tr_state.block_level,
                block.end,
            );
        }

        let _ = ev_tx.send(TrainerEvent::BlockDone { idx: block_idx, repeats: repeats_used });

        // Check if Block Level should change.
        let change = decide_level_change(&tr_state.recent_pre_dpc, &tr_state.recent_repeats_used);
        match change {
            syntrail_lm::trainer::adaptive::LevelChange::Upgrade => {
                let new_level = tr_state.block_level.upgrade();
                if new_level != tr_state.block_level {
                    tr_state.block_level = new_level;
                    splitter.set_level(new_level);
                    let _ = ev_tx.send(TrainerEvent::LevelChanged(new_level));
                }
            }
            syntrail_lm::trainer::adaptive::LevelChange::Downgrade => {
                let new_level = tr_state.block_level.downgrade();
                if new_level != tr_state.block_level {
                    tr_state.block_level = new_level;
                    splitter.set_level(new_level);
                    let _ = ev_tx.send(TrainerEvent::LevelChanged(new_level));
                }
            }
            syntrail_lm::trainer::adaptive::LevelChange::Keep => {}
        }

        // Save after each block.
        tr_state.model_fingerprint = model.state_fingerprint();
        do_save(&model, &model_path, &tr_state, &ev_tx);

        block_idx += 1;
    }
}

/// Save model + trainer state atomically; send Saved on success, Error on failure (§110, §111).
fn do_save(
    model: &ModelState,
    model_path: &PathBuf,
    tr_state: &TrainerState,
    ev_tx: &Sender<TrainerEvent>,
) -> bool {
    if let Err(e) = save_model_file(model, model_path) {
        let _ = ev_tx.send(TrainerEvent::Error(format!("Model save failed: {e}")));
        return false;
    }
    if let Err(e) = tr_state.save(&PathBuf::from(&tr_state.dataset_path)) {
        let _ = ev_tx.send(TrainerEvent::Error(format!("State save failed: {e}")));
        return false;
    }
    let _ = ev_tx.send(TrainerEvent::Saved);
    true
}

/// Save on worker exit; updates fingerprint, reports errors, sends Stopped (§112).
fn save_and_exit(
    model: &mut ModelState,
    model_path: &PathBuf,
    tr_state: &mut TrainerState,
    ev_tx: &Sender<TrainerEvent>,
) {
    tr_state.model_fingerprint = model.state_fingerprint();
    if let Err(e) = save_model_file(model, model_path) {
        eprintln!("save_and_exit: model save failed: {e}");
        let _ = ev_tx.send(TrainerEvent::Error(format!("Exit save failed: {e}")));
    } else if let Err(e) = tr_state.save(&PathBuf::from(&tr_state.dataset_path)) {
        eprintln!("save_and_exit: state save failed: {e}");
        let _ = ev_tx.send(TrainerEvent::Error(format!("Exit save failed: {e}")));
    } else {
        let _ = ev_tx.send(TrainerEvent::Saved);
    }
    let _ = ev_tx.send(TrainerEvent::Stopped);
}

// ── GUI state ─────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct TrainerProgress {
    block_idx: usize,
    source_bytes_done: usize,
    total_bytes: usize,
    repeat: u32,
    checkpoint: u32,
    pre_dpc: f64,
    current_dpc: f64,
    accuracy: f64,
    block_level: BlockLevel,
    block_preview: String,
    analytics: Analytics,
    last_event: String,
}

#[allow(dead_code)]
enum AppState {
    Idle,
    Ready { model_loaded: bool, dataset_loaded: bool },
    Running,
    Paused,
    Completed(usize),
    Error(String),
}

struct TrainerApp {
    model_path: String,
    dataset_path: String,
    state: AppState,
    worker: Option<JoinHandle<()>>,
    cmd_tx: Option<Sender<TrainerCommand>>,
    event_rx: Option<Receiver<TrainerEvent>>,
    progress: TrainerProgress,
    status_msg: String,
    // Loaded objects (only valid in Ready/before start)
    loaded_model: Option<ModelState>,
    loaded_dataset: Option<Dataset>,
    // Detected existing trainer state
    resume_state: Option<Result<TrainerState, String>>,
    #[cfg(all(target_os = "windows", feature = "gui"))]
    native_menu: NativeMenu,
}

impl TrainerApp {
    fn new() -> Self {
        Self {
            model_path: "model.json".to_owned(),
            dataset_path: String::new(),
            state: AppState::Idle,
            worker: None,
            cmd_tx: None,
            event_rx: None,
            progress: TrainerProgress::default(),
            status_msg: "Select a model and a text file to begin.".to_owned(),
            loaded_model: None,
            loaded_dataset: None,
            resume_state: None,
            #[cfg(all(target_os = "windows", feature = "gui"))]
            native_menu: NativeMenu::build(),
        }
    }

    fn try_load_model(&mut self) {
        let p = PathBuf::from(&self.model_path);
        match load_model_file(&p) {
            Ok(m) => {
                self.status_msg = format!("Model loaded: {}", p.display());
                self.loaded_model = Some(m);
                self.check_ready();
            }
            Err(e) => {
                self.status_msg = format!("Model load failed: {e}");
                self.loaded_model = None;
            }
        }
    }

    fn try_load_dataset(&mut self) {
        let p = PathBuf::from(&self.dataset_path);
        match Dataset::load(&p) {
            Ok(ds) => {
                self.status_msg = format!(
                    "Dataset loaded: {} chars, {} bytes",
                    ds.normalized.chars().count(),
                    ds.normalized.len()
                );
                // Check for existing trainer state.
                self.resume_state = Some(TrainerState::load(&p));
                self.loaded_dataset = Some(ds);
                self.check_ready();
            }
            Err(e) => {
                self.status_msg = format!("Dataset load failed: {e}");
                self.loaded_dataset = None;
                self.resume_state = None;
            }
        }
    }

    fn check_ready(&mut self) {
        let model_ok = self.loaded_model.is_some();
        let dataset_ok = self.loaded_dataset.is_some();
        self.state = AppState::Ready { model_loaded: model_ok, dataset_loaded: dataset_ok };
    }

    fn start_training(&mut self, resume: bool) {
        let model = match self.loaded_model.take() {
            Some(m) => m,
            None => { self.status_msg = "No model loaded".to_owned(); return; }
        };
        let dataset = match self.loaded_dataset.take() {
            Some(d) => d,
            None => { self.status_msg = "No dataset loaded".to_owned(); return; }
        };

        let model_fp = model.state_fingerprint();
        let dataset_path = PathBuf::from(&self.dataset_path);
        let model_path = PathBuf::from(&self.model_path);

        let tr_state = if resume {
            match &self.resume_state {
                Some(Ok(saved)) => {
                    match saved.verify_resume(dataset.fingerprint, &model_fp) {
                        Ok(()) => saved.clone(),
                        Err(e) => {
                            self.status_msg = e;
                            self.loaded_model = Some(model);
                            self.loaded_dataset = Some(dataset);
                            self.check_ready();
                            return;
                        }
                    }
                }
                _ => {
                    self.status_msg = "No valid saved state to resume from.".to_owned();
                    self.loaded_model = Some(model);
                    self.loaded_dataset = Some(dataset);
                    self.check_ready();
                    return;
                }
            }
        } else {
            TrainerState::new(&dataset_path, dataset.fingerprint, dataset.normalized.len(), &model_path, model_fp)
        };

        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (ev_tx, ev_rx) = mpsc::channel();

        let ev_tx2 = ev_tx.clone();
        let handle = std::thread::spawn(move || {
            worker_main(model, dataset, tr_state, model_path, cmd_rx, ev_tx2);
        });

        self.cmd_tx = Some(cmd_tx);
        self.event_rx = Some(ev_rx);
        self.worker = Some(handle);
        self.state = AppState::Running;
        self.status_msg = "Training started.".to_owned();
    }

    fn send_cmd(&self, cmd: TrainerCommand) {
        if let Some(tx) = &self.cmd_tx { let _ = tx.send(cmd); }
    }

    fn poll_events(&mut self) {
        let events: Vec<TrainerEvent> = match &self.event_rx {
            Some(rx) => {
                let mut v = Vec::new();
                while let Ok(ev) = rx.try_recv() { v.push(ev); }
                v
            }
            None => return,
        };
        for ev in events { self.handle_event(ev); }
    }

    fn handle_event(&mut self, ev: TrainerEvent) {
        match ev {
            TrainerEvent::BlockStarted(info) => {
                self.progress.block_idx = info.idx;
                self.progress.source_bytes_done = info.source_bytes_done;
                self.progress.total_bytes = info.total_bytes;
                self.progress.pre_dpc = info.pre_dpc;
                self.progress.current_dpc = info.pre_dpc;
                self.progress.accuracy = info.pre_accuracy;
                self.progress.block_preview = info.preview;
                self.progress.repeat = 0;
                self.progress.checkpoint = 4;
                self.progress.last_event = format!("Block {}", info.idx);
            }
            TrainerEvent::ExposureDone(n) => {
                self.progress.repeat = n;
            }
            TrainerEvent::Checkpoint(ci) => {
                self.progress.repeat = ci.repeat;
                self.progress.checkpoint = ci.checkpoint;
                self.progress.current_dpc = ci.dpc;
                self.progress.accuracy = ci.accuracy;
                self.progress.last_event = format!("Checkpoint {} — dpc {:.4}", ci.checkpoint, ci.dpc);
            }
            TrainerEvent::BlockDone { idx, repeats } => {
                self.progress.last_event = format!("Block {} done ({} repeats)", idx, repeats);
            }
            TrainerEvent::LevelChanged(lvl) => {
                self.progress.block_level = lvl;
                self.status_msg = format!("Block level changed to {}", lvl.label());
            }
            TrainerEvent::Saved => {
                self.status_msg = format!("Saved (block {})", self.progress.block_idx);
            }
            TrainerEvent::Paused => {
                self.state = AppState::Paused;
                self.status_msg = "Paused and saved.".to_owned();
            }
            TrainerEvent::Stopped => {
                // Worker has exited; reclaim resources and return to Ready.
                if let Some(h) = self.worker.take() { let _ = h.join(); }
                self.cmd_tx = None;
                self.event_rx = None;
                // Re-instate loaded objects from model_path so UI is consistent.
                let model_path = PathBuf::from(&self.model_path);
                if let Ok(m) = syntrail_lm::app::load_model_file(&model_path) {
                    self.loaded_model = Some(m);
                }
                self.check_ready();
                if self.status_msg.starts_with("Error") {
                    // keep error message
                } else {
                    self.status_msg = "Stopped and saved.".to_owned();
                }
            }
            TrainerEvent::Completed(n) => {
                if let Some(h) = self.worker.take() { let _ = h.join(); }
                self.cmd_tx = None;
                self.event_rx = None;
                self.state = AppState::Completed(n);
                self.status_msg = format!("Training complete — {} blocks processed.", n);
            }
            TrainerEvent::Error(e) => {
                self.state = AppState::Error(e.clone());
                self.status_msg = format!("Error: {e}");
            }
            TrainerEvent::Analytics(a) => {
                self.progress.analytics = *a;
            }
        }
    }
}

// ── eframe App ────────────────────────────────────────────────────────────

impl eframe::App for TrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_events();

        // Native menu: attach on first frame, then poll each frame (§88–94)
        #[cfg(all(target_os = "windows", feature = "gui"))]
        {
            use raw_window_handle::{HasWindowHandle, RawWindowHandle};
            let training_active_menu = matches!(&self.state, AppState::Running | AppState::Paused);
            if let Ok(handle) = _frame.window_handle() {
                if let RawWindowHandle::Win32(h) = handle.as_raw() {
                    let hwnd = h.hwnd.get() as isize;
                    let _ = unsafe { self.native_menu.attach(hwnd) };
                }
            }
            if let Some(cmd) = self.native_menu.poll() {
                match cmd {
                    FileCommand::New => {
                        if !training_active_menu {
                            self.loaded_model = None;
                            self.loaded_dataset = None;
                            self.resume_state = None;
                            self.state = AppState::Idle;
                            self.status_msg = "Ready for new model and dataset.".to_owned();
                        } else {
                            self.status_msg = "Stop training before starting new session.".to_owned();
                        }
                    }
                    FileCommand::Open => {
                        if !training_active_menu {
                            if let Some(p) = rfd::FileDialog::new()
                                .set_title("モデルを開く")
                                .add_filter("Model files", &["json", "db", "sqlite"])
                                .pick_file()
                            {
                                self.model_path = p.to_string_lossy().to_string();
                                self.try_load_model();
                            }
                        } else {
                            self.status_msg = "Stop training before opening a new model.".to_owned();
                        }
                    }
                    FileCommand::Save => {
                        let p = PathBuf::from(&self.model_path);
                        if let Some(m) = &self.loaded_model {
                            match save_model_file(m, &p) {
                                Ok(()) => self.status_msg = format!("Saved: {}", p.display()),
                                Err(e) => self.status_msg = format!("Save failed: {e}"),
                            }
                        }
                    }
                    FileCommand::SaveAs => {
                        if let Some(p) = rfd::FileDialog::new()
                            .set_title("名前を付けて保存")
                            .add_filter("Model JSON", &["json"])
                            .save_file()
                        {
                            if let Some(m) = &self.loaded_model {
                                match save_model_file(m, &p) {
                                    Ok(()) => {
                                        self.model_path = p.to_string_lossy().to_string();
                                        self.status_msg = format!("Saved: {}", p.display());
                                    }
                                    Err(e) => self.status_msg = format!("Save failed: {e}"),
                                }
                            }
                        }
                    }
                    FileCommand::LoadPath(p) => {
                        if !training_active_menu {
                            self.model_path = p.to_string_lossy().to_string();
                            self.try_load_model();
                        }
                    }
                }
            }
        }

        // D&D: block changes while Running or Paused (§108)
        let training_active = matches!(&self.state, AppState::Running | AppState::Paused);
        ctx.input(|i| {
            if i.raw.dropped_files.is_empty() { return; }
            if training_active {
                // set status after closure; can't mutate self.status_msg here
                return;
            }
            for file in &i.raw.dropped_files {
                if let Some(path) = &file.path {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if ext == "txt" {
                        self.dataset_path = path.to_string_lossy().to_string();
                        self.try_load_dataset();
                    } else if matches!(ext, "json" | "db" | "sqlite") {
                        self.model_path = path.to_string_lossy().to_string();
                        self.try_load_model();
                    }
                }
            }
        });
        if training_active {
            ctx.input(|i| {
                if !i.raw.dropped_files.is_empty() {
                    self.status_msg = "Stop training before replacing model/dataset.".to_owned();
                }
            });
        }

        // ── Status bar ────────────────────────────────────────────────────
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&self.status_msg).small());
            });
        });

        // ── Central content ───────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("SynTrail Trainer");
            ui.separator();

            // ── File selection — read-only display during training (§116) ──
            let file_ops_enabled = !matches!(&self.state, AppState::Running | AppState::Paused);
            ui.horizontal(|ui| {
                ui.label("Model:");
                // Read-only display; Open is the only way to change path (§116)
                ui.add_enabled(false,
                    egui::TextEdit::singleline(&mut self.model_path.clone()).desired_width(260.0));
                if ui.add_enabled(file_ops_enabled, egui::Button::new("Open…")).clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .set_title("モデルを開く")
                        .add_filter("SynTrail JSON", &["json"])
                        .add_filter("SynTrail DB", &["db", "sqlite"])
                        .pick_file()
                    {
                        self.model_path = p.to_string_lossy().to_string();
                        self.try_load_model();
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Dataset:");
                ui.add_enabled(false,
                    egui::TextEdit::singleline(&mut self.dataset_path.clone()).desired_width(260.0));
                if ui.add_enabled(file_ops_enabled, egui::Button::new("Open…")).clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .set_title("テキストファイルを開く")
                        .add_filter("Text", &["txt"])
                        .pick_file()
                    {
                        self.dataset_path = p.to_string_lossy().to_string();
                        self.try_load_dataset();
                    }
                }
            });

            ui.add_space(4.0);
            ui.label(format!(
                "Block Level: {}  ({} lines / {} chars)",
                self.progress.block_level.label(),
                self.progress.block_level.target_lines(),
                self.progress.block_level.max_chars(),
            ));
            ui.add_space(4.0);

            // ── Control buttons ───────────────────────────────────────────
            let can_start = matches!(&self.state, AppState::Ready { model_loaded: true, dataset_loaded: true, .. });
            let is_running = matches!(&self.state, AppState::Running);
            let is_paused = matches!(&self.state, AppState::Paused);
            let has_save = matches!(&self.resume_state, Some(Ok(_)));

            ui.horizontal(|ui| {
                if ui.add_enabled(can_start, egui::Button::new("Start")).clicked() {
                    self.start_training(false);
                }
                // Resume Saved: create new worker from saved trainer state (§107)
                if ui.add_enabled(can_start && has_save, egui::Button::new("Resume Saved")).clicked() {
                    self.start_training(true);
                }
                if ui.add_enabled(is_running, egui::Button::new("Pause")).clicked() {
                    self.send_cmd(TrainerCommand::Pause);
                }
                // Resume: send command to existing paused worker (§107)
                if ui.add_enabled(is_paused, egui::Button::new("Resume")).clicked() {
                    self.send_cmd(TrainerCommand::Resume);
                    self.state = AppState::Running;
                    self.status_msg = "Resumed.".to_owned();
                }
                if ui.add_enabled(is_running || is_paused, egui::Button::new("Stop")).clicked() {
                    self.send_cmd(TrainerCommand::Stop);
                }
            });

            ui.separator();

            // ── Progress panel ────────────────────────────────────────────
            let p = &self.progress;
            let pct = if p.total_bytes > 0 { p.source_bytes_done as f64 / p.total_bytes as f64 } else { 0.0 };

            egui::Grid::new("progress").num_columns(2).striped(true).show(ui, |ui| {
                ui.label("Block");            ui.label(p.block_idx.to_string());             ui.end_row();
                ui.label("Source");          ui.label(format!("{:.1}%", pct * 100.0));       ui.end_row();
                ui.label("Repeat");          ui.label(format!("{} / max {}", p.repeat, 32)); ui.end_row();
                ui.label("Checkpoint");      ui.label(p.checkpoint.to_string());             ui.end_row();
                ui.end_row(); ui.end_row();
                ui.label("pre dpc");         ui.label(format!("{:.4}", p.pre_dpc));          ui.end_row();
                ui.label("current dpc");     ui.label(format!("{:.4}", p.current_dpc));      ui.end_row();
                ui.label("accuracy");        ui.label(format!("{:.4}", p.accuracy));         ui.end_row();
                ui.end_row(); ui.end_row();
                let a = &p.analytics;
                ui.label("Chunk");           ui.label(a.chunk_count.to_string());            ui.end_row();
                ui.label("HOT / SLEEP");     ui.label(format!("{} / {}", a.hot_count, a.sleep_count)); ui.end_row();
                ui.label("Avg Exp Len");     ui.label(format!("{:.2}", a.avg_expanded_length)); ui.end_row();
                ui.label("T0/T1/T2");        ui.label(format!("{}/{}/{}", a.t0_count, a.t1_count, a.t2_count)); ui.end_row();
                ui.label("dpc (model)");     ui.label(format!("{:.6}", a.dpc));              ui.end_row();
                ui.label("Total Chars");     ui.label(a.total_characters.to_string());       ui.end_row();
            });

            ui.add(egui::ProgressBar::new(pct as f32).show_percentage());

            ui.separator();
            ui.label("Current Block:");
            egui::ScrollArea::vertical().id_salt("block_preview").max_height(160.0).show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(&p.block_preview).monospace())
                        .selectable(true)
                        .wrap(),
                );
            });
        });

        // Keep refreshing while running so events are processed promptly.
        if matches!(&self.state, AppState::Running) {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
    }
}
