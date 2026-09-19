/// SynTrail Trainer GUI — adaptive multi-line exposure training engine.
///
/// Build:  cargo build --features gui --bin syntrail-trainer
/// Run:    cargo run   --features gui --bin syntrail-trainer
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use syntrail_lm::app::{load_model_file, model_analytics, save_model_file, save_model_file_with_generation, Analytics, DocumentState};
use syntrail_lm::desktop::dialogs::{
    open_dataset_dialog, open_model_dialog, save_model_dialog,
    confirm_save_discard_cancel_dialog, ConfirmResult,
};
use syntrail_lm::desktop::drop::{AcceptedKinds, DropResult, route_drop};
use syntrail_lm::desktop::file_ops::{FileCommand, FileKind};
use syntrail_lm::desktop::fonts::setup_fonts;
use syntrail_lm::eval::evaluate_sample_frozen;
use syntrail_lm::model::ModelState;
use syntrail_lm::trainer::adaptive::{decide_level_change, BlockLevel};
use syntrail_lm::trainer::dataset::Dataset;
use syntrail_lm::trainer::scheduler::{CheckpointOutcome, TrainingScheduler};
use syntrail_lm::trainer::splitter::BlockSplitter;
use syntrail_lm::trainer::state::{TrainerState, TrainerStatus};

#[cfg(all(target_os = "windows", feature = "gui"))]
use syntrail_lm::desktop::platform::windows::TrainerMenu;

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

// ── Worker ────────────────────────────────────────────────────────────────

/// §29: Why a save was requested — controls failure severity and UI state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SaveKind {
    /// User-initiated save while running (recoverable if it fails).
    Manual,
    /// User-initiated save-as while running (recoverable if it fails).
    SaveAs,
    /// Scheduled checkpoint/block save (fatal if it fails — worker stops).
    Checkpoint,
    /// Save triggered by Pause command (failure → ErrorPaused, not Paused).
    Pause,
    /// Save triggered by Stop command (fatal, but model is still transferred).
    Stop,
}

/// §31: Result of the exit save, carried with TrainerEvent::Stopped.
#[derive(Debug)]
enum SaveResult {
    Saved(PathBuf),
    Failed { path: PathBuf, error: String },
}

#[derive(Debug)]
#[allow(dead_code)]
enum TrainerCommand {
    Pause,
    Resume,
    Stop,
    /// §28: Save to the current canonical model path.
    Save,
    /// §28: Save to a new path and make it the new canonical path.
    SaveAs(std::path::PathBuf),
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

#[allow(dead_code, clippy::enum_variant_names)]
enum TrainerEvent {
    BlockStarted(BlockInfo),
    ExposureDone(u32),
    Checkpoint(CheckpointInfo),
    BlockDone { idx: usize, repeats: u32 },
    LevelChanged(BlockLevel),
    /// §29: Save succeeded — carries path and classification.
    Saved { path: PathBuf, kind: SaveKind },
    /// §30: Save failed — carries path, operation type, and whether the worker can continue.
    SaveFailed { path: PathBuf, operation: SaveKind, recoverable: bool },
    Paused,
    /// §31: Worker has exited — carries the latest in-memory model so UI never loses it.
    Stopped { model: Box<ModelState>, save_result: SaveResult },
    /// §39: carries the finished model so the UI takes ownership without a disk re-read.
    Completed(usize, Box<ModelState>),
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
    // §28: worker owns the canonical model path; SaveAs updates it here and in tr_state.
    let mut current_model_path = model_path;
    let total_bytes = dataset.normalized.len();

    // Determine starting position.
    let resume_block_start = tr_state.current_block_start;
    let resume_repeat = tr_state.current_block_repeat;
    let resume_checkpoint_dpcs = tr_state.checkpoint_dpcs;
    let resume_pre_dpc = tr_state.current_block_pre_dpc;
    let is_resuming = resume_repeat > 0 && resume_block_start < total_bytes;

    let mut splitter = BlockSplitter::with_seed(
        dataset.normalized.clone(),
        tr_state.block_level,
        if is_resuming { resume_block_start } else { tr_state.cursor },
        tr_state.split_seed,
    );

    macro_rules! check_cmd {
        () => {{
            use std::sync::mpsc::TryRecvError;
            loop {
                match cmd_rx.try_recv() {
                    Ok(TrainerCommand::Pause) => {
                        tr_state.status = TrainerStatus::Paused;
                        tr_state.model_fingerprint = model.state_fingerprint();
                        tr_state.checkpoint_generation = model.tick; // §32
                        // §4: always enter command-wait after Pause regardless of save result.
                        // On success: send Paused → UI stays Paused.
                        // On failure: SaveFailed already sent → UI transitions to ErrorPaused.
                        // Both paths wait for Resume or Stop.
                        if do_save(&model, &current_model_path, &tr_state, SaveKind::Pause, &ev_tx) {
                            let _ = ev_tx.send(TrainerEvent::Paused);
                        }
                        // Wait for Resume or Stop.
                        loop {
                            match cmd_rx.recv() {
                                Ok(TrainerCommand::Resume) => {
                                    tr_state.status = TrainerStatus::Running;
                                    break;
                                }
                                Ok(TrainerCommand::Stop) | Err(_) => {
                                    save_and_exit(model, &current_model_path, &mut tr_state, &ev_tx);
                                    return;
                                }
                                Ok(TrainerCommand::Save) => {
                                    tr_state.model_fingerprint = model.state_fingerprint();
                                    do_save(&model, &current_model_path, &tr_state, SaveKind::Manual, &ev_tx);
                                }
                                Ok(TrainerCommand::SaveAs(new_path)) => {
                                    tr_state.model_fingerprint = model.state_fingerprint();
                                    // §28: update canonical path on confirmed save only
                                    if do_save(&model, &new_path, &tr_state, SaveKind::SaveAs, &ev_tx) {
                                        tr_state.model_path = new_path.to_string_lossy().to_string();
                                        current_model_path = new_path;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(TrainerCommand::Stop) => {
                        save_and_exit(model, &current_model_path, &mut tr_state, &ev_tx);
                        return;
                    }
                    Ok(TrainerCommand::Resume) => {}  // already running
                    Ok(TrainerCommand::Save) => {
                        // §28: on-demand save to current canonical path; failure is recoverable.
                        tr_state.model_fingerprint = model.state_fingerprint();
                        do_save(&model, &current_model_path, &tr_state, SaveKind::Manual, &ev_tx);
                    }
                    Ok(TrainerCommand::SaveAs(new_path)) => {
                        // §28: on-demand save-as; update canonical path only on success.
                        tr_state.model_fingerprint = model.state_fingerprint();
                        if do_save(&model, &new_path, &tr_state, SaveKind::SaveAs, &ev_tx) {
                            tr_state.model_path = new_path.to_string_lossy().to_string();
                            current_model_path = new_path;
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => { save_and_exit(model, &current_model_path, &mut tr_state, &ev_tx); return; }
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
            let mut tmp = BlockSplitter::with_seed(
                dataset.normalized.clone(),
                tr_state.block_level,
                resume_block_start,
                tr_state.split_seed,
            );
            tmp.next_block()
        } else {
            splitter.next_block()
        } {
            Some(b) => b,
            None => {
                tr_state.status = TrainerStatus::Completed;
                tr_state.model_fingerprint = model.state_fingerprint();
                tr_state.checkpoint_generation = model.tick; // §32
                do_save(&model, &current_model_path, &tr_state, SaveKind::Checkpoint, &ev_tx);
                // §39: transfer model ownership to UI; no disk re-read needed.
                let _ = ev_tx.send(TrainerEvent::Completed(block_idx, Box::new(model)));
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

                // Save at each checkpoint. §37/§30/§32: failure stops the worker.
                tr_state.model_fingerprint = model.state_fingerprint();
                tr_state.checkpoint_generation = model.tick; // §32
                if !do_save(&model, &current_model_path, &tr_state, SaveKind::Checkpoint, &ev_tx) {
                    let _ = ev_tx.send(TrainerEvent::Stopped {
                        model: Box::new(model),
                        save_result: SaveResult::Failed {
                            path: current_model_path.clone(),
                            error: "checkpoint save failed".to_owned(),
                        },
                    });
                    return;
                }
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

        // Save after each block. §37/§30/§32: failure stops the worker.
        tr_state.model_fingerprint = model.state_fingerprint();
        tr_state.checkpoint_generation = model.tick; // §32
        if !do_save(&model, &current_model_path, &tr_state, SaveKind::Checkpoint, &ev_tx) {
            let _ = ev_tx.send(TrainerEvent::Stopped {
                model: Box::new(model),
                save_result: SaveResult::Failed {
                    path: current_model_path.clone(),
                    error: "block save failed".to_owned(),
                },
            });
            return;
        }

        block_idx += 1;
    }
}

/// §29/§30/§32: Save model + trainer state; send typed Saved/SaveFailed events.
/// For Checkpoint/Pause/Stop saves, embeds a checkpoint_generation in both files.
/// Returns true on success, false on failure (failure event already sent).
fn do_save(
    model: &ModelState,
    model_path: &PathBuf,
    tr_state: &TrainerState,
    kind: SaveKind,
    ev_tx: &Sender<TrainerEvent>,
) -> bool {
    let recoverable = matches!(kind, SaveKind::Manual | SaveKind::SaveAs | SaveKind::Pause);
    // §32: pair model + state file under a common generation ID for checkpoint saves.
    let generation = tr_state.checkpoint_generation;
    if let Err(e) = save_model_file_with_generation(model, model_path, generation) {
        let _ = ev_tx.send(TrainerEvent::SaveFailed {
            path: model_path.clone(),
            operation: kind,
            recoverable,
        });
        eprintln!("do_save: model save failed ({kind:?}): {e}");
        return false;
    }
    if let Err(e) = tr_state.save(&PathBuf::from(&tr_state.dataset_path)) {
        let _ = ev_tx.send(TrainerEvent::SaveFailed {
            path: model_path.clone(),
            operation: kind,
            recoverable,
        });
        eprintln!("do_save: state save failed ({kind:?}): {e}");
        return false;
    }
    let _ = ev_tx.send(TrainerEvent::Saved { path: model_path.clone(), kind });
    true
}

/// §31/§32: Save on worker exit; sends Stopped with the in-memory model regardless of save outcome.
fn save_and_exit(
    model: ModelState,
    current_model_path: &PathBuf,
    tr_state: &mut TrainerState,
    ev_tx: &Sender<TrainerEvent>,
) {
    tr_state.model_fingerprint = model.state_fingerprint();
    tr_state.checkpoint_generation = model.tick; // §32
    let generation = tr_state.checkpoint_generation;
    let save_result = if let Err(e) = save_model_file_with_generation(&model, current_model_path, generation) {
        eprintln!("save_and_exit: model save failed: {e}");
        let _ = ev_tx.send(TrainerEvent::SaveFailed {
            path: current_model_path.clone(),
            operation: SaveKind::Stop,
            recoverable: false,
        });
        SaveResult::Failed { path: current_model_path.clone(), error: e.to_string() }
    } else if let Err(e) = tr_state.save(&PathBuf::from(&tr_state.dataset_path)) {
        eprintln!("save_and_exit: state save failed: {e}");
        let _ = ev_tx.send(TrainerEvent::SaveFailed {
            path: current_model_path.clone(),
            operation: SaveKind::Stop,
            recoverable: false,
        });
        SaveResult::Failed { path: current_model_path.clone(), error: e.to_string() }
    } else {
        let _ = ev_tx.send(TrainerEvent::Saved { path: current_model_path.clone(), kind: SaveKind::Stop });
        SaveResult::Saved(current_model_path.clone())
    };
    let _ = ev_tx.send(TrainerEvent::Stopped { model: Box::new(model), save_result });
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
    /// §40: Stop was sent; waiting for the worker to confirm with Stopped.
    Stopping,
    /// §40: Worker hit a non-fatal error and paused; user can Resume or Stop.
    ErrorPaused(String),
    Completed(usize),
    Error(String),
}

/// §17: How the user chooses a start position for new training runs.
#[derive(Default, Clone, Copy, PartialEq)]
enum StartMode {
    #[default]
    Beginning,
    Line,
    Percent,
}

struct TrainerApp {
    doc: DocumentState,
    dataset_path: String,
    /// §17: new-training start position mode.
    start_mode: StartMode,
    /// §17: Line-mode input (1-based line number as string).
    start_line_str: String,
    /// §17: Percent-mode input (0–100 as string).
    start_pct_str: String,
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
    native_menu: TrainerMenu,
}

impl TrainerApp {
    fn new() -> Self {
        Self {
            doc: DocumentState::from_path(PathBuf::from("model.json")),
            dataset_path: String::new(),
            start_mode: StartMode::Beginning,
            start_line_str: "1".to_owned(),
            start_pct_str: "0.0".to_owned(),
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
            native_menu: TrainerMenu::build(),
        }
    }

    fn try_load_model(&mut self) {
        let p = self.doc.path.clone().unwrap_or_else(|| PathBuf::from("model.json"));
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
        // §23: show diagnostic summary when both model and dataset are loaded.
        if model_ok && dataset_ok {
            let m = self.loaded_model.as_ref().unwrap();
            let ds = self.loaded_dataset.as_ref().unwrap();
            self.status_msg = format!(
                "Ready — model: {} chunks / {} primitives / tick {} | dataset: {} chars",
                m.chunk_count(), m.primitive_count(), m.tick,
                ds.normalized.chars().count()
            );
        }
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
        let model_path = self.doc.path.clone().unwrap_or_else(|| PathBuf::from("model.json"));

        let tr_state = if resume {
            match &self.resume_state {
                Some(Ok(saved)) => {
                    // §32: read checkpoint_generation from the model file to detect mismatched pairs.
                    let model_gen = syntrail_lm::persistence::load_checkpoint_generation(&model_path)
                        .ok(); // None if file doesn't exist or is legacy
                    match saved.verify_resume_with_generation(dataset.fingerprint, &model_fp, model_gen) {
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
            let mut ts = TrainerState::new(&dataset_path, dataset.fingerprint, dataset.normalized.len(), &model_path, model_fp);
            // §17: apply UI start position for new runs (§19: resume always uses saved cursor).
            ts.cursor = compute_start_cursor(&dataset.normalized, self.start_mode, &self.start_line_str, &self.start_pct_str);
            ts
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
            TrainerEvent::Saved { path, kind } => {
                // §28: SaveAs confirmed — update canonical path in UI now.
                if kind == SaveKind::SaveAs {
                    self.doc.mark_saved(path.clone());
                    self.status_msg = format!("Saved As: {}", path.display());
                } else {
                    self.doc.dirty = false;
                    self.status_msg = format!("Saved (block {})", self.progress.block_idx);
                }
            }
            TrainerEvent::SaveFailed { path, operation, recoverable } => {
                let op_name = match operation {
                    SaveKind::Manual | SaveKind::SaveAs => "Save",
                    SaveKind::Checkpoint => "Checkpoint save",
                    SaveKind::Pause => "Pause save",
                    SaveKind::Stop => "Stop save",
                };
                self.status_msg = format!("{op_name} failed: {}", path.display());
                // §30: Pause failure → ErrorPaused (worker stayed Running).
                if operation == SaveKind::Pause {
                    self.state = AppState::ErrorPaused(format!("{op_name} failed"));
                }
                // Non-recoverable: a Stopped event will follow — no state change here.
                let _ = recoverable;
            }
            TrainerEvent::Paused => {
                self.state = AppState::Paused;
                self.status_msg = "Paused and saved.".to_owned();
            }
            TrainerEvent::Stopped { model, save_result } => {
                // §31: Worker has exited — take in-memory model; no disk re-read needed.
                if let Some(h) = self.worker.take() { let _ = h.join(); }
                self.cmd_tx = None;
                self.event_rx = None;
                self.loaded_model = Some(*model);
                self.check_ready();
                // §41: model is dirty only if the final save failed.
                self.doc.dirty = matches!(save_result, SaveResult::Failed { .. });
                if self.status_msg.starts_with("Error") || self.status_msg.contains("failed") {
                    // keep error message
                } else {
                    self.status_msg = match &save_result {
                        SaveResult::Saved(_) => "Stopped and saved.".to_owned(),
                        SaveResult::Failed { error, .. } => format!("Stopped (save failed: {error})"),
                    };
                }
            }
            TrainerEvent::Completed(n, trained_model) => {
                if let Some(h) = self.worker.take() { let _ = h.join(); }
                self.cmd_tx = None;
                self.event_rx = None;
                // §39: take ownership of the trained model so Save/Save As uses it.
                self.loaded_model = Some(*trained_model);
                self.doc.dirty = false; // §41: always checkpoint-saved before Completed
                self.state = AppState::Completed(n);
                self.status_msg = format!("Training complete — {} blocks processed.", n);
            }
            TrainerEvent::Error(e) => {
                // §40: error while running → ErrorPaused (worker already paused or stopped);
                // error in other states → terminal Error.
                if matches!(&self.state, AppState::Running | AppState::Stopping) {
                    self.state = AppState::ErrorPaused(e.clone());
                } else {
                    self.state = AppState::Error(e.clone());
                }
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
            let training_active_menu = matches!(&self.state, AppState::Running | AppState::Paused | AppState::Stopping | AppState::ErrorPaused(_));
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
                            // §41: guard dirty model before discarding it.
                            if self.doc.dirty {
                                match confirm_save_discard_cancel_dialog() {
                                    ConfirmResult::Save => {
                                        let p = self.doc.path.clone().unwrap_or_else(|| PathBuf::from("model.json"));
                                        match save_model_file(self.loaded_model.as_ref().unwrap(), &p) {
                                            Ok(()) => { self.doc.dirty = false; }
                                            Err(e) => {
                                                self.status_msg = format!("Save failed: {e}");
                                                return; // abort New
                                            }
                                        }
                                    }
                                    ConfirmResult::Discard => {}
                                    ConfirmResult::Cancel => return,
                                }
                            }
                            self.loaded_model = None;
                            self.loaded_dataset = None;
                            self.resume_state = None;
                            self.state = AppState::Idle;
                            self.doc.dirty = false;
                            self.status_msg = "Ready for new model and dataset.".to_owned();
                        } else {
                            self.status_msg = "Stop training before starting new session.".to_owned();
                        }
                    }
                    FileCommand::Open => {
                        if !training_active_menu {
                            // §41: guard dirty model before replacing it.
                            if self.doc.dirty {
                                match confirm_save_discard_cancel_dialog() {
                                    ConfirmResult::Save => {
                                        let p = self.doc.path.clone().unwrap_or_else(|| PathBuf::from("model.json"));
                                        match save_model_file(self.loaded_model.as_ref().unwrap(), &p) {
                                            Ok(()) => { self.doc.dirty = false; }
                                            Err(e) => {
                                                self.status_msg = format!("Save failed: {e}");
                                                return; // abort Open
                                            }
                                        }
                                    }
                                    ConfirmResult::Discard => {}
                                    ConfirmResult::Cancel => return,
                                }
                            }
                            if let Some(p) = open_model_dialog(self.doc.path_str()) {
                                self.doc.path = Some(p.clone());
                                self.doc.dirty = false;
                                self.try_load_model();
                            }
                        } else {
                            self.status_msg = "Stop training before opening a new model.".to_owned();
                        }
                    }
                    FileCommand::Save => {
                        let p = self.doc.path.clone().unwrap_or_else(|| PathBuf::from("model.json"));
                        if training_active_menu {
                            // §28: delegate save to worker; path stays canonical until SavedAs confirmed.
                            self.send_cmd(TrainerCommand::Save);
                            self.status_msg = format!("Save requested: {}", p.display());
                        } else if let Some(m) = &self.loaded_model {
                            match save_model_file(m, &p) {
                                Ok(()) => {
                                    self.doc.dirty = false;
                                    self.status_msg = format!("Saved: {}", p.display());
                                }
                                Err(e) => self.status_msg = format!("Save failed: {e}"),
                            }
                        }
                    }
                    FileCommand::SaveAs => {
                        if let Some(p) = save_model_dialog(self.doc.path_str()) {
                            if training_active_menu {
                                // §28: delegate SaveAs to worker; do NOT update doc.path yet.
                                // UI path updates only on receiving Saved { kind: SaveAs }.
                                self.send_cmd(TrainerCommand::SaveAs(p.clone()));
                                self.status_msg = format!("Save As requested: {}", p.display());
                            } else if let Some(m) = &self.loaded_model {
                                match save_model_file(m, &p) {
                                    Ok(()) => {
                                        self.doc.mark_saved(p.clone());
                                        self.status_msg = format!("Saved: {}", p.display());
                                    }
                                    Err(e) => self.status_msg = format!("Save failed: {e}"),
                                }
                            }
                        }
                    }
                    FileCommand::OpenDataset => {
                        if !training_active_menu {
                            if let Some(p) = open_dataset_dialog(&self.dataset_path) {
                                self.dataset_path = p.to_string_lossy().to_string();
                                self.try_load_dataset();
                            }
                        } else {
                            self.status_msg = "Stop training before opening a new dataset.".to_owned();
                        }
                    }
                    FileCommand::LoadPath(p) => {
                        if !training_active_menu {
                            self.doc.path = Some(p.clone());
                            self.doc.dirty = false;
                            self.try_load_model();
                        }
                    }
                    FileCommand::NewConversation => {} // not applicable in Trainer
                }
            }
        }

        // D&D — routed via desktop::drop::route_drop (§108).
        let training_active = matches!(&self.state, AppState::Running | AppState::Paused | AppState::Stopping | AppState::ErrorPaused(_));
        let dropped: Vec<_> = ctx.input(|i| {
            i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect()
        });
        if !dropped.is_empty() {
            if training_active {
                self.status_msg = "Stop training before replacing model/dataset.".to_owned();
            } else {
                match route_drop(dropped, &AcceptedKinds::trainer()) {
                    DropResult::Command(FileCommand::LoadPath(p)) => {
                        match FileKind::detect(&p) {
                            FileKind::DatasetText => {
                                self.dataset_path = p.to_string_lossy().to_string();
                                self.try_load_dataset();
                            }
                            _ => {
                                // §22: guard dirty model before D&D replacement.
                                if self.doc.dirty {
                                    match confirm_save_discard_cancel_dialog() {
                                        ConfirmResult::Save => {
                                            let save_p = self.doc.path.clone().unwrap_or_else(|| PathBuf::from("model.json"));
                                            match save_model_file(self.loaded_model.as_ref().unwrap(), &save_p) {
                                                Ok(()) => { self.doc.dirty = false; }
                                                Err(e) => {
                                                    self.status_msg = format!("Save failed: {e}");
                                                    return;
                                                }
                                            }
                                        }
                                        ConfirmResult::Discard => {}
                                        ConfirmResult::Cancel => return,
                                    }
                                }
                                self.doc.path = Some(p.clone());
                                self.doc.dirty = false;
                                self.try_load_model();
                            }
                        }
                    }
                    DropResult::ModelAndDataset { model, dataset } => {
                        // §43: load both in one drop.
                        // §22: guard dirty model before D&D replacement.
                        if self.doc.dirty {
                            match confirm_save_discard_cancel_dialog() {
                                ConfirmResult::Save => {
                                    let save_p = self.doc.path.clone().unwrap_or_else(|| PathBuf::from("model.json"));
                                    match save_model_file(self.loaded_model.as_ref().unwrap(), &save_p) {
                                        Ok(()) => { self.doc.dirty = false; }
                                        Err(e) => {
                                            self.status_msg = format!("Save failed: {e}");
                                            return;
                                        }
                                    }
                                }
                                ConfirmResult::Discard => {}
                                ConfirmResult::Cancel => return,
                            }
                        }
                        self.doc.path = Some(model.clone());
                        self.doc.dirty = false;
                        self.dataset_path = dataset.to_string_lossy().to_string();
                        self.try_load_model();
                        self.try_load_dataset();
                    }
                    DropResult::MultipleFiles => {
                        self.status_msg = "Drop one file at a time (or one model + one dataset together).".to_owned();
                    }
                    DropResult::Unsupported(p) => {
                        self.status_msg = format!(
                            "Unsupported file: {}",
                            p.file_name().and_then(|n| n.to_str()).unwrap_or("?")
                        );
                    }
                    _ => {}
                }
            }
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
            // §38: On Windows (native menu present), toolbar Open buttons are hidden;
            //      use File menu instead.  Non-Windows keeps the buttons as the fallback.
            let file_ops_enabled = !matches!(&self.state, AppState::Running | AppState::Paused | AppState::Stopping);
            ui.horizontal(|ui| {
                ui.label("Model:");
                // Read-only display; Open is the only way to change path (§116)
                let mut model_display = self.doc.path_str().to_owned();
                ui.add_enabled(false,
                    egui::TextEdit::singleline(&mut model_display).desired_width(260.0));
                #[cfg(not(all(target_os = "windows", feature = "gui")))]
                if ui.add_enabled(file_ops_enabled, egui::Button::new("Open…")).clicked() {
                    if let Some(p) = open_model_dialog(self.doc.path_str()) {
                        self.doc.path = Some(p.clone());
                        self.doc.dirty = false;
                        self.try_load_model();
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Dataset:");
                ui.add_enabled(false,
                    egui::TextEdit::singleline(&mut self.dataset_path.clone()).desired_width(260.0));
                #[cfg(not(all(target_os = "windows", feature = "gui")))]
                if ui.add_enabled(file_ops_enabled, egui::Button::new("Open…")).clicked() {
                    if let Some(p) = open_dataset_dialog(&self.dataset_path) {
                        self.dataset_path = p.to_string_lossy().to_string();
                        self.try_load_dataset();
                    }
                }
            });

            // §17/§18: Start Position selector + preview (new training only, hidden while training).
            let dataset_loaded = matches!(&self.state, AppState::Ready { dataset_loaded: true, .. })
                || matches!(&self.state, AppState::Idle);
            if dataset_loaded && !training_active {
                ui.horizontal(|ui| {
                    ui.label("Start:");
                    ui.selectable_value(&mut self.start_mode, StartMode::Beginning, "Beginning");
                    ui.selectable_value(&mut self.start_mode, StartMode::Line, "Line");
                    ui.selectable_value(&mut self.start_mode, StartMode::Percent, "Percent");
                    match self.start_mode {
                        StartMode::Line => {
                            ui.add(egui::TextEdit::singleline(&mut self.start_line_str).desired_width(70.0));
                        }
                        StartMode::Percent => {
                            ui.add(egui::TextEdit::singleline(&mut self.start_pct_str).desired_width(60.0));
                            ui.label("%");
                        }
                        StartMode::Beginning => {}
                    }
                });
                // §18: show line / % and first few lines after start position.
                if let Some(ds) = &self.loaded_dataset {
                    let cursor = compute_start_cursor(&ds.normalized, self.start_mode, &self.start_line_str, &self.start_pct_str);
                    let line_num = ds.normalized[..cursor].chars().filter(|&c| c == '\n').count() + 1;
                    let pct = if ds.normalized.is_empty() { 0.0 } else { cursor as f64 / ds.normalized.len() as f64 * 100.0 };
                    let first_lines: Vec<&str> = ds.normalized[cursor..].lines().take(3).collect();
                    let preview_str = first_lines.join("\n");
                    ui.label(egui::RichText::new(format!("→ Line {} ({:.1}%)", line_num, pct)).small());
                    if !preview_str.is_empty() {
                        egui::ScrollArea::vertical().id_salt("start_preview").max_height(48.0).show(ui, |ui| {
                            ui.add(egui::Label::new(egui::RichText::new(&preview_str).monospace().small()).wrap());
                        });
                    }
                }
                ui.add_space(4.0);
            }

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
                let is_stopping = matches!(&self.state, AppState::Stopping);
                let is_error_paused = matches!(&self.state, AppState::ErrorPaused(_));
                // Resume from error-pause: try to continue.
                if ui.add_enabled(is_error_paused, egui::Button::new("Resume")).clicked() {
                    self.send_cmd(TrainerCommand::Resume);
                    self.state = AppState::Running;
                    self.status_msg = "Resumed after error.".to_owned();
                }
                if ui.add_enabled((is_running || is_paused || is_error_paused) && !is_stopping,
                                  egui::Button::new("Stop")).clicked() {
                    self.send_cmd(TrainerCommand::Stop);
                    self.state = AppState::Stopping; // §40
                    self.status_msg = "Stopping…".to_owned();
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

// ── §17: Start Position helpers ───────────────────────────────────────────

/// Snap `byte_pos` back to the start of its containing line.
fn snap_to_line_start(source: &str, byte_pos: usize) -> usize {
    let pos = byte_pos.min(source.len());
    match source[..pos].rfind('\n') {
        Some(nl) => nl + 1,
        None => 0,
    }
}

/// Compute a byte cursor into `source` from the user-selected start mode.
/// Always returns a valid UTF-8 line boundary.
fn compute_start_cursor(source: &str, mode: StartMode, line_str: &str, pct_str: &str) -> usize {
    match mode {
        StartMode::Beginning => 0,
        StartMode::Line => {
            let n: usize = line_str.trim().parse::<usize>().unwrap_or(1).saturating_sub(1);
            let mut byte_pos = 0usize;
            for (i, line) in source.split('\n').enumerate() {
                if i == n { break; }
                byte_pos += line.len() + 1;
            }
            byte_pos.min(source.len())
        }
        StartMode::Percent => {
            let pct = pct_str.trim().parse::<f64>().unwrap_or(0.0).clamp(0.0, 100.0);
            let raw = (pct / 100.0 * source.len() as f64) as usize;
            snap_to_line_start(source, raw)
        }
    }
}
