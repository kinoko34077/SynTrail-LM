/// SynTrail-LM GUI — eframe/egui simple research interface.
///
/// Build:  cargo build --features gui --bin syntrail-gui
/// Run:    cargo run   --features gui --bin syntrail-gui
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::Duration;
use syntrail_lm::app::{Analytics, AppHandle};
use syntrail_lm::db::TurnRow;
use syntrail_lm::desktop::dialogs::{confirm_discard_dialog, open_model_dialog, save_model_dialog};
use syntrail_lm::desktop::drop::{AcceptedKinds, DropResult, route_drop};
use syntrail_lm::desktop::fonts::setup_fonts;
use syntrail_lm::feedback::FeedbackSign;

#[cfg(all(target_os = "windows", feature = "gui"))]
use syntrail_lm::desktop::platform::windows::NativeMenu;

const DEFAULT_MODEL: &str = "model.json";
const DEFAULT_HISTORY: &str = "history.sqlite";
const DEFAULT_REFRESH: u32 = 5;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 680.0])
            .with_title("SynTrail-LM")
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "SynTrail-LM",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            Ok(Box::new(SynTrailApp::new()))
        }),
    )
}

// ── Data types ────────────────────────────────────────────────────────────

struct ChatEntry {
    is_user: bool,
    text: String,
}

#[derive(Clone, Copy)]
enum FeedbackState {
    None,
    Pending(i64),
}

enum Action {
    None,
    New,
    NewConversation,
    OpenLoadDialog,
    Save,
    OpenSaveDialog,
    Feedback(i64, FeedbackSign),
    RefreshAnalytics,
    LoadPath(PathBuf),
    Send,
}

struct GenerationResult {
    handle: AppHandle,
    result: Result<(i64, String), String>,
}

enum ChatRequestState {
    Idle,
    Generating(Receiver<GenerationResult>),
}

// ── App ───────────────────────────────────────────────────────────────────

struct SynTrailApp {
    /// Generation temporarily owns the handle on a background thread.
    handle: Option<AppHandle>,
    request_state: ChatRequestState,
    model_path_display: String,
    input: String,
    chat_history: Vec<ChatEntry>,
    feedback_state: FeedbackState,
    analytics: Analytics,
    refresh_interval: u32,
    turns_since_refresh: u32,
    status: String,
    #[cfg(all(target_os = "windows", feature = "gui"))]
    native_menu: NativeMenu,
}

impl SynTrailApp {
    fn new() -> Self {
        let model_path = PathBuf::from(DEFAULT_MODEL);
        let handle = AppHandle::new(model_path, DEFAULT_HISTORY)
            .unwrap_or_else(|e| panic!("Failed to init AppHandle: {e}"));
        let chat_history = load_history_entries(&handle);
        let analytics = handle.get_analytics();
        let model_path_display = handle.doc.path_str().to_string();
        Self {
            handle: Some(handle),
            request_state: ChatRequestState::Idle,
            model_path_display,
            input: String::new(),
            chat_history,
            feedback_state: FeedbackState::None,
            analytics,
            refresh_interval: DEFAULT_REFRESH,
            turns_since_refresh: 0,
            status: "Ready".to_string(),
            #[cfg(all(target_os = "windows", feature = "gui"))]
            native_menu: NativeMenu::build(),
        }
    }

    fn is_generating(&self) -> bool {
        matches!(self.request_state, ChatRequestState::Generating(_))
    }

    fn start_generation(&mut self) {
        if self.is_generating() {
            self.status = "Generation in progress — wait for the current response.".to_string();
            return;
        }

        let raw = self.input.clone(); // pass raw string to model (§100)
        if raw.trim().is_empty() {
            return;
        }

        let Some(mut handle) = self.handle.take() else {
            self.status = "Generation unavailable: model handle is not ready.".to_string();
            return;
        };
        self.model_path_display = handle.doc.path_str().to_string();
        self.input.clear();
        self.feedback_state = FeedbackState::None; // auto-skip prior feedback
        self.chat_history.push(ChatEntry {
            is_user: true,
            text: raw.clone(),
        });
        self.status = "Generating…".to_string();

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            // Always return ownership of AppHandle even if generation itself panics.
            let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                handle.generate_turn(&raw)
            })) {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(error)) => Err(error.to_string()),
                Err(_) => Err("Generation worker panicked".to_string()),
            };
            let _ = tx.send(GenerationResult { handle, result });
        });
        self.request_state = ChatRequestState::Generating(rx);
    }

    fn poll_generation(&mut self, ctx: &egui::Context) {
        let received = match &self.request_state {
            ChatRequestState::Idle => return,
            ChatRequestState::Generating(rx) => rx.try_recv(),
        };

        match received {
            Ok(done) => {
                self.handle = Some(done.handle);
                self.request_state = ChatRequestState::Idle;
                match done.result {
                    Ok((turn_id, output)) => {
                        let Some(handle) = self.handle.as_ref() else {
                            self.status = "Generation completed but model handle was not restored."
                                .to_string();
                            return;
                        };
                        let a = handle.get_analytics();
                        self.status =
                            format!("Turn {}  |  {} decisions", turn_id, a.last_decision_count);
                        self.chat_history.push(ChatEntry {
                            is_user: false,
                            text: output,
                        });
                        self.feedback_state = FeedbackState::Pending(turn_id);
                        self.turns_since_refresh += 1;
                        if self.turns_since_refresh >= self.refresh_interval {
                            self.analytics = a;
                            self.turns_since_refresh = 0;
                        }
                    }
                    Err(error) => {
                        self.chat_history.pop();
                        self.status = format!("Generation failed: {error}");
                    }
                }
                ctx.request_repaint();
            }
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(33));
            }
            Err(TryRecvError::Disconnected) => {
                self.request_state = ChatRequestState::Idle;
                self.chat_history.pop();
                self.status = "Generation failed: worker disconnected before returning the model handle. Restart the app before continuing.".to_string();
                ctx.request_repaint();
            }
        }
    }

    fn do_feedback(&mut self, turn_id: i64, sign: FeedbackSign) {
        let Some(handle) = self.handle.as_mut() else {
            self.status =
                "Generation in progress — feedback is temporarily unavailable.".to_string();
            return;
        };
        match handle.apply_feedback(turn_id, sign) {
            Ok(()) => {
                let label = match sign {
                    FeedbackSign::Positive => "○",
                    FeedbackSign::Negative => "×",
                };
                self.status = format!("Feedback {label} → turn {turn_id}");
                self.analytics = handle.get_analytics();
            }
            Err(e) => self.status = format!("Feedback failed: {e}"),
        }
        self.feedback_state = FeedbackState::None;
    }

    fn do_load(&mut self, path: PathBuf) {
        if !self.confirm_discard_if_dirty() {
            return;
        }
        let Some(handle) = self.handle.as_mut() else {
            self.status = "Generation in progress — model replacement is temporarily unavailable."
                .to_string();
            return;
        };
        match handle.load_model_from(&path) {
            Ok(()) => {
                self.analytics = handle.get_analytics();
                self.chat_history.clear();
                self.feedback_state = FeedbackState::None;
                self.status = format!("Loaded: {}", path.display());
            }
            Err(e) => self.status = format!("Load failed: {e}"),
        }
    }

    fn do_save(&mut self) {
        let Some(handle) = self.handle.as_mut() else {
            self.status = "Generation in progress — save is temporarily unavailable.".to_string();
            return;
        };
        match handle.save_model() {
            Ok(()) => self.status = format!("Saved: {}", handle.doc.path_str()),
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    fn do_save_as(&mut self, path: PathBuf) {
        let Some(handle) = self.handle.as_mut() else {
            self.status = "Generation in progress — save is temporarily unavailable.".to_string();
            return;
        };
        match handle.save_model_to(&path) {
            Ok(()) => self.status = format!("Saved: {}", path.display()),
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    /// Returns true if the user confirmed or no confirmation was needed.
    fn confirm_discard_if_dirty(&self) -> bool {
        let Some(handle) = self.handle.as_ref() else {
            return false;
        };
        if !handle.doc.dirty {
            return true;
        }
        confirm_discard_dialog()
    }

    fn do_new(&mut self) {
        if !self.confirm_discard_if_dirty() {
            return;
        }
        let Some(handle) = self.handle.as_mut() else {
            self.status = "Generation in progress — model replacement is temporarily unavailable."
                .to_string();
            return;
        };
        handle.reset_model();
        self.chat_history.clear();
        self.feedback_state = FeedbackState::None;
        self.analytics = handle.get_analytics();
        self.status = "New model (unsaved)".to_string();
    }

    fn do_new_conversation(&mut self) {
        let Some(handle) = self.handle.as_mut() else {
            self.status = "Generation in progress — conversation reset is temporarily unavailable."
                .to_string();
            return;
        };
        handle.new_conversation();
        self.chat_history.clear();
        self.feedback_state = FeedbackState::None;
        self.status = "New conversation".to_string();
    }
}

fn load_history_entries(handle: &AppHandle) -> Vec<ChatEntry> {
    let rows: Vec<TurnRow> = handle.history(60).unwrap_or_default();
    let mut v = Vec::with_capacity(rows.len() * 2);
    for row in rows.into_iter().rev() {
        v.push(ChatEntry {
            is_user: true,
            text: row.input_text,
        });
        v.push(ChatEntry {
            is_user: false,
            text: row.output_text,
        });
    }
    v
}

// ── egui App ──────────────────────────────────────────────────────────────

impl eframe::App for SynTrailApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_generation(ctx);
        if let Some(handle) = self.handle.as_ref() {
            self.model_path_display = handle.doc.path_str().to_string();
        }

        let mut action = Action::None;
        let is_generating = self.is_generating();

        // Native menu: attach on first frame, then poll each frame (§88–94)
        #[cfg(all(target_os = "windows", feature = "gui"))]
        {
            use raw_window_handle::{HasWindowHandle, RawWindowHandle};
            if let Ok(handle) = _frame.window_handle() {
                if let RawWindowHandle::Win32(h) = handle.as_raw() {
                    let hwnd = h.hwnd.get() as isize;
                    let _ = unsafe { self.native_menu.attach(hwnd) };
                }
            }
            if let Some(cmd) = self.native_menu.poll() {
                action = match cmd {
                    syntrail_lm::desktop::FileCommand::New => Action::New,
                    syntrail_lm::desktop::FileCommand::NewConversation => Action::NewConversation,
                    syntrail_lm::desktop::FileCommand::Open => Action::OpenLoadDialog,
                    syntrail_lm::desktop::FileCommand::Save => Action::Save,
                    syntrail_lm::desktop::FileCommand::SaveAs => Action::OpenSaveDialog,
                    syntrail_lm::desktop::FileCommand::LoadPath(p) => Action::LoadPath(p),
                    syntrail_lm::desktop::FileCommand::OpenDataset => Action::None,
                };
            }
        }

        // File drag-and-drop — routed via desktop::drop::route_drop.
        let dropped: Vec<_> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            match route_drop(dropped, &AcceptedKinds::chat()) {
                DropResult::Command(syntrail_lm::desktop::FileCommand::LoadPath(p)) => {
                    action = Action::LoadPath(p);
                }
                DropResult::MultipleFiles => {
                    self.status =
                        "Drop failed: Chat accepts one model file at a time.".to_string();
                }
                DropResult::Unsupported(path) => {
                    self.status = format!(
                        "Drop failed: unsupported Chat file type: {}. Drop one supported model file.",
                        path.display()
                    );
                }
                DropResult::ModelAndDataset { .. } => {
                    self.status =
                        "Drop failed: Chat accepts one model file; dataset pairing is Trainer-only."
                            .to_string();
                }
                _ => {}
            }
        }

        let model_path_str = self.model_path_display.clone();

        // ── Top toolbar ───────────────────────────────────────────────────
        // §38: File-operation buttons shown only on platforms without a native menu.
        //      On Windows + gui, the native menu (NativeMenu) is the primary file UI.
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                #[cfg(not(all(target_os = "windows", feature = "gui")))]
                {
                    if ui.button("New Model").clicked() {
                        action = Action::New;
                    }
                    if ui.button("New Conversation").clicked() {
                        action = Action::NewConversation;
                    }
                    if ui.button("Open…").clicked() {
                        action = Action::OpenLoadDialog;
                    }
                    if ui.button("Save").clicked() {
                        action = Action::Save;
                    }
                    if ui.button("Save As…").clicked() {
                        action = Action::OpenSaveDialog;
                    }
                    ui.separator();
                }
                ui.label(egui::RichText::new(&model_path_str).small().weak());
                ui.separator();
                let a = &self.analytics;
                ui.label(format!(
                    "Gen: {}  Tick: {}  Chunk: {}(HOT:{})  dpc: {:.4}",
                    a.generation, a.tick, a.chunk_count, a.hot_count, a.dpc
                ));
            });
        });

        // ── Bottom input + feedback panel ─────────────────────────────────
        egui::TopBottomPanel::bottom("input_panel")
            .min_height(110.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&self.status).small());
                    if let FeedbackState::Pending(tid) = self.feedback_state {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add_enabled(!is_generating, egui::Button::new("  ×  "))
                                .on_hover_text("Negative feedback (−1)")
                                .clicked()
                            {
                                action = Action::Feedback(tid, FeedbackSign::Negative);
                            }
                            if ui
                                .add_enabled(!is_generating, egui::Button::new("  ○  "))
                                .on_hover_text("Positive feedback (+1)")
                                .clicked()
                            {
                                action = Action::Feedback(tid, FeedbackSign::Positive);
                            }
                        });
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    let avail = ui.available_width();
                    let response = ui.add(
                        egui::TextEdit::multiline(&mut self.input)
                            .desired_rows(3)
                            .desired_width(avail - 82.0),
                    );
                    let enter_pressed = !is_generating
                        && response.has_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                    let send_clicked = ui
                        .add_enabled(
                            !is_generating,
                            egui::Button::new("Send").min_size(egui::vec2(72.0, 60.0)),
                        )
                        .clicked();
                    if send_clicked || enter_pressed {
                        action = Action::Send;
                    }
                });
            });

        // ── Right analytics panel (scrollable) ────────────────────────────
        egui::SidePanel::right("analytics")
            .default_width(235.0)
            .min_width(180.0)
            .show(ctx, |ui| {
                ui.heading("Analytics");
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("analytics_scroll")
                    .show(ui, |ui| {
                        let a = &self.analytics;
                        egui::Grid::new("ag")
                            .num_columns(2)
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Generation");
                                ui.label(a.generation.to_string());
                                ui.end_row();
                                ui.label("Tick");
                                ui.label(a.tick.to_string());
                                ui.end_row();
                                ui.end_row();
                                ui.end_row();
                                ui.label("Primitive");
                                ui.label(a.primitive_count.to_string());
                                ui.end_row();
                                ui.label("Chunk");
                                ui.label(a.chunk_count.to_string());
                                ui.end_row();
                                ui.label("HOT");
                                ui.label(a.hot_count.to_string());
                                ui.end_row();
                                ui.label("SLEEP");
                                ui.label(a.sleep_count.to_string());
                                ui.end_row();
                                ui.end_row();
                                ui.end_row();
                                ui.label("Association");
                                ui.label(a.association_count.to_string());
                                ui.end_row();
                                ui.label("Route (edges)");
                                ui.label(a.edge_count.to_string());
                                ui.end_row();
                                ui.end_row();
                                ui.end_row();
                                ui.label("T0");
                                ui.label(a.t0_count.to_string());
                                ui.end_row();
                                ui.label("T1");
                                ui.label(a.t1_count.to_string());
                                ui.end_row();
                                ui.label("T2");
                                ui.label(a.t2_count.to_string());
                                ui.end_row();
                                ui.end_row();
                                ui.end_row();
                                ui.label("Avg Exp Len");
                                ui.label(format!("{:.2}", a.avg_expanded_length));
                                ui.end_row();
                                ui.end_row();
                                ui.end_row();
                                ui.label("Total Chars");
                                ui.label(a.total_characters.to_string());
                                ui.end_row();
                                ui.label("Total Decisions");
                                ui.label(a.total_decisions.to_string());
                                ui.end_row();
                                ui.label("dpc");
                                ui.label(format!("{:.6}", a.dpc));
                                ui.end_row();
                                ui.end_row();
                                ui.end_row();
                                ui.label("Last Decisions");
                                ui.label(a.last_decision_count.to_string());
                                ui.end_row();
                                ui.label("Last Out Chars");
                                ui.label(a.last_output_chars.to_string());
                                ui.end_row();
                                ui.end_row();
                                ui.end_row();
                                ui.label("Positive FB");
                                ui.label(a.pos_feedback_count.to_string());
                                ui.end_row();
                                ui.label("Negative FB");
                                ui.label(a.neg_feedback_count.to_string());
                                ui.end_row();
                            });
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Refresh /");
                            ui.add(egui::DragValue::new(&mut self.refresh_interval).range(1..=200));
                            ui.label("turns");
                        });
                        if ui.button("Update Now").clicked() {
                            action = Action::RefreshAnalytics;
                        }
                    });
            });

        // ── Central chat panel ────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("chat_scroll")
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for entry in &self.chat_history {
                        let (prefix, color) = if entry.is_user {
                            ("User:", egui::Color32::from_rgb(100, 180, 255))
                        } else {
                            ("Model:", egui::Color32::from_rgb(100, 220, 130))
                        };
                        ui.horizontal_wrapped(|ui| {
                            ui.colored_label(color, prefix);
                            ui.add(egui::Label::new(&entry.text).selectable(true).wrap());
                        });
                        ui.add_space(3.0);
                    }
                });
        });

        // The background worker exclusively owns AppHandle during generation.
        // Reject all actions that require that handle rather than blocking the UI on it.
        if is_generating && !matches!(action, Action::None) {
            self.status = "Generation in progress — wait for the current response.".to_string();
            action = Action::None;
        }

        // ── Deferred action dispatch ──────────────────────────────────────
        match action {
            Action::None => {}
            Action::New => self.do_new(),
            Action::NewConversation => self.do_new_conversation(),
            Action::Save => self.do_save(),
            Action::Send => self.start_generation(),
            Action::Feedback(id, s) => self.do_feedback(id, s),
            Action::RefreshAnalytics => {
                if let Some(handle) = self.handle.as_ref() {
                    self.analytics = handle.get_analytics();
                }
            }
            Action::LoadPath(path) => self.do_load(path),

            // File dialogs — blocking native dialog; runs after frame is rendered
            Action::OpenLoadDialog => {
                if let Some(path) = open_model_dialog(&model_path_str) {
                    self.do_load(path);
                }
            }
            Action::OpenSaveDialog => {
                if let Some(path) = save_model_dialog(&model_path_str) {
                    self.do_save_as(path);
                }
            }
        }

        if self.is_generating() {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
    }
}
