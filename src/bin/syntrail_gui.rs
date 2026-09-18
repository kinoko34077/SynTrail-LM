/// SynTrail-LM GUI — eframe/egui simple research interface.
///
/// Build:  cargo build --features gui --bin syntrail-gui
/// Run:    cargo run   --features gui --bin syntrail-gui
use eframe::egui;
use std::path::PathBuf;
use syntrail_lm::app::{Analytics, AppHandle};
use syntrail_lm::db::TurnRow;
use syntrail_lm::feedback::FeedbackSign;

const DEFAULT_MODEL: &str = "model.json";
const DEFAULT_HISTORY: &str = "history.sqlite";
const DEFAULT_REFRESH: u32 = 5;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([980.0, 660.0])
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

// ── Font setup (Japanese / CJK) ───────────────────────────────────────────

fn setup_fonts(ctx: &egui::Context) {
    let candidates: &[&str] = &[
        // Windows
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\msmincho.ttc",
        // macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        // Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
    ];

    let Some(font_data) = candidates.iter().find_map(|p| std::fs::read(p).ok()) else {
        return; // no CJK font found; Latin-only fallback
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "cjk".to_owned(),
        egui::FontData::from_owned(font_data),
    );
    // Append after the default Latin font so Latin glyphs stay sharp
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("cjk".to_owned());
    }
    ctx.set_fonts(fonts);
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
    Load,
    Save,
    Send,
    Feedback(i64, FeedbackSign),
    RefreshAnalytics,
    LoadPath(PathBuf),
}

// ── App ───────────────────────────────────────────────────────────────────

struct SynTrailApp {
    handle: AppHandle,
    input: String,
    chat_history: Vec<ChatEntry>,
    feedback_state: FeedbackState,
    analytics: Analytics,
    refresh_interval: u32,
    turns_since_refresh: u32,
    status: String,
    model_path_input: String,
}

impl SynTrailApp {
    fn new() -> Self {
        let model_path = PathBuf::from(DEFAULT_MODEL);
        let model_path_input = model_path.display().to_string();
        let handle = AppHandle::new(model_path, DEFAULT_HISTORY)
            .unwrap_or_else(|e| panic!("Failed to init AppHandle: {e}"));
        let chat_history = load_history_entries(&handle);
        let analytics = handle.get_analytics();
        Self {
            handle,
            input: String::new(),
            chat_history,
            feedback_state: FeedbackState::None,
            analytics,
            refresh_interval: DEFAULT_REFRESH,
            turns_since_refresh: 0,
            status: "Ready".to_string(),
            model_path_input,
        }
    }

    fn send_message(&mut self) {
        let input = self.input.trim().to_string();
        if input.is_empty() { return; }
        self.input.clear();
        // Previous pending feedback is silently skipped — no button press needed.
        self.feedback_state = FeedbackState::None;
        self.chat_history.push(ChatEntry { is_user: true, text: input.clone() });

        match self.handle.generate_turn(&input) {
            Ok((turn_id, output)) => {
                let a = self.handle.get_analytics();
                self.status = format!("Turn {}  |  {} decisions", turn_id, a.last_decision_count);
                self.chat_history.push(ChatEntry { is_user: false, text: output });
                self.feedback_state = FeedbackState::Pending(turn_id);
                self.turns_since_refresh += 1;
                if self.turns_since_refresh >= self.refresh_interval {
                    self.analytics = a;
                    self.turns_since_refresh = 0;
                }
            }
            Err(e) => {
                self.chat_history.pop();
                self.status = format!("Generation failed: {e}");
            }
        }
    }

    fn do_feedback(&mut self, turn_id: i64, sign: FeedbackSign) {
        match self.handle.apply_feedback(turn_id, sign) {
            Ok(()) => {
                let label = match sign { FeedbackSign::Positive => "○", FeedbackSign::Negative => "×" };
                self.status = format!("Feedback {label} → turn {turn_id}");
                self.analytics = self.handle.get_analytics();
            }
            Err(e) => self.status = format!("Feedback failed: {e}"),
        }
        self.feedback_state = FeedbackState::None;
    }

    fn do_load(&mut self, path: PathBuf) {
        match self.handle.load_model(&path) {
            Ok(()) => {
                self.model_path_input = path.display().to_string();
                self.analytics = self.handle.get_analytics();
                self.status = format!("Loaded: {}", path.display());
            }
            Err(e) => self.status = format!("Load failed: {e}"),
        }
    }

    fn do_save(&mut self) {
        match self.handle.save_model() {
            Ok(()) => self.status = format!("Saved: {}", self.handle.model_path.display()),
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    fn do_new(&mut self) {
        self.handle.reset_model();
        self.chat_history.clear();
        self.feedback_state = FeedbackState::None;
        self.analytics = self.handle.get_analytics();
        self.status = "New model (unsaved)".to_string();
    }
}

fn load_history_entries(handle: &AppHandle) -> Vec<ChatEntry> {
    let rows: Vec<TurnRow> = handle.history(60).unwrap_or_default();
    let mut v = Vec::with_capacity(rows.len() * 2);
    for row in rows.into_iter().rev() {
        v.push(ChatEntry { is_user: true,  text: row.input_text });
        v.push(ChatEntry { is_user: false, text: row.output_text });
    }
    v
}

// ── egui App ──────────────────────────────────────────────────────────────

impl eframe::App for SynTrailApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut action = Action::None;

        // ── File drag-and-drop (model .json) ─────────────────────────────
        ctx.input(|i| {
            for file in &i.raw.dropped_files {
                if let Some(path) = &file.path {
                    if path.extension().map_or(false, |e| e == "json") {
                        action = Action::LoadPath(path.clone());
                    }
                }
            }
        });

        // ── Top toolbar ───────────────────────────────────────────────────
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("New").clicked()  { action = Action::New; }
                if ui.button("Load").clicked() { action = Action::Load; }
                if ui.button("Save").clicked() { action = Action::Save; }
                ui.separator();
                ui.label("Model:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.model_path_input)
                        .desired_width(180.0),
                );
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
                // Status + feedback buttons (○/× — no skip button; just send next msg)
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&self.status).small());
                    if let FeedbackState::Pending(tid) = self.feedback_state {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("  ×  ")
                                .on_hover_text("Negative feedback (−1)")
                                .clicked()
                            {
                                action = Action::Feedback(tid, FeedbackSign::Negative);
                            }
                            if ui.button("  ○  ")
                                .on_hover_text("Positive feedback (+1)")
                                .clicked()
                            {
                                action = Action::Feedback(tid, FeedbackSign::Positive);
                            }
                        });
                    }
                });
                ui.separator();

                // Input + Send
                ui.horizontal(|ui| {
                    let avail = ui.available_width();
                    let response = ui.add(
                        egui::TextEdit::multiline(&mut self.input)
                            .desired_rows(3)
                            .desired_width(avail - 82.0),
                    );
                    let enter_pressed = response.has_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                    if ui.add_sized([72.0, 60.0], egui::Button::new("Send")).clicked()
                        || enter_pressed
                    {
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
                        egui::Grid::new("ag").num_columns(2).striped(true).show(ui, |ui| {
                            ui.label("Generation");      ui.label(a.generation.to_string());          ui.end_row();
                            ui.label("Tick");            ui.label(a.tick.to_string());                ui.end_row();
                            ui.end_row(); ui.end_row();
                            ui.label("Primitive");       ui.label(a.primitive_count.to_string());     ui.end_row();
                            ui.label("Chunk");           ui.label(a.chunk_count.to_string());         ui.end_row();
                            ui.label("HOT");             ui.label(a.hot_count.to_string());           ui.end_row();
                            ui.label("SLEEP");           ui.label(a.sleep_count.to_string());         ui.end_row();
                            ui.end_row(); ui.end_row();
                            ui.label("Association");     ui.label(a.association_count.to_string());   ui.end_row();
                            ui.label("Route (edges)");   ui.label(a.edge_count.to_string());          ui.end_row();
                            ui.end_row(); ui.end_row();
                            ui.label("T0");              ui.label(a.t0_count.to_string());            ui.end_row();
                            ui.label("T1");              ui.label(a.t1_count.to_string());            ui.end_row();
                            ui.label("T2");              ui.label(a.t2_count.to_string());            ui.end_row();
                            ui.end_row(); ui.end_row();
                            ui.label("Avg Exp Len");     ui.label(format!("{:.2}", a.avg_expanded_length)); ui.end_row();
                            ui.end_row(); ui.end_row();
                            ui.label("Total Chars");     ui.label(a.total_characters.to_string());   ui.end_row();
                            ui.label("Total Decisions"); ui.label(a.total_decisions.to_string());    ui.end_row();
                            ui.label("dpc");             ui.label(format!("{:.6}", a.dpc));          ui.end_row();
                            ui.end_row(); ui.end_row();
                            ui.label("Last Decisions");  ui.label(a.last_decision_count.to_string()); ui.end_row();
                            ui.label("Last Out Len");    ui.label(a.last_output_len.to_string());    ui.end_row();
                            ui.end_row(); ui.end_row();
                            ui.label("Positive FB");     ui.label(a.pos_feedback_count.to_string()); ui.end_row();
                            ui.label("Negative FB");     ui.label(a.neg_feedback_count.to_string()); ui.end_row();
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

        // ── Central chat panel (scrollable, selectable text) ─────────────
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
                            ui.add(
                                egui::Label::new(&entry.text)
                                    .selectable(true)
                                    .wrap(),
                            );
                        });
                        ui.add_space(3.0);
                    }
                });
        });

        // ── Deferred action dispatch ──────────────────────────────────────
        match action {
            Action::None             => {}
            Action::New              => self.do_new(),
            Action::Load             => {
                let path = PathBuf::from(&self.model_path_input);
                self.do_load(path);
            }
            Action::Save             => self.do_save(),
            Action::Send             => self.send_message(),
            Action::Feedback(id, s)  => self.do_feedback(id, s),
            Action::RefreshAnalytics => self.analytics = self.handle.get_analytics(),
            Action::LoadPath(path)   => self.do_load(path),
        }
    }
}
