/// Application API — shared by CLI, GUI, and Trainer (§11).
///
/// Free functions (`save_model_file`, `load_model_file`, `model_analytics`)
/// contain the shared logic so both AppHandle and the Trainer worker can
/// call the same implementation without duplication (§5).
use std::error::Error;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::db::{Database, TurnRow};
use crate::feedback::{FeedbackSign, FeedbackSource};
use crate::model::ModelState;
use crate::persistence;
use crate::session::Session;
use crate::trace::now_secs;

// ── Shared free functions (used by both AppHandle and Trainer) ────────────

/// Extension-aware save: `.stm` → binary; `.db`/`.sqlite` → snapshot row; anything else → JSON.
pub fn save_model_file(model: &ModelState, path: &Path) -> Result<(), Box<dyn Error>> {
    let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("stm") => { persistence::save_binary(model, path)?; }
        Some("db") | Some("sqlite") => {
            let db = Database::open(&path.to_string_lossy())?;
            let snap = persistence::to_snapshot(model);
            let json = serde_json::to_string(&snap)?;
            let fp = model.state_fingerprint();
            db.insert_snapshot(None, model.tick, &fp, &json, now_secs())?;
        }
        _ => { persistence::save(model, path)?; }
    }
    Ok(())
}

/// Extension-aware load: `.stm` → binary; `.db`/`.sqlite` → latest snapshot; anything else → JSON.
pub fn load_model_file(path: &Path) -> Result<ModelState, Box<dyn Error>> {
    let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("stm") => Ok(persistence::load_binary(path)?),
        Some("db") | Some("sqlite") => {
            let db = Database::open(&path.to_string_lossy())?;
            let json = db.load_latest_snapshot_json()?
                .ok_or("no snapshot found in database")?;
            let snap: persistence::ModelSnapshot = serde_json::from_str(&json)?;
            Ok(persistence::from_snapshot(snap))
        }
        _ => Ok(persistence::load(path)?),
    }
}

/// Build an Analytics snapshot from a ModelState alone.
/// AppHandle-specific counters (generation, last_decision_count, …) are set to 0;
/// the caller fills them in when needed.
pub fn model_analytics(model: &ModelState) -> Analytics {
    let mut tier = [0u32; 3];
    let mut total_exp: u64 = 0;
    let chunk_count = model.chunk_count();
    for c in model.chunks.iter_all() {
        tier[c.tier as usize] += 1;
        total_exp += c.expanded_length as u64;
    }
    Analytics {
        generation: 0,
        tick: model.tick,
        primitive_count: model.primitive_count(),
        chunk_count,
        hot_count: model.hot_chunk_count(),
        sleep_count: model.sleep_chunk_count(),
        association_count: model.association_count(),
        edge_count: model.edge_count(),
        t0_count: tier[0],
        t1_count: tier[1],
        t2_count: tier[2],
        avg_expanded_length: if chunk_count > 0 { total_exp as f64 / chunk_count as f64 } else { 0.0 },
        total_characters: model.metrics.total_characters,
        total_decisions: model.metrics.total_decisions,
        dpc: model.metrics.decision_per_character(),
        last_decision_count: 0,
        last_output_len: 0,
        last_output_chars: 0,
        pos_feedback_count: 0,
        neg_feedback_count: 0,
    }
}

// ── Analytics ─────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct Analytics {
    pub generation: u64,
    pub tick: u64,
    pub primitive_count: usize,
    pub chunk_count: usize,
    pub hot_count: usize,
    pub sleep_count: usize,
    pub association_count: usize,
    pub edge_count: usize,
    pub t0_count: u32,
    pub t1_count: u32,
    pub t2_count: u32,
    pub avg_expanded_length: f64,
    pub total_characters: u64,
    pub total_decisions: u64,
    pub dpc: f64,
    pub last_decision_count: usize,
    pub last_output_len: usize,
    pub last_output_chars: usize,
    pub pos_feedback_count: u64,
    pub neg_feedback_count: u64,
}

// ── AppHandle ─────────────────────────────────────────────────────────────

/// Shared application state. Owns the model (via Session) and the model file path.
/// History is stored in `history_db_path` (separate from the model file — §10).
pub struct AppHandle {
    pub session: Session,
    pub model_path: PathBuf,
    last_decision_count: usize,
    last_output_len: usize,
    last_output_chars: usize,
    pos_feedback_count: u64,
    neg_feedback_count: u64,
}

impl AppHandle {
    /// Create a new handle.
    /// Loads `model_path` if it exists; otherwise starts with an empty model.
    /// History is kept in `history_db_path` (SQLite).
    pub fn new(model_path: PathBuf, history_db_path: &str) -> Result<Self, Box<dyn Error>> {
        let db = Database::open(history_db_path)?;
        let model = if model_path.exists() {
            persistence::load(&model_path)?
        } else {
            ModelState::new()
        };
        let session = Session { model, db, config: Config::default_v04(), turn_count: 0 };
        Ok(Self {
            session,
            model_path,
            last_decision_count: 0,
            last_output_len: 0,
            last_output_chars: 0,
            pos_feedback_count: 0,
            neg_feedback_count: 0,
        })
    }

    /// Run one conversation turn. Returns `(turn_id, output_text)`.
    pub fn generate_turn(&mut self, input: &str) -> Result<(i64, String), Box<dyn Error>> {
        let (turn_id, output, trace) = self.session.turn(input)?;
        self.last_decision_count = trace.decision_count;
        self.last_output_len = output.len();
        self.last_output_chars = output.chars().count();
        Ok((turn_id, output))
    }

    /// Apply ±1 feedback to a past turn.
    pub fn apply_feedback(
        &mut self,
        turn_id: i64,
        sign: FeedbackSign,
    ) -> Result<(), Box<dyn Error>> {
        match sign {
            FeedbackSign::Positive => self.pos_feedback_count += 1,
            FeedbackSign::Negative => self.neg_feedback_count += 1,
        }
        self.session.feedback(turn_id, sign, FeedbackSource::User, None)
    }

    /// Quick-save to `self.model_path` (extension-aware).
    pub fn save_model(&self) -> Result<(), Box<dyn Error>> {
        self.save_model_to(&self.model_path.clone())
    }

    /// Save As: write to `path` (json or db/sqlite). Does NOT update `self.model_path`.
    pub fn save_model_to(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        save_model_file(&self.session.model, path)
    }

    /// Load model from `path` (json or db/sqlite). Alias for `load_model_from`.
    pub fn load_model(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        self.load_model_from(path)
    }

    /// Extension-aware load; updates `self.model_path` on success.
    pub fn load_model_from(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        self.session.model = load_model_file(path)?;
        self.model_path = path.to_path_buf();
        self.last_decision_count = 0;
        self.last_output_len = 0;
        self.last_output_chars = 0;
        Ok(())
    }

    /// Replace model with a blank one, reset counters.
    pub fn reset_model(&mut self) {
        self.session.model = ModelState::new();
        self.session.turn_count = 0;
        self.last_decision_count = 0;
        self.last_output_len = 0;
        self.last_output_chars = 0;
        self.pos_feedback_count = 0;
        self.neg_feedback_count = 0;
    }

    /// Snapshot the current model state into a read-only Analytics value.
    pub fn get_analytics(&self) -> Analytics {
        let mut a = model_analytics(&self.session.model);
        a.generation = self.session.turn_count;
        a.last_decision_count = self.last_decision_count;
        a.last_output_len = self.last_output_len;
        a.last_output_chars = self.last_output_chars;
        a.pos_feedback_count = self.pos_feedback_count;
        a.neg_feedback_count = self.neg_feedback_count;
        a
    }

    /// Return the last `limit` turns from history (newest first).
    pub fn history(&self, limit: usize) -> Result<Vec<TurnRow>, rusqlite::Error> {
        self.session.history(limit)
    }
}
