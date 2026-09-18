/// Application API — shared by CLI and GUI (§11).
///
/// AppHandle owns a Session (model + history DB) and a model file path.
/// CLI and GUI both call the methods here; neither duplicates Core logic.
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::db::{Database, TurnRow};
use crate::feedback::{FeedbackSign, FeedbackSource};
use crate::model::ModelState;
use crate::persistence;
use crate::session::Session;

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
            pos_feedback_count: 0,
            neg_feedback_count: 0,
        })
    }

    /// Run one conversation turn. Returns `(turn_id, output_text)`.
    pub fn generate_turn(&mut self, input: &str) -> Result<(i64, String), Box<dyn Error>> {
        let (turn_id, output, trace) = self.session.turn(input)?;
        self.last_decision_count = trace.decision_count;
        self.last_output_len = output.len();
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

    /// Save model to `self.model_path`.
    pub fn save_model(&self) -> io::Result<()> {
        persistence::save(&self.session.model, &self.model_path)
    }

    /// Load model from `path` and update `self.model_path`.
    pub fn load_model(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        self.session.model = persistence::load(path)?;
        self.model_path = path.to_path_buf();
        self.last_decision_count = 0;
        self.last_output_len = 0;
        Ok(())
    }

    /// Replace model with a blank one, reset counters.
    pub fn reset_model(&mut self) {
        self.session.model = ModelState::new();
        self.session.turn_count = 0;
        self.last_decision_count = 0;
        self.last_output_len = 0;
        self.pos_feedback_count = 0;
        self.neg_feedback_count = 0;
    }

    /// Snapshot the current model state into a read-only Analytics value.
    pub fn get_analytics(&self) -> Analytics {
        let m = &self.session.model;
        let mut tier = [0u32; 3];
        let mut total_exp: u64 = 0;
        let chunk_count = m.chunk_count();
        for c in m.chunks.iter_all() {
            tier[c.tier as usize] += 1;
            total_exp += c.expanded_length as u64;
        }
        Analytics {
            generation: self.session.turn_count,
            tick: m.tick,
            primitive_count: m.primitive_count(),
            chunk_count,
            hot_count: m.hot_chunk_count(),
            sleep_count: m.sleep_chunk_count(),
            association_count: m.association_count(),
            edge_count: m.edge_count(),
            t0_count: tier[0],
            t1_count: tier[1],
            t2_count: tier[2],
            avg_expanded_length: if chunk_count > 0 { total_exp as f64 / chunk_count as f64 } else { 0.0 },
            total_characters: m.metrics.total_characters,
            total_decisions: m.metrics.total_decisions,
            dpc: m.metrics.decision_per_character(),
            last_decision_count: self.last_decision_count,
            last_output_len: self.last_output_len,
            pos_feedback_count: self.pos_feedback_count,
            neg_feedback_count: self.neg_feedback_count,
        }
    }

    /// Return the last `limit` turns from history (newest first).
    pub fn history(&self, limit: usize) -> Result<Vec<TurnRow>, rusqlite::Error> {
        self.session.history(limit)
    }
}
