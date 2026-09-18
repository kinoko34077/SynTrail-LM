/// Phase 8-9: Session — owns ModelState + Database + Config.
///
/// Session is the public façade for the chat workflow:
///   turn(input) → (output, trace)
///   feedback(turn_id, sign) → apply ±1 credit to past turn
///   snapshot() → save current model state to DB
///   restore(snapshot_id) → load model state from DB snapshot
use std::path::Path;

use crate::config::Config;
use crate::db::{Database, TurnRow};
use crate::feedback::{distribute, FeedbackEvent, FeedbackSign, FeedbackSource};
use crate::model::ModelState;
use crate::persistence;
use crate::trace::{TurnTrace, now_secs};

pub struct Session {
    pub model: ModelState,
    pub db: Database,
    pub config: Config,
    /// Turn counter (total turns processed this session).
    pub turn_count: u64,
}

impl Session {
    /// Create a new session with an in-memory database.
    pub fn new_in_memory(config: Config) -> Result<Self, rusqlite::Error> {
        Ok(Self {
            model: ModelState::new(),
            db: Database::open_in_memory()?,
            config,
            turn_count: 0,
        })
    }

    /// Open a persistent session at `db_path`, loading the latest model snapshot
    /// if one exists.
    pub fn open(db_path: &str, config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let db = Database::open(db_path)?;
        let model = if let Some(json) = db.load_latest_snapshot_json()? {
            let snap: persistence::ModelSnapshot = serde_json::from_str(&json)?;
            persistence::from_snapshot(snap)
        } else {
            ModelState::new()
        };
        Ok(Self { model, db, config, turn_count: 0 })
    }

    // ── Turn ─────────────────────────────────────────────────────────────

    /// Process one conversation turn:
    /// 1. generate_with_trace (frozen — &self)
    /// 2. expose input to model (mutating)
    /// 3. log turn + steps to DB
    /// 4. auto-snapshot every `snapshot_interval_turns` turns
    ///
    /// Returns (turn_id, output_text, trace).
    pub fn turn(
        &mut self,
        input: &str,
    ) -> Result<(i64, String, TurnTrace), Box<dyn std::error::Error>> {
        let trace_id = self.model.alloc_trace_id();
        let max_units = self.config.default_max_units;

        let (output, trace) = self.model.generate_with_trace(input, input, max_units, trace_id);

        // Expose input to model (phase 3: post-turn exposure of INPUT not output)
        self.model.expose(input);

        // Log to DB
        let turn_id = self.db.insert_turn(&trace)?;
        self.db.insert_trace_steps(turn_id, &trace)?;

        self.turn_count += 1;

        // Auto-snapshot
        if self.turn_count % self.config.snapshot_interval_turns == 0 {
            self.snapshot_to_db(Some(turn_id))?;
        }

        Ok((turn_id, output, trace))
    }

    // ── Feedback ─────────────────────────────────────────────────────────

    /// Apply signed feedback to a past turn.
    ///
    /// Retrieves the DecisionSteps from the DB, distributes uniform credits,
    /// applies them to the model, and logs the FeedbackEvent.
    pub fn feedback(
        &mut self,
        turn_id: i64,
        sign: FeedbackSign,
        source: FeedbackSource,
        note: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let step_rows = self.db.query_steps_for_turn(turn_id)?;
        if step_rows.is_empty() {
            return Ok(());
        }

        let mock_trace_id = turn_id as u64;

        let dummy_trace = crate::trace::TurnTrace::new(
            mock_trace_id,
            0,
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            step_rows.iter().map(Database::step_row_to_decision).collect(),
            now_secs(),
        );

        let credits = distribute(sign, &dummy_trace, &self.config);
        let decay = self.config.feedback_decay;
        self.model.apply_feedback_to_trace(&dummy_trace, &credits, decay);

        // Log feedback event
        let ev = FeedbackEvent {
            feedback_id: None,
            turn_id,
            trace_id: mock_trace_id,
            timestamp: now_secs(),
            sign,
            magnitude: self.config.feedback_base_magnitude,
            source,
            note,
        };
        self.db.insert_feedback_event(&ev)?;

        Ok(())
    }

    // ── Snapshot ─────────────────────────────────────────────────────────

    /// Save the current model state to the DB; returns snapshot_id.
    pub fn snapshot_to_db(
        &self,
        turn_id: Option<i64>,
    ) -> Result<i64, Box<dyn std::error::Error>> {
        let snap = persistence::to_snapshot(&self.model);
        let json = serde_json::to_string(&snap)?;
        let fp = self.model.state_fingerprint();
        let sid = self.db.insert_snapshot(
            turn_id,
            self.model.tick,
            &fp,
            &json,
            now_secs(),
        )?;
        Ok(sid)
    }

    /// Save a snapshot to a JSON file on disk (for `syntrail snapshot` CLI).
    pub fn snapshot_to_file(&self, path: &Path) -> std::io::Result<()> {
        persistence::save(&self.model, path)
    }

    /// Restore model state from a snapshot_id in the DB.
    pub fn restore_from_db(
        &mut self,
        snapshot_id: i64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.db.load_snapshot_json(snapshot_id)?;
        let snap: persistence::ModelSnapshot = serde_json::from_str(&json)?;
        self.model = persistence::from_snapshot(snap);
        Ok(())
    }

    /// Restore model state from a JSON file (for `syntrail restore` CLI).
    pub fn restore_from_file(&mut self, path: &Path) -> std::io::Result<()> {
        self.model = persistence::load(path)?;
        Ok(())
    }

    // ── History ───────────────────────────────────────────────────────────

    pub fn history(&self, limit: usize) -> Result<Vec<TurnRow>, rusqlite::Error> {
        self.db.query_turns(limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session() -> Session {
        let mut s = Session::new_in_memory(Config::default_v02()).unwrap();
        // Pre-train so generation produces non-trivial output
        for _ in 0..20 { s.model.train("hello world"); }
        s
    }

    #[test]
    fn test_turn_returns_output() {
        let mut s = make_session();
        let (turn_id, output, trace) = s.turn("hello").unwrap();
        assert!(turn_id > 0);
        assert!(!output.is_empty());
        assert_eq!(trace.input_text, "hello");
    }

    #[test]
    fn test_turn_increments_counter() {
        let mut s = make_session();
        s.turn("a").unwrap();
        s.turn("b").unwrap();
        assert_eq!(s.turn_count, 2);
    }

    #[test]
    fn test_history_records_turns() {
        let mut s = make_session();
        s.turn("hello").unwrap();
        s.turn("world").unwrap();
        let history = s.history(10).unwrap();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_feedback_positive() {
        let mut s = make_session();
        let (turn_id, _, trace) = s.turn("hello").unwrap();
        if trace.decision_count > 0 {
            s.feedback(turn_id, FeedbackSign::Positive, FeedbackSource::User, None).unwrap();
            let fb_rows = s.db.query_feedback_for_turn(turn_id).unwrap();
            assert_eq!(fb_rows.len(), 1);
            assert_eq!(fb_rows[0].sign, FeedbackSign::Positive);
        }
    }

    #[test]
    fn test_snapshot_to_db_and_restore() {
        let mut s = make_session();
        s.turn("hello").unwrap();
        let tick_before = s.model.tick;
        let sid = s.snapshot_to_db(None).unwrap();
        // Expose more data to change state
        s.model.train("world world");
        assert_ne!(s.model.tick, tick_before);
        // Restore to snapshot
        s.restore_from_db(sid).unwrap();
        assert_eq!(s.model.tick, tick_before);
    }

    #[test]
    fn test_auto_snapshot_triggers() {
        let mut s = Session::new_in_memory(Config {
            snapshot_interval_turns: 2,
            ..Config::default_v02()
        }).unwrap();
        for _ in 0..20 { s.model.train("hello world"); }
        s.turn("a").unwrap();
        s.turn("b").unwrap(); // snapshot should fire here
        let latest = s.db.load_latest_snapshot_json().unwrap();
        assert!(latest.is_some(), "auto-snapshot should have fired at turn 2");
    }
}
