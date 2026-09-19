/// Phase 4: SQLite conversation log (§17).
///
/// Tables:
///   turns            — one row per conversation turn
///   trace_steps      — one row per DecisionStep in a turn's TurnTrace
///   feedback_events  — one row per ±1 feedback application
///   snapshots        — model state snapshots
use rusqlite::{Connection, Result, params};

use crate::feedback::{FeedbackEvent, FeedbackSign, FeedbackSource};
use crate::trace::{DecisionStep, RouteKind, TurnTrace};

pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open (or create) a SQLite database at `path`.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.create_tables()?;
        Ok(db)
    }

    /// Open an in-memory database for testing.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self { conn };
        db.create_tables()?;
        Ok(db)
    }

    /// §29: add route_provenance columns to trace_steps if they don't exist yet.
    fn migrate_trace_steps(&self) {
        // ALTER TABLE ADD COLUMN fails if the column already exists; ignore that error.
        let _ = self.conn.execute_batch(
            "ALTER TABLE trace_steps ADD COLUMN rsrc_is_chunk INTEGER NOT NULL DEFAULT 0;
             ALTER TABLE trace_steps ADD COLUMN rsrc_raw      INTEGER NOT NULL DEFAULT 0;
             ALTER TABLE trace_steps ADD COLUMN route_kind    INTEGER NOT NULL DEFAULT 0;"
        );
    }

    fn create_tables(&self) -> Result<()> {
        self.conn.execute_batch("
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS turns (
                turn_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id      INTEGER NOT NULL,
                input_text    TEXT NOT NULL,
                output_text   TEXT NOT NULL,
                decision_count INTEGER NOT NULL,
                state_tick    INTEGER NOT NULL,
                created_at    INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trace_steps (
                step_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id           INTEGER NOT NULL REFERENCES turns(turn_id),
                trace_id          INTEGER NOT NULL,
                step_index        INTEGER NOT NULL,
                unit_is_chunk     INTEGER NOT NULL,
                unit_raw          INTEGER NOT NULL,
                ctx_is_chunk      INTEGER NOT NULL,
                ctx_raw           INTEGER NOT NULL,
                score             REAL    NOT NULL,
                rsrc_is_chunk     INTEGER NOT NULL DEFAULT 0,
                rsrc_raw          INTEGER NOT NULL DEFAULT 0,
                route_kind        INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS feedback_events (
                feedback_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id       INTEGER NOT NULL REFERENCES turns(turn_id),
                trace_id      INTEGER NOT NULL,
                timestamp     INTEGER NOT NULL,
                sign          INTEGER NOT NULL,
                magnitude     REAL    NOT NULL,
                source        TEXT    NOT NULL,
                note          TEXT
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id       INTEGER,
                state_tick    INTEGER NOT NULL,
                fingerprint   TEXT    NOT NULL,
                json_blob     TEXT    NOT NULL,
                created_at    INTEGER NOT NULL
            );
        ")?;
        self.migrate_trace_steps();
        Ok(())
    }

    // ── Turn insertion ────────────────────────────────────────────────────

    /// Insert a completed turn; returns the new turn_id.
    pub fn insert_turn(&self, trace: &TurnTrace) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO turns (trace_id, input_text, output_text, decision_count, state_tick, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                trace.trace_id as i64,
                trace.input_text,
                // Store emitted-only text (dialogue output) not full seed+emitted.
                trace.emitted_text,
                trace.decision_count as i64,
                trace.state_before_tick as i64,
                trace.created_at as i64,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Insert all DecisionSteps for a turn.
    pub fn insert_trace_steps(&self, turn_id: i64, trace: &TurnTrace) -> Result<()> {
        for step in &trace.decision_steps {
            self.conn.execute(
                "INSERT INTO trace_steps
                 (turn_id, trace_id, step_index, unit_is_chunk, unit_raw, ctx_is_chunk, ctx_raw, score,
                  rsrc_is_chunk, rsrc_raw, route_kind)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    turn_id,
                    trace.trace_id as i64,
                    step.step_index as i64,
                    step.unit.is_chunk() as i64,
                    step.unit.raw() as i64,
                    step.context.is_chunk() as i64,
                    step.context.raw() as i64,
                    step.score,
                    step.route_source.is_chunk() as i64,
                    step.route_source.raw() as i64,
                    route_kind_to_i64(step.route_kind),
                ],
            )?;
        }
        Ok(())
    }

    // ── Feedback insertion ────────────────────────────────────────────────

    /// Insert a feedback event; returns the new feedback_id.
    pub fn insert_feedback_event(&self, ev: &FeedbackEvent) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO feedback_events (turn_id, trace_id, timestamp, sign, magnitude, source, note)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                ev.turn_id,
                ev.trace_id as i64,
                ev.timestamp as i64,
                ev.sign.as_i8() as i64,
                ev.magnitude,
                ev.source.as_str(),
                ev.note.as_deref(),
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ── Snapshot insertion / retrieval ────────────────────────────────────

    /// Store a model snapshot; returns the new snapshot_id.
    pub fn insert_snapshot(
        &self,
        turn_id: Option<i64>,
        state_tick: u64,
        fingerprint: &str,
        json_blob: &str,
        created_at: u64,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO snapshots (turn_id, state_tick, fingerprint, json_blob, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                turn_id,
                state_tick as i64,
                fingerprint,
                json_blob,
                created_at as i64,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Load the JSON blob for a snapshot by ID.
    pub fn load_snapshot_json(&self, snapshot_id: i64) -> Result<String> {
        self.conn.query_row(
            "SELECT json_blob FROM snapshots WHERE snapshot_id = ?1",
            params![snapshot_id],
            |row| row.get(0),
        )
    }

    /// Load the most recent snapshot JSON blob.
    pub fn load_latest_snapshot_json(&self) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT json_blob FROM snapshots ORDER BY snapshot_id DESC LIMIT 1"
        )?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    // ── History query ─────────────────────────────────────────────────────

    /// Return the last `limit` turns as (turn_id, input, output, decision_count) rows.
    pub fn query_turns(&self, limit: usize) -> Result<Vec<TurnRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT turn_id, input_text, output_text, decision_count, created_at
             FROM turns ORDER BY turn_id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(TurnRow {
                turn_id: row.get(0)?,
                input_text: row.get(1)?,
                output_text: row.get(2)?,
                decision_count: row.get::<_, i64>(3)? as usize,
                created_at: row.get::<_, i64>(4)? as u64,
            })
        })?;
        rows.collect()
    }

    /// Return feedback events for a given turn.
    pub fn query_feedback_for_turn(&self, turn_id: i64) -> Result<Vec<FeedbackRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT feedback_id, sign, magnitude, source, note, timestamp
             FROM feedback_events WHERE turn_id = ?1 ORDER BY feedback_id"
        )?;
        let rows = stmt.query_map(params![turn_id], |row| {
            let sign_i: i64 = row.get(1)?;
            let source_s: String = row.get(3)?;
            Ok(FeedbackRow {
                feedback_id: row.get(0)?,
                sign: if sign_i >= 0 { FeedbackSign::Positive } else { FeedbackSign::Negative },
                magnitude: row.get(2)?,
                source: FeedbackSource::from_str(&source_s),
                note: row.get(4)?,
                timestamp: row.get::<_, i64>(5)? as u64,
            })
        })?;
        rows.collect()
    }

    /// Return the DecisionSteps for a given turn_id.
    pub fn query_steps_for_turn(&self, turn_id: i64) -> Result<Vec<StepRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT step_index, unit_is_chunk, unit_raw, ctx_is_chunk, ctx_raw, score,
                    rsrc_is_chunk, rsrc_raw, route_kind
             FROM trace_steps WHERE turn_id = ?1 ORDER BY step_index"
        )?;
        let rows = stmt.query_map(params![turn_id], |row| {
            Ok(StepRow {
                step_index: row.get::<_, i64>(0)? as usize,
                unit_is_chunk: row.get::<_, i64>(1)? != 0,
                unit_raw: row.get::<_, i64>(2)? as u32,
                ctx_is_chunk: row.get::<_, i64>(3)? != 0,
                ctx_raw: row.get::<_, i64>(4)? as u32,
                score: row.get(5)?,
                rsrc_is_chunk: row.get::<_, i64>(6)? != 0,
                rsrc_raw: row.get::<_, i64>(7)? as u32,
                route_kind: row.get::<_, i64>(8)?,
            })
        })?;
        rows.collect()
    }

    /// Convert a stored StepRow back to a DecisionStep.
    pub fn step_row_to_decision(row: &StepRow) -> DecisionStep {
        use crate::units::UnitId;
        let unit = if row.unit_is_chunk { UnitId::chunk(row.unit_raw) } else { UnitId::primitive(row.unit_raw) };
        let context = if row.ctx_is_chunk { UnitId::chunk(row.ctx_raw) } else { UnitId::primitive(row.ctx_raw) };
        let route_source = if row.rsrc_is_chunk { UnitId::chunk(row.rsrc_raw) } else { UnitId::primitive(row.rsrc_raw) };
        DecisionStep {
            step_index: row.step_index,
            unit,
            context,
            route_source,
            score: row.score,
            route_kind: route_kind_from_i64(row.route_kind),
        }
    }
}

// ── Plain data structs for query results ──────────────────────────────────

#[derive(Debug, Clone)]
pub struct TurnRow {
    pub turn_id: i64,
    pub input_text: String,
    pub output_text: String,
    pub decision_count: usize,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub struct FeedbackRow {
    pub feedback_id: i64,
    pub sign: FeedbackSign,
    pub magnitude: f64,
    pub source: FeedbackSource,
    pub note: Option<String>,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct StepRow {
    pub step_index: usize,
    pub unit_is_chunk: bool,
    pub unit_raw: u32,
    pub ctx_is_chunk: bool,
    pub ctx_raw: u32,
    pub score: f64,
    /// §29: route provenance fields.
    pub rsrc_is_chunk: bool,
    pub rsrc_raw: u32,
    pub route_kind: i64,
}

fn route_kind_to_i64(k: RouteKind) -> i64 {
    match k {
        RouteKind::Direct       => 0,
        RouteKind::RepFallback  => 1,
        RouteKind::RecallBridge => 2,
        RouteKind::Generalize   => 3,
    }
}

fn route_kind_from_i64(v: i64) -> RouteKind {
    match v {
        1 => RouteKind::RepFallback,
        2 => RouteKind::RecallBridge,
        3 => RouteKind::Generalize,
        _ => RouteKind::Direct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{TurnTrace, DecisionStep};
    use crate::units::UnitId;
    use crate::feedback::FeedbackSource;

    fn dummy_trace(id: u64) -> TurnTrace {
        let ctx = UnitId::primitive(2);
        let step = DecisionStep {
            step_index: 0,
            unit: UnitId::primitive(1),
            context: ctx,
            route_source: ctx,
            score: 0.9,
            route_kind: crate::trace::RouteKind::Direct,
        };
        TurnTrace::new(id, 1, "seed".into(), "input".into(), "output".into(), String::new(), vec![step], 1000)
    }

    #[test]
    fn test_create_tables() {
        Database::open_in_memory().unwrap();
    }

    #[test]
    fn test_insert_and_query_turn() {
        let db = Database::open_in_memory().unwrap();
        let trace = dummy_trace(42);
        let turn_id = db.insert_turn(&trace).unwrap();
        assert_eq!(turn_id, 1);
        let rows = db.query_turns(10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].input_text, "input");
        assert_eq!(rows[0].decision_count, 1);
    }

    #[test]
    fn test_insert_trace_steps() {
        let db = Database::open_in_memory().unwrap();
        let trace = dummy_trace(1);
        let turn_id = db.insert_turn(&trace).unwrap();
        db.insert_trace_steps(turn_id, &trace).unwrap();
        let steps = db.query_steps_for_turn(turn_id).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].step_index, 0);
        assert!((steps[0].score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_insert_feedback_event() {
        let db = Database::open_in_memory().unwrap();
        let trace = dummy_trace(1);
        let turn_id = db.insert_turn(&trace).unwrap();
        let ev = FeedbackEvent {
            feedback_id: None,
            turn_id,
            trace_id: 1,
            timestamp: 999,
            sign: FeedbackSign::Positive,
            magnitude: 1.0,
            source: FeedbackSource::User,
            note: None,
        };
        let fb_id = db.insert_feedback_event(&ev).unwrap();
        assert_eq!(fb_id, 1);
        let fb_rows = db.query_feedback_for_turn(turn_id).unwrap();
        assert_eq!(fb_rows.len(), 1);
        assert_eq!(fb_rows[0].sign, FeedbackSign::Positive);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let db = Database::open_in_memory().unwrap();
        let sid = db.insert_snapshot(None, 42, "fp-abc", "{}", 1000).unwrap();
        let json = db.load_snapshot_json(sid).unwrap();
        assert_eq!(json, "{}");
        let latest = db.load_latest_snapshot_json().unwrap();
        assert_eq!(latest, Some("{}".to_string()));
    }
}
