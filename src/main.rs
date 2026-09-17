/// SynTrail-LM CLI — v0.2
/// Commands: train / generate / inspect / evaluate / chat / feedback / snapshot / restore / history
use std::path::{Path, PathBuf};
use std::process;

use syntrail_lm::config::Config;
use syntrail_lm::feedback::{FeedbackSign, FeedbackSource};
use syntrail_lm::model::ModelState;
use syntrail_lm::persistence;
use syntrail_lm::session::Session;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", USAGE);
        process::exit(1);
    }

    match args[1].as_str() {
        "train"    => cmd_train(&args[2..]),
        "generate" => cmd_generate(&args[2..]),
        "inspect"  => cmd_inspect(&args[2..]),
        "evaluate" => cmd_evaluate(&args[2..]),
        "chat"     => cmd_chat(&args[2..]),
        "feedback" => cmd_feedback(&args[2..]),
        "snapshot" => cmd_snapshot(&args[2..]),
        "restore"  => cmd_restore(&args[2..]),
        "history"  => cmd_history(&args[2..]),
        _ => {
            eprintln!("Unknown command: {}\n{}", args[1], USAGE);
            process::exit(1);
        }
    }
}

const USAGE: &str = "\
Usage: syntrail <command> [options]

Commands:
  train      --input <file> [--model <path>]
  generate   --seed <text>  [--model <path>] [--max-units N]
  inspect    [--model <path>]
  evaluate   --input <file> [--model <path>]
  chat       --input <text> [--db <path>] [--model <path>]
  feedback   --turn-id <N> --sign <+|-|1|-1> [--db <path>]
  snapshot   [--db <path>] [--model <path>]
  restore    --snapshot-id <N> [--db <path>] [--model <path>]
  history    [--limit N] [--db <path>]
";

fn default_model_path() -> PathBuf { PathBuf::from("model.json") }
fn default_db_path() -> String { "syntrail.db".to_string() }

fn load_model(path: &Path) -> ModelState {
    if path.exists() {
        persistence::load(path).unwrap_or_else(|e| {
            eprintln!("Failed to load model from {}: {e}", path.display());
            process::exit(1);
        })
    } else {
        ModelState::new()
    }
}

fn save_model(model: &ModelState, path: &Path) {
    persistence::save(model, path).unwrap_or_else(|e| {
        eprintln!("Failed to save model to {}: {e}", path.display());
        process::exit(1);
    });
}

fn open_session(args: &[String]) -> Session {
    let db_path = flag_value(args, "--db").unwrap_or_else(default_db_path);
    let config = Config::default_v02();
    Session::open(&db_path, config).unwrap_or_else(|e| {
        eprintln!("Failed to open session at {db_path}: {e}");
        process::exit(1);
    })
}

// ── train ─────────────────────────────────────────────────────────────────

fn cmd_train(args: &[String]) {
    let input = flag_value(args, "--input").unwrap_or_else(|| {
        eprintln!("--input <file> required"); process::exit(1);
    });
    let model_path = flag_path(args, "--model").unwrap_or_else(default_model_path);
    let text = std::fs::read_to_string(&input).unwrap_or_else(|e| {
        eprintln!("Cannot read {input}: {e}"); process::exit(1);
    });
    let mut model = load_model(&model_path);
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    for (i, line) in lines.iter().enumerate() {
        model.expose(line);
        if (i + 1) % 1000 == 0 || i + 1 == n {
            eprintln!(
                "[{}/{}] primitives={} chunks={} edges={} dpc={:.4}",
                i + 1, n,
                model.primitive_count(), model.chunk_count(),
                model.edge_count(), model.metrics.decision_per_character()
            );
        }
    }
    save_model(&model, &model_path);
    println!(
        "Saved to {}  (primitives={} chunks={} edges={} dpc={:.4})",
        model_path.display(),
        model.primitive_count(), model.chunk_count(),
        model.edge_count(), model.metrics.decision_per_character()
    );
}

// ── generate ──────────────────────────────────────────────────────────────

fn cmd_generate(args: &[String]) {
    let seed = flag_value(args, "--seed").unwrap_or_default();
    let max_units: usize = flag_value(args, "--max-units")
        .and_then(|v| v.parse().ok()).unwrap_or(50);
    let model_path = flag_path(args, "--model").unwrap_or_else(default_model_path);
    let model = load_model(&model_path);
    let output = model.generate(&seed, max_units);
    println!("{output}");
}

// ── inspect ───────────────────────────────────────────────────────────────

fn cmd_inspect(args: &[String]) {
    let model_path = flag_path(args, "--model").unwrap_or_else(default_model_path);
    let model = load_model(&model_path);
    println!("Model file    : {}", model_path.display());
    println!("Primitives    : {}", model.primitive_count());
    println!("Chunks        : {}", model.chunk_count());
    println!("Pred. edges   : {}", model.edge_count());
    println!("Tick          : {}", model.tick);
    println!("Total chars   : {}", model.metrics.total_characters);
    println!("Total decisions: {}", model.metrics.total_decisions);
    println!("dpc           : {:.6}", model.metrics.decision_per_character());
    let mut t = [0u32; 3];
    for chunk in model.chunks.iter_all() { t[chunk.tier as usize] += 1; }
    println!("Chunk tiers   : T0={} T1={} T2={}", t[0], t[1], t[2]);
}

// ── evaluate ──────────────────────────────────────────────────────────────

fn cmd_evaluate(args: &[String]) {
    let input = flag_value(args, "--input").unwrap_or_else(|| {
        eprintln!("--input <file> required"); process::exit(1);
    });
    let model_path = flag_path(args, "--model").unwrap_or_else(default_model_path);
    let text = std::fs::read_to_string(&input).unwrap_or_else(|e| {
        eprintln!("Cannot read {input}: {e}"); process::exit(1);
    });
    let mut eval_model = load_model(&model_path);
    let mut decisions = 0u64;
    let mut chars = 0u64;
    for line in text.lines() {
        if line.is_empty() { continue; }
        let before_d = eval_model.metrics.total_decisions;
        let before_c = eval_model.metrics.total_characters;
        eval_model.expose(line);
        decisions += eval_model.metrics.total_decisions - before_d;
        chars += eval_model.metrics.total_characters - before_c;
    }
    let dpc = if chars == 0 { 0.0 } else { decisions as f64 / chars as f64 };
    println!("Lines evaluated : {}", text.lines().count());
    println!("Characters      : {chars}");
    println!("Decisions       : {decisions}");
    println!("dpc             : {dpc:.6}");
}

// ── chat ──────────────────────────────────────────────────────────────────

fn cmd_chat(args: &[String]) {
    let input = flag_value(args, "--input").unwrap_or_else(|| {
        eprintln!("--input <text> required"); process::exit(1);
    });
    let mut session = open_session(args);
    let (turn_id, output, trace) = session.turn(&input).unwrap_or_else(|e| {
        eprintln!("chat error: {e}"); process::exit(1);
    });
    println!("{output}");
    eprintln!("[turn_id={turn_id} decisions={} tick={}]",
        trace.decision_count, session.model.tick);
}

// ── feedback ──────────────────────────────────────────────────────────────

fn cmd_feedback(args: &[String]) {
    let turn_id: i64 = flag_value(args, "--turn-id")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| { eprintln!("--turn-id <N> required"); process::exit(1); });
    let sign_str = flag_value(args, "--sign").unwrap_or_else(|| {
        eprintln!("--sign <+|-|1|-1> required"); process::exit(1);
    });
    let sign = match sign_str.as_str() {
        "+" | "1" | "+1" => FeedbackSign::Positive,
        "-" | "-1"       => FeedbackSign::Negative,
        _ => { eprintln!("invalid sign: {sign_str}"); process::exit(1); }
    };
    let mut session = open_session(args);
    session.feedback(turn_id, sign, FeedbackSource::User, None).unwrap_or_else(|e| {
        eprintln!("feedback error: {e}"); process::exit(1);
    });
    println!("Feedback applied to turn {turn_id}.");
}

// ── snapshot ──────────────────────────────────────────────────────────────

fn cmd_snapshot(args: &[String]) {
    let model_path = flag_path(args, "--model");
    let session = open_session(args);
    let sid = session.snapshot_to_db(None).unwrap_or_else(|e| {
        eprintln!("snapshot error: {e}"); process::exit(1);
    });
    println!("Snapshot saved: id={sid} tick={}", session.model.tick);
    if let Some(path) = model_path {
        persistence::save(&session.model, &path).unwrap_or_else(|e| {
            eprintln!("save error: {e}"); process::exit(1);
        });
        println!("Model also saved to {}", path.display());
    }
}

// ── restore ───────────────────────────────────────────────────────────────

fn cmd_restore(args: &[String]) {
    let snapshot_id: i64 = flag_value(args, "--snapshot-id")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| { eprintln!("--snapshot-id <N> required"); process::exit(1); });
    let model_path = flag_path(args, "--model");
    let mut session = open_session(args);
    session.restore_from_db(snapshot_id).unwrap_or_else(|e| {
        eprintln!("restore error: {e}"); process::exit(1);
    });
    println!("Restored from snapshot {snapshot_id}. tick={}", session.model.tick);
    if let Some(path) = model_path {
        persistence::save(&session.model, &path).unwrap_or_else(|e| {
            eprintln!("save error: {e}"); process::exit(1);
        });
        println!("Model also saved to {}", path.display());
    }
}

// ── history ───────────────────────────────────────────────────────────────

fn cmd_history(args: &[String]) {
    let limit: usize = flag_value(args, "--limit")
        .and_then(|v| v.parse().ok()).unwrap_or(20);
    let session = open_session(args);
    let rows = session.history(limit).unwrap_or_else(|e| {
        eprintln!("history error: {e}"); process::exit(1);
    });
    if rows.is_empty() {
        println!("(no turns yet)");
        return;
    }
    for r in &rows {
        println!("[{}] input={:?}  output={:?}  decisions={}",
            r.turn_id,
            truncate(&r.input_text, 40),
            truncate(&r.output_text, 40),
            r.decision_count,
        );
    }
}

fn truncate(s: &str, max: usize) -> &str {
    let end = s.char_indices()
        .nth(max)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..end]
}

// ── Argument parsing helpers ───────────────────────────────────────────────

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn flag_path(args: &[String], flag: &str) -> Option<PathBuf> {
    flag_value(args, flag).map(PathBuf::from)
}
