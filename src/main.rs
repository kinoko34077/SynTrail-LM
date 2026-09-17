/// SynTrail-LM CLI — §E §15: train / generate / inspect / evaluate
use std::path::{Path, PathBuf};
use std::process;

use syntrail_lm::model::ModelState;
use syntrail_lm::persistence;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", USAGE);
        process::exit(1);
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "generate" => cmd_generate(&args[2..]),
        "inspect" => cmd_inspect(&args[2..]),
        "evaluate" => cmd_evaluate(&args[2..]),
        _ => {
            eprintln!("Unknown command: {}\n{}", args[1], USAGE);
            process::exit(1);
        }
    }
}

const USAGE: &str = "\
Usage: syntrail <command> [options]

Commands:
  train      --input <file> [--model <path>]   Train on text file, save model
  generate   --seed <text>  [--model <path>] [--max-units N]
  inspect    [--model <path>]                  Print model statistics
  evaluate   --input <file> [--model <path>]   Compute metrics on text file
";

fn default_model_path() -> PathBuf {
    PathBuf::from("model.json")
}

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

// ── train ─────────────────────────────────────────────────────────────────

fn cmd_train(args: &[String]) {
    let input = flag_value(args, "--input").unwrap_or_else(|| {
        eprintln!("--input <file> required");
        process::exit(1);
    });
    let model_path = flag_path(args, "--model").unwrap_or_else(default_model_path);

    let text = std::fs::read_to_string(&input).unwrap_or_else(|e| {
        eprintln!("Cannot read {input}: {e}");
        process::exit(1);
    });

    let mut model = load_model(&model_path);

    // Train line by line for progress reporting
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    for (i, line) in lines.iter().enumerate() {
        model.train(line);
        if (i + 1) % 1000 == 0 || i + 1 == n {
            eprintln!(
                "[{}/{}] primitives={} chunks={} edges={} dpc={:.4}",
                i + 1,
                n,
                model.primitive_count(),
                model.chunk_count(),
                model.edge_count(),
                model.metrics.decision_per_character()
            );
        }
    }

    save_model(&model, &model_path);
    println!(
        "Saved to {}  (primitives={} chunks={} edges={} dpc={:.4})",
        model_path.display(),
        model.primitive_count(),
        model.chunk_count(),
        model.edge_count(),
        model.metrics.decision_per_character()
    );
}

// ── generate ──────────────────────────────────────────────────────────────

fn cmd_generate(args: &[String]) {
    let seed = flag_value(args, "--seed").unwrap_or_default();
    let max_units: usize = flag_value(args, "--max-units")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
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
    println!(
        "dpc           : {:.6}",
        model.metrics.decision_per_character()
    );

    // Tier breakdown
    let mut t = [0u32; 3];
    for chunk in model.chunks.iter_all() {
        t[chunk.tier as usize] += 1;
    }
    println!("Chunk tiers   : T0={} T1={} T2={}", t[0], t[1], t[2]);
}

// ── evaluate ──────────────────────────────────────────────────────────────

fn cmd_evaluate(args: &[String]) {
    let input = flag_value(args, "--input").unwrap_or_else(|| {
        eprintln!("--input <file> required");
        process::exit(1);
    });
    let model_path = flag_path(args, "--model").unwrap_or_else(default_model_path);

    let text = std::fs::read_to_string(&input).unwrap_or_else(|e| {
        eprintln!("Cannot read {input}: {e}");
        process::exit(1);
    });

    // Evaluate on a fresh scratch model trained on the text, then measure dpc
    let mut eval_model = load_model(&model_path);
    let mut decisions = 0u64;
    let mut chars = 0u64;

    for line in text.lines() {
        if line.is_empty() { continue; }
        let before_d = eval_model.metrics.total_decisions;
        let before_c = eval_model.metrics.total_characters;
        eval_model.train(line);
        decisions += eval_model.metrics.total_decisions - before_d;
        chars += eval_model.metrics.total_characters - before_c;
    }

    let dpc = if chars == 0 { 0.0 } else { decisions as f64 / chars as f64 };
    println!("Lines evaluated : {}", text.lines().count());
    println!("Characters      : {chars}");
    println!("Decisions       : {decisions}");
    println!("dpc             : {dpc:.6}");
}

// ── Argument parsing helpers ───────────────────────────────────────────────

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}

fn flag_path(args: &[String], flag: &str) -> Option<PathBuf> {
    flag_value(args, flag).map(PathBuf::from)
}
