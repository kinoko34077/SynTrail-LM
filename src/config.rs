/// Memory budget tunables (§14-§16).
#[derive(Debug, Clone)]
pub struct MemoryBudgetConfig {
    /// Maximum number of HOT chunks globally. 0 = unlimited.
    pub hot_chunks_max: usize,
}

impl Default for MemoryBudgetConfig {
    fn default() -> Self { Self { hot_chunks_max: 0 } }
}

/// Learning tunables — controls merge pressure and factorization (§17).
#[derive(Debug, Clone)]
pub struct LearningConfig {
    /// Penalty per additional left context that shares the same right unit (§17).
    /// High value → right-reuse resists merge (factorization pressure).
    pub factorization_scale: f64,
}

impl Default for LearningConfig {
    fn default() -> Self { Self { factorization_scale: 1.0 } }
}

/// Generation-specific tunables (§13 — moved from model.rs constants).
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Top-K route candidates per step.
    pub route_top_k: usize,
    /// Score multiplier for a route that appears in recent_routes.
    pub cycle_penalty: f64,
    /// Ring-buffer depth for cycle detection.
    pub recent_routes_max: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            route_top_k: 5,
            cycle_penalty: 0.05,
            recent_routes_max: 8,
        }
    }
}

/// Runtime configuration (§31).
/// All tunables are here — never hardcode them in business logic.
#[derive(Debug, Clone)]
pub struct Config {
    /// Base magnitude M(context) = 1.0 in v0.2 (§9).
    pub feedback_base_magnitude: f64,
    /// Decay λ for feedback_value updates: V_new = λ·V_old + r  (§15).
    pub feedback_decay: f64,
    /// How often (in turns) to auto-snapshot (§20.2).
    pub snapshot_interval_turns: u64,
    /// Credit distribution strategy (§10).
    pub feedback_distribution: FeedbackDistribution,
    /// Whether the model may evaluate its own output (§19, §31).
    pub self_feedback_enabled: bool,
    /// Default max units per generate call from the chat CLI.
    pub default_max_units: usize,
    // ── v0.3 Association (§7-§11) ──────────────────────────────────────────
    /// Maximum associations stored per UnitId (budget).
    pub association_top_k: usize,
    /// Decay for association strength: S_new = decay * S_old + 1.0
    pub association_decay: f64,
    /// Decay for the avoidance field (§19).
    pub avoidance_decay: f64,
    /// Generation tunables (route_top_k, cycle_penalty, recent_routes_max).
    pub generation: GenerationConfig,
    // ── v0.4 Residency (§10, §41) ──────────────────────────────────────────
    /// Maximum number of HOT chunks. 0 = unlimited.
    pub hot_budget: usize,
    // ── v0.5 Memory Budget (§14-§16) ──────────────────────────────────────
    /// Memory budget tunables (HOT chunk cap etc.).
    pub memory_budget: MemoryBudgetConfig,
    // ── v0.5 Learning tunables (§17) ──────────────────────────────────────
    /// Learning tunables (factorization_scale etc.).
    pub learning: LearningConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackDistribution {
    /// r_i = R / N  (v0.2 standard)
    Uniform,
}

impl Config {
    /// v0.2 standard defaults (backward compat).
    pub fn default_v02() -> Self {
        Self {
            feedback_base_magnitude: 1.0,
            feedback_decay: 0.99,
            snapshot_interval_turns: 16,
            feedback_distribution: FeedbackDistribution::Uniform,
            self_feedback_enabled: false,
            default_max_units: 50,
            association_top_k: 32,
            association_decay: 0.99,
            avoidance_decay: 0.99,
            generation: GenerationConfig::default(),
            hot_budget: 0,
            memory_budget: MemoryBudgetConfig::default(),
            learning: LearningConfig::default(),
        }
    }

    /// v0.3 standard defaults.
    pub fn default_v03() -> Self { Self::default_v02() }

    /// v0.4 standard defaults.
    pub fn default_v04() -> Self { Self::default_v02() }

    /// Compute M(context). v0.2: always 1.0.
    pub fn feedback_magnitude(&self) -> f64 {
        self.feedback_base_magnitude
    }

    /// Compute R = sign × M(context).
    pub fn reward(&self, sign: i8) -> f64 {
        sign as f64 * self.feedback_magnitude()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::default_v02()
    }
}
