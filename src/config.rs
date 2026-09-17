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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackDistribution {
    /// r_i = R / N  (v0.2 standard)
    Uniform,
}

impl Config {
    /// v0.2 standard defaults.
    pub fn default_v02() -> Self {
        Self {
            feedback_base_magnitude: 1.0,
            feedback_decay: 0.99,
            snapshot_interval_turns: 16,
            feedback_distribution: FeedbackDistribution::Uniform,
            self_feedback_enabled: false,
            default_max_units: 50,
        }
    }

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
