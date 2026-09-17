/// Phase 5-6: Signed feedback and uniform credit distribution (§8-§12).
use crate::config::Config;
use crate::trace::TurnTrace;

/// +1 or -1 (§8).  None = no feedback (§4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackSign {
    Positive = 1,
    Negative = -1,
}

impl FeedbackSign {
    pub fn as_i8(self) -> i8 {
        self as i8
    }
}

/// Feedback event record (§17 feedback_events).
#[derive(Debug, Clone)]
pub struct FeedbackEvent {
    pub feedback_id: Option<i64>,  // set after DB insert
    pub turn_id: i64,
    pub trace_id: u64,
    pub timestamp: u64,
    pub sign: FeedbackSign,
    pub magnitude: f64,
    pub source: FeedbackSource,
    pub note: Option<String>,
}

/// Where the feedback came from (§18).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedbackSource {
    User,
    Self_,
    Tool,
    GroundTruth,
    System,
}

impl FeedbackSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Self_ => "self",
            Self::Tool => "tool",
            Self::GroundTruth => "ground_truth",
            Self::System => "system",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "self" => Self::Self_,
            "tool" => Self::Tool,
            "ground_truth" => Self::GroundTruth,
            "system" => Self::System,
            _ => Self::User,
        }
    }
}

/// Evaluate the turn's total reward (§9).
/// R = sign × M(context)
pub fn total_reward(sign: FeedbackSign, config: &Config) -> f64 {
    config.reward(sign.as_i8())
}

/// Distribute R uniformly over N decisions (§10, §11).
/// Returns r_i = R / N for each decision.
/// Σ r_i == R (within floating-point tolerance).
pub fn distribute_uniform(sign: FeedbackSign, decision_count: usize, config: &Config) -> Vec<f64> {
    if decision_count == 0 {
        return vec![];
    }
    let r = total_reward(sign, config);
    let r_i = r / decision_count as f64;
    vec![r_i; decision_count]
}

/// Distribute according to config.feedback_distribution.
pub fn distribute(sign: FeedbackSign, trace: &TurnTrace, config: &Config) -> Vec<f64> {
    use crate::config::FeedbackDistribution;
    match config.feedback_distribution {
        FeedbackDistribution::Uniform => {
            distribute_uniform(sign, trace.decision_count, config)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config { Config::default_v02() }

    #[test]
    fn test_positive_reward() {
        let r = total_reward(FeedbackSign::Positive, &cfg());
        assert_eq!(r, 1.0);
    }

    #[test]
    fn test_negative_reward() {
        let r = total_reward(FeedbackSign::Negative, &cfg());
        assert_eq!(r, -1.0);
    }

    #[test]
    fn test_symmetry() {
        let pos = total_reward(FeedbackSign::Positive, &cfg());
        let neg = total_reward(FeedbackSign::Negative, &cfg());
        assert_eq!(pos, -neg, "T-07 symmetry");
    }

    #[test]
    fn test_uniform_distribution_sums_to_r() {
        let parts = distribute_uniform(FeedbackSign::Negative, 4, &cfg());
        assert_eq!(parts.len(), 4);
        let sum: f64 = parts.iter().sum();
        assert!((sum - (-1.0)).abs() < 1e-12, "T-06: sum={sum}");
        assert!((parts[0] - (-0.25)).abs() < 1e-12, "T-08: each=-0.25");
    }

    #[test]
    fn test_uniform_positive() {
        let parts = distribute_uniform(FeedbackSign::Positive, 5, &cfg());
        let sum: f64 = parts.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "T-05: sum={sum}");
    }

    #[test]
    fn test_zero_decisions_empty() {
        let parts = distribute_uniform(FeedbackSign::Positive, 0, &cfg());
        assert!(parts.is_empty());
    }
}
