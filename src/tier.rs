/// 3-tier classification for Chunks.
/// Tier is about verification status, not string length.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Tier {
    /// New / exploratory — not yet verified.
    #[default]
    T0 = 0,
    /// Being verified — accumulating evidence.
    T1 = 1,
    /// Established — high-confidence, reused freely.
    T2 = 2,
}

/// Provisional promotion thresholds (§B §4).
pub struct TierThresholds;

impl TierThresholds {
    /// Minimum successes to consider T0 → T1.
    pub const T0_TO_T1_SUCCESSES: u32 = 16;
    /// Minimum accuracy (successes/total) to consider T0 → T1.
    pub const T0_TO_T1_ACCURACY: f64 = 0.90;

    /// Minimum successes to consider T1 → T2.
    pub const T1_TO_T2_SUCCESSES: u32 = 64;
    /// Minimum accuracy to consider T1 → T2.
    pub const T1_TO_T2_ACCURACY: f64 = 0.98;

    /// Below this accuracy a T1 chunk may be demoted to T0.
    pub const T1_TO_T0_ACCURACY: f64 = 0.80;
    /// Below this accuracy a T2 chunk may be demoted to T1.
    pub const T2_TO_T1_ACCURACY: f64 = 0.90;
}

impl Tier {
    /// Evaluate whether a chunk should be promoted based on its statistics.
    /// Returns the new Tier (unchanged if no promotion applies).
    pub fn maybe_promote(self, successes: u32, total: u32) -> Tier {
        let accuracy = if total == 0 { 0.0 } else { successes as f64 / total as f64 };
        match self {
            Tier::T0
                if successes >= TierThresholds::T0_TO_T1_SUCCESSES
                    && accuracy >= TierThresholds::T0_TO_T1_ACCURACY =>
            {
                Tier::T1
            }
            Tier::T1
                if successes >= TierThresholds::T1_TO_T2_SUCCESSES
                    && accuracy >= TierThresholds::T1_TO_T2_ACCURACY =>
            {
                Tier::T2
            }
            _ => self,
        }
    }

    /// Evaluate whether a chunk should be demoted based on accuracy.
    /// Returns the new Tier (unchanged if no demotion applies).
    pub fn maybe_demote(self, successes: u32, total: u32) -> Tier {
        let accuracy = if total == 0 { 0.0 } else { successes as f64 / total as f64 };
        match self {
            Tier::T2 if accuracy < TierThresholds::T2_TO_T1_ACCURACY => Tier::T1,
            Tier::T1 if accuracy < TierThresholds::T1_TO_T0_ACCURACY => Tier::T0,
            _ => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promotion_t0_to_t1() {
        // Exactly at threshold: 16 successes, 90% accuracy (total=16 → 100%)
        assert_eq!(Tier::T0.maybe_promote(16, 16), Tier::T1);
    }

    #[test]
    fn test_no_promotion_below_count() {
        assert_eq!(Tier::T0.maybe_promote(15, 15), Tier::T0);
    }

    #[test]
    fn test_no_promotion_below_accuracy() {
        // 16 successes but accuracy only 88.9% (16/18)
        assert_eq!(Tier::T0.maybe_promote(16, 18), Tier::T0);
    }

    #[test]
    fn test_promotion_t1_to_t2() {
        // 64 successes, 100% accuracy
        assert_eq!(Tier::T1.maybe_promote(64, 64), Tier::T2);
    }

    #[test]
    fn test_no_promotion_t2() {
        // Already T2, no further promotion
        assert_eq!(Tier::T2.maybe_promote(1000, 1000), Tier::T2);
    }

    #[test]
    fn test_demotion_t2_to_t1() {
        // accuracy 89% < 90% threshold
        assert_eq!(Tier::T2.maybe_demote(89, 100), Tier::T1);
    }

    #[test]
    fn test_demotion_t1_to_t0() {
        // accuracy 79% < 80% threshold
        assert_eq!(Tier::T1.maybe_demote(79, 100), Tier::T0);
    }

    #[test]
    fn test_no_demotion_t0() {
        assert_eq!(Tier::T0.maybe_demote(0, 100), Tier::T0);
    }

    #[test]
    fn test_promote_then_demote_cycle() {
        let t = Tier::T0.maybe_promote(16, 16);
        assert_eq!(t, Tier::T1);
        let t = t.maybe_promote(64, 64);
        assert_eq!(t, Tier::T2);
        // drop accuracy
        let t = t.maybe_demote(89, 100);
        assert_eq!(t, Tier::T1);
        let t = t.maybe_demote(79, 100);
        assert_eq!(t, Tier::T0);
    }
}
