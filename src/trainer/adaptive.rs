/// Adaptive control — Block Level selection (TR-SIZE-01..03) and
/// repeat-stop decisions (TR-ADAPT-01..05).
use serde::{Deserialize, Serialize};

// ── Block Level ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockLevel { S, M, L, XL }

impl Default for BlockLevel { fn default() -> Self { Self::S } }

impl BlockLevel {
    pub fn target_lines(self) -> usize {
        match self { Self::S => 4, Self::M => 8, Self::L => 16, Self::XL => 32 }
    }
    pub fn max_chars(self) -> usize {
        match self { Self::S => 800, Self::M => 1600, Self::L => 3200, Self::XL => 6400 }
    }
    pub fn upgrade(self) -> Self {
        match self { Self::S => Self::M, Self::M => Self::L, Self::L => Self::XL, Self::XL => Self::XL }
    }
    pub fn downgrade(self) -> Self {
        match self { Self::S => Self::S, Self::M => Self::S, Self::L => Self::M, Self::XL => Self::L }
    }
    pub fn label(self) -> &'static str {
        match self { Self::S => "S", Self::M => "M", Self::L => "L", Self::XL => "XL" }
    }
}

// ── Block-size adaptive decision ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelChange { Upgrade, Downgrade, Keep }

fn median_f64(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 { return 0.0; }
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

fn median_u32_as_f64(v: &[u32]) -> f64 {
    let f: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    median_f64(&f)
}

/// Decide whether to change Block Level based on the last 8 blocks' metrics.
/// Returns `Keep` if fewer than 8 blocks have been completed (TR-T15).
pub fn decide_level_change(recent_pre_dpc: &[f64], recent_repeats: &[u32]) -> LevelChange {
    if recent_pre_dpc.len() < 8 || recent_repeats.len() < 8 { return LevelChange::Keep; }
    let tail_dpc = &recent_pre_dpc[recent_pre_dpc.len() - 8..];
    let tail_rep = &recent_repeats[recent_repeats.len() - 8..];
    let med_dpc = median_f64(tail_dpc);
    let med_rep = median_u32_as_f64(tail_rep);

    if med_dpc <= 0.70 && med_rep <= 8.0 {
        LevelChange::Upgrade     // TR-SIZE-01
    } else if med_dpc >= 0.90 && med_rep >= 16.0 {
        LevelChange::Downgrade   // TR-SIZE-02
    } else {
        LevelChange::Keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tr_t15_fewer_than_8_no_change() {
        let dpcs = vec![0.50, 0.60, 0.55]; // only 3
        let reps = vec![4u32, 6, 5];
        assert_eq!(decide_level_change(&dpcs, &reps), LevelChange::Keep, "TR-T15");
    }

    #[test]
    fn tr_t16_upgrade_condition() {
        let dpcs = vec![0.60f64; 8];
        let reps = vec![6u32; 8];
        assert_eq!(decide_level_change(&dpcs, &reps), LevelChange::Upgrade, "TR-T16");
    }

    #[test]
    fn tr_t17_downgrade_condition() {
        let dpcs = vec![0.95f64; 8];
        let reps = vec![20u32; 8];
        assert_eq!(decide_level_change(&dpcs, &reps), LevelChange::Downgrade, "TR-T17");
    }

    #[test]
    fn no_change_middle_range() {
        let dpcs = vec![0.80f64; 8];
        let reps = vec![10u32; 8];
        assert_eq!(decide_level_change(&dpcs, &reps), LevelChange::Keep);
    }

    #[test]
    fn level_upgrade_chain() {
        assert_eq!(BlockLevel::S.upgrade(), BlockLevel::M);
        assert_eq!(BlockLevel::M.upgrade(), BlockLevel::L);
        assert_eq!(BlockLevel::L.upgrade(), BlockLevel::XL);
        assert_eq!(BlockLevel::XL.upgrade(), BlockLevel::XL); // capped
    }

    #[test]
    fn level_downgrade_chain() {
        assert_eq!(BlockLevel::XL.downgrade(), BlockLevel::L);
        assert_eq!(BlockLevel::L.downgrade(), BlockLevel::M);
        assert_eq!(BlockLevel::M.downgrade(), BlockLevel::S);
        assert_eq!(BlockLevel::S.downgrade(), BlockLevel::S); // floored
    }
}
