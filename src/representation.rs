/// Representation Lineage — acquisition history of how a Content was processed.
///
/// SPEC: Representation Lineage (Phase B)
///
/// DISTINCT FROM Structural Lineage (lineage.rs):
///   Structural Lineage: Chunk ABC → left=AB, right=C  (Chunk structure tree)
///   Representation Lineage: "same Content, different processing granularities, ordered"
///
/// Example for Content "ABCDE":
///   R0 = [A][B][C][D][E]   (primitive-only, first observation)
///   R1 = [A][BC][D][E]     (after BC chunk forms via Replay)
///   R2 = [AB][CDE]         (after further Replay)
///
/// Invariants:
/// - predecessor is always within the same Identity
/// - acquisition order is preserved (oldest rep_id < newest)
/// - save/load must preserve the full chain
/// - legacy models have no synthetic predecessors
use std::collections::HashMap;

use crate::identity::IdentityId;
use crate::units::UnitId;

pub type RepId = u32;

/// One concrete representation of a Content: a specific unit sequence.
#[derive(Debug, Clone)]
pub struct RepresentationEntry {
    pub rep_id: RepId,
    pub identity_id: IdentityId,
    /// The unit sequence for this representation.
    pub units: Vec<UnitId>,
    /// Model tick at which this representation was first acquired.
    pub acquired_at: u64,
    /// Practice count — incremented by Replay and recognition hits.
    pub practice_count: u32,
    /// Confidence in this representation (0.0–1.0).
    /// Increases with successful practice; decreases with failure.
    pub confidence: f32,
    /// Previous representation in the lineage chain for this Content.
    pub predecessor_rep_id: Option<RepId>,
}

impl RepresentationEntry {
    fn new(
        rep_id: RepId,
        identity_id: IdentityId,
        units: Vec<UnitId>,
        acquired_at: u64,
        predecessor_rep_id: Option<RepId>,
    ) -> Self {
        Self {
            rep_id,
            identity_id,
            units,
            acquired_at,
            practice_count: 0,
            confidence: 0.5,
            predecessor_rep_id,
        }
    }

    /// Execution cost proxy: number of decisions this representation requires.
    pub fn cost(&self) -> usize {
        self.units.len()
    }

    /// Preference score: balances confidence against cost.
    ///
    /// Higher is better. Uses a simple formula:
    ///   score = confidence / (1 + cost_factor * cost)
    /// where cost_factor = 0.1 so a 5-unit rep with confidence 1.0 scores
    /// lower than a 3-unit rep with confidence 0.9.
    pub fn preference_score(&self) -> f32 {
        const COST_FACTOR: f32 = 0.1;
        self.confidence / (1.0 + COST_FACTOR * self.cost() as f32)
    }

    /// Record a successful usage of this representation.
    pub fn record_success(&mut self) {
        self.practice_count += 1;
        self.confidence = (self.confidence + 0.05).min(1.0);
    }

    /// Record a failure (representation could not be applied successfully).
    pub fn record_failure(&mut self) {
        self.confidence = (self.confidence - 0.1).max(0.0);
    }
}

/// Stores and manages Representation Lineage for all known Identities.
///
/// Entry ordering within `by_identity[id]` reflects acquisition order:
/// index 0 = oldest (R0), last index = most recently acquired.
#[derive(Debug, Default)]
pub struct RepresentationStore {
    entries: Vec<RepresentationEntry>,
    /// identity_id → list of rep_ids in acquisition order (oldest first).
    by_identity: HashMap<IdentityId, Vec<RepId>>,
    /// (identity_id, units_fingerprint) → rep_id — deduplication key.
    by_key: HashMap<(IdentityId, u64), RepId>,
}

fn fingerprint_units(units: &[UnitId]) -> u64 {
    // FNV-1a variant over the compact tagged representation.
    let mut h: u64 = 0xcbf29ce484222325;
    for u in units {
        let bits = u.raw() as u64 | ((u.is_chunk() as u64) << 32);
        h ^= bits;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl RepresentationStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a representation for `(identity_id, units)` acquired at `tick`.
    ///
    /// Returns `(rep_id, is_new)`. If the same unit sequence was already
    /// registered for this identity, returns the existing rep_id.
    pub fn intern(
        &mut self,
        identity_id: IdentityId,
        units: Vec<UnitId>,
        acquired_at: u64,
    ) -> (RepId, bool) {
        let key = (identity_id, fingerprint_units(&units));
        if let Some(&existing) = self.by_key.get(&key) {
            return (existing, false);
        }
        let rep_id = self.entries.len() as RepId;
        let predecessor = self
            .by_identity
            .get(&identity_id)
            .and_then(|v| v.last().copied());
        self.entries.push(RepresentationEntry::new(
            rep_id,
            identity_id,
            units,
            acquired_at,
            predecessor,
        ));
        self.by_identity
            .entry(identity_id)
            .or_default()
            .push(rep_id);
        self.by_key.insert(key, rep_id);
        (rep_id, true)
    }

    /// Get an entry for mutation (e.g., to record success/failure or set confidence).
    pub fn get_mut(&mut self, rep_id: RepId) -> Option<&mut RepresentationEntry> {
        self.entries.get_mut(rep_id as usize)
    }

    pub fn get(&self, rep_id: RepId) -> Option<&RepresentationEntry> {
        self.entries.get(rep_id as usize)
    }

    /// All representations for an identity, in acquisition order (R0 first).
    pub fn for_identity(&self, identity_id: IdentityId) -> Vec<&RepresentationEntry> {
        self.by_identity
            .get(&identity_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|&id| self.entries.get(id as usize))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Return the preferred representation for an identity: highest preference_score.
    ///
    /// Returns None if no representations are registered for the identity.
    pub fn preferred(&self, identity_id: IdentityId) -> Option<&RepresentationEntry> {
        self.for_identity(identity_id)
            .into_iter()
            .max_by(|a, b| {
                a.preference_score()
                    .partial_cmp(&b.preference_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Follow the predecessor chain from `rep_id` until a representation other
    /// than `exclude` (the one that failed) is found.
    ///
    /// Returns the first viable predecessor, or None if the chain is exhausted.
    pub fn predecessor_of(&self, rep_id: RepId) -> Option<&RepresentationEntry> {
        let entry = self.entries.get(rep_id as usize)?;
        let pred_id = entry.predecessor_rep_id?;
        self.entries.get(pred_id as usize)
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &RepresentationEntry> {
        self.entries.iter()
    }

    // ── Bulk access for persistence ───────────────────────────────────────

    pub fn all_entries_raw(&self) -> &[RepresentationEntry] {
        &self.entries
    }

    /// Reconstruct from a saved list of entries.
    ///
    /// Entries must be provided in rep_id order (0, 1, 2, ...) so that
    /// predecessor indices remain valid.
    pub fn from_bulk(entries: Vec<RepresentationEntry>) -> Self {
        let mut by_identity: HashMap<IdentityId, Vec<RepId>> = HashMap::new();
        let mut by_key: HashMap<(IdentityId, u64), RepId> = HashMap::new();
        for e in &entries {
            by_identity
                .entry(e.identity_id)
                .or_default()
                .push(e.rep_id);
            by_key.insert((e.identity_id, fingerprint_units(&e.units)), e.rep_id);
        }
        Self { entries, by_identity, by_key }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(id: u32) -> UnitId { UnitId::primitive(id) }
    fn c(id: u32) -> UnitId { UnitId::chunk(id) }

    #[test]
    fn intern_is_dedup() {
        let mut store = RepresentationStore::new();
        let (id1, new1) = store.intern(0, vec![p(1), p(2)], 1);
        let (id2, new2) = store.intern(0, vec![p(1), p(2)], 2);
        assert_eq!(id1, id2);
        assert!(new1);
        assert!(!new2);
    }

    #[test]
    fn acquisition_order_and_predecessor() {
        let mut store = RepresentationStore::new();
        let (r0, _) = store.intern(0, vec![p(1), p(2), p(3)], 1);
        let (r1, _) = store.intern(0, vec![p(1), c(0)], 5);
        let (r2, _) = store.intern(0, vec![c(1)], 10);
        assert_eq!(store.get(r1).unwrap().predecessor_rep_id, Some(r0));
        assert_eq!(store.get(r2).unwrap().predecessor_rep_id, Some(r1));
    }

    #[test]
    fn preferred_balances_confidence_and_cost() {
        let mut store = RepresentationStore::new();
        // R0: high confidence, high cost
        let (r0, _) = store.intern(0, vec![p(1), p(2), p(3), p(4), p(5)], 1); // cost=5
        store.get_mut(r0).unwrap().confidence = 1.0;
        // R1: medium confidence, medium cost
        let (r1, _) = store.intern(0, vec![p(1), c(0), p(5)], 5); // cost=3
        store.get_mut(r1).unwrap().confidence = 0.9;
        // R2: low confidence, low cost
        let (r2, _) = store.intern(0, vec![c(1)], 10); // cost=1
        store.get_mut(r2).unwrap().confidence = 0.2;

        let preferred = store.preferred(0).unwrap();
        assert_eq!(preferred.rep_id, r1,
            "R1 should be preferred: conf=0.9,cost=3 beats R0 conf=1.0,cost=5 and R2 conf=0.2,cost=1");
    }

    #[test]
    fn predecessor_of_returns_prior() {
        let mut store = RepresentationStore::new();
        let (r0, _) = store.intern(0, vec![p(1), p(2)], 1);
        let (r1, _) = store.intern(0, vec![c(0)], 5);
        let pred = store.predecessor_of(r1).unwrap();
        assert_eq!(pred.rep_id, r0);
    }

    #[test]
    fn predecessor_of_r0_is_none() {
        let mut store = RepresentationStore::new();
        let (r0, _) = store.intern(0, vec![p(1), p(2)], 1);
        assert!(store.predecessor_of(r0).is_none());
    }

    #[test]
    fn from_bulk_roundtrip_preserves_order() {
        let mut store = RepresentationStore::new();
        store.intern(0, vec![p(1), p(2), p(3)], 1);
        store.intern(0, vec![c(0), p(3)], 5);
        let raw: Vec<RepresentationEntry> = store.iter_all().cloned().collect();
        let restored = RepresentationStore::from_bulk(raw);
        assert_eq!(restored.entry_count(), store.entry_count());
        let reps = restored.for_identity(0);
        assert_eq!(reps.len(), 2);
        assert_eq!(reps[0].acquired_at, 1);
        assert_eq!(reps[1].acquired_at, 5);
        assert_eq!(reps[1].predecessor_rep_id, Some(reps[0].rep_id));
    }
}
