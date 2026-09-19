/// Phase 12: Transform / Inverse / Composition.
///
/// A Transform is a directed relation `f: IdentityA → IdentityB`.
///
/// Phase F (§17): TransformKind controls whether Identity merge is allowed.
///   - EquivalentView: the two Identities are the same content seen differently → merge allowed
///   - Mapping: a directional transform (e.g., singular→plural) → no merge
///   - Inverse / Composed: derived transforms → no merge
///
/// Only EquivalentView triggers merge_identities. Mapping and derived kinds
/// record the relation without collapsing the two Identities.
use std::collections::HashMap;

use crate::identity::{IdentityId, IdentityStore};

pub type TransformId = u32;

/// Phase F: classification of transform semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformKind {
    /// The two Identities are equivalent views of the same Content.
    /// Registering this kind triggers merge_identities(source, target).
    EquivalentView,
    /// A directional mapping (e.g., inflection, derivation).
    /// Does NOT trigger merge_identities.
    Mapping,
    /// Derived as the inverse of another Transform.
    Inverse(TransformId),
    /// Derived as the composition of two Transforms.
    Composed(TransformId, TransformId),
}

impl TransformKind {
    /// Whether this kind allows merging the two Identity endpoints.
    pub fn allows_merge(&self) -> bool {
        matches!(self, TransformKind::EquivalentView)
    }
}

/// A single Transform edge: `source → target`.
#[derive(Debug, Clone)]
pub struct Transform {
    pub id: TransformId,
    pub source: IdentityId,
    pub target: IdentityId,
    pub kind: TransformKind,
    /// Accumulated transform value (e.g., log-ratio, distance). Updated by callers.
    pub value: f64,
    /// Evidence/confidence weight for this transform. Updated by callers.
    pub evidence: f64,
}

/// Registry of all Transforms.
///
/// Index layout:
///   `by_pair[(src, tgt)]` — deduplication / lookup by (source, target).
#[derive(Debug, Default, Clone)]
pub struct TransformStore {
    transforms: Vec<Transform>,
    by_pair: HashMap<(IdentityId, IdentityId), TransformId>,
}

impl TransformStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a direct Transform `source → target` with the given kind.
    ///
    /// Calls `identities.merge_identities(source, target)` ONLY when
    /// `kind == EquivalentView` (§17 fix: no unconditional merge).
    ///
    /// Returns the existing TransformId if the pair was already registered.
    pub fn register(
        &mut self,
        source: IdentityId,
        target: IdentityId,
        kind: TransformKind,
        identities: &mut IdentityStore,
    ) -> TransformId {
        if let Some(&tid) = self.by_pair.get(&(source, target)) {
            return tid;
        }
        if kind.allows_merge() {
            identities.merge_identities(source, target);
        }
        let id = self.transforms.len() as TransformId;
        self.transforms.push(Transform { id, source, target, kind, value: 0.0, evidence: 0.0 });
        self.by_pair.insert((source, target), id);
        id
    }

    /// Return (or derive) the inverse Transform `target → source`.
    ///
    /// If the inverse already exists in the store, the existing id is returned.
    /// Otherwise a new derived Transform is created and stored.
    pub fn inverse(&mut self, tid: TransformId) -> Option<TransformId> {
        let (src, tgt, orig_value, orig_evidence) = {
            let t = self.transforms.get(tid as usize)?;
            (t.source, t.target, t.value, t.evidence)
        };
        if let Some(&inv_id) = self.by_pair.get(&(tgt, src)) {
            return Some(inv_id);
        }
        let inv_id = self.transforms.len() as TransformId;
        self.transforms.push(Transform {
            id: inv_id,
            source: tgt,
            target: src,
            kind: TransformKind::Inverse(tid),
            // §18: numeric inverse — negate the original scalar value.
            value: -orig_value,
            evidence: orig_evidence,
        });
        self.by_pair.insert((tgt, src), inv_id);
        Some(inv_id)
    }

    /// Return (or derive) the composition `f ; g` where `f.target == g.source`.
    ///
    /// Returns `None` if the chain does not connect (`f.target != g.source`).
    pub fn compose(&mut self, f: TransformId, g: TransformId) -> Option<TransformId> {
        let (f_src, f_tgt, f_val, f_ev) = {
            let t = self.transforms.get(f as usize)?;
            (t.source, t.target, t.value, t.evidence)
        };
        let (g_src, g_tgt, g_val, g_ev) = {
            let t = self.transforms.get(g as usize)?;
            (t.source, t.target, t.value, t.evidence)
        };
        if f_tgt != g_src {
            return None;
        }
        if let Some(&existing) = self.by_pair.get(&(f_src, g_tgt)) {
            return Some(existing);
        }
        let cid = self.transforms.len() as TransformId;
        self.transforms.push(Transform {
            id: cid,
            source: f_src,
            target: g_tgt,
            kind: TransformKind::Composed(f, g),
            // §18: composed numeric value = f.value + g.value (additive in log space).
            value: f_val + g_val,
            evidence: f_ev.min(g_ev),
        });
        self.by_pair.insert((f_src, g_tgt), cid);
        Some(cid)
    }

    /// Look up a Transform by id.
    pub fn get(&self, tid: TransformId) -> Option<&Transform> {
        self.transforms.get(tid as usize)
    }

    /// Mutable access to a Transform by id (for updating value/evidence).
    pub fn get_mut(&mut self, tid: TransformId) -> Option<&mut Transform> {
        self.transforms.get_mut(tid as usize)
    }

    /// Find an existing Transform by (source, target) pair.
    pub fn find(&self, source: IdentityId, target: IdentityId) -> Option<TransformId> {
        self.by_pair.get(&(source, target)).copied()
    }

    pub fn transform_count(&self) -> usize {
        self.transforms.len()
    }

    /// Iterate all transforms for serialisation.
    pub fn iter_all(&self) -> impl Iterator<Item = &Transform> {
        self.transforms.iter()
    }

    /// §19: Restore a persisted entry during deserialization.
    /// Does NOT trigger merge_identities — caller is responsible for Identity store state.
    pub fn restore_entry(
        &mut self,
        source: IdentityId,
        target: IdentityId,
        kind: TransformKind,
        value: f64,
        evidence: f64,
    ) {
        if self.by_pair.contains_key(&(source, target)) {
            return;
        }
        let id = self.transforms.len() as TransformId;
        self.transforms.push(Transform { id, source, target, kind, value, evidence });
        self.by_pair.insert((source, target), id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_identities() -> (TransformStore, IdentityStore) {
        let mut ids = IdentityStore::new();
        ids.intern_identity(&[1]);
        ids.intern_identity(&[2]);
        ids.intern_identity(&[3]);
        (TransformStore::new(), ids)
    }

    #[test]
    fn tr01_register_direct() {
        let (mut ts, mut ids) = store_with_identities();
        let tid = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let t = ts.get(tid).unwrap();
        assert_eq!(t.source, 0);
        assert_eq!(t.target, 1);
        assert!(matches!(t.kind, TransformKind::Mapping));
    }

    #[test]
    fn tr02_equivalent_view_merges_identities() {
        let (mut ts, mut ids) = store_with_identities();
        ts.register(0, 1, TransformKind::EquivalentView, &mut ids);
        // EquivalentView triggers merge_identities.
        assert_eq!(ids.canonical(0), ids.canonical(1));
    }

    #[test]
    fn tr02b_mapping_does_not_merge_identities() {
        let (mut ts, mut ids) = store_with_identities();
        ts.register(0, 1, TransformKind::Mapping, &mut ids);
        // Mapping must NOT merge.
        assert_ne!(ids.canonical(0), ids.canonical(1));
    }

    #[test]
    fn tr03_register_idempotent() {
        let (mut ts, mut ids) = store_with_identities();
        let t1 = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let t2 = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        assert_eq!(t1, t2);
        assert_eq!(ts.transform_count(), 1);
    }

    #[test]
    fn tr04_inverse_derives_reverse() {
        let (mut ts, mut ids) = store_with_identities();
        let fwd = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let inv = ts.inverse(fwd).unwrap();
        let t = ts.get(inv).unwrap();
        assert_eq!(t.source, 1);
        assert_eq!(t.target, 0);
        assert!(matches!(t.kind, TransformKind::Inverse(f) if f == fwd));
    }

    #[test]
    fn tr05_inverse_idempotent() {
        let (mut ts, mut ids) = store_with_identities();
        let fwd = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let inv1 = ts.inverse(fwd).unwrap();
        let inv2 = ts.inverse(fwd).unwrap();
        assert_eq!(inv1, inv2);
        assert_eq!(ts.transform_count(), 2); // fwd + inverse only
    }

    #[test]
    fn tr06_compose_connects_chain() {
        let (mut ts, mut ids) = store_with_identities();
        let f = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let g = ts.register(1, 2, TransformKind::Mapping, &mut ids);
        let h = ts.compose(f, g).unwrap();
        let t = ts.get(h).unwrap();
        assert_eq!(t.source, 0);
        assert_eq!(t.target, 2);
        assert!(matches!(t.kind, TransformKind::Composed(a, b) if a == f && b == g));
    }

    #[test]
    fn tr07_compose_fails_on_gap() {
        let (mut ts, mut ids) = store_with_identities();
        let f = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let g = ts.register(0, 2, TransformKind::Mapping, &mut ids);
        assert!(ts.compose(f, g).is_none());
    }

    #[test]
    fn tr08_compose_idempotent() {
        let (mut ts, mut ids) = store_with_identities();
        let f = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let g = ts.register(1, 2, TransformKind::Mapping, &mut ids);
        let h1 = ts.compose(f, g).unwrap();
        let h2 = ts.compose(f, g).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn tr09_find_by_pair() {
        let (mut ts, mut ids) = store_with_identities();
        let tid = ts.register(0, 2, TransformKind::Mapping, &mut ids);
        assert_eq!(ts.find(0, 2), Some(tid));
        assert_eq!(ts.find(2, 0), None);
    }

    #[test]
    fn tr10_inverse_of_direct_is_direct() {
        let (mut ts, mut ids) = store_with_identities();
        let fwd = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let rev = ts.register(1, 0, TransformKind::Mapping, &mut ids);
        let inv = ts.inverse(fwd).unwrap();
        assert_eq!(inv, rev);
        assert_eq!(ts.transform_count(), 2);
    }

    #[test]
    fn tr11_value_and_evidence_start_at_zero() {
        let (mut ts, mut ids) = store_with_identities();
        let tid = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let t = ts.get(tid).unwrap();
        assert_eq!(t.value, 0.0);
        assert_eq!(t.evidence, 0.0);
    }

    #[test]
    fn tr12_value_evidence_can_be_updated() {
        let (mut ts, mut ids) = store_with_identities();
        let tid = ts.register(0, 1, TransformKind::Mapping, &mut ids);
        let t = ts.get_mut(tid).unwrap();
        t.value = 1.5;
        t.evidence = 0.8;
        let t = ts.get(tid).unwrap();
        assert!((t.value - 1.5).abs() < 1e-9);
        assert!((t.evidence - 0.8).abs() < 1e-9);
    }
}
