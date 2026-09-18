/// Phase 12: Transform / Inverse / Composition.
///
/// A Transform is a directed relation `f: IdentityA → IdentityB` asserting
/// that the two Identities are equivalent under some mapping.  Registering a
/// Transform calls `IdentityStore::merge_identities` so that both ends share
/// a canonical IdentityId.
///
/// Inverses and compositions are derived lazily:
///   inverse(f: A→B)            = g: B→A
///   compose(f: A→B, g: B→C)   = h: A→C  (only when f.target == g.source)
///
/// Derived transforms (inverse / composition results) are stored in the same
/// table and carry a `DerivedFrom` tag so callers can distinguish them from
/// directly registered transforms.
use std::collections::HashMap;

use crate::identity::{IdentityId, IdentityStore};

pub type TransformId = u32;

/// How a Transform was created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformOrigin {
    /// Registered directly by the caller.
    Direct,
    /// Derived as the inverse of another Transform.
    Inverse(TransformId),
    /// Derived as the composition of two Transforms.
    Composed(TransformId, TransformId),
}

/// A single Transform edge: `source → target`.
#[derive(Debug, Clone)]
pub struct Transform {
    pub id: TransformId,
    pub source: IdentityId,
    pub target: IdentityId,
    pub origin: TransformOrigin,
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

    /// Register a direct Transform `source → target`.
    ///
    /// Also calls `identities.merge_identities(source, target)` so that the
    /// two Identities share a canonical root (Phase 11 integration).
    ///
    /// Returns the existing TransformId if the pair was already registered.
    pub fn register(
        &mut self,
        source: IdentityId,
        target: IdentityId,
        identities: &mut IdentityStore,
    ) -> TransformId {
        if let Some(&tid) = self.by_pair.get(&(source, target)) {
            return tid;
        }
        identities.merge_identities(source, target);
        let id = self.transforms.len() as TransformId;
        self.transforms.push(Transform { id, source, target, origin: TransformOrigin::Direct });
        self.by_pair.insert((source, target), id);
        id
    }

    /// Return (or derive) the inverse Transform `target → source`.
    ///
    /// If the inverse already exists in the store, the existing id is returned.
    /// Otherwise a new derived Transform is created and stored.
    pub fn inverse(&mut self, tid: TransformId) -> Option<TransformId> {
        let (src, tgt) = {
            let t = self.transforms.get(tid as usize)?;
            (t.source, t.target)
        };
        if let Some(&inv_id) = self.by_pair.get(&(tgt, src)) {
            return Some(inv_id);
        }
        let inv_id = self.transforms.len() as TransformId;
        self.transforms.push(Transform {
            id: inv_id,
            source: tgt,
            target: src,
            origin: TransformOrigin::Inverse(tid),
        });
        self.by_pair.insert((tgt, src), inv_id);
        Some(inv_id)
    }

    /// Return (or derive) the composition `f ; g` where `f.target == g.source`.
    ///
    /// Returns `None` if the chain does not connect (`f.target != g.source`).
    pub fn compose(&mut self, f: TransformId, g: TransformId) -> Option<TransformId> {
        let (f_src, f_tgt) = {
            let t = self.transforms.get(f as usize)?;
            (t.source, t.target)
        };
        let (g_src, g_tgt) = {
            let t = self.transforms.get(g as usize)?;
            (t.source, t.target)
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
            origin: TransformOrigin::Composed(f, g),
        });
        self.by_pair.insert((f_src, g_tgt), cid);
        Some(cid)
    }

    /// Look up a Transform by id.
    pub fn get(&self, tid: TransformId) -> Option<&Transform> {
        self.transforms.get(tid as usize)
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
        let tid = ts.register(0, 1, &mut ids);
        let t = ts.get(tid).unwrap();
        assert_eq!(t.source, 0);
        assert_eq!(t.target, 1);
        assert_eq!(t.origin, TransformOrigin::Direct);
    }

    #[test]
    fn tr02_register_merges_identities() {
        let (mut ts, mut ids) = store_with_identities();
        ts.register(0, 1, &mut ids);
        // After registering 0→1, their canonical ids must be the same.
        assert_eq!(ids.canonical(0), ids.canonical(1));
    }

    #[test]
    fn tr03_register_idempotent() {
        let (mut ts, mut ids) = store_with_identities();
        let t1 = ts.register(0, 1, &mut ids);
        let t2 = ts.register(0, 1, &mut ids);
        assert_eq!(t1, t2);
        assert_eq!(ts.transform_count(), 1);
    }

    #[test]
    fn tr04_inverse_derives_reverse() {
        let (mut ts, mut ids) = store_with_identities();
        let fwd = ts.register(0, 1, &mut ids);
        let inv = ts.inverse(fwd).unwrap();
        let t = ts.get(inv).unwrap();
        assert_eq!(t.source, 1);
        assert_eq!(t.target, 0);
        assert_eq!(t.origin, TransformOrigin::Inverse(fwd));
    }

    #[test]
    fn tr05_inverse_idempotent() {
        let (mut ts, mut ids) = store_with_identities();
        let fwd = ts.register(0, 1, &mut ids);
        let inv1 = ts.inverse(fwd).unwrap();
        let inv2 = ts.inverse(fwd).unwrap();
        assert_eq!(inv1, inv2);
        assert_eq!(ts.transform_count(), 2); // fwd + inverse only
    }

    #[test]
    fn tr06_compose_connects_chain() {
        let (mut ts, mut ids) = store_with_identities();
        let f = ts.register(0, 1, &mut ids); // 0→1
        let g = ts.register(1, 2, &mut ids); // 1→2
        let h = ts.compose(f, g).unwrap();
        let t = ts.get(h).unwrap();
        assert_eq!(t.source, 0);
        assert_eq!(t.target, 2);
        assert_eq!(t.origin, TransformOrigin::Composed(f, g));
    }

    #[test]
    fn tr07_compose_fails_on_gap() {
        let (mut ts, mut ids) = store_with_identities();
        let f = ts.register(0, 1, &mut ids); // 0→1
        let g = ts.register(0, 2, &mut ids); // 0→2 (not 1→2)
        assert!(ts.compose(f, g).is_none());
    }

    #[test]
    fn tr08_compose_idempotent() {
        let (mut ts, mut ids) = store_with_identities();
        let f = ts.register(0, 1, &mut ids);
        let g = ts.register(1, 2, &mut ids);
        let h1 = ts.compose(f, g).unwrap();
        let h2 = ts.compose(f, g).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn tr09_find_by_pair() {
        let (mut ts, mut ids) = store_with_identities();
        let tid = ts.register(0, 2, &mut ids);
        assert_eq!(ts.find(0, 2), Some(tid));
        assert_eq!(ts.find(2, 0), None);
    }

    #[test]
    fn tr10_inverse_of_direct_is_direct() {
        // Registering B→A directly, then asking for inverse of A→B should
        // return the existing B→A Transform (not create a new one).
        let (mut ts, mut ids) = store_with_identities();
        let fwd = ts.register(0, 1, &mut ids); // 0→1
        let rev = ts.register(1, 0, &mut ids); // 1→0 (already direct)
        let inv = ts.inverse(fwd).unwrap();
        assert_eq!(inv, rev); // same id
        assert_eq!(ts.transform_count(), 2);
    }
}
