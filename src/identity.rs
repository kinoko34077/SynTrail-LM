/// Phase 2: Identity and View registry — Exact Identity only.
///
/// Identity: canonical Primitive expansion (Vec<PrimitiveId>).
/// View: specific Chunk tree / segmentation (Vec<UnitId>).
///
/// Exact Identity rule: two Views share the same IdentityId iff their
/// Primitive expansions are identical.
///
/// Phase 11+ will add Cross-View Identity (semantic similarity).
use std::collections::HashMap;

use crate::primitives::PrimitiveId;
use crate::units::UnitId;

pub type IdentityId = u32;
pub type ViewId = u32;

/// Registry of Identities and Views.
#[derive(Debug, Default, Clone)]
pub struct IdentityStore {
    // identity side
    prim_seq_to_id: HashMap<Vec<PrimitiveId>, IdentityId>,
    identity_prims: Vec<Vec<PrimitiveId>>,

    // view side
    unit_seq_to_id: HashMap<Vec<UnitId>, ViewId>,
    view_units: Vec<Vec<UnitId>>,
    view_identity: Vec<IdentityId>,
}

impl IdentityStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a Primitive sequence, returning a stable IdentityId.
    pub fn intern_identity(&mut self, prims: &[PrimitiveId]) -> IdentityId {
        if let Some(&id) = self.prim_seq_to_id.get(prims) {
            return id;
        }
        let id = self.identity_prims.len() as IdentityId;
        self.identity_prims.push(prims.to_vec());
        self.prim_seq_to_id.insert(prims.to_vec(), id);
        id
    }

    /// Intern a View (UnitId sequence) bound to `identity`.
    ///
    /// If the same unit sequence was already interned, returns its existing
    /// ViewId — the identity binding is immutable (first writer wins).
    pub fn intern_view(&mut self, units: &[UnitId], identity: IdentityId) -> ViewId {
        if let Some(&id) = self.unit_seq_to_id.get(units) {
            return id;
        }
        let id = self.view_units.len() as ViewId;
        self.view_units.push(units.to_vec());
        self.unit_seq_to_id.insert(units.to_vec(), id);
        self.view_identity.push(identity);
        id
    }

    /// Return the IdentityId bound to a ViewId.
    pub fn identity_of_view(&self, view: ViewId) -> Option<IdentityId> {
        self.view_identity.get(view as usize).copied()
    }

    /// Return the canonical Primitive sequence for an IdentityId.
    pub fn primitives_of_identity(&self, identity: IdentityId) -> Option<&[PrimitiveId]> {
        self.identity_prims.get(identity as usize).map(Vec::as_slice)
    }

    pub fn identity_count(&self) -> usize {
        self.identity_prims.len()
    }

    pub fn view_count(&self) -> usize {
        self.view_units.len()
    }

    // ── Bulk access for persistence ───────────────────────────────────────

    pub fn all_identities(&self) -> &[Vec<PrimitiveId>] {
        &self.identity_prims
    }

    pub fn all_views(&self) -> impl Iterator<Item = (&[UnitId], IdentityId)> {
        self.view_units
            .iter()
            .zip(self.view_identity.iter())
            .map(|(u, &id)| (u.as_slice(), id))
    }

    /// Reconstruct from raw bulk data (called by the persistence layer).
    pub fn from_bulk(
        identities: Vec<Vec<PrimitiveId>>,
        views: Vec<(Vec<UnitId>, IdentityId)>,
    ) -> Self {
        let mut store = Self::new();
        for prims in identities {
            let id = store.identity_prims.len() as IdentityId;
            store.prim_seq_to_id.insert(prims.clone(), id);
            store.identity_prims.push(prims);
        }
        for (units, identity) in views {
            let id = store.view_units.len() as ViewId;
            store.unit_seq_to_id.insert(units.clone(), id);
            store.view_units.push(units);
            store.view_identity.push(identity);
        }
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_prims_same_identity() {
        let mut store = IdentityStore::new();
        let id1 = store.intern_identity(&[1, 2, 3]);
        let id2 = store.intern_identity(&[1, 2, 3]);
        assert_eq!(id1, id2);
    }

    #[test]
    fn different_prims_different_identity() {
        let mut store = IdentityStore::new();
        let id1 = store.intern_identity(&[1, 2, 3]);
        let id2 = store.intern_identity(&[1, 2, 4]);
        assert_ne!(id1, id2);
    }

    #[test]
    fn different_chunk_trees_same_identity() {
        // ABCDE / AB+CDE / ABC+DE all expand to primitives [1,2,3,4,5]
        let mut store = IdentityStore::new();
        let prims = vec![1u32, 2, 3, 4, 5];
        let identity = store.intern_identity(&prims);

        // View A: all primitives
        let view_a_units = vec![
            UnitId::primitive(1),
            UnitId::primitive(2),
            UnitId::primitive(3),
            UnitId::primitive(4),
            UnitId::primitive(5),
        ];
        // View B: chunk(AB) + primitives(C,D,E) — C(0) represents AB
        let view_b_units = vec![
            UnitId::chunk(0),
            UnitId::primitive(3),
            UnitId::primitive(4),
            UnitId::primitive(5),
        ];
        // View C: chunk(ABC) + chunk(DE) — C(1) represents ABC, C(2) represents DE
        let view_c_units = vec![UnitId::chunk(1), UnitId::chunk(2)];

        let va = store.intern_view(&view_a_units, identity);
        let vb = store.intern_view(&view_b_units, identity);
        let vc = store.intern_view(&view_c_units, identity);

        // All three views are distinct objects
        assert_ne!(va, vb);
        assert_ne!(vb, vc);
        assert_ne!(va, vc);

        // All three resolve to the same Identity
        assert_eq!(store.identity_of_view(va), Some(identity));
        assert_eq!(store.identity_of_view(vb), Some(identity));
        assert_eq!(store.identity_of_view(vc), Some(identity));
    }

    #[test]
    fn same_unit_seq_same_view() {
        let mut store = IdentityStore::new();
        let units = vec![UnitId::primitive(1), UnitId::primitive(2)];
        let iid = store.intern_identity(&[1, 2]);
        let v1 = store.intern_view(&units, iid);
        let v2 = store.intern_view(&units, iid);
        assert_eq!(v1, v2);
    }

    #[test]
    fn identity_primitives_roundtrip() {
        let mut store = IdentityStore::new();
        let prims = vec![10u32, 20, 30];
        let id = store.intern_identity(&prims);
        assert_eq!(store.primitives_of_identity(id), Some(prims.as_slice()));
    }

    #[test]
    fn from_bulk_roundtrip() {
        let mut orig = IdentityStore::new();
        let id0 = orig.intern_identity(&[1, 2]);
        let id1 = orig.intern_identity(&[3, 4]);
        let units_a = vec![UnitId::primitive(1), UnitId::primitive(2)];
        let units_b = vec![UnitId::chunk(0)];
        orig.intern_view(&units_a, id0);
        orig.intern_view(&units_b, id0);
        orig.intern_view(&vec![UnitId::primitive(3), UnitId::primitive(4)], id1);

        // Reconstruct from bulk
        let idents: Vec<_> = orig.all_identities().to_vec();
        let views: Vec<_> = orig.all_views().map(|(u, id)| (u.to_vec(), id)).collect();
        let mut restored = IdentityStore::from_bulk(idents, views);

        assert_eq!(restored.identity_count(), orig.identity_count());
        assert_eq!(restored.view_count(), orig.view_count());
        assert_eq!(
            restored.intern_identity(&[1, 2]),
            orig.prim_seq_to_id[&vec![1u32, 2]]
        );
    }
}
