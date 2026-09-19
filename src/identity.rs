/// Identity and View registry.
///
/// Phase 2: Exact Identity — two Views share the same IdentityId iff their
/// Primitive expansions are identical.
///
/// Phase 11: Cross-View Identity — Union-Find merging allows explicitly
/// declaring two Identities equivalent (e.g. via a Transform relation).
/// `merge_identities(a, b)` unions them; `canonical(id)` resolves to the
/// root. Phase 12 Transforms will call merge_identities when they are
/// registered.
use std::collections::HashMap;

use crate::primitives::PrimitiveId;
use crate::units::UnitId;

pub type IdentityId = u32;
pub type ViewId = u32;

/// Registry of Identities and Views.
///
/// `parent[i]` is the Union-Find parent of Identity `i`.
/// A root satisfies `parent[i] == i`.
#[derive(Debug, Default, Clone)]
pub struct IdentityStore {
    // identity side
    prim_seq_to_id: HashMap<Vec<PrimitiveId>, IdentityId>,
    identity_prims: Vec<Vec<PrimitiveId>>,
    // Phase 11: Union-Find parent array (parallel to identity_prims)
    parent: Vec<IdentityId>,

    // view side
    unit_seq_to_id: HashMap<Vec<UnitId>, ViewId>,
    view_units: Vec<Vec<UnitId>>,
    view_identity: Vec<IdentityId>,

    // §7: Content→Identity direct index — O(1) UnitId→IdentityId lookup.
    // Populated at chunk/primitive registration time; derived on load.
    unit_to_identity: HashMap<UnitId, IdentityId>,
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
        self.parent.push(id); // each new Identity is its own root
        id
    }

    /// Intern a View (UnitId sequence) bound to `identity`.
    ///
    /// If the same unit sequence was already interned, returns its existing
    /// ViewId — the identity binding is immutable (first writer wins).
    /// Single-element views also populate the §7 direct index.
    pub fn intern_view(&mut self, units: &[UnitId], identity: IdentityId) -> ViewId {
        if let Some(&id) = self.unit_seq_to_id.get(units) {
            return id;
        }
        let id = self.view_units.len() as ViewId;
        self.view_units.push(units.to_vec());
        self.unit_seq_to_id.insert(units.to_vec(), id);
        self.view_identity.push(identity);
        if units.len() == 1 {
            self.unit_to_identity.entry(units[0]).or_insert(identity);
        }
        id
    }

    // ── Phase 11: Union-Find ──────────────────────────────────────────────

    /// Return the canonical (root) IdentityId for `id`.
    ///
    /// Follows the parent chain without mutating (no path compression here —
    /// chains are short in practice).
    pub fn canonical(&self, mut id: IdentityId) -> IdentityId {
        while (id as usize) < self.parent.len() && self.parent[id as usize] != id {
            id = self.parent[id as usize];
        }
        id
    }

    /// Declare that Identities `a` and `b` are equivalent (Cross-View merge).
    ///
    /// The canonical root of `a`'s cluster becomes the canonical root of the
    /// merged cluster. Idempotent if they are already in the same cluster.
    pub fn merge_identities(&mut self, a: IdentityId, b: IdentityId) {
        let ra = self.canonical(a);
        let rb = self.canonical(b);
        if ra != rb {
            // Make ra the canonical root of both clusters.
            if (rb as usize) < self.parent.len() {
                self.parent[rb as usize] = ra;
            }
        }
    }

    /// Return the IdentityId bound to a ViewId (raw, not canonicalized).
    pub fn identity_of_view(&self, view: ViewId) -> Option<IdentityId> {
        self.view_identity.get(view as usize).copied()
    }

    /// Return the *canonical* IdentityId for the given ViewId.
    ///
    /// Phase 11: follows the Union-Find chain so that views whose Identities
    /// were merged by `merge_identities` resolve to the same canonical root.
    pub fn canonical_identity_of_view(&self, view: ViewId) -> Option<IdentityId> {
        self.identity_of_view(view).map(|id| self.canonical(id))
    }

    /// Return the canonical Primitive sequence for an IdentityId.
    pub fn primitives_of_identity(&self, identity: IdentityId) -> Option<&[PrimitiveId]> {
        self.identity_prims.get(identity as usize).map(Vec::as_slice)
    }

    /// Look up an existing IdentityId for `prims` without inserting (read-only).
    pub fn find_identity(&self, prims: &[PrimitiveId]) -> Option<IdentityId> {
        self.prim_seq_to_id.get(prims).copied()
    }

    // ── §7: Content→Identity direct index ────────────────────────────────

    /// Register a direct UnitId→IdentityId mapping (§7).
    ///
    /// Call this when a chunk is formed or a primitive is first encoded so
    /// that `identity_of_unit` resolves in O(1) without lineage traversal.
    /// First writer wins; idempotent for the same (unit, identity) pair.
    pub fn register_unit_identity(&mut self, unit: UnitId, identity: IdentityId) {
        self.unit_to_identity.entry(unit).or_insert(identity);
    }

    /// O(1) UnitId→IdentityId lookup (§7 direct index).
    ///
    /// Returns None if the unit was never explicitly registered via
    /// `register_unit_identity` or `intern_view` (single-element).
    pub fn identity_of_unit(&self, unit: UnitId) -> Option<IdentityId> {
        self.unit_to_identity.get(&unit).copied()
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

    /// Phase 11: return the full Union-Find parent array for serialisation.
    pub fn all_parents(&self) -> &[IdentityId] {
        &self.parent
    }

    pub fn all_views(&self) -> impl Iterator<Item = (&[UnitId], IdentityId)> {
        self.view_units
            .iter()
            .zip(self.view_identity.iter())
            .map(|(u, &id)| (u.as_slice(), id))
    }

    /// Reconstruct from raw bulk data (called by the persistence layer).
    ///
    /// `parent` is the serialised Union-Find array (Phase 11); pass `None`
    /// or an empty vec to start with all roots (pre-Phase-11 snapshots).
    pub fn from_bulk(
        identities: Vec<Vec<PrimitiveId>>,
        views: Vec<(Vec<UnitId>, IdentityId)>,
        parent: Option<Vec<IdentityId>>,
    ) -> Self {
        let mut store = Self::new();
        for prims in identities {
            let id = store.identity_prims.len() as IdentityId;
            store.prim_seq_to_id.insert(prims.clone(), id);
            store.identity_prims.push(prims);
            store.parent.push(id); // self-root default
        }
        // Override parent array if a valid one was provided.
        if let Some(p) = parent {
            if p.len() == store.parent.len() {
                store.parent = p;
            }
        }
        for (units, identity) in views {
            let id = store.view_units.len() as ViewId;
            if units.len() == 1 {
                store.unit_to_identity.entry(units[0]).or_insert(identity);
            }
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
        let mut store = IdentityStore::new();
        let prims = vec![1u32, 2, 3, 4, 5];
        let identity = store.intern_identity(&prims);

        let view_a_units = vec![
            UnitId::primitive(1), UnitId::primitive(2), UnitId::primitive(3),
            UnitId::primitive(4), UnitId::primitive(5),
        ];
        let view_b_units = vec![
            UnitId::chunk(0), UnitId::primitive(3), UnitId::primitive(4), UnitId::primitive(5),
        ];
        let view_c_units = vec![UnitId::chunk(1), UnitId::chunk(2)];

        let va = store.intern_view(&view_a_units, identity);
        let vb = store.intern_view(&view_b_units, identity);
        let vc = store.intern_view(&view_c_units, identity);

        assert_ne!(va, vb);
        assert_ne!(vb, vc);
        assert_ne!(va, vc);

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

        let idents: Vec<_> = orig.all_identities().to_vec();
        let views: Vec<_> = orig.all_views().map(|(u, id)| (u.to_vec(), id)).collect();
        let mut restored = IdentityStore::from_bulk(idents, views, None);

        assert_eq!(restored.identity_count(), orig.identity_count());
        assert_eq!(restored.view_count(), orig.view_count());
        assert_eq!(
            restored.intern_identity(&[1, 2]),
            orig.prim_seq_to_id[&vec![1u32, 2]]
        );
    }

    // ── Phase 11: Cross-View Identity tests ──────────────────────────────

    #[test]
    fn ci11_merge_identities_canonical() {
        let mut store = IdentityStore::new();
        let a = store.intern_identity(&[1, 2]);
        let b = store.intern_identity(&[3, 4]);
        assert_ne!(store.canonical(a), store.canonical(b), "before merge");
        store.merge_identities(a, b);
        assert_eq!(store.canonical(a), store.canonical(b), "after merge");
    }

    #[test]
    fn ci11_canonical_identity_of_view() {
        let mut store = IdentityStore::new();
        let a = store.intern_identity(&[1, 2]);
        let b = store.intern_identity(&[3, 4]);
        let va = store.intern_view(&[UnitId::primitive(1), UnitId::primitive(2)], a);
        let vb = store.intern_view(&[UnitId::primitive(3), UnitId::primitive(4)], b);

        assert_ne!(
            store.canonical_identity_of_view(va),
            store.canonical_identity_of_view(vb),
            "before merge"
        );
        store.merge_identities(a, b);
        assert_eq!(
            store.canonical_identity_of_view(va),
            store.canonical_identity_of_view(vb),
            "after merge"
        );
    }

    #[test]
    fn ci11_merge_transitive() {
        // A=B then B=C should give A, B, C all the same canonical.
        let mut store = IdentityStore::new();
        let a = store.intern_identity(&[1]);
        let b = store.intern_identity(&[2]);
        let c = store.intern_identity(&[3]);
        store.merge_identities(a, b);
        store.merge_identities(b, c);
        let ca = store.canonical(a);
        let cb = store.canonical(b);
        let cc = store.canonical(c);
        assert_eq!(ca, cb);
        assert_eq!(cb, cc);
    }

    #[test]
    fn ci11_merge_idempotent() {
        let mut store = IdentityStore::new();
        let a = store.intern_identity(&[1]);
        let b = store.intern_identity(&[2]);
        store.merge_identities(a, b);
        let c1 = store.canonical(a);
        store.merge_identities(a, b);
        let c2 = store.canonical(a);
        assert_eq!(c1, c2);
    }

    #[test]
    fn ci11_persistence_roundtrip_with_aliases() {
        let mut orig = IdentityStore::new();
        let a = orig.intern_identity(&[10, 20]);
        let b = orig.intern_identity(&[30, 40]);
        let c = orig.intern_identity(&[50, 60]);
        orig.merge_identities(a, b); // a == b
        // c remains its own root

        let idents = orig.all_identities().to_vec();
        let views: Vec<_> = orig.all_views().map(|(u, id)| (u.to_vec(), id)).collect();
        let parent = orig.all_parents().to_vec();

        let restored = IdentityStore::from_bulk(idents, views, Some(parent));
        assert_eq!(restored.canonical(a), restored.canonical(b), "alias preserved");
        assert_ne!(restored.canonical(a), restored.canonical(c), "non-alias preserved");
    }
}
