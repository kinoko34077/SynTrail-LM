/// Phase 7: Unified Relation Core.
///
/// All relations between UnitIds share a common edge structure tagged by
/// RelationKind.  Phase 12 (Transform) and Phase 13 (Micro-ISA REL.*) will
/// build directly on this foundation.
///
/// Current kinds:
///   Route      — sequential next-unit prediction (from PredictionStore)
///   Adjacency  — window-based co-occurrence    (from AssociationStore)
///   DerivedFrom — chunk derivation lineage      (from LineageStore)
use std::collections::HashMap;

use crate::units::UnitId;

/// Tag identifying the semantics of a relation edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationKind {
    /// Sequential prediction: context → likely next unit.
    Route,
    /// Co-occurrence within a context window.
    Adjacency,
    /// Derivation: a Chunk was formed by merging source and a sibling.
    DerivedFrom,
}

/// A single relation edge in the unified store.
#[derive(Debug, Clone)]
pub struct RelationEdge {
    pub source: UnitId,
    pub target: UnitId,
    pub kind: RelationKind,
    /// Decaying strength accumulator (semantics depend on kind).
    pub strength: f64,
    pub use_count: u32,
}

impl RelationEdge {
    pub fn new(source: UnitId, target: UnitId, kind: RelationKind) -> Self {
        Self { source, target, kind, strength: 0.0, use_count: 0 }
    }
}

/// Unified storage for all relation edges across all kinds.
///
/// Primary key: (source, target, kind) — one edge per unique triple.
/// Secondary index: (source, kind) → edges — for O(K) fan-out queries.
#[derive(Debug, Default, Clone)]
pub struct RelationStore {
    edges: HashMap<(UnitId, UnitId, RelationKind), RelationEdge>,
    /// (source, kind) → list of target UnitIds
    source_kind_index: HashMap<(UnitId, RelationKind), Vec<UnitId>>,
}

impl RelationStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update an edge, bumping strength and use_count.
    pub fn observe(&mut self, source: UnitId, target: UnitId, kind: RelationKind, strength_delta: f64) {
        let key = (source, target, kind);
        let is_new = !self.edges.contains_key(&key);
        let edge = self.edges
            .entry(key)
            .or_insert_with(|| RelationEdge::new(source, target, kind));
        edge.strength += strength_delta;
        edge.use_count += 1;
        if is_new {
            self.source_kind_index.entry((source, kind)).or_default().push(target);
        }
    }

    /// Insert a derivation edge (strength = 1.0, idempotent).
    pub fn record_derivation(&mut self, source: UnitId, target: UnitId) {
        let key = (source, target, RelationKind::DerivedFrom);
        if !self.edges.contains_key(&key) {
            self.edges.insert(key, RelationEdge {
                source, target,
                kind: RelationKind::DerivedFrom,
                strength: 1.0,
                use_count: 1,
            });
            self.source_kind_index
                .entry((source, RelationKind::DerivedFrom))
                .or_default()
                .push(target);
        }
    }

    /// Return all targets of `source` for `kind`, in insertion order.
    pub fn targets_of(&self, source: UnitId, kind: RelationKind) -> &[UnitId] {
        self.source_kind_index
            .get(&(source, kind))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Return an edge if it exists.
    pub fn get(&self, source: UnitId, target: UnitId, kind: RelationKind) -> Option<&RelationEdge> {
        self.edges.get(&(source, target, kind))
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &RelationEdge> {
        self.edges.values()
    }

    pub fn iter_kind(&self, kind: RelationKind) -> impl Iterator<Item = &RelationEdge> {
        self.edges.values().filter(move |e| e.kind == kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(id: u32) -> UnitId { UnitId::primitive(id) }
    fn c(id: u32) -> UnitId { UnitId::chunk(id) }

    #[test]
    fn observe_creates_edge() {
        let mut rs = RelationStore::new();
        rs.observe(p(1), p(2), RelationKind::Route, 1.0);
        assert_eq!(rs.edge_count(), 1);
    }

    #[test]
    fn observe_accumulates_strength() {
        let mut rs = RelationStore::new();
        rs.observe(p(1), p(2), RelationKind::Route, 1.0);
        rs.observe(p(1), p(2), RelationKind::Route, 1.0);
        let edge = rs.get(p(1), p(2), RelationKind::Route).unwrap();
        assert!((edge.strength - 2.0).abs() < 1e-9);
        assert_eq!(edge.use_count, 2);
    }

    #[test]
    fn different_kinds_different_edges() {
        let mut rs = RelationStore::new();
        rs.observe(p(1), p(2), RelationKind::Route, 1.0);
        rs.observe(p(1), p(2), RelationKind::Adjacency, 1.0);
        assert_eq!(rs.edge_count(), 2);
    }

    #[test]
    fn targets_of_returns_correct_kind() {
        let mut rs = RelationStore::new();
        rs.observe(p(1), p(2), RelationKind::Route, 1.0);
        rs.observe(p(1), p(3), RelationKind::Route, 1.0);
        rs.observe(p(1), p(4), RelationKind::Adjacency, 1.0);
        let routes = rs.targets_of(p(1), RelationKind::Route);
        assert_eq!(routes.len(), 2);
        assert!(routes.contains(&p(2)));
        assert!(routes.contains(&p(3)));
        let adj = rs.targets_of(p(1), RelationKind::Adjacency);
        assert_eq!(adj.len(), 1);
    }

    #[test]
    fn record_derivation_is_idempotent() {
        let mut rs = RelationStore::new();
        rs.record_derivation(c(0), p(1));
        rs.record_derivation(c(0), p(1));
        assert_eq!(rs.edge_count(), 1);
    }

    #[test]
    fn iter_kind_filters_correctly() {
        let mut rs = RelationStore::new();
        rs.observe(p(1), p(2), RelationKind::Route, 1.0);
        rs.observe(p(3), p(4), RelationKind::Adjacency, 1.0);
        rs.record_derivation(c(0), p(5));
        let routes: Vec<_> = rs.iter_kind(RelationKind::Route).collect();
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].kind, RelationKind::Route);
    }

    #[test]
    fn all_three_kinds_in_one_store() {
        // Completion condition for Phase 7: Route/Adjacency/DerivedFrom coexist.
        let mut rs = RelationStore::new();
        rs.observe(p(1), p(2), RelationKind::Route, 1.0);
        rs.observe(p(1), p(3), RelationKind::Adjacency, 1.0);
        rs.record_derivation(c(0), p(1));
        assert_eq!(rs.iter_kind(RelationKind::Route).count(), 1);
        assert_eq!(rs.iter_kind(RelationKind::Adjacency).count(), 1);
        assert_eq!(rs.iter_kind(RelationKind::DerivedFrom).count(), 1);
    }
}
