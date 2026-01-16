//! Common data structures for the BHC compiler.
//!
//! This crate re-exports and provides wrappers around commonly used
//! data structures, ensuring consistent hashing and performance.

#![warn(missing_docs)]

use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;

/// A hash map using `FxHasher` for fast hashing.
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// A hash set using `FxHasher` for fast hashing.
pub type FxHashSet<T> = std::collections::HashSet<T, BuildHasherDefault<FxHasher>>;

/// An insertion-ordered hash map.
pub type FxIndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// An insertion-ordered hash set.
pub type FxIndexSet<T> = indexmap::IndexSet<T, BuildHasherDefault<FxHasher>>;

// Re-export commonly used types
pub use indexmap::{IndexMap, IndexSet};
pub use parking_lot::{Mutex, RwLock};
pub use smallvec::SmallVec;

/// A small vector optimized for holding 0-4 elements inline.
pub type TinyVec<T> = SmallVec<[T; 4]>;

/// A small vector optimized for holding 0-8 elements inline.
pub type SmallVec8<T> = SmallVec<[T; 8]>;

/// Extension trait for creating `FxHashMap` instances.
pub trait FxHashMapExt<K, V> {
    /// Create a new empty map.
    fn new() -> Self;
    /// Create a new map with the given capacity.
    fn with_capacity(capacity: usize) -> Self;
}

impl<K, V> FxHashMapExt<K, V> for FxHashMap<K, V> {
    fn new() -> Self {
        Self::default()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, BuildHasherDefault::default())
    }
}

/// Extension trait for creating `FxHashSet` instances.
pub trait FxHashSetExt<T> {
    /// Create a new empty set.
    fn new() -> Self;
    /// Create a new set with the given capacity.
    fn with_capacity(capacity: usize) -> Self;
}

impl<T> FxHashSetExt<T> for FxHashSet<T> {
    fn new() -> Self {
        Self::default()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, BuildHasherDefault::default())
    }
}

/// A work queue for graph traversals.
#[derive(Debug, Clone)]
pub struct WorkQueue<T> {
    queue: std::collections::VecDeque<T>,
    seen: FxHashSet<T>,
}

impl<T: std::hash::Hash + Eq + Clone> WorkQueue<T> {
    /// Create a new empty work queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: std::collections::VecDeque::new(),
            seen: FxHashSet::default(),
        }
    }

    /// Add an item to the queue if it hasn't been seen before.
    pub fn push(&mut self, item: T) -> bool {
        if self.seen.insert(item.clone()) {
            self.queue.push_back(item);
            true
        } else {
            false
        }
    }

    /// Pop an item from the queue.
    pub fn pop(&mut self) -> Option<T> {
        self.queue.pop_front()
    }

    /// Check if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl<T: std::hash::Hash + Eq + Clone> Default for WorkQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A frozen hash map that becomes immutable after construction.
#[derive(Debug, Clone)]
pub struct FrozenMap<K, V> {
    map: FxHashMap<K, V>,
}

impl<K: std::hash::Hash + Eq, V> FrozenMap<K, V> {
    /// Create a frozen map from a hash map.
    #[must_use]
    pub fn new(map: FxHashMap<K, V>) -> Self {
        Self { map }
    }

    /// Get a value by key.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Check if a key exists.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Get the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate over the entries.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }
}

impl<K: std::hash::Hash + Eq, V> FromIterator<(K, V)> for FrozenMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

/// A union-find (disjoint set) data structure.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Create a new union-find structure with `n` elements.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the representative of the set containing `x`.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing `x` and `y`.
    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);

        if rx == ry {
            return;
        }

        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }

    /// Check if `x` and `y` are in the same set.
    pub fn same_set(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_queue() {
        let mut wq: WorkQueue<i32> = WorkQueue::new();

        assert!(wq.push(1));
        assert!(wq.push(2));
        assert!(!wq.push(1)); // Already seen

        assert_eq!(wq.pop(), Some(1));
        assert_eq!(wq.pop(), Some(2));
        assert_eq!(wq.pop(), None);
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);

        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 3);

        assert!(uf.same_set(0, 3));
        assert!(!uf.same_set(0, 4));
    }
}
