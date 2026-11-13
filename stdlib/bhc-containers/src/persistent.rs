//! Persistent ordered maps and sets, behind `Data.Map` and `Data.Set`.
//!
//! `insert`, `delete` and the like give a new map and leave the old one as
//! it was. Over a `BTreeMap` that meant copying every entry first, so one
//! insert cost O(n) time and, with no collector, O(n) memory for good. Here
//! a map is a weight-balanced binary tree with shared nodes: a copy is a
//! reference to the same root, and an insert or delete rebuilds only the
//! path from the root to the key, O(log n) nodes.
//!
//! The balancing is that of Haskell's `containers` (Adams' trees with
//! `delta = 3`, `ratio = 2`), so the shape and the costs are the ones a
//! Haskell programmer expects of `Data.Map`.
//!
//! The method names follow `BTreeMap` and `BTreeSet`, which this replaced.

use std::cmp::Ordering;
use std::sync::Arc;

/// A subtree may be at most this many times the size of its sibling.
const DELTA: usize = 3;
/// Whether a single or a double rotation restores the balance.
const RATIO: usize = 2;

struct Node<K, V> {
    size: usize,
    key: K,
    value: V,
    left: Tree<K, V>,
    right: Tree<K, V>,
}

type Tree<K, V> = Option<Arc<Node<K, V>>>;

fn size<K, V>(tree: &Tree<K, V>) -> usize {
    tree.as_ref().map_or(0, |node| node.size)
}

fn node<K, V>(left: Tree<K, V>, key: K, value: V, right: Tree<K, V>) -> Tree<K, V> {
    Some(Arc::new(Node {
        size: 1 + size(&left) + size(&right),
        key,
        value,
        left,
        right,
    }))
}

/// `node`, after an insert or delete changed the size of one side by one.
fn balance<K: Clone, V: Clone>(
    left: Tree<K, V>,
    key: K,
    value: V,
    right: Tree<K, V>,
) -> Tree<K, V> {
    let (ls, rs) = (size(&left), size(&right));
    if ls + rs <= 1 {
        return node(left, key, value, right);
    }
    if rs > DELTA * ls {
        // The right side is too heavy: rotate left.
        let Some(r) = right else {
            return node(left, key, value, right);
        };
        if size(&r.left) < RATIO * size(&r.right) {
            let inner = node(left, key, value, r.left.clone());
            return node(inner, r.key.clone(), r.value.clone(), r.right.clone());
        }
        let Some(rl) = r.left.as_ref() else {
            return node(left, key, value, Some(r));
        };
        let new_left = node(left, key, value, rl.left.clone());
        let new_right = node(
            rl.right.clone(),
            r.key.clone(),
            r.value.clone(),
            r.right.clone(),
        );
        return node(new_left, rl.key.clone(), rl.value.clone(), new_right);
    }
    if ls > DELTA * rs {
        // The left side is too heavy: rotate right.
        let Some(l) = left else {
            return node(left, key, value, right);
        };
        if size(&l.right) < RATIO * size(&l.left) {
            let inner = node(l.right.clone(), key, value, right);
            return node(l.left.clone(), l.key.clone(), l.value.clone(), inner);
        }
        let Some(lr) = l.right.as_ref() else {
            return node(Some(l), key, value, right);
        };
        let new_left = node(
            l.left.clone(),
            l.key.clone(),
            l.value.clone(),
            lr.left.clone(),
        );
        let new_right = node(lr.right.clone(), key, value, right);
        return node(new_left, lr.key.clone(), lr.value.clone(), new_right);
    }
    node(left, key, value, right)
}

fn insert<K: Ord + Clone, V: Clone>(
    tree: &Tree<K, V>,
    key: K,
    value: V,
) -> (Tree<K, V>, Option<V>) {
    let Some(n) = tree else {
        return (node(None, key, value, None), None);
    };
    match key.cmp(&n.key) {
        Ordering::Less => {
            let (left, old) = insert(&n.left, key, value);
            let tree = if old.is_some() {
                node(left, n.key.clone(), n.value.clone(), n.right.clone())
            } else {
                balance(left, n.key.clone(), n.value.clone(), n.right.clone())
            };
            (tree, old)
        }
        Ordering::Greater => {
            let (right, old) = insert(&n.right, key, value);
            let tree = if old.is_some() {
                node(n.left.clone(), n.key.clone(), n.value.clone(), right)
            } else {
                balance(n.left.clone(), n.key.clone(), n.value.clone(), right)
            };
            (tree, old)
        }
        // The new key replaces the old one, as `BTreeMap::insert` keeps the
        // old key but the two are equal; the value is the new one.
        Ordering::Equal => (
            node(n.left.clone(), n.key.clone(), value, n.right.clone()),
            Some(n.value.clone()),
        ),
    }
}

/// The tree without its least entry, and that entry.
fn remove_min<K: Clone, V: Clone>(n: &Node<K, V>) -> (Tree<K, V>, (K, V)) {
    match &n.left {
        None => (n.right.clone(), (n.key.clone(), n.value.clone())),
        Some(left) => {
            let (left, min) = remove_min(left);
            (
                balance(left, n.key.clone(), n.value.clone(), n.right.clone()),
                min,
            )
        }
    }
}

/// The tree without its greatest entry, and that entry.
fn remove_max<K: Clone, V: Clone>(n: &Node<K, V>) -> (Tree<K, V>, (K, V)) {
    match &n.right {
        None => (n.left.clone(), (n.key.clone(), n.value.clone())),
        Some(right) => {
            let (right, max) = remove_max(right);
            (
                balance(n.left.clone(), n.key.clone(), n.value.clone(), right),
                max,
            )
        }
    }
}

/// Two trees, every key of the first below every key of the second, as one.
fn glue<K: Clone, V: Clone>(left: &Tree<K, V>, right: &Tree<K, V>) -> Tree<K, V> {
    match (left, right) {
        (None, _) => right.clone(),
        (_, None) => left.clone(),
        (Some(l), Some(r)) => {
            if l.size > r.size {
                let (left, (key, value)) = remove_max(l);
                balance(left, key, value, right.clone())
            } else {
                let (right, (key, value)) = remove_min(r);
                balance(left.clone(), key, value, right)
            }
        }
    }
}

fn remove<K: Ord + Clone, V: Clone>(tree: &Tree<K, V>, key: &K) -> (Tree<K, V>, Option<V>) {
    let Some(n) = tree else {
        return (None, None);
    };
    match key.cmp(&n.key) {
        Ordering::Less => {
            let (left, old) = remove(&n.left, key);
            if old.is_none() {
                return (tree.clone(), None);
            }
            (
                balance(left, n.key.clone(), n.value.clone(), n.right.clone()),
                old,
            )
        }
        Ordering::Greater => {
            let (right, old) = remove(&n.right, key);
            if old.is_none() {
                return (tree.clone(), None);
            }
            (
                balance(n.left.clone(), n.key.clone(), n.value.clone(), right),
                old,
            )
        }
        Ordering::Equal => (glue(&n.left, &n.right), Some(n.value.clone())),
    }
}

/// A persistent ordered map. `clone` is O(1) and shares every node.
pub struct Map<K, V> {
    root: Tree<K, V>,
}

impl<K, V> Clone for Map<K, V> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
        }
    }
}

impl<K, V> Default for Map<K, V> {
    fn default() -> Self {
        Self { root: None }
    }
}

impl<K: Ord + Clone, V: Clone> Map<K, V> {
    /// The empty map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of entries. O(1).
    #[must_use]
    pub fn len(&self) -> usize {
        size(&self.root)
    }

    /// Whether the map has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// The value at `key`. O(log n).
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut tree = &self.root;
        while let Some(n) = tree {
            match key.cmp(&n.key) {
                Ordering::Less => tree = &n.left,
                Ordering::Greater => tree = &n.right,
                Ordering::Equal => return Some(&n.value),
            }
        }
        None
    }

    /// Whether `key` has a value.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Set the value at `key`, returning the one it replaced. O(log n), and
    /// maps this one was cloned from are unchanged.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (root, old) = insert(&self.root, key, value);
        self.root = root;
        old
    }

    /// Remove the entry at `key`, returning its value. O(log n).
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (root, old) = remove(&self.root, key);
        self.root = root;
        old
    }

    /// Remove and return the least entry.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let n = self.root.clone()?;
        let (root, min) = remove_min(&n);
        self.root = root;
        Some(min)
    }

    /// Remove and return the greatest entry.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        let n = self.root.clone()?;
        let (root, max) = remove_max(&n);
        self.root = root;
        Some(max)
    }

    /// The entries in ascending key order.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K, V> {
        let mut iter = Iter { stack: Vec::new() };
        iter.descend(&self.root);
        iter
    }

    /// The entries in descending key order.
    #[must_use]
    pub fn iter_rev(&self) -> RevIter<'_, K, V> {
        let mut iter = RevIter { stack: Vec::new() };
        iter.descend(&self.root);
        iter
    }

    /// The greatest entry. O(log n).
    #[must_use]
    pub fn last(&self) -> Option<(&K, &V)> {
        let mut n = self.root.as_ref()?;
        while let Some(right) = &n.right {
            n = right;
        }
        Some((&n.key, &n.value))
    }

    /// The keys in ascending order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(key, _)| key)
    }

    /// The values in ascending key order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, value)| value)
    }
}

impl<'a, K: Ord + Clone, V: Clone> IntoIterator for &'a Map<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Ord + Clone, V: Clone> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(entries: I) -> Self {
        let mut map = Self::new();
        for (key, value) in entries {
            map.insert(key, value);
        }
        map
    }
}

/// The entries of a [`Map`] in ascending key order.
pub struct Iter<'a, K, V> {
    /// The nodes still to visit, each with its right subtree, nearest last.
    stack: Vec<&'a Node<K, V>>,
}

impl<'a, K, V> Iter<'a, K, V> {
    fn descend(&mut self, mut tree: &'a Tree<K, V>) {
        while let Some(n) = tree {
            self.stack.push(n);
            tree = &n.left;
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.stack.pop()?;
        self.descend(&n.right);
        Some((&n.key, &n.value))
    }
}

/// The entries of a [`Map`] in descending key order.
pub struct RevIter<'a, K, V> {
    /// The nodes still to visit, each with its left subtree, nearest last.
    stack: Vec<&'a Node<K, V>>,
}

impl<'a, K, V> RevIter<'a, K, V> {
    fn descend(&mut self, mut tree: &'a Tree<K, V>) {
        while let Some(n) = tree {
            self.stack.push(n);
            tree = &n.right;
        }
    }
}

impl<'a, K, V> Iterator for RevIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.stack.pop()?;
        self.descend(&n.left);
        Some((&n.key, &n.value))
    }
}

/// A persistent ordered set. `clone` is O(1) and shares every node.
pub struct Set<K> {
    map: Map<K, ()>,
}

impl<K> Clone for Set<K> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<K> Default for Set<K> {
    fn default() -> Self {
        Self {
            map: Map::default(),
        }
    }
}

impl<K: Ord + Clone> Set<K> {
    /// The empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of elements. O(1).
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the set has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Whether `key` is an element. O(log n).
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Add `key`; whether it was new. O(log n).
    pub fn insert(&mut self, key: K) -> bool {
        self.map.insert(key, ()).is_none()
    }

    /// Remove `key`; whether it was an element. O(log n).
    pub fn remove(&mut self, key: &K) -> bool {
        self.map.remove(key).is_some()
    }

    /// Remove and return the least element.
    pub fn pop_first(&mut self) -> Option<K> {
        self.map.pop_first().map(|(key, ())| key)
    }

    /// Remove and return the greatest element.
    pub fn pop_last(&mut self) -> Option<K> {
        self.map.pop_last().map(|(key, ())| key)
    }

    /// The elements in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    /// The elements in descending order.
    pub fn iter_rev(&self) -> impl Iterator<Item = &K> {
        self.map.iter_rev().map(|(key, ())| key)
    }

    /// The greatest element. O(log n).
    #[must_use]
    pub fn last(&self) -> Option<&K> {
        self.map.last().map(|(key, ())| key)
    }

    /// The elements of either set, in ascending order.
    pub fn union<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a K> {
        let mut merged: Vec<&K> = Vec::with_capacity(self.len() + other.len());
        let (mut a, mut b) = (self.iter().peekable(), other.iter().peekable());
        loop {
            let next = match (a.peek(), b.peek()) {
                (Some(x), Some(y)) => match x.cmp(y) {
                    Ordering::Less => a.next(),
                    Ordering::Greater => b.next(),
                    Ordering::Equal => {
                        b.next();
                        a.next()
                    }
                },
                (Some(_), None) => a.next(),
                (None, _) => b.next(),
            };
            match next {
                Some(key) => merged.push(key),
                None => break,
            }
        }
        merged.into_iter()
    }

    /// The elements of both sets, in ascending order.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a K> {
        self.iter().filter(move |key| other.contains(key))
    }

    /// The elements of this set that are not in `other`, in ascending order.
    pub fn difference<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a K> {
        self.iter().filter(move |key| !other.contains(key))
    }

    /// Whether every element of this set is in `other`.
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.iter().all(|key| other.contains(key))
    }
}

impl<'a, K: Ord + Clone> IntoIterator for &'a Set<K> {
    type Item = &'a K;
    type IntoIter = std::iter::Map<Iter<'a, K, ()>, fn((&'a K, &'a ())) -> &'a K>;

    fn into_iter(self) -> Self::IntoIter {
        fn key<'a, K>((key, ()): (&'a K, &'a ())) -> &'a K {
            key
        }
        self.map.iter().map(key)
    }
}

impl<K: Ord + Clone> FromIterator<K> for Set<K> {
    fn from_iter<I: IntoIterator<Item = K>>(keys: I) -> Self {
        let mut set = Self::new();
        for key in keys {
            set.insert(key);
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// The size of every node is right and no side outweighs the other by
    /// more than `DELTA`.
    fn check<K: Ord, V>(tree: &Tree<K, V>) -> usize {
        let Some(n) = tree else {
            return 0;
        };
        let (ls, rs) = (check(&n.left), check(&n.right));
        assert_eq!(n.size, 1 + ls + rs);
        if ls + rs > 1 {
            assert!(
                ls <= DELTA * rs && rs <= DELTA * ls,
                "unbalanced: {ls} / {rs}"
            );
        }
        if let Some(l) = &n.left {
            assert!(l.key < n.key);
        }
        if let Some(r) = &n.right {
            assert!(r.key > n.key);
        }
        n.size
    }

    /// A small deterministic generator, so the test needs no seed.
    fn numbers(count: usize, modulus: u64) -> Vec<u64> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state % modulus
            })
            .collect()
    }

    #[test]
    fn a_map_agrees_with_btreemap_and_stays_balanced() {
        let mut ours: Map<u64, u64> = Map::new();
        let mut theirs: BTreeMap<u64, u64> = BTreeMap::new();
        for (step, key) in numbers(4000, 700).into_iter().enumerate() {
            let value = step as u64;
            if step % 3 == 2 {
                assert_eq!(ours.remove(&key), theirs.remove(&key));
            } else {
                assert_eq!(ours.insert(key, value), theirs.insert(key, value));
            }
            assert_eq!(ours.len(), theirs.len());
            if step % 97 == 0 {
                check(&ours.root);
                assert!(ours.iter().eq(theirs.iter()));
            }
        }
        check(&ours.root);
        assert!(ours.iter().eq(theirs.iter()));
        assert!(ours.keys().eq(theirs.keys()));
        assert!(ours.values().eq(theirs.values()));
        assert!(ours.iter_rev().eq(theirs.iter().rev()));
        assert_eq!(ours.last(), theirs.last_key_value());
        for key in 0..700 {
            assert_eq!(ours.get(&key), theirs.get(&key));
            assert_eq!(ours.contains_key(&key), theirs.contains_key(&key));
        }
        assert_eq!(ours.pop_first(), theirs.pop_first());
        assert_eq!(ours.pop_last(), theirs.pop_last());
        check(&ours.root);
        assert!(ours.iter().eq(theirs.iter()));
    }

    #[test]
    fn ascending_and_descending_inserts_stay_balanced() {
        let up: Map<u32, ()> = (0..5000).map(|k| (k, ())).collect();
        let down: Map<u32, ()> = (0..5000).rev().map(|k| (k, ())).collect();
        assert_eq!(check(&up.root), 5000);
        assert_eq!(check(&down.root), 5000);
        assert!(up.keys().eq(down.keys()));
    }

    #[test]
    fn a_change_leaves_the_map_it_started_from_alone() {
        let before: Map<u32, &str> = [(1, "one"), (2, "two"), (3, "three")].into_iter().collect();
        let mut after = before.clone();
        after.insert(2, "TWO");
        after.insert(4, "four");
        after.remove(&1);
        assert_eq!(
            before.iter().collect::<Vec<_>>(),
            [(&1, &"one"), (&2, &"two"), (&3, &"three")]
        );
        assert_eq!(
            after.iter().collect::<Vec<_>>(),
            [(&2, &"TWO"), (&3, &"three"), (&4, &"four")]
        );
        let mut empty: Map<u32, &str> = Map::new();
        assert!(empty.is_empty());
        assert_eq!(empty.remove(&1), None);
        assert_eq!(empty.pop_first(), None);
    }

    #[test]
    fn a_set_agrees_with_btreeset() {
        let a_keys = numbers(600, 400);
        let b_keys: Vec<u64> = numbers(900, 400).into_iter().skip(300).collect();
        let (a, b): (Set<u64>, Set<u64>) = (
            a_keys.iter().copied().collect(),
            b_keys.iter().copied().collect(),
        );
        let (x, y): (BTreeSet<u64>, BTreeSet<u64>) = (
            a_keys.iter().copied().collect(),
            b_keys.iter().copied().collect(),
        );
        assert_eq!(a.len(), x.len());
        assert!(a.iter().eq(x.iter()));
        assert!(a.union(&b).eq(x.union(&y)));
        assert!(a.intersection(&b).eq(x.intersection(&y)));
        assert!(a.difference(&b).eq(x.difference(&y)));
        assert_eq!(a.is_subset(&b), x.is_subset(&y));
        let both: Set<u64> = a.intersection(&b).copied().collect();
        assert!(both.is_subset(&a) && both.is_subset(&b));
        let (mut a, mut x) = (a, x);
        assert_eq!(a.pop_first(), x.pop_first());
        assert_eq!(a.pop_last(), x.pop_last());
        assert_eq!(a.insert(1_000), x.insert(1_000));
        assert_eq!(a.insert(1_000), x.insert(1_000));
        assert_eq!(a.remove(&1_000), x.remove(&1_000));
        assert_eq!(a.contains(&7), x.contains(&7));
        assert!(a.iter_rev().eq(x.iter().rev()));
        assert_eq!(a.last(), x.last());
        assert!(a.iter().eq(x.iter()));
    }
}
