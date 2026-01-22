//! Immutable ordered maps
//!
//! This module provides an immutable map data structure based on weight-balanced
//! binary search trees (also known as Adams trees or bounded balance trees).
//!
//! # Performance
//!
//! | Operation | Complexity |
//! |-----------|------------|
//! | lookup    | O(log n)   |
//! | insert    | O(log n)   |
//! | delete    | O(log n)   |
//! | union     | O(m log(n/m + 1)), m <= n |
//!
//! # Example
//!
//! ```ignore
//! let m = Map::empty()
//!     .insert(1, "one")
//!     .insert(2, "two")
//!     .insert(3, "three");
//!
//! assert_eq!(m.lookup(&2), Some(&"two"));
//! assert_eq!(m.size(), 3);
//! ```

use std::cmp::Ordering;
use std::fmt::{self, Debug};
use std::iter::FromIterator;
use std::rc::Rc;

// Weight balance parameters (Adams' balance criterion)
const DELTA: usize = 3;
const GAMMA: usize = 2;

/// An immutable ordered map based on weight-balanced binary search trees.
#[derive(Clone)]
pub struct Map<K, V> {
    root: Option<Rc<Node<K, V>>>,
}

#[derive(Clone)]
struct Node<K, V> {
    key: K,
    value: V,
    size: usize,
    left: Option<Rc<Node<K, V>>>,
    right: Option<Rc<Node<K, V>>>,
}

impl<K, V> Node<K, V> {
    fn new(key: K, value: V, left: Option<Rc<Node<K, V>>>, right: Option<Rc<Node<K, V>>>) -> Self {
        let size = 1 + node_size(&left) + node_size(&right);
        Node {
            key,
            value,
            size,
            left,
            right,
        }
    }

    fn singleton(key: K, value: V) -> Self {
        Node {
            key,
            value,
            size: 1,
            left: None,
            right: None,
        }
    }
}

fn node_size<K, V>(node: &Option<Rc<Node<K, V>>>) -> usize {
    node.as_ref().map_or(0, |n| n.size)
}

impl<K: Ord, V> Map<K, V> {
    /// Create an empty map.
    ///
    /// O(1) time and space.
    #[inline]
    pub fn empty() -> Self {
        Map { root: None }
    }

    /// Create a map with a single element.
    ///
    /// O(1) time and space.
    #[inline]
    pub fn singleton(key: K, value: V) -> Self {
        Map {
            root: Some(Rc::new(Node::singleton(key, value))),
        }
    }

    /// Check if the map is empty.
    ///
    /// O(1) time.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Return the number of elements in the map.
    ///
    /// O(1) time.
    #[inline]
    pub fn size(&self) -> usize {
        node_size(&self.root)
    }

    /// Lookup a key in the map.
    ///
    /// O(log n) time.
    pub fn lookup(&self, key: &K) -> Option<&V> {
        let mut current = &self.root;
        while let Some(node) = current {
            match key.cmp(&node.key) {
                Ordering::Less => current = &node.left,
                Ordering::Greater => current = &node.right,
                Ordering::Equal => return Some(&node.value),
            }
        }
        None
    }

    /// Check if a key is in the map.
    ///
    /// O(log n) time.
    #[inline]
    pub fn member(&self, key: &K) -> bool {
        self.lookup(key).is_some()
    }

    /// Insert a key-value pair into the map.
    /// If the key already exists, the value is replaced.
    ///
    /// O(log n) time.
    pub fn insert(&self, key: K, value: V) -> Self
    where
        K: Clone,
        V: Clone,
    {
        Map {
            root: Some(insert_node(&self.root, key, value)),
        }
    }

    /// Insert with a combining function.
    /// If the key already exists, `f(new_value, old_value)` is used.
    ///
    /// O(log n) time.
    pub fn insert_with<F>(&self, key: K, value: V, f: F) -> Self
    where
        K: Clone,
        V: Clone,
        F: FnOnce(V, V) -> V,
    {
        Map {
            root: Some(insert_with_node(&self.root, key, value, f)),
        }
    }

    /// Delete a key from the map.
    ///
    /// O(log n) time.
    pub fn delete(&self, key: &K) -> Self
    where
        K: Clone,
        V: Clone,
    {
        Map {
            root: delete_node(&self.root, key),
        }
    }

    /// Update a value at a specific key.
    ///
    /// O(log n) time.
    pub fn adjust<F>(&self, key: &K, f: F) -> Self
    where
        K: Clone,
        V: Clone,
        F: FnOnce(&V) -> V,
    {
        Map {
            root: adjust_node(&self.root, key, f),
        }
    }

    /// Lookup and update.
    /// If the function returns None, the element is deleted.
    ///
    /// O(log n) time.
    pub fn update<F>(&self, key: &K, f: F) -> Self
    where
        K: Clone,
        V: Clone,
        F: FnOnce(&V) -> Option<V>,
    {
        Map {
            root: update_node(&self.root, key, f),
        }
    }

    /// Get the minimum key and value.
    ///
    /// O(log n) time.
    pub fn min(&self) -> Option<(&K, &V)> {
        min_node(&self.root)
    }

    /// Get the maximum key and value.
    ///
    /// O(log n) time.
    pub fn max(&self) -> Option<(&K, &V)> {
        max_node(&self.root)
    }

    /// Create a map from a list of key-value pairs.
    ///
    /// O(n log n) time.
    pub fn from_list(pairs: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Clone,
        V: Clone,
    {
        let mut map = Map::empty();
        for (k, v) in pairs {
            map = map.insert(k, v);
        }
        map
    }

    /// Convert to a list of key-value pairs in ascending key order.
    ///
    /// O(n) time.
    pub fn to_list(&self) -> Vec<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        let mut result = Vec::with_capacity(self.size());
        to_list_node(&self.root, &mut result);
        result
    }

    /// Get all keys in ascending order.
    ///
    /// O(n) time.
    pub fn keys(&self) -> Vec<K>
    where
        K: Clone,
    {
        let mut result = Vec::with_capacity(self.size());
        keys_node(&self.root, &mut result);
        result
    }

    /// Get all values in ascending key order.
    ///
    /// O(n) time.
    pub fn elems(&self) -> Vec<V>
    where
        V: Clone,
    {
        let mut result = Vec::with_capacity(self.size());
        elems_node(&self.root, &mut result);
        result
    }

    /// Map a function over all values in the map.
    ///
    /// O(n) time.
    pub fn map<U, F>(&self, f: F) -> Map<K, U>
    where
        K: Clone,
        F: Fn(&V) -> U,
    {
        Map {
            root: map_node(&self.root, &f),
        }
    }

    /// Map a function over all keys and values.
    ///
    /// O(n) time.
    pub fn map_with_key<U, F>(&self, f: F) -> Map<K, U>
    where
        K: Clone,
        F: Fn(&K, &V) -> U,
    {
        Map {
            root: map_with_key_node(&self.root, &f),
        }
    }

    /// Fold the map in ascending key order.
    ///
    /// O(n) time.
    pub fn foldr<B, F>(&self, f: F, init: B) -> B
    where
        F: Fn(&K, &V, B) -> B,
    {
        foldr_node(&self.root, &f, init)
    }

    /// Fold the map in descending key order.
    ///
    /// O(n) time.
    pub fn foldl<B, F>(&self, f: F, init: B) -> B
    where
        F: Fn(B, &K, &V) -> B,
    {
        foldl_node(&self.root, &f, init)
    }

    /// Filter elements by a predicate.
    ///
    /// O(n) time.
    pub fn filter<F>(&self, pred: F) -> Self
    where
        K: Clone,
        V: Clone,
        F: Fn(&K, &V) -> bool,
    {
        Map {
            root: filter_node(&self.root, &pred),
        }
    }

    /// Union of two maps. If a key exists in both, the value from self is used.
    ///
    /// O(m log(n/m + 1)) time where m <= n.
    pub fn union(&self, other: &Self) -> Self
    where
        K: Clone,
        V: Clone,
    {
        Map {
            root: union_node(&self.root, &other.root),
        }
    }

    /// Union with a combining function.
    ///
    /// O(m log(n/m + 1)) time.
    pub fn union_with<F>(&self, other: &Self, f: F) -> Self
    where
        K: Clone,
        V: Clone,
        F: Fn(&V, &V) -> V + Clone,
    {
        Map {
            root: union_with_node(&self.root, &other.root, &f),
        }
    }

    /// Intersection of two maps.
    ///
    /// O(m log(n/m + 1)) time.
    pub fn intersection(&self, other: &Self) -> Self
    where
        K: Clone,
        V: Clone,
    {
        Map {
            root: intersection_node(&self.root, &other.root),
        }
    }

    /// Difference of two maps (elements in self but not in other).
    ///
    /// O(m log(n/m + 1)) time.
    pub fn difference(&self, other: &Self) -> Self
    where
        K: Clone,
        V: Clone,
    {
        Map {
            root: difference_node(&self.root, &other.root),
        }
    }

    /// Iterate over key-value pairs in ascending key order.
    pub fn iter(&self) -> MapIter<'_, K, V> {
        let mut stack = Vec::new();
        push_left(&self.root, &mut stack);
        MapIter { stack }
    }
}

// Internal functions

fn insert_node<K: Ord + Clone, V: Clone>(
    node: &Option<Rc<Node<K, V>>>,
    key: K,
    value: V,
) -> Rc<Node<K, V>> {
    match node {
        None => Rc::new(Node::singleton(key, value)),
        Some(n) => match key.cmp(&n.key) {
            Ordering::Less => {
                let new_left = insert_node(&n.left, key, value);
                balance(n.key.clone(), n.value.clone(), Some(new_left), n.right.clone())
            }
            Ordering::Greater => {
                let new_right = insert_node(&n.right, key, value);
                balance(n.key.clone(), n.value.clone(), n.left.clone(), Some(new_right))
            }
            Ordering::Equal => Rc::new(Node::new(key, value, n.left.clone(), n.right.clone())),
        },
    }
}

fn insert_with_node<K: Ord + Clone, V: Clone, F: FnOnce(V, V) -> V>(
    node: &Option<Rc<Node<K, V>>>,
    key: K,
    value: V,
    f: F,
) -> Rc<Node<K, V>> {
    match node {
        None => Rc::new(Node::singleton(key, value)),
        Some(n) => match key.cmp(&n.key) {
            Ordering::Less => {
                let new_left = insert_with_node(&n.left, key, value, f);
                balance(n.key.clone(), n.value.clone(), Some(new_left), n.right.clone())
            }
            Ordering::Greater => {
                let new_right = insert_with_node(&n.right, key, value, f);
                balance(n.key.clone(), n.value.clone(), n.left.clone(), Some(new_right))
            }
            Ordering::Equal => {
                let new_value = f(value, n.value.clone());
                Rc::new(Node::new(key, new_value, n.left.clone(), n.right.clone()))
            }
        },
    }
}

fn delete_node<K: Ord + Clone, V: Clone>(
    node: &Option<Rc<Node<K, V>>>,
    key: &K,
) -> Option<Rc<Node<K, V>>> {
    match node {
        None => None,
        Some(n) => match key.cmp(&n.key) {
            Ordering::Less => {
                let new_left = delete_node(&n.left, key);
                Some(balance(
                    n.key.clone(),
                    n.value.clone(),
                    new_left,
                    n.right.clone(),
                ))
            }
            Ordering::Greater => {
                let new_right = delete_node(&n.right, key);
                Some(balance(
                    n.key.clone(),
                    n.value.clone(),
                    n.left.clone(),
                    new_right,
                ))
            }
            Ordering::Equal => glue(&n.left, &n.right),
        },
    }
}

fn adjust_node<K: Ord + Clone, V: Clone, F: FnOnce(&V) -> V>(
    node: &Option<Rc<Node<K, V>>>,
    key: &K,
    f: F,
) -> Option<Rc<Node<K, V>>> {
    match node {
        None => None,
        Some(n) => match key.cmp(&n.key) {
            Ordering::Less => {
                let new_left = adjust_node(&n.left, key, f);
                Some(Rc::new(Node::new(
                    n.key.clone(),
                    n.value.clone(),
                    new_left,
                    n.right.clone(),
                )))
            }
            Ordering::Greater => {
                let new_right = adjust_node(&n.right, key, f);
                Some(Rc::new(Node::new(
                    n.key.clone(),
                    n.value.clone(),
                    n.left.clone(),
                    new_right,
                )))
            }
            Ordering::Equal => {
                let new_value = f(&n.value);
                Some(Rc::new(Node::new(
                    n.key.clone(),
                    new_value,
                    n.left.clone(),
                    n.right.clone(),
                )))
            }
        },
    }
}

fn update_node<K: Ord + Clone, V: Clone, F: FnOnce(&V) -> Option<V>>(
    node: &Option<Rc<Node<K, V>>>,
    key: &K,
    f: F,
) -> Option<Rc<Node<K, V>>> {
    match node {
        None => None,
        Some(n) => match key.cmp(&n.key) {
            Ordering::Less => {
                let new_left = update_node(&n.left, key, f);
                Some(Rc::new(Node::new(
                    n.key.clone(),
                    n.value.clone(),
                    new_left,
                    n.right.clone(),
                )))
            }
            Ordering::Greater => {
                let new_right = update_node(&n.right, key, f);
                Some(Rc::new(Node::new(
                    n.key.clone(),
                    n.value.clone(),
                    n.left.clone(),
                    new_right,
                )))
            }
            Ordering::Equal => match f(&n.value) {
                Some(new_value) => Some(Rc::new(Node::new(
                    n.key.clone(),
                    new_value,
                    n.left.clone(),
                    n.right.clone(),
                ))),
                None => glue(&n.left, &n.right),
            },
        },
    }
}

fn min_node<K, V>(node: &Option<Rc<Node<K, V>>>) -> Option<(&K, &V)> {
    let node = node.as_ref()?;
    match &node.left {
        None => Some((&node.key, &node.value)),
        Some(_) => min_node(&node.left),
    }
}

fn max_node<K, V>(node: &Option<Rc<Node<K, V>>>) -> Option<(&K, &V)> {
    let node = node.as_ref()?;
    match &node.right {
        None => Some((&node.key, &node.value)),
        Some(_) => max_node(&node.right),
    }
}

fn to_list_node<K: Clone, V: Clone>(node: &Option<Rc<Node<K, V>>>, result: &mut Vec<(K, V)>) {
    if let Some(n) = node {
        to_list_node(&n.left, result);
        result.push((n.key.clone(), n.value.clone()));
        to_list_node(&n.right, result);
    }
}

fn keys_node<K: Clone, V>(node: &Option<Rc<Node<K, V>>>, result: &mut Vec<K>) {
    if let Some(n) = node {
        keys_node(&n.left, result);
        result.push(n.key.clone());
        keys_node(&n.right, result);
    }
}

fn elems_node<K, V: Clone>(node: &Option<Rc<Node<K, V>>>, result: &mut Vec<V>) {
    if let Some(n) = node {
        elems_node(&n.left, result);
        result.push(n.value.clone());
        elems_node(&n.right, result);
    }
}

fn map_node<K: Clone, V, U, F: Fn(&V) -> U>(
    node: &Option<Rc<Node<K, V>>>,
    f: &F,
) -> Option<Rc<Node<K, U>>> {
    node.as_ref().map(|n| {
        Rc::new(Node::new(
            n.key.clone(),
            f(&n.value),
            map_node(&n.left, f),
            map_node(&n.right, f),
        ))
    })
}

fn map_with_key_node<K: Clone, V, U, F: Fn(&K, &V) -> U>(
    node: &Option<Rc<Node<K, V>>>,
    f: &F,
) -> Option<Rc<Node<K, U>>> {
    node.as_ref().map(|n| {
        Rc::new(Node::new(
            n.key.clone(),
            f(&n.key, &n.value),
            map_with_key_node(&n.left, f),
            map_with_key_node(&n.right, f),
        ))
    })
}

fn foldr_node<K, V, B, F: Fn(&K, &V, B) -> B>(
    node: &Option<Rc<Node<K, V>>>,
    f: &F,
    init: B,
) -> B {
    match node {
        None => init,
        Some(n) => {
            let right_result = foldr_node(&n.right, f, init);
            let mid_result = f(&n.key, &n.value, right_result);
            foldr_node(&n.left, f, mid_result)
        }
    }
}

fn foldl_node<K, V, B, F: Fn(B, &K, &V) -> B>(
    node: &Option<Rc<Node<K, V>>>,
    f: &F,
    init: B,
) -> B {
    match node {
        None => init,
        Some(n) => {
            let left_result = foldl_node(&n.left, f, init);
            let mid_result = f(left_result, &n.key, &n.value);
            foldl_node(&n.right, f, mid_result)
        }
    }
}

fn filter_node<K: Ord + Clone, V: Clone, F: Fn(&K, &V) -> bool>(
    node: &Option<Rc<Node<K, V>>>,
    pred: &F,
) -> Option<Rc<Node<K, V>>> {
    match node {
        None => None,
        Some(n) => {
            let new_left = filter_node(&n.left, pred);
            let new_right = filter_node(&n.right, pred);
            if pred(&n.key, &n.value) {
                Some(balance(n.key.clone(), n.value.clone(), new_left, new_right))
            } else {
                glue(&new_left, &new_right)
            }
        }
    }
}

fn union_node<K: Ord + Clone, V: Clone>(
    t1: &Option<Rc<Node<K, V>>>,
    t2: &Option<Rc<Node<K, V>>>,
) -> Option<Rc<Node<K, V>>> {
    match (t1, t2) {
        (None, t2) => t2.clone(),
        (t1, None) => t1.clone(),
        (Some(n1), Some(_n2)) => {
            let (lt, gt) = split(&n1.key, t2);
            let new_left = union_node(&n1.left, &lt);
            let new_right = union_node(&n1.right, &gt);
            Some(join(n1.key.clone(), n1.value.clone(), new_left, new_right))
        }
    }
}

fn union_with_node<K: Ord + Clone, V: Clone, F: Fn(&V, &V) -> V + Clone>(
    t1: &Option<Rc<Node<K, V>>>,
    t2: &Option<Rc<Node<K, V>>>,
    f: &F,
) -> Option<Rc<Node<K, V>>> {
    match (t1, t2) {
        (None, t2) => t2.clone(),
        (t1, None) => t1.clone(),
        (Some(n1), Some(_)) => {
            let (lt, found, gt) = split_lookup(&n1.key, t2);
            let value = match found {
                Some(v2) => f(&n1.value, &v2),
                None => n1.value.clone(),
            };
            let new_left = union_with_node(&n1.left, &lt, f);
            let new_right = union_with_node(&n1.right, &gt, f);
            Some(join(n1.key.clone(), value, new_left, new_right))
        }
    }
}

fn intersection_node<K: Ord + Clone, V: Clone>(
    t1: &Option<Rc<Node<K, V>>>,
    t2: &Option<Rc<Node<K, V>>>,
) -> Option<Rc<Node<K, V>>> {
    match (t1, t2) {
        (None, _) | (_, None) => None,
        (Some(n1), Some(_)) => {
            let (lt, found, gt) = split_lookup(&n1.key, t2);
            let new_left = intersection_node(&n1.left, &lt);
            let new_right = intersection_node(&n1.right, &gt);
            match found {
                Some(_) => Some(join(n1.key.clone(), n1.value.clone(), new_left, new_right)),
                None => glue(&new_left, &new_right),
            }
        }
    }
}

fn difference_node<K: Ord + Clone, V: Clone>(
    t1: &Option<Rc<Node<K, V>>>,
    t2: &Option<Rc<Node<K, V>>>,
) -> Option<Rc<Node<K, V>>> {
    match (t1, t2) {
        (None, _) => None,
        (t1, None) => t1.clone(),
        (Some(n1), Some(_)) => {
            let (lt, found, gt) = split_lookup(&n1.key, t2);
            let new_left = difference_node(&n1.left, &lt);
            let new_right = difference_node(&n1.right, &gt);
            match found {
                Some(_) => glue(&new_left, &new_right),
                None => Some(join(n1.key.clone(), n1.value.clone(), new_left, new_right)),
            }
        }
    }
}

fn split<K: Ord + Clone, V: Clone>(
    key: &K,
    node: &Option<Rc<Node<K, V>>>,
) -> (Option<Rc<Node<K, V>>>, Option<Rc<Node<K, V>>>) {
    match node {
        None => (None, None),
        Some(n) => match key.cmp(&n.key) {
            Ordering::Less => {
                let (lt, gt) = split(key, &n.left);
                (lt, Some(join(n.key.clone(), n.value.clone(), gt, n.right.clone())))
            }
            Ordering::Greater => {
                let (lt, gt) = split(key, &n.right);
                (Some(join(n.key.clone(), n.value.clone(), n.left.clone(), lt)), gt)
            }
            Ordering::Equal => (n.left.clone(), n.right.clone()),
        },
    }
}

fn split_lookup<K: Ord + Clone, V: Clone>(
    key: &K,
    node: &Option<Rc<Node<K, V>>>,
) -> (Option<Rc<Node<K, V>>>, Option<V>, Option<Rc<Node<K, V>>>) {
    match node {
        None => (None, None, None),
        Some(n) => match key.cmp(&n.key) {
            Ordering::Less => {
                let (lt, found, gt) = split_lookup(key, &n.left);
                (lt, found, Some(join(n.key.clone(), n.value.clone(), gt, n.right.clone())))
            }
            Ordering::Greater => {
                let (lt, found, gt) = split_lookup(key, &n.right);
                (Some(join(n.key.clone(), n.value.clone(), n.left.clone(), lt)), found, gt)
            }
            Ordering::Equal => (n.left.clone(), Some(n.value.clone()), n.right.clone()),
        },
    }
}

// Balancing operations

fn balance<K: Clone, V: Clone>(
    key: K,
    value: V,
    left: Option<Rc<Node<K, V>>>,
    right: Option<Rc<Node<K, V>>>,
) -> Rc<Node<K, V>> {
    let ls = node_size(&left);
    let rs = node_size(&right);

    if ls + rs <= 1 {
        Rc::new(Node::new(key, value, left, right))
    } else if rs > DELTA * ls {
        rotate_left(key, value, left, right.unwrap())
    } else if ls > DELTA * rs {
        rotate_right(key, value, left.unwrap(), right)
    } else {
        Rc::new(Node::new(key, value, left, right))
    }
}

fn rotate_left<K: Clone, V: Clone>(
    key: K,
    value: V,
    left: Option<Rc<Node<K, V>>>,
    right: Rc<Node<K, V>>,
) -> Rc<Node<K, V>> {
    let rls = node_size(&right.left);
    let rrs = node_size(&right.right);

    if rls < GAMMA * rrs {
        single_left(key, value, left, right)
    } else {
        double_left(key, value, left, right)
    }
}

fn rotate_right<K: Clone, V: Clone>(
    key: K,
    value: V,
    left: Rc<Node<K, V>>,
    right: Option<Rc<Node<K, V>>>,
) -> Rc<Node<K, V>> {
    let lls = node_size(&left.left);
    let lrs = node_size(&left.right);

    if lrs < GAMMA * lls {
        single_right(key, value, left, right)
    } else {
        double_right(key, value, left, right)
    }
}

fn single_left<K: Clone, V: Clone>(
    key: K,
    value: V,
    left: Option<Rc<Node<K, V>>>,
    right: Rc<Node<K, V>>,
) -> Rc<Node<K, V>> {
    let new_left = Rc::new(Node::new(key, value, left, right.left.clone()));
    Rc::new(Node::new(
        right.key.clone(),
        right.value.clone(),
        Some(new_left),
        right.right.clone(),
    ))
}

fn single_right<K: Clone, V: Clone>(
    key: K,
    value: V,
    left: Rc<Node<K, V>>,
    right: Option<Rc<Node<K, V>>>,
) -> Rc<Node<K, V>> {
    let new_right = Rc::new(Node::new(key, value, left.right.clone(), right));
    Rc::new(Node::new(
        left.key.clone(),
        left.value.clone(),
        left.left.clone(),
        Some(new_right),
    ))
}

fn double_left<K: Clone, V: Clone>(
    key: K,
    value: V,
    left: Option<Rc<Node<K, V>>>,
    right: Rc<Node<K, V>>,
) -> Rc<Node<K, V>> {
    match &right.left {
        Some(rl) => {
            let new_left = Rc::new(Node::new(key, value, left, rl.left.clone()));
            let new_right = Rc::new(Node::new(
                right.key.clone(),
                right.value.clone(),
                rl.right.clone(),
                right.right.clone(),
            ));
            Rc::new(Node::new(
                rl.key.clone(),
                rl.value.clone(),
                Some(new_left),
                Some(new_right),
            ))
        }
        None => single_left(key, value, left, right),
    }
}

fn double_right<K: Clone, V: Clone>(
    key: K,
    value: V,
    left: Rc<Node<K, V>>,
    right: Option<Rc<Node<K, V>>>,
) -> Rc<Node<K, V>> {
    match &left.right {
        Some(lr) => {
            let new_left = Rc::new(Node::new(
                left.key.clone(),
                left.value.clone(),
                left.left.clone(),
                lr.left.clone(),
            ));
            let new_right = Rc::new(Node::new(key, value, lr.right.clone(), right));
            Rc::new(Node::new(
                lr.key.clone(),
                lr.value.clone(),
                Some(new_left),
                Some(new_right),
            ))
        }
        None => single_right(key, value, left, right),
    }
}

fn glue<K: Clone, V: Clone>(
    left: &Option<Rc<Node<K, V>>>,
    right: &Option<Rc<Node<K, V>>>,
) -> Option<Rc<Node<K, V>>> {
    match (left, right) {
        (None, r) => r.clone(),
        (l, None) => l.clone(),
        (Some(l), Some(r)) => {
            if l.size > r.size {
                let (k, v) = max_node(left).unwrap();
                let new_left = delete_max(left);
                Some(balance(k.clone(), v.clone(), new_left, right.clone()))
            } else {
                let (k, v) = min_node(right).unwrap();
                let new_right = delete_min(right);
                Some(balance(k.clone(), v.clone(), left.clone(), new_right))
            }
        }
    }
}

fn delete_min<K: Clone, V: Clone>(node: &Option<Rc<Node<K, V>>>) -> Option<Rc<Node<K, V>>> {
    match node {
        None => None,
        Some(n) => match &n.left {
            None => n.right.clone(),
            Some(_) => {
                let new_left = delete_min(&n.left);
                Some(balance(
                    n.key.clone(),
                    n.value.clone(),
                    new_left,
                    n.right.clone(),
                ))
            }
        },
    }
}

fn delete_max<K: Clone, V: Clone>(node: &Option<Rc<Node<K, V>>>) -> Option<Rc<Node<K, V>>> {
    match node {
        None => None,
        Some(n) => match &n.right {
            None => n.left.clone(),
            Some(_) => {
                let new_right = delete_max(&n.right);
                Some(balance(
                    n.key.clone(),
                    n.value.clone(),
                    n.left.clone(),
                    new_right,
                ))
            }
        },
    }
}

fn join<K: Ord + Clone, V: Clone>(
    key: K,
    value: V,
    left: Option<Rc<Node<K, V>>>,
    right: Option<Rc<Node<K, V>>>,
) -> Rc<Node<K, V>> {
    match (&left, &right) {
        (None, None) => Rc::new(Node::singleton(key, value)),
        (None, Some(r)) => insert_min(key, value, r),
        (Some(l), None) => insert_max(key, value, l),
        (Some(l), Some(r)) => {
            let ls = l.size;
            let rs = r.size;
            if DELTA * ls < rs {
                balance(
                    r.key.clone(),
                    r.value.clone(),
                    Some(join(key, value, left, r.left.clone())),
                    r.right.clone(),
                )
            } else if DELTA * rs < ls {
                balance(
                    l.key.clone(),
                    l.value.clone(),
                    l.left.clone(),
                    Some(join(key, value, l.right.clone(), right)),
                )
            } else {
                Rc::new(Node::new(key, value, left, right))
            }
        }
    }
}

fn insert_min<K: Clone, V: Clone>(key: K, value: V, node: &Rc<Node<K, V>>) -> Rc<Node<K, V>> {
    match &node.left {
        None => Rc::new(Node::new(
            node.key.clone(),
            node.value.clone(),
            Some(Rc::new(Node::singleton(key, value))),
            node.right.clone(),
        )),
        Some(l) => {
            balance(
                node.key.clone(),
                node.value.clone(),
                Some(insert_min(key, value, l)),
                node.right.clone(),
            )
        }
    }
}

fn insert_max<K: Clone, V: Clone>(key: K, value: V, node: &Rc<Node<K, V>>) -> Rc<Node<K, V>> {
    match &node.right {
        None => Rc::new(Node::new(
            node.key.clone(),
            node.value.clone(),
            node.left.clone(),
            Some(Rc::new(Node::singleton(key, value))),
        )),
        Some(r) => {
            balance(
                node.key.clone(),
                node.value.clone(),
                node.left.clone(),
                Some(insert_max(key, value, r)),
            )
        }
    }
}

// Iterator

/// Iterator over a Map in ascending key order.
pub struct MapIter<'a, K, V> {
    stack: Vec<&'a Node<K, V>>,
}

fn push_left<'a, K, V>(node: &'a Option<Rc<Node<K, V>>>, stack: &mut Vec<&'a Node<K, V>>) {
    let mut current = node;
    while let Some(n) = current {
        stack.push(n);
        current = &n.left;
    }
}

impl<'a, K, V> Iterator for MapIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;
        push_left(&node.right, &mut self.stack);
        Some((&node.key, &node.value))
    }
}

// Trait implementations

impl<K: Ord, V> Default for Map<K, V> {
    fn default() -> Self {
        Map::empty()
    }
}

impl<K: Ord + Clone, V: Clone> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Map::from_list(iter)
    }
}

impl<K: Debug + Ord + Clone, V: Debug> Debug for Map<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Ord + PartialEq, V: PartialEq> PartialEq for Map<K, V>
where
    K: Clone,
    V: Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.size() == other.size() && self.to_list() == other.to_list()
    }
}

impl<K: Ord + Eq, V: Eq> Eq for Map<K, V>
where
    K: Clone,
    V: Clone,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let m: Map<i32, &str> = Map::empty();
        assert!(m.is_empty());
        assert_eq!(m.size(), 0);
        assert_eq!(m.lookup(&1), None);
    }

    #[test]
    fn test_singleton() {
        let m = Map::singleton(1, "one");
        assert!(!m.is_empty());
        assert_eq!(m.size(), 1);
        assert_eq!(m.lookup(&1), Some(&"one"));
    }

    #[test]
    fn test_insert_lookup() {
        let m = Map::empty()
            .insert(3, "three")
            .insert(1, "one")
            .insert(2, "two");

        assert_eq!(m.size(), 3);
        assert_eq!(m.lookup(&1), Some(&"one"));
        assert_eq!(m.lookup(&2), Some(&"two"));
        assert_eq!(m.lookup(&3), Some(&"three"));
        assert_eq!(m.lookup(&4), None);
    }

    #[test]
    fn test_insert_replaces() {
        let m1 = Map::singleton(1, "one");
        let m2 = m1.insert(1, "ONE");

        assert_eq!(m2.lookup(&1), Some(&"ONE"));
        assert_eq!(m1.lookup(&1), Some(&"one")); // Original unchanged
    }

    #[test]
    fn test_delete() {
        let m = Map::empty()
            .insert(1, "one")
            .insert(2, "two")
            .insert(3, "three");

        let m2 = m.delete(&2);
        assert_eq!(m2.size(), 2);
        assert_eq!(m2.lookup(&2), None);
        assert_eq!(m2.lookup(&1), Some(&"one"));
        assert_eq!(m2.lookup(&3), Some(&"three"));

        // Original unchanged
        assert_eq!(m.size(), 3);
        assert_eq!(m.lookup(&2), Some(&"two"));
    }

    #[test]
    fn test_min_max() {
        let m = Map::empty()
            .insert(3, "three")
            .insert(1, "one")
            .insert(2, "two");

        assert_eq!(m.min(), Some((&1, &"one")));
        assert_eq!(m.max(), Some((&3, &"three")));

        let empty: Map<i32, &str> = Map::empty();
        assert_eq!(empty.min(), None);
        assert_eq!(empty.max(), None);
    }

    #[test]
    fn test_to_list() {
        let m = Map::empty()
            .insert(3, "three")
            .insert(1, "one")
            .insert(2, "two");

        let list = m.to_list();
        assert_eq!(list, vec![(1, "one"), (2, "two"), (3, "three")]);
    }

    #[test]
    fn test_from_list() {
        let m = Map::from_list(vec![(1, "one"), (2, "two"), (3, "three")]);
        assert_eq!(m.size(), 3);
        assert_eq!(m.lookup(&2), Some(&"two"));
    }

    #[test]
    fn test_map() {
        let m = Map::from_list(vec![(1, 10), (2, 20), (3, 30)]);
        let m2 = m.map(|v| v * 2);

        assert_eq!(m2.lookup(&1), Some(&20));
        assert_eq!(m2.lookup(&2), Some(&40));
        assert_eq!(m2.lookup(&3), Some(&60));
    }

    #[test]
    fn test_filter() {
        let m = Map::from_list(vec![(1, 10), (2, 20), (3, 30), (4, 40)]);
        let m2 = m.filter(|k, _| k % 2 == 0);

        assert_eq!(m2.size(), 2);
        assert_eq!(m2.lookup(&2), Some(&20));
        assert_eq!(m2.lookup(&4), Some(&40));
        assert_eq!(m2.lookup(&1), None);
    }

    #[test]
    fn test_union() {
        let m1 = Map::from_list(vec![(1, "a"), (2, "b")]);
        let m2 = Map::from_list(vec![(2, "B"), (3, "c")]);

        let u = m1.union(&m2);
        assert_eq!(u.size(), 3);
        assert_eq!(u.lookup(&1), Some(&"a"));
        assert_eq!(u.lookup(&2), Some(&"b")); // m1's value wins
        assert_eq!(u.lookup(&3), Some(&"c"));
    }

    #[test]
    fn test_intersection() {
        let m1 = Map::from_list(vec![(1, "a"), (2, "b"), (3, "c")]);
        let m2 = Map::from_list(vec![(2, "B"), (3, "C"), (4, "d")]);

        let i = m1.intersection(&m2);
        assert_eq!(i.size(), 2);
        assert_eq!(i.lookup(&2), Some(&"b"));
        assert_eq!(i.lookup(&3), Some(&"c"));
        assert_eq!(i.lookup(&1), None);
    }

    #[test]
    fn test_difference() {
        let m1 = Map::from_list(vec![(1, "a"), (2, "b"), (3, "c")]);
        let m2 = Map::from_list(vec![(2, "B"), (3, "C")]);

        let d = m1.difference(&m2);
        assert_eq!(d.size(), 1);
        assert_eq!(d.lookup(&1), Some(&"a"));
        assert_eq!(d.lookup(&2), None);
    }

    #[test]
    fn test_iter() {
        let m = Map::from_list(vec![(3, "c"), (1, "a"), (2, "b")]);
        let pairs: Vec<_> = m.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(pairs, vec![(1, "a"), (2, "b"), (3, "c")]);
    }

    #[test]
    fn test_foldr() {
        let m = Map::from_list(vec![(1, 10), (2, 20), (3, 30)]);
        let sum = m.foldr(|_, v, acc| acc + v, 0);
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_balance_many_inserts() {
        let mut m = Map::empty();
        for i in 0..1000 {
            m = m.insert(i, i * 2);
        }
        assert_eq!(m.size(), 1000);

        for i in 0..1000 {
            assert_eq!(m.lookup(&i), Some(&(i * 2)));
        }
    }
}
