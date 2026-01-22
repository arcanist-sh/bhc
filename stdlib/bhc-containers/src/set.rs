//! Immutable ordered sets
//!
//! This module provides an immutable set data structure based on weight-balanced
//! binary search trees, implemented as a wrapper around `Map<T, ()>`.
//!
//! # Performance
//!
//! | Operation    | Complexity |
//! |--------------|------------|
//! | member       | O(log n)   |
//! | insert       | O(log n)   |
//! | delete       | O(log n)   |
//! | union        | O(m log(n/m + 1)), m <= n |
//!
//! # Example
//!
//! ```ignore
//! let s = Set::empty()
//!     .insert(1)
//!     .insert(2)
//!     .insert(3);
//!
//! assert!(s.member(&2));
//! assert_eq!(s.size(), 3);
//! ```

use crate::map::Map;
use std::fmt::{self, Debug};
use std::iter::FromIterator;

/// An immutable ordered set based on weight-balanced binary search trees.
#[derive(Clone)]
pub struct Set<T> {
    map: Map<T, ()>,
}

impl<T: Ord> Set<T> {
    /// Create an empty set.
    ///
    /// O(1) time and space.
    #[inline]
    pub fn empty() -> Self {
        Set { map: Map::empty() }
    }

    /// Create a set with a single element.
    ///
    /// O(1) time and space.
    #[inline]
    pub fn singleton(x: T) -> Self {
        Set {
            map: Map::singleton(x, ()),
        }
    }

    /// Check if the set is empty.
    ///
    /// O(1) time.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return the number of elements in the set.
    ///
    /// O(1) time.
    #[inline]
    pub fn size(&self) -> usize {
        self.map.size()
    }

    /// Check if an element is in the set.
    ///
    /// O(log n) time.
    #[inline]
    pub fn member(&self, x: &T) -> bool {
        self.map.member(x)
    }

    /// Check if an element is not in the set.
    ///
    /// O(log n) time.
    #[inline]
    pub fn not_member(&self, x: &T) -> bool {
        !self.member(x)
    }

    /// Insert an element into the set.
    ///
    /// O(log n) time.
    pub fn insert(&self, x: T) -> Self
    where
        T: Clone,
    {
        Set {
            map: self.map.insert(x, ()),
        }
    }

    /// Delete an element from the set.
    ///
    /// O(log n) time.
    pub fn delete(&self, x: &T) -> Self
    where
        T: Clone,
    {
        Set {
            map: self.map.delete(x),
        }
    }

    /// Get the minimum element.
    ///
    /// O(log n) time.
    pub fn min(&self) -> Option<&T> {
        self.map.min().map(|(k, _)| k)
    }

    /// Get the maximum element.
    ///
    /// O(log n) time.
    pub fn max(&self) -> Option<&T> {
        self.map.max().map(|(k, _)| k)
    }

    /// Create a set from a list of elements.
    ///
    /// O(n log n) time.
    pub fn from_list(xs: impl IntoIterator<Item = T>) -> Self
    where
        T: Clone,
    {
        let mut set = Set::empty();
        for x in xs {
            set = set.insert(x);
        }
        set
    }

    /// Convert to a list of elements in ascending order.
    ///
    /// O(n) time.
    pub fn to_list(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.map.keys()
    }

    /// Map a function over all elements.
    ///
    /// Note: This may produce a smaller set if the function maps distinct
    /// elements to the same value.
    ///
    /// O(n log n) time.
    pub fn map<U: Ord + Clone, F>(&self, f: F) -> Set<U>
    where
        T: Clone,
        F: Fn(&T) -> U,
    {
        Set::from_list(self.to_list().iter().map(|x| f(x)))
    }

    /// Filter elements by a predicate.
    ///
    /// O(n) time.
    pub fn filter<F>(&self, pred: F) -> Self
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        Set {
            map: self.map.filter(|k, _| pred(k)),
        }
    }

    /// Partition the set by a predicate.
    ///
    /// Returns (elements satisfying pred, elements not satisfying pred).
    ///
    /// O(n) time.
    pub fn partition<F>(&self, pred: F) -> (Self, Self)
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        let list = self.to_list();
        let (yes, no): (Vec<_>, Vec<_>) = list.into_iter().partition(|x| pred(x));
        (Set::from_list(yes), Set::from_list(no))
    }

    /// Fold the set in ascending order.
    ///
    /// O(n) time.
    pub fn foldr<B, F>(&self, f: F, init: B) -> B
    where
        F: Fn(&T, B) -> B,
    {
        self.map.foldr(|k, _, acc| f(k, acc), init)
    }

    /// Fold the set in descending order.
    ///
    /// O(n) time.
    pub fn foldl<B, F>(&self, f: F, init: B) -> B
    where
        F: Fn(B, &T) -> B,
    {
        self.map.foldl(|acc, k, _| f(acc, k), init)
    }

    /// Union of two sets.
    ///
    /// O(m log(n/m + 1)) time where m <= n.
    pub fn union(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        Set {
            map: self.map.union(&other.map),
        }
    }

    /// Intersection of two sets.
    ///
    /// O(m log(n/m + 1)) time.
    pub fn intersection(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        Set {
            map: self.map.intersection(&other.map),
        }
    }

    /// Difference of two sets (elements in self but not in other).
    ///
    /// O(m log(n/m + 1)) time.
    pub fn difference(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        Set {
            map: self.map.difference(&other.map),
        }
    }

    /// Symmetric difference (elements in exactly one of the sets).
    ///
    /// O(m log(n/m + 1)) time.
    pub fn symmetric_difference(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        let union = self.union(other);
        let intersection = self.intersection(other);
        union.difference(&intersection)
    }

    /// Check if this set is a subset of another.
    ///
    /// O(n) time.
    pub fn is_subset_of(&self, other: &Self) -> bool
    where
        T: Clone,
    {
        self.size() <= other.size() && self.to_list().iter().all(|x| other.member(x))
    }

    /// Check if this set is a proper subset of another.
    ///
    /// O(n) time.
    pub fn is_proper_subset_of(&self, other: &Self) -> bool
    where
        T: Clone,
    {
        self.size() < other.size() && self.is_subset_of(other)
    }

    /// Check if two sets are disjoint (have no common elements).
    ///
    /// O(n) time.
    pub fn is_disjoint(&self, other: &Self) -> bool
    where
        T: Clone,
    {
        self.intersection(other).is_empty()
    }

    /// Iterate over elements in ascending order.
    pub fn iter(&self) -> SetIter<'_, T> {
        SetIter {
            inner: self.map.iter(),
        }
    }
}

/// Iterator over a Set in ascending order.
pub struct SetIter<'a, T> {
    inner: crate::map::MapIter<'a, T, ()>,
}

impl<'a, T> Iterator for SetIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

// Trait implementations

impl<T: Ord> Default for Set<T> {
    fn default() -> Self {
        Set::empty()
    }
}

impl<T: Ord + Clone> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Set::from_list(iter)
    }
}

impl<T: Debug + Ord + Clone> Debug for Set<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Ord + PartialEq + Clone> PartialEq for Set<T> {
    fn eq(&self, other: &Self) -> bool {
        self.size() == other.size() && self.to_list() == other.to_list()
    }
}

impl<T: Ord + Eq + Clone> Eq for Set<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let s: Set<i32> = Set::empty();
        assert!(s.is_empty());
        assert_eq!(s.size(), 0);
        assert!(!s.member(&1));
    }

    #[test]
    fn test_singleton() {
        let s = Set::singleton(42);
        assert!(!s.is_empty());
        assert_eq!(s.size(), 1);
        assert!(s.member(&42));
        assert!(!s.member(&1));
    }

    #[test]
    fn test_insert() {
        let s = Set::empty().insert(1).insert(2).insert(3);
        assert_eq!(s.size(), 3);
        assert!(s.member(&1));
        assert!(s.member(&2));
        assert!(s.member(&3));
        assert!(!s.member(&4));
    }

    #[test]
    fn test_insert_duplicate() {
        let s = Set::empty().insert(1).insert(1).insert(1);
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn test_delete() {
        let s = Set::from_list(vec![1, 2, 3]);
        let s2 = s.delete(&2);

        assert_eq!(s2.size(), 2);
        assert!(!s2.member(&2));
        assert!(s2.member(&1));
        assert!(s2.member(&3));

        // Original unchanged
        assert_eq!(s.size(), 3);
        assert!(s.member(&2));
    }

    #[test]
    fn test_min_max() {
        let s = Set::from_list(vec![3, 1, 4, 1, 5, 9, 2, 6]);
        assert_eq!(s.min(), Some(&1));
        assert_eq!(s.max(), Some(&9));

        let empty: Set<i32> = Set::empty();
        assert_eq!(empty.min(), None);
        assert_eq!(empty.max(), None);
    }

    #[test]
    fn test_to_list() {
        let s = Set::from_list(vec![3, 1, 2]);
        assert_eq!(s.to_list(), vec![1, 2, 3]);
    }

    #[test]
    fn test_filter() {
        let s = Set::from_list(vec![1, 2, 3, 4, 5, 6]);
        let evens = s.filter(|x| x % 2 == 0);

        assert_eq!(evens.size(), 3);
        assert_eq!(evens.to_list(), vec![2, 4, 6]);
    }

    #[test]
    fn test_union() {
        let s1 = Set::from_list(vec![1, 2, 3]);
        let s2 = Set::from_list(vec![2, 3, 4, 5]);

        let u = s1.union(&s2);
        assert_eq!(u.to_list(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_intersection() {
        let s1 = Set::from_list(vec![1, 2, 3, 4]);
        let s2 = Set::from_list(vec![2, 4, 6]);

        let i = s1.intersection(&s2);
        assert_eq!(i.to_list(), vec![2, 4]);
    }

    #[test]
    fn test_difference() {
        let s1 = Set::from_list(vec![1, 2, 3, 4]);
        let s2 = Set::from_list(vec![2, 4]);

        let d = s1.difference(&s2);
        assert_eq!(d.to_list(), vec![1, 3]);
    }

    #[test]
    fn test_symmetric_difference() {
        let s1 = Set::from_list(vec![1, 2, 3]);
        let s2 = Set::from_list(vec![2, 3, 4]);

        let sd = s1.symmetric_difference(&s2);
        assert_eq!(sd.to_list(), vec![1, 4]);
    }

    #[test]
    fn test_subset() {
        let s1 = Set::from_list(vec![1, 2]);
        let s2 = Set::from_list(vec![1, 2, 3]);
        let s3 = Set::from_list(vec![1, 2]);

        assert!(s1.is_subset_of(&s2));
        assert!(s1.is_subset_of(&s3));
        assert!(!s2.is_subset_of(&s1));

        assert!(s1.is_proper_subset_of(&s2));
        assert!(!s1.is_proper_subset_of(&s3)); // Equal, not proper
    }

    #[test]
    fn test_disjoint() {
        let s1 = Set::from_list(vec![1, 2]);
        let s2 = Set::from_list(vec![3, 4]);
        let s3 = Set::from_list(vec![2, 3]);

        assert!(s1.is_disjoint(&s2));
        assert!(!s1.is_disjoint(&s3));
    }

    #[test]
    fn test_partition() {
        let s = Set::from_list(vec![1, 2, 3, 4, 5, 6]);
        let (evens, odds) = s.partition(|x| x % 2 == 0);

        assert_eq!(evens.to_list(), vec![2, 4, 6]);
        assert_eq!(odds.to_list(), vec![1, 3, 5]);
    }

    #[test]
    fn test_foldr() {
        let s = Set::from_list(vec![1, 2, 3]);
        let sum = s.foldr(|x, acc| x + acc, 0);
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_iter() {
        let s = Set::from_list(vec![3, 1, 2]);
        let items: Vec<_> = s.iter().copied().collect();
        assert_eq!(items, vec![1, 2, 3]);
    }

    #[test]
    fn test_from_iter() {
        let s: Set<i32> = vec![3, 1, 2, 1].into_iter().collect();
        assert_eq!(s.size(), 3);
        assert_eq!(s.to_list(), vec![1, 2, 3]);
    }
}
