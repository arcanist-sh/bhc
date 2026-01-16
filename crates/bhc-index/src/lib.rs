//! Typed indices for efficient and safe indexing.
//!
//! This crate provides a pattern for creating type-safe indices that
//! prevent mixing up indices from different collections.

#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

/// Trait for typed indices.
///
/// Types implementing this trait can be used as indices into `IndexVec`.
/// This provides type-safe indexing that prevents mixing up indices
/// from different collections.
pub trait Idx: Copy + Eq + Hash {
    /// Create a new index from a usize.
    fn new(idx: usize) -> Self;

    /// Get the underlying index as a usize.
    fn index(self) -> usize;
}

/// A macro for defining new index types.
///
/// # Example
///
/// ```
/// use bhc_index::define_index;
///
/// define_index! {
///     /// Index into the expression arena.
///     pub struct ExprId;
///
///     /// Index into the type arena.
///     pub struct TypeId;
/// }
/// ```
#[macro_export]
macro_rules! define_index {
    ($($(#[$attr:meta])* $vis:vis struct $name:ident;)*) => {
        $(
            $(#[$attr])*
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            #[repr(transparent)]
            $vis struct $name(u32);

            impl $name {
                /// Create a new index from a raw value.
                #[must_use]
                #[inline]
                pub const fn new(idx: u32) -> Self {
                    Self(idx)
                }

                /// Create a new index from a usize.
                #[must_use]
                #[inline]
                pub const fn from_usize(idx: usize) -> Self {
                    Self(idx as u32)
                }

                /// Get the raw index value.
                #[must_use]
                #[inline]
                pub const fn as_u32(self) -> u32 {
                    self.0
                }

                /// Get the index as a usize.
                #[must_use]
                #[inline]
                pub const fn as_usize(self) -> usize {
                    self.0 as usize
                }
            }

            impl std::fmt::Debug for $name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}({})", stringify!($name), self.0)
                }
            }

            impl std::fmt::Display for $name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}", self.0)
                }
            }

            impl From<u32> for $name {
                fn from(idx: u32) -> Self {
                    Self(idx)
                }
            }

            impl From<usize> for $name {
                fn from(idx: usize) -> Self {
                    Self(idx as u32)
                }
            }

            impl From<$name> for u32 {
                fn from(idx: $name) -> u32 {
                    idx.0
                }
            }

            impl From<$name> for usize {
                fn from(idx: $name) -> usize {
                    idx.0 as usize
                }
            }

            impl $crate::Idx for $name {
                fn new(idx: usize) -> Self {
                    Self(idx as u32)
                }

                fn index(self) -> usize {
                    self.0 as usize
                }
            }

            impl serde::Serialize for $name {
                fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                    self.0.serialize(s)
                }
            }

            impl<'de> serde::Deserialize<'de> for $name {
                fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                    u32::deserialize(d).map(Self)
                }
            }
        )*
    };
}

/// A vector indexed by a typed index.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IndexVec<I, T> {
    data: Vec<T>,
    #[serde(skip)]
    _marker: PhantomData<fn(&I)>,
}

impl<I, T: fmt::Debug> fmt::Debug for IndexVec<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.data.iter()).finish()
    }
}

impl<I, T> Default for IndexVec<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I, T> IndexVec<I, T> {
    /// Create a new empty index vector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Create an index vector with the given capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            _marker: PhantomData,
        }
    }

    /// Create an index vector from a raw vector.
    #[must_use]
    pub fn from_vec(data: Vec<T>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    /// Get the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Push a value and return its index.
    pub fn push(&mut self, value: T) -> I
    where
        I: From<usize>,
    {
        let idx = self.data.len();
        self.data.push(value);
        I::from(idx)
    }

    /// Get a reference to the value at the given index.
    #[must_use]
    pub fn get(&self, idx: I) -> Option<&T>
    where
        I: Into<usize>,
    {
        self.data.get(idx.into())
    }

    /// Get a mutable reference to the value at the given index.
    #[must_use]
    pub fn get_mut(&mut self, idx: I) -> Option<&mut T>
    where
        I: Into<usize>,
    {
        self.data.get_mut(idx.into())
    }

    /// Iterate over the values.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Iterate over the values mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Iterate over indices and values.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (I, &T)>
    where
        I: From<usize>,
    {
        self.data.iter().enumerate().map(|(i, v)| (I::from(i), v))
    }

    /// Iterate over indices.
    pub fn indices(&self) -> impl Iterator<Item = I>
    where
        I: From<usize>,
    {
        (0..self.data.len()).map(I::from)
    }

    /// Get the next index that would be assigned.
    #[must_use]
    pub fn next_index(&self) -> I
    where
        I: From<usize>,
    {
        I::from(self.data.len())
    }

    /// Get the raw data vector.
    #[must_use]
    pub fn raw(&self) -> &Vec<T> {
        &self.data
    }

    /// Get the raw data vector mutably.
    #[must_use]
    pub fn raw_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Convert into the raw data vector.
    #[must_use]
    pub fn into_raw(self) -> Vec<T> {
        self.data
    }
}

impl<I: Into<usize>, T> std::ops::Index<I> for IndexVec<I, T> {
    type Output = T;

    fn index(&self, idx: I) -> &Self::Output {
        &self.data[idx.into()]
    }
}

impl<I: Into<usize>, T> std::ops::IndexMut<I> for IndexVec<I, T> {
    fn index_mut(&mut self, idx: I) -> &mut Self::Output {
        &mut self.data[idx.into()]
    }
}

impl<I, T> FromIterator<T> for IndexVec<I, T> {
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        Self {
            data: iter.into_iter().collect(),
            _marker: PhantomData,
        }
    }
}

impl<I, T> IntoIterator for IndexVec<I, T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, I, T> IntoIterator for &'a IndexVec<I, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

/// A map indexed by a typed index, backed by a vector.
pub type IndexMap<I, T> = IndexVec<I, Option<T>>;

#[cfg(test)]
mod tests {
    use super::*;

    define_index! {
        /// Test index.
        pub struct TestId;
    }

    #[test]
    fn test_index_vec() {
        let mut vec: IndexVec<TestId, &str> = IndexVec::new();

        let id1 = vec.push("hello");
        let id2 = vec.push("world");

        assert_eq!(vec[id1], "hello");
        assert_eq!(vec[id2], "world");
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_index_enumerated() {
        let mut vec: IndexVec<TestId, i32> = IndexVec::new();
        vec.push(10);
        vec.push(20);
        vec.push(30);

        let items: Vec<_> = vec.iter_enumerated().collect();
        assert_eq!(items.len(), 3);
        assert_eq!(*items[0].1, 10);
        assert_eq!(*items[1].1, 20);
        assert_eq!(*items[2].1, 30);
    }
}
