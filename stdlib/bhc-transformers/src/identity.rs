//! Identity and IdentityT
//!
//! The identity monad and its transformer. IdentityT is the base transformer
//! that adds no effects - useful as a building block.
//!
//! # Example
//!
//! ```ignore
//! let id = Identity::new(42);
//! let mapped = id.map(|x| x * 2);
//! assert_eq!(mapped.run(), 84);
//! ```

use std::fmt::{self, Debug};
use std::marker::PhantomData;

/// The Identity monad - a trivial wrapper that adds no effects.
///
/// Identity is the base case for monad transformers. When you stack
/// transformers and need a "pure" base, Identity is the choice.
#[derive(Clone, PartialEq, Eq)]
pub struct Identity<A> {
    value: A,
}

impl<A> Identity<A> {
    /// Wrap a value in Identity.
    #[inline]
    pub fn new(value: A) -> Self {
        Identity { value }
    }

    /// Extract the value from Identity.
    #[inline]
    pub fn run(self) -> A {
        self.value
    }

    /// Get a reference to the inner value.
    #[inline]
    pub fn as_ref(&self) -> &A {
        &self.value
    }

    /// Map a function over the identity.
    ///
    /// Functor instance.
    #[inline]
    pub fn map<B, F>(self, f: F) -> Identity<B>
    where
        F: FnOnce(A) -> B,
    {
        Identity::new(f(self.value))
    }

    /// Applicative pure - same as new.
    #[inline]
    pub fn pure(value: A) -> Self {
        Identity::new(value)
    }

    /// Apply a wrapped function to a wrapped value.
    ///
    /// Applicative instance.
    #[inline]
    pub fn ap<B, F>(self, f: Identity<F>) -> Identity<B>
    where
        F: FnOnce(A) -> B,
    {
        Identity::new((f.value)(self.value))
    }

    /// Monadic bind.
    #[inline]
    pub fn and_then<B, F>(self, f: F) -> Identity<B>
    where
        F: FnOnce(A) -> Identity<B>,
    {
        f(self.value)
    }

    /// Alias for and_then (Haskell's >>=).
    #[inline]
    pub fn bind<B, F>(self, f: F) -> Identity<B>
    where
        F: FnOnce(A) -> Identity<B>,
    {
        self.and_then(f)
    }

    /// Sequence two identity computations, discarding the first result.
    #[inline]
    pub fn then<B>(self, other: Identity<B>) -> Identity<B> {
        other
    }
}

impl<A: Debug> Debug for Identity<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Identity").field(&self.value).finish()
    }
}

impl<A: Default> Default for Identity<A> {
    fn default() -> Self {
        Identity::new(A::default())
    }
}

/// IdentityT monad transformer.
///
/// IdentityT wraps an inner monad without adding any effects.
/// It's primarily useful as a base case for monad transformer stacks.
///
/// In Rust, we model this using a phantom type for the monad
/// and storing the actual computation.
pub struct IdentityT<M, A> {
    inner: A,
    _phantom: PhantomData<M>,
}

impl<M, A> IdentityT<M, A> {
    /// Wrap a value in IdentityT.
    #[inline]
    pub fn new(value: A) -> Self {
        IdentityT {
            inner: value,
            _phantom: PhantomData,
        }
    }

    /// Extract the inner value.
    #[inline]
    pub fn run(self) -> A {
        self.inner
    }

    /// Map a function over the transformer.
    #[inline]
    pub fn map<B, F>(self, f: F) -> IdentityT<M, B>
    where
        F: FnOnce(A) -> B,
    {
        IdentityT::new(f(self.inner))
    }

    /// Lift a monadic value into IdentityT.
    #[inline]
    pub fn lift(value: A) -> Self {
        IdentityT::new(value)
    }
}

impl<M, A: Clone> Clone for IdentityT<M, A> {
    fn clone(&self) -> Self {
        IdentityT::new(self.inner.clone())
    }
}

impl<M, A: PartialEq> PartialEq for IdentityT<M, A> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<M, A: Eq> Eq for IdentityT<M, A> {}

impl<M, A: Debug> Debug for IdentityT<M, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IdentityT").field(&self.inner).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_new_run() {
        let id = Identity::new(42);
        assert_eq!(id.run(), 42);
    }

    #[test]
    fn test_identity_map() {
        let id = Identity::new(10);
        let mapped = id.map(|x| x * 2);
        assert_eq!(mapped.run(), 20);
    }

    #[test]
    fn test_identity_and_then() {
        let id = Identity::new(5);
        let result = id.and_then(|x| Identity::new(x + 1));
        assert_eq!(result.run(), 6);
    }

    #[test]
    fn test_identity_functor_identity_law() {
        // fmap id = id
        let id = Identity::new(42);
        let mapped = id.clone().map(|x| x);
        assert_eq!(mapped.run(), id.run());
    }

    #[test]
    fn test_identity_functor_composition_law() {
        // fmap (f . g) = fmap f . fmap g
        let id = Identity::new(10);
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;

        let left = id.clone().map(|x| f(g(x)));
        let right = id.map(g).map(f);

        assert_eq!(left.run(), right.run());
    }

    #[test]
    fn test_identity_monad_left_identity() {
        // return a >>= f = f a
        let f = |x: i32| Identity::new(x * 2);
        let a = 5;

        let left = Identity::new(a).and_then(f);
        let right = f(a);

        assert_eq!(left.run(), right.run());
    }

    #[test]
    fn test_identity_monad_right_identity() {
        // m >>= return = m
        let m = Identity::new(42);
        let result = m.clone().and_then(Identity::new);
        assert_eq!(result.run(), m.run());
    }

    #[test]
    fn test_identity_monad_associativity() {
        // (m >>= f) >>= g = m >>= (\x -> f x >>= g)
        let m = Identity::new(5);
        let f = |x: i32| Identity::new(x + 1);
        let g = |x: i32| Identity::new(x * 2);

        let left = m.clone().and_then(f).and_then(g);
        let right = m.and_then(|x| f(x).and_then(g));

        assert_eq!(left.run(), right.run());
    }

    #[test]
    fn test_identity_t_new_run() {
        let id: IdentityT<(), i32> = IdentityT::new(42);
        assert_eq!(id.run(), 42);
    }

    #[test]
    fn test_identity_t_map() {
        let id: IdentityT<(), i32> = IdentityT::new(10);
        let mapped = id.map(|x| x * 2);
        assert_eq!(mapped.run(), 20);
    }
}
