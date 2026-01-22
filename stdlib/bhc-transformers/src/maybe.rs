//! MaybeT transformer
//!
//! The MaybeT monad transformer adds optional/failure semantics to any monad.
//!
//! # Example
//!
//! ```ignore
//! let computation = MaybeT::just(42)
//!     .and_then(|x| if x > 0 { MaybeT::just(x * 2) } else { MaybeT::nothing() });
//! assert_eq!(computation.run(), Some(84));
//! ```

use crate::identity::Identity;
use std::fmt::{self, Debug};
use std::marker::PhantomData;

/// MaybeT monad transformer.
///
/// MaybeT adds optional/failure semantics to any monad.
/// A computation can either produce a value (Just) or fail (Nothing).
///
/// In Haskell: `newtype MaybeT m a = MaybeT { runMaybeT :: m (Maybe a) }`
#[derive(Clone)]
pub struct MaybeT<M, A> {
    inner: Option<A>,
    _phantom: PhantomData<M>,
}

impl<M, A> MaybeT<M, A> {
    /// Create a MaybeT that holds a value.
    pub fn just(value: A) -> Self {
        MaybeT {
            inner: Some(value),
            _phantom: PhantomData,
        }
    }

    /// Create a MaybeT that represents failure.
    pub fn nothing() -> Self {
        MaybeT {
            inner: None,
            _phantom: PhantomData,
        }
    }

    /// Alias for just.
    pub fn pure(value: A) -> Self {
        MaybeT::just(value)
    }

    /// Create from an Option.
    pub fn from_option(opt: Option<A>) -> Self {
        MaybeT {
            inner: opt,
            _phantom: PhantomData,
        }
    }

    /// Run the MaybeT and get the Option result.
    pub fn run(self) -> Option<A> {
        self.inner
    }

    /// Check if this MaybeT holds a value.
    pub fn is_just(&self) -> bool {
        self.inner.is_some()
    }

    /// Check if this MaybeT represents failure.
    pub fn is_nothing(&self) -> bool {
        self.inner.is_none()
    }

    /// Map a function over the value.
    ///
    /// Functor instance.
    pub fn map<B, F>(self, f: F) -> MaybeT<M, B>
    where
        F: FnOnce(A) -> B,
    {
        MaybeT::from_option(self.inner.map(f))
    }

    /// Monadic bind.
    pub fn and_then<B, F>(self, f: F) -> MaybeT<M, B>
    where
        F: FnOnce(A) -> MaybeT<M, B>,
    {
        match self.inner {
            Some(a) => f(a),
            None => MaybeT::nothing(),
        }
    }

    /// Alias for and_then.
    pub fn bind<B, F>(self, f: F) -> MaybeT<M, B>
    where
        F: FnOnce(A) -> MaybeT<M, B>,
    {
        self.and_then(f)
    }

    /// Lift a monadic value into MaybeT.
    pub fn lift(value: A) -> Self {
        MaybeT::just(value)
    }

    /// Provide an alternative if this computation fails.
    ///
    /// This is the `<|>` operation from Alternative.
    pub fn or(self, other: MaybeT<M, A>) -> MaybeT<M, A> {
        match self.inner {
            Some(a) => MaybeT::just(a),
            None => other,
        }
    }

    /// Provide a lazy alternative if this computation fails.
    pub fn or_else<F>(self, f: F) -> MaybeT<M, A>
    where
        F: FnOnce() -> MaybeT<M, A>,
    {
        match self.inner {
            Some(a) => MaybeT::just(a),
            None => f(),
        }
    }

    /// Get the value or a default.
    pub fn unwrap_or(self, default: A) -> A {
        self.inner.unwrap_or(default)
    }

    /// Get the value or compute a default.
    pub fn unwrap_or_else<F>(self, f: F) -> A
    where
        F: FnOnce() -> A,
    {
        self.inner.unwrap_or_else(f)
    }

    /// Filter the value with a predicate.
    pub fn filter<P>(self, predicate: P) -> MaybeT<M, A>
    where
        P: FnOnce(&A) -> bool,
    {
        MaybeT::from_option(self.inner.filter(predicate))
    }

    /// Require a boolean condition, failing if false.
    pub fn guard(condition: bool) -> MaybeT<M, ()> {
        if condition {
            MaybeT::just(())
        } else {
            MaybeT::nothing()
        }
    }
}

impl<M, A: PartialEq> PartialEq for MaybeT<M, A> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<M, A: Eq> Eq for MaybeT<M, A> {}

impl<M, A: Debug> Debug for MaybeT<M, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            Some(a) => f.debug_tuple("Just").field(a).finish(),
            None => write!(f, "Nothing"),
        }
    }
}

impl<M, A> From<Option<A>> for MaybeT<M, A> {
    fn from(opt: Option<A>) -> Self {
        MaybeT::from_option(opt)
    }
}

impl<M, A> From<MaybeT<M, A>> for Option<A> {
    fn from(maybe_t: MaybeT<M, A>) -> Self {
        maybe_t.run()
    }
}

/// Type alias for MaybeT using Identity as the base monad.
pub type MaybeI<A> = MaybeT<Identity<Option<A>>, A>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_t_just() {
        let m: MaybeT<(), i32> = MaybeT::just(42);
        assert!(m.is_just());
        assert!(!m.is_nothing());
        assert_eq!(m.run(), Some(42));
    }

    #[test]
    fn test_maybe_t_nothing() {
        let m: MaybeT<(), i32> = MaybeT::nothing();
        assert!(!m.is_just());
        assert!(m.is_nothing());
        assert_eq!(m.run(), None);
    }

    #[test]
    fn test_maybe_t_from_option() {
        let m1: MaybeT<(), i32> = MaybeT::from_option(Some(42));
        assert_eq!(m1.run(), Some(42));

        let m2: MaybeT<(), i32> = MaybeT::from_option(None);
        assert_eq!(m2.run(), None);
    }

    #[test]
    fn test_maybe_t_map() {
        let m: MaybeT<(), i32> = MaybeT::just(10);
        let mapped = m.map(|x| x * 2);
        assert_eq!(mapped.run(), Some(20));

        let m2: MaybeT<(), i32> = MaybeT::nothing();
        let mapped2 = m2.map(|x| x * 2);
        assert_eq!(mapped2.run(), None);
    }

    #[test]
    fn test_maybe_t_and_then() {
        let m: MaybeT<(), i32> = MaybeT::just(10);
        let result = m.and_then(|x| {
            if x > 5 {
                MaybeT::just(x * 2)
            } else {
                MaybeT::nothing()
            }
        });
        assert_eq!(result.run(), Some(20));

        let m2: MaybeT<(), i32> = MaybeT::just(3);
        let result2 = m2.and_then(|x| {
            if x > 5 {
                MaybeT::just(x * 2)
            } else {
                MaybeT::nothing()
            }
        });
        assert_eq!(result2.run(), None);

        let m3: MaybeT<(), i32> = MaybeT::nothing();
        let result3 = m3.and_then(|x| MaybeT::just(x * 2));
        assert_eq!(result3.run(), None);
    }

    #[test]
    fn test_maybe_t_or() {
        let m1: MaybeT<(), i32> = MaybeT::just(42);
        let m2: MaybeT<(), i32> = MaybeT::just(0);
        assert_eq!(m1.or(m2).run(), Some(42));

        let m3: MaybeT<(), i32> = MaybeT::nothing();
        let m4: MaybeT<(), i32> = MaybeT::just(0);
        assert_eq!(m3.or(m4).run(), Some(0));

        let m5: MaybeT<(), i32> = MaybeT::nothing();
        let m6: MaybeT<(), i32> = MaybeT::nothing();
        assert_eq!(m5.or(m6).run(), None);
    }

    #[test]
    fn test_maybe_t_or_else() {
        let m: MaybeT<(), i32> = MaybeT::nothing();
        let result = m.or_else(|| MaybeT::just(42));
        assert_eq!(result.run(), Some(42));
    }

    #[test]
    fn test_maybe_t_unwrap_or() {
        let m1: MaybeT<(), i32> = MaybeT::just(42);
        assert_eq!(m1.unwrap_or(0), 42);

        let m2: MaybeT<(), i32> = MaybeT::nothing();
        assert_eq!(m2.unwrap_or(0), 0);
    }

    #[test]
    fn test_maybe_t_filter() {
        let m1: MaybeT<(), i32> = MaybeT::just(10);
        let filtered = m1.filter(|x| *x > 5);
        assert_eq!(filtered.run(), Some(10));

        let m2: MaybeT<(), i32> = MaybeT::just(3);
        let filtered2 = m2.filter(|x| *x > 5);
        assert_eq!(filtered2.run(), None);
    }

    #[test]
    fn test_maybe_t_guard() {
        let guarded: MaybeT<(), ()> = MaybeT::<(), ()>::guard(true);
        assert_eq!(guarded.run(), Some(()));

        let not_guarded: MaybeT<(), ()> = MaybeT::<(), ()>::guard(false);
        assert_eq!(not_guarded.run(), None);
    }

    #[test]
    fn test_maybe_t_functor_identity() {
        let m: MaybeT<(), i32> = MaybeT::just(42);
        let left = m.clone().run();
        let right = m.map(|x| x).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_maybe_t_functor_composition() {
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;

        let m: MaybeT<(), i32> = MaybeT::just(5);
        let left = m.clone().map(|x| f(g(x))).run();
        let right = m.map(g).map(f).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_maybe_t_monad_left_identity() {
        let f = |x: i32| MaybeT::<(), i32>::just(x * 2);
        let a = 5;

        let left = MaybeT::just(a).and_then(f).run();
        let right = f(a).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_maybe_t_monad_right_identity() {
        let m: MaybeT<(), i32> = MaybeT::just(42);
        let left = m.clone().run();
        let right = m.and_then(MaybeT::just).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_maybe_t_monad_associativity() {
        let m: MaybeT<(), i32> = MaybeT::just(5);
        let f = |x: i32| MaybeT::<(), i32>::just(x + 1);
        let g = |x: i32| MaybeT::<(), i32>::just(x * 2);

        let left = m.clone().and_then(f).and_then(g).run();
        let right = m.and_then(|x| f(x).and_then(g)).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_maybe_t_chain_example() {
        // Example: lookup chain that can fail at any point
        fn lookup_user(_id: i32) -> MaybeT<(), String> {
            MaybeT::just(String::from("Alice"))
        }

        fn lookup_address(_name: &str) -> MaybeT<(), String> {
            MaybeT::just(String::from("123 Main St"))
        }

        fn lookup_zip(address: &str) -> MaybeT<(), String> {
            if address.contains("Main") {
                MaybeT::just(String::from("12345"))
            } else {
                MaybeT::nothing()
            }
        }

        let result = lookup_user(1)
            .and_then(|name| lookup_address(&name))
            .and_then(|addr| lookup_zip(&addr));

        assert_eq!(result.run(), Some(String::from("12345")));
    }
}
