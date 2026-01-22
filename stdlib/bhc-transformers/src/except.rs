//! Except and ExceptT
//!
//! The Except monad for computations that may fail with an error.
//! ExceptT is the transformer version for stacking with other monads.
//!
//! # Example
//!
//! ```ignore
//! fn divide(a: i32, b: i32) -> Except<String, i32> {
//!     if b == 0 {
//!         Except::throw("division by zero".to_string())
//!     } else {
//!         Except::pure(a / b)
//!     }
//! }
//!
//! let result = divide(10, 2);
//! assert_eq!(result.run(), Ok(5));
//! ```

use crate::identity::Identity;
use std::fmt::{self, Debug};
use std::marker::PhantomData;

/// The Except monad - computations that may fail with an error.
///
/// Except encapsulates a computation that can either succeed with a value
/// or fail with an error. It's essentially `Either E A` with error-biased operations.
#[derive(Clone, PartialEq, Eq)]
pub enum Except<E, A> {
    /// Error case.
    Error(E),
    /// Success case.
    Ok(A),
}

impl<E, A> Except<E, A> {
    /// Create a successful computation.
    pub fn ok(value: A) -> Self {
        Except::Ok(value)
    }

    /// Create a failed computation (alias for ok).
    pub fn pure(value: A) -> Self {
        Except::Ok(value)
    }

    /// Throw an error.
    pub fn throw(error: E) -> Self {
        Except::Error(error)
    }

    /// Run the computation and get the result.
    pub fn run(self) -> Result<A, E> {
        match self {
            Except::Ok(a) => Ok(a),
            Except::Error(e) => Err(e),
        }
    }

    /// Check if the computation succeeded.
    pub fn is_ok(&self) -> bool {
        matches!(self, Except::Ok(_))
    }

    /// Check if the computation failed.
    pub fn is_error(&self) -> bool {
        matches!(self, Except::Error(_))
    }

    /// Map a function over the success value.
    ///
    /// Functor instance.
    pub fn map<B, F>(self, f: F) -> Except<E, B>
    where
        F: FnOnce(A) -> B,
    {
        match self {
            Except::Ok(a) => Except::Ok(f(a)),
            Except::Error(e) => Except::Error(e),
        }
    }

    /// Map a function over the error.
    pub fn map_error<F, G>(self, f: G) -> Except<F, A>
    where
        G: FnOnce(E) -> F,
    {
        match self {
            Except::Ok(a) => Except::Ok(a),
            Except::Error(e) => Except::Error(f(e)),
        }
    }

    /// Monadic bind.
    pub fn and_then<B, F>(self, f: F) -> Except<E, B>
    where
        F: FnOnce(A) -> Except<E, B>,
    {
        match self {
            Except::Ok(a) => f(a),
            Except::Error(e) => Except::Error(e),
        }
    }

    /// Alias for and_then.
    pub fn bind<B, F>(self, f: F) -> Except<E, B>
    where
        F: FnOnce(A) -> Except<E, B>,
    {
        self.and_then(f)
    }

    /// Handle an error.
    ///
    /// This is the `catchE` operation in Haskell.
    pub fn catch<F>(self, handler: F) -> Except<E, A>
    where
        F: FnOnce(E) -> Except<E, A>,
    {
        match self {
            Except::Ok(a) => Except::Ok(a),
            Except::Error(e) => handler(e),
        }
    }

    /// Transform the error with a new error type.
    pub fn with_except<F, G>(self, f: G) -> Except<F, A>
    where
        G: FnOnce(E) -> F,
    {
        self.map_error(f)
    }

    /// Unwrap the success value or use a default.
    pub fn unwrap_or(self, default: A) -> A {
        match self {
            Except::Ok(a) => a,
            Except::Error(_) => default,
        }
    }

    /// Unwrap the success value or compute a default.
    pub fn unwrap_or_else<F>(self, f: F) -> A
    where
        F: FnOnce(E) -> A,
    {
        match self {
            Except::Ok(a) => a,
            Except::Error(e) => f(e),
        }
    }

    /// Convert from Result.
    pub fn from_result(result: Result<A, E>) -> Self {
        match result {
            Ok(a) => Except::Ok(a),
            Err(e) => Except::Error(e),
        }
    }
}

impl<E: Debug, A: Debug> Debug for Except<E, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Except::Ok(a) => f.debug_tuple("Ok").field(a).finish(),
            Except::Error(e) => f.debug_tuple("Error").field(e).finish(),
        }
    }
}

impl<E, A> From<Result<A, E>> for Except<E, A> {
    fn from(result: Result<A, E>) -> Self {
        Except::from_result(result)
    }
}

impl<E, A> From<Except<E, A>> for Result<A, E> {
    fn from(except: Except<E, A>) -> Self {
        except.run()
    }
}

/// ExceptT monad transformer.
///
/// ExceptT adds error handling capabilities to any monad.
///
/// In Haskell: `newtype ExceptT e m a = ExceptT { runExceptT :: m (Either e a) }`
#[derive(Clone)]
pub struct ExceptT<E, M, A> {
    inner: Except<E, A>,
    _phantom: PhantomData<M>,
}

impl<E, M, A> ExceptT<E, M, A> {
    /// Create an ExceptT from an Except.
    pub fn new(inner: Except<E, A>) -> Self {
        ExceptT {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Create a successful ExceptT.
    pub fn ok(value: A) -> Self {
        ExceptT::new(Except::Ok(value))
    }

    /// Create a successful ExceptT (alias for ok).
    pub fn pure(value: A) -> Self {
        ExceptT::ok(value)
    }

    /// Throw an error.
    pub fn throw(error: E) -> Self {
        ExceptT::new(Except::Error(error))
    }

    /// Run the ExceptT.
    pub fn run(self) -> Result<A, E> {
        self.inner.run()
    }

    /// Map a function over the success value.
    pub fn map<B, F>(self, f: F) -> ExceptT<E, M, B>
    where
        F: FnOnce(A) -> B,
    {
        ExceptT::new(self.inner.map(f))
    }

    /// Map a function over the error.
    pub fn map_error<F, G>(self, f: G) -> ExceptT<F, M, A>
    where
        G: FnOnce(E) -> F,
    {
        ExceptT::new(self.inner.map_error(f))
    }

    /// Lift a monadic value into ExceptT.
    pub fn lift(value: A) -> Self {
        ExceptT::ok(value)
    }

    /// Handle an error.
    pub fn catch<F>(self, handler: F) -> ExceptT<E, M, A>
    where
        F: FnOnce(E) -> ExceptT<E, M, A>,
    {
        match self.inner {
            Except::Ok(a) => ExceptT::ok(a),
            Except::Error(e) => handler(e),
        }
    }
}

impl<E: PartialEq, M, A: PartialEq> PartialEq for ExceptT<E, M, A> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<E: Eq, M, A: Eq> Eq for ExceptT<E, M, A> {}

impl<E: Debug, M, A: Debug> Debug for ExceptT<E, M, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExceptT").field(&self.inner).finish()
    }
}

/// Type alias for Except using Identity as the base monad.
pub type ExceptI<E, A> = ExceptT<E, Identity<Result<A, E>>, A>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_except_ok() {
        let e: Except<String, i32> = Except::ok(42);
        assert!(e.is_ok());
        assert!(!e.is_error());
        assert_eq!(e.run(), Ok(42));
    }

    #[test]
    fn test_except_throw() {
        let e: Except<String, i32> = Except::throw(String::from("error"));
        assert!(!e.is_ok());
        assert!(e.is_error());
        assert_eq!(e.run(), Err(String::from("error")));
    }

    #[test]
    fn test_except_map() {
        let e: Except<String, i32> = Except::ok(10);
        let mapped = e.map(|x| x * 2);
        assert_eq!(mapped.run(), Ok(20));

        let e2: Except<String, i32> = Except::throw(String::from("error"));
        let mapped2 = e2.map(|x| x * 2);
        assert_eq!(mapped2.run(), Err(String::from("error")));
    }

    #[test]
    fn test_except_map_error() {
        let e: Except<String, i32> = Except::throw(String::from("error"));
        let mapped = e.map_error(|s| s.len());
        assert_eq!(mapped.run(), Err(5));
    }

    #[test]
    fn test_except_and_then() {
        fn safe_div(a: i32, b: i32) -> Except<String, i32> {
            if b == 0 {
                Except::throw(String::from("division by zero"))
            } else {
                Except::ok(a / b)
            }
        }

        let result = Except::ok(10).and_then(|x| safe_div(x, 2));
        assert_eq!(result.run(), Ok(5));

        let result2 = Except::ok(10).and_then(|x| safe_div(x, 0));
        assert_eq!(result2.run(), Err(String::from("division by zero")));

        let result3: Except<String, i32> =
            Except::throw(String::from("initial error")).and_then(|x| safe_div(x, 2));
        assert_eq!(result3.run(), Err(String::from("initial error")));
    }

    #[test]
    fn test_except_catch() {
        let e: Except<String, i32> = Except::throw(String::from("error"));
        let caught = e.catch(|_| Except::ok(0));
        assert_eq!(caught.run(), Ok(0));

        let e2: Except<String, i32> = Except::ok(42);
        let not_caught = e2.catch(|_| Except::ok(0));
        assert_eq!(not_caught.run(), Ok(42));
    }

    #[test]
    fn test_except_unwrap_or() {
        let e: Except<String, i32> = Except::ok(42);
        assert_eq!(e.unwrap_or(0), 42);

        let e2: Except<String, i32> = Except::throw(String::from("error"));
        assert_eq!(e2.unwrap_or(0), 0);
    }

    #[test]
    fn test_except_unwrap_or_else() {
        let e: Except<String, i32> = Except::ok(42);
        assert_eq!(e.unwrap_or_else(|_| 0), 42);

        let e2: Except<String, i32> = Except::throw(String::from("error"));
        assert_eq!(e2.unwrap_or_else(|s| s.len() as i32), 5);
    }

    #[test]
    fn test_except_from_result() {
        let ok: Result<i32, String> = Ok(42);
        let except = Except::from_result(ok);
        assert_eq!(except.run(), Ok(42));

        let err: Result<i32, String> = Err(String::from("error"));
        let except2 = Except::from_result(err);
        assert_eq!(except2.run(), Err(String::from("error")));
    }

    #[test]
    fn test_except_functor_identity() {
        let e: Except<String, i32> = Except::ok(42);
        let left = e.clone().run();
        let right = e.map(|x| x).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_except_functor_composition() {
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;

        let e: Except<String, i32> = Except::ok(5);
        let left = e.clone().map(|x| f(g(x))).run();
        let right = e.map(g).map(f).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_except_monad_left_identity() {
        let f = |x: i32| Except::<String, i32>::ok(x * 2);
        let a = 5;

        let left = Except::ok(a).and_then(f).run();
        let right = f(a).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_except_monad_right_identity() {
        let m: Except<String, i32> = Except::ok(42);
        let left = m.clone().run();
        let right = m.and_then(Except::ok).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_except_monad_associativity() {
        let m: Except<String, i32> = Except::ok(5);
        let f = |x: i32| Except::<String, i32>::ok(x + 1);
        let g = |x: i32| Except::<String, i32>::ok(x * 2);

        let left = m.clone().and_then(f).and_then(g).run();
        let right = m.and_then(|x| f(x).and_then(g)).run();
        assert_eq!(left, right);
    }

    #[test]
    fn test_except_t_ok() {
        let e: ExceptT<String, (), i32> = ExceptT::ok(42);
        assert_eq!(e.run(), Ok(42));
    }

    #[test]
    fn test_except_t_throw() {
        let e: ExceptT<String, (), i32> = ExceptT::throw(String::from("error"));
        assert_eq!(e.run(), Err(String::from("error")));
    }

    #[test]
    fn test_except_t_map() {
        let e: ExceptT<String, (), i32> = ExceptT::ok(10);
        let mapped = e.map(|x| x * 2);
        assert_eq!(mapped.run(), Ok(20));
    }

    #[test]
    fn test_except_t_map_error() {
        let e: ExceptT<String, (), i32> = ExceptT::throw(String::from("error"));
        let mapped = e.map_error(|s| s.len());
        assert_eq!(mapped.run(), Err(5));
    }

    #[test]
    fn test_except_t_lift() {
        let e: ExceptT<String, (), i32> = ExceptT::lift(42);
        assert_eq!(e.run(), Ok(42));
    }

    #[test]
    fn test_except_t_catch() {
        let e: ExceptT<String, (), i32> = ExceptT::throw(String::from("error"));
        let caught = e.catch(|_| ExceptT::ok(0));
        assert_eq!(caught.run(), Ok(0));
    }
}
