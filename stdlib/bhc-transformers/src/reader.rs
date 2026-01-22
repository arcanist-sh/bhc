//! Reader and ReaderT
//!
//! The Reader monad for computations with access to a shared environment.
//! ReaderT is the transformer version that can be stacked with other monads.
//!
//! # Example
//!
//! ```ignore
//! // A computation that reads from config
//! fn get_name(config: &Config) -> String {
//!     config.name.clone()
//! }
//!
//! let reader = Reader::new(get_name);
//! let name = reader.run(&config);
//! ```

use crate::identity::Identity;
use std::fmt::{self, Debug};
use std::marker::PhantomData;

/// The Reader monad - computations with read-only access to an environment.
///
/// Reader encapsulates functions of the form `r -> a`, where `r` is the
/// environment type that can be read from.
pub struct Reader<R, A> {
    run_fn: Box<dyn Fn(&R) -> A>,
}

impl<R: 'static, A> Reader<R, A> {
    /// Create a new Reader from a function.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&R) -> A + 'static,
    {
        Reader { run_fn: Box::new(f) }
    }

    /// Run the reader with the given environment.
    pub fn run(&self, env: &R) -> A {
        (self.run_fn)(env)
    }

    /// Get the environment.
    ///
    /// This is the `ask` operation in Haskell.
    pub fn ask() -> Reader<R, R>
    where
        R: Clone + 'static,
    {
        Reader::new(|r: &R| r.clone())
    }

    /// Get a specific component of the environment.
    ///
    /// This is the `asks` operation in Haskell.
    pub fn asks<B, F>(f: F) -> Reader<R, B>
    where
        F: Fn(&R) -> B + 'static,
    {
        Reader::new(f)
    }

    /// Wrap a pure value in Reader (ignores the environment).
    pub fn pure(value: A) -> Self
    where
        A: Clone + 'static,
    {
        Reader::new(move |_: &R| value.clone())
    }

    /// Map a function over the result.
    ///
    /// Functor instance.
    pub fn map<B, F>(self, f: F) -> Reader<R, B>
    where
        F: Fn(A) -> B + 'static,
        A: 'static,
    {
        Reader::new(move |r: &R| f((self.run_fn)(r)))
    }

    /// Monadic bind.
    pub fn and_then<B, F>(self, f: F) -> Reader<R, B>
    where
        F: Fn(A) -> Reader<R, B> + 'static,
        A: 'static,
    {
        Reader::new(move |r: &R| {
            let a = (self.run_fn)(r);
            f(a).run(r)
        })
    }

    /// Alias for and_then.
    pub fn bind<B, F>(self, f: F) -> Reader<R, B>
    where
        F: Fn(A) -> Reader<R, B> + 'static,
        A: 'static,
    {
        self.and_then(f)
    }

    /// Execute a reader with a modified environment.
    ///
    /// This is the `local` operation in Haskell.
    pub fn local<F>(self, modify: F) -> Reader<R, A>
    where
        F: Fn(&R) -> R + 'static,
        A: 'static,
    {
        Reader::new(move |r: &R| {
            let modified = modify(r);
            (self.run_fn)(&modified)
        })
    }
}

impl<R: 'static, A: Debug + 'static> Debug for Reader<R, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reader(<function>)")
    }
}

/// ReaderT monad transformer.
///
/// ReaderT adds read-only environment access to any monad.
///
/// In Haskell: `newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }`
///
/// Since Rust doesn't have higher-kinded types, we use a callback-based approach
/// where the inner monad is represented by the result type.
pub struct ReaderT<R, M, A> {
    run_fn: Box<dyn Fn(&R) -> A>,
    _phantom: PhantomData<M>,
}

impl<R: 'static, M, A> ReaderT<R, M, A> {
    /// Create a new ReaderT from a function.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&R) -> A + 'static,
    {
        ReaderT {
            run_fn: Box::new(f),
            _phantom: PhantomData,
        }
    }

    /// Run the ReaderT with the given environment.
    pub fn run(&self, env: &R) -> A {
        (self.run_fn)(env)
    }

    /// Get the environment.
    pub fn ask() -> ReaderT<R, M, R>
    where
        R: Clone + 'static,
    {
        ReaderT::new(|r: &R| r.clone())
    }

    /// Get a specific component of the environment.
    pub fn asks<B, F>(f: F) -> ReaderT<R, M, B>
    where
        F: Fn(&R) -> B + 'static,
    {
        ReaderT::new(f)
    }

    /// Map a function over the result.
    pub fn map<B, F>(self, f: F) -> ReaderT<R, M, B>
    where
        F: Fn(A) -> B + 'static,
        A: 'static,
    {
        ReaderT::new(move |r: &R| f((self.run_fn)(r)))
    }

    /// Lift a monadic value into ReaderT.
    pub fn lift(ma: A) -> Self
    where
        A: Clone + 'static,
    {
        ReaderT::new(move |_: &R| ma.clone())
    }

    /// Execute with a modified environment.
    pub fn local<F>(self, modify: F) -> ReaderT<R, M, A>
    where
        F: Fn(&R) -> R + 'static,
        A: 'static,
    {
        ReaderT::new(move |r: &R| {
            let modified = modify(r);
            (self.run_fn)(&modified)
        })
    }
}

impl<R: 'static, M, A: Debug + 'static> Debug for ReaderT<R, M, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReaderT(<function>)")
    }
}

/// Type alias for Reader using Identity as the base monad.
pub type ReaderI<R, A> = ReaderT<R, Identity<A>, A>;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct Config {
        name: String,
        port: u16,
    }

    #[test]
    fn test_reader_new_run() {
        let reader: Reader<i32, i32> = Reader::new(|x: &i32| x * 2);
        assert_eq!(reader.run(&5), 10);
    }

    #[test]
    fn test_reader_ask() {
        let reader: Reader<String, String> = Reader::<String, String>::ask();
        assert_eq!(reader.run(&String::from("hello")), "hello");
    }

    #[test]
    fn test_reader_asks() {
        let config = Config {
            name: String::from("test"),
            port: 8080,
        };
        let reader: Reader<Config, u16> = Reader::<Config, u16>::asks(|c: &Config| c.port);
        assert_eq!(reader.run(&config), 8080);
    }

    #[test]
    fn test_reader_pure() {
        let reader: Reader<String, i32> = Reader::pure(42);
        assert_eq!(reader.run(&String::from("ignored")), 42);
    }

    #[test]
    fn test_reader_map() {
        let reader: Reader<i32, i32> = Reader::new(|x: &i32| *x);
        let mapped = reader.map(|x| x * 2);
        assert_eq!(mapped.run(&5), 10);
    }

    #[test]
    fn test_reader_and_then() {
        let reader: Reader<i32, i32> = Reader::new(|x: &i32| *x);
        let result = reader.and_then(|a| Reader::new(move |x: &i32| a + *x));
        assert_eq!(result.run(&5), 10); // 5 + 5
    }

    #[test]
    fn test_reader_local() {
        let reader: Reader<i32, i32> = Reader::new(|x: &i32| *x);
        let local_reader = reader.local(|x: &i32| x * 2);
        assert_eq!(local_reader.run(&5), 10); // reads from modified env
    }

    #[test]
    fn test_reader_composition() {
        let config = Config {
            name: String::from("app"),
            port: 3000,
        };

        let get_greeting: Reader<Config, String> = Reader::<Config, String>::asks(|c: &Config| c.name.clone())
            .and_then(|name| Reader::pure(format!("Hello, {}!", name)));

        assert_eq!(get_greeting.run(&config), "Hello, app!");
    }

    #[test]
    fn test_reader_functor_identity() {
        let reader: Reader<i32, i32> = Reader::new(|x: &i32| *x);
        let result1 = reader.run(&5);

        let reader2: Reader<i32, i32> = Reader::new(|x: &i32| *x);
        let result2 = reader2.map(|x| x).run(&5);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_reader_functor_composition() {
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;

        let reader: Reader<i32, i32> = Reader::new(|x: &i32| *x);
        let left = reader.map(move |x| f(g(x))).run(&5);

        let reader2: Reader<i32, i32> = Reader::new(|x: &i32| *x);
        let right = reader2.map(g).map(f).run(&5);

        assert_eq!(left, right);
    }

    #[test]
    fn test_reader_t_new_run() {
        let reader: ReaderT<i32, (), i32> = ReaderT::new(|x: &i32| x * 2);
        assert_eq!(reader.run(&5), 10);
    }

    #[test]
    fn test_reader_t_ask() {
        let reader: ReaderT<String, (), String> = ReaderT::<String, (), String>::ask();
        assert_eq!(reader.run(&String::from("hello")), "hello");
    }

    #[test]
    fn test_reader_t_asks() {
        let config = Config {
            name: String::from("test"),
            port: 8080,
        };
        let reader: ReaderT<Config, (), String> = ReaderT::<Config, (), String>::asks(|c: &Config| c.name.clone());
        assert_eq!(reader.run(&config), "test");
    }

    #[test]
    fn test_reader_t_map() {
        let reader: ReaderT<i32, (), i32> = ReaderT::new(|x: &i32| *x);
        let mapped = reader.map(|x| x * 2);
        assert_eq!(mapped.run(&5), 10);
    }

    #[test]
    fn test_reader_t_lift() {
        let reader: ReaderT<String, (), i32> = ReaderT::lift(42);
        assert_eq!(reader.run(&String::from("ignored")), 42);
    }

    #[test]
    fn test_reader_t_local() {
        let reader: ReaderT<i32, (), i32> = ReaderT::new(|x: &i32| *x);
        let local_reader = reader.local(|x: &i32| x * 2);
        assert_eq!(local_reader.run(&5), 10);
    }
}
