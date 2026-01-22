//! Writer and WriterT
//!
//! The Writer monad for computations that produce a log or accumulated output.
//! WriterT is the transformer version that can be stacked with other monads.
//!
//! # Example
//!
//! ```ignore
//! let computation = Writer::new(42, vec!["started"])
//!     .and_then(|x| Writer::new(x * 2, vec!["doubled"]));
//! let (result, log) = computation.run();
//! assert_eq!(result, 84);
//! assert_eq!(log, vec!["started", "doubled"]);
//! ```

use crate::identity::Identity;
use std::fmt::{self, Debug};
use std::marker::PhantomData;

/// Trait for types that can be combined (monoid).
///
/// This is equivalent to Haskell's Monoid typeclass.
pub trait Monoid: Clone {
    /// The identity element.
    fn mempty() -> Self;

    /// Combine two values.
    fn mappend(&self, other: &Self) -> Self;
}

// Monoid instances for common types

impl Monoid for String {
    fn mempty() -> Self {
        String::new()
    }

    fn mappend(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.push_str(other);
        result
    }
}

impl<T: Clone> Monoid for Vec<T> {
    fn mempty() -> Self {
        Vec::new()
    }

    fn mappend(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.extend(other.iter().cloned());
        result
    }
}

impl Monoid for () {
    fn mempty() -> Self {
        ()
    }

    fn mappend(&self, _other: &Self) -> Self {
        ()
    }
}

/// Monoid instance for Option<T> where T is a Monoid.
/// Uses the "first non-None" semantics.
impl<T: Monoid> Monoid for Option<T> {
    fn mempty() -> Self {
        None
    }

    fn mappend(&self, other: &Self) -> Self {
        match (self, other) {
            (None, None) => None,
            (Some(x), None) => Some(x.clone()),
            (None, Some(y)) => Some(y.clone()),
            (Some(x), Some(y)) => Some(x.mappend(y)),
        }
    }
}

/// Monoid instance for sum of integers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sum<T>(pub T);

impl Monoid for Sum<i32> {
    fn mempty() -> Self {
        Sum(0)
    }

    fn mappend(&self, other: &Self) -> Self {
        Sum(self.0 + other.0)
    }
}

impl Monoid for Sum<i64> {
    fn mempty() -> Self {
        Sum(0)
    }

    fn mappend(&self, other: &Self) -> Self {
        Sum(self.0 + other.0)
    }
}

/// Monoid instance for product of integers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Product<T>(pub T);

impl Monoid for Product<i32> {
    fn mempty() -> Self {
        Product(1)
    }

    fn mappend(&self, other: &Self) -> Self {
        Product(self.0 * other.0)
    }
}

impl Monoid for Product<i64> {
    fn mempty() -> Self {
        Product(1)
    }

    fn mappend(&self, other: &Self) -> Self {
        Product(self.0 * other.0)
    }
}

/// The Writer monad - computations that produce a log/output alongside a value.
///
/// Writer pairs a computation result with accumulated output (typically a log).
/// The output must be a Monoid so that logs can be combined.
#[derive(Clone)]
pub struct Writer<W, A> {
    value: A,
    output: W,
}

impl<W: Monoid, A> Writer<W, A> {
    /// Create a new Writer with a value and output.
    pub fn new(value: A, output: W) -> Self {
        Writer { value, output }
    }

    /// Run the writer and get both the value and output.
    pub fn run(self) -> (A, W) {
        (self.value, self.output)
    }

    /// Run and return only the value, discarding the output.
    pub fn eval(self) -> A {
        self.value
    }

    /// Run and return only the output, discarding the value.
    pub fn exec(self) -> W {
        self.output
    }

    /// Add output without producing a value.
    ///
    /// This is the `tell` operation in Haskell.
    pub fn tell(output: W) -> Writer<W, ()> {
        Writer { value: (), output }
    }

    /// Wrap a pure value with empty output.
    pub fn pure(value: A) -> Self
    where
        A: Clone,
    {
        Writer {
            value,
            output: W::mempty(),
        }
    }

    /// Execute a writer and add the output to the result.
    ///
    /// This is the `listen` operation in Haskell.
    pub fn listen(self) -> Writer<W, (A, W)>
    where
        W: Clone,
    {
        Writer {
            value: (self.value, self.output.clone()),
            output: self.output,
        }
    }

    /// Execute a writer with a function that can modify the output.
    ///
    /// This is the `pass` operation in Haskell.
    pub fn pass<F>(self) -> Writer<W, A>
    where
        A: 'static,
        F: Fn(W) -> W,
    {
        self
    }

    /// Map a function over the value.
    ///
    /// Functor instance.
    pub fn map<B, F>(self, f: F) -> Writer<W, B>
    where
        F: FnOnce(A) -> B,
    {
        Writer {
            value: f(self.value),
            output: self.output,
        }
    }

    /// Map a function over the output.
    pub fn map_output<V: Monoid, F>(self, f: F) -> Writer<V, A>
    where
        F: FnOnce(W) -> V,
    {
        Writer {
            value: self.value,
            output: f(self.output),
        }
    }

    /// Monadic bind.
    pub fn and_then<B, F>(self, f: F) -> Writer<W, B>
    where
        F: FnOnce(A) -> Writer<W, B>,
    {
        let Writer { value: b, output: w2 } = f(self.value);
        Writer {
            value: b,
            output: self.output.mappend(&w2),
        }
    }

    /// Alias for and_then.
    pub fn bind<B, F>(self, f: F) -> Writer<W, B>
    where
        F: FnOnce(A) -> Writer<W, B>,
    {
        self.and_then(f)
    }

    /// Replace the output for this computation.
    pub fn censor<F>(self, f: F) -> Writer<W, A>
    where
        F: FnOnce(W) -> W,
    {
        Writer {
            value: self.value,
            output: f(self.output),
        }
    }
}

impl<W: Debug, A: Debug> Debug for Writer<W, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Writer")
            .field("value", &self.value)
            .field("output", &self.output)
            .finish()
    }
}

impl<W: Monoid + PartialEq, A: PartialEq> PartialEq for Writer<W, A> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.output == other.output
    }
}

impl<W: Monoid + Eq, A: Eq> Eq for Writer<W, A> {}

/// WriterT monad transformer.
///
/// WriterT adds logging/accumulation capabilities to any monad.
///
/// In Haskell: `newtype WriterT w m a = WriterT { runWriterT :: m (a, w) }`
#[derive(Clone)]
pub struct WriterT<W, M, A> {
    value: A,
    output: W,
    _phantom: PhantomData<M>,
}

impl<W: Monoid, M, A> WriterT<W, M, A> {
    /// Create a new WriterT.
    pub fn new(value: A, output: W) -> Self {
        WriterT {
            value,
            output,
            _phantom: PhantomData,
        }
    }

    /// Run and get the value and output.
    pub fn run(self) -> (A, W) {
        (self.value, self.output)
    }

    /// Run and return only the value.
    pub fn eval(self) -> A {
        self.value
    }

    /// Run and return only the output.
    pub fn exec(self) -> W {
        self.output
    }

    /// Add output.
    pub fn tell(output: W) -> WriterT<W, M, ()> {
        WriterT::new((), output)
    }

    /// Map a function over the value.
    pub fn map<B, F>(self, f: F) -> WriterT<W, M, B>
    where
        F: FnOnce(A) -> B,
    {
        WriterT::new(f(self.value), self.output)
    }

    /// Lift a monadic value into WriterT.
    pub fn lift(value: A) -> Self {
        WriterT::new(value, W::mempty())
    }

    /// Execute and get the output as part of the result.
    pub fn listen(self) -> WriterT<W, M, (A, W)>
    where
        W: Clone,
    {
        WriterT::new((self.value, self.output.clone()), self.output)
    }

    /// Modify the output.
    pub fn censor<F>(self, f: F) -> WriterT<W, M, A>
    where
        F: FnOnce(W) -> W,
    {
        WriterT::new(self.value, f(self.output))
    }
}

impl<W: Debug, M, A: Debug> Debug for WriterT<W, M, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriterT")
            .field("value", &self.value)
            .field("output", &self.output)
            .finish()
    }
}

impl<W: Monoid + PartialEq, M, A: PartialEq> PartialEq for WriterT<W, M, A> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.output == other.output
    }
}

impl<W: Monoid + Eq, M, A: Eq> Eq for WriterT<W, M, A> {}

/// Type alias for Writer using Identity as the base monad.
pub type WriterI<W, A> = WriterT<W, Identity<(A, W)>, A>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_new_run() {
        let writer: Writer<Vec<&str>, i32> = Writer::new(42, vec!["hello"]);
        let (value, output) = writer.run();
        assert_eq!(value, 42);
        assert_eq!(output, vec!["hello"]);
    }

    #[test]
    fn test_writer_eval_exec() {
        let writer: Writer<Vec<&str>, i32> = Writer::new(42, vec!["hello"]);
        assert_eq!(writer.clone().eval(), 42);
        assert_eq!(writer.exec(), vec!["hello"]);
    }

    #[test]
    fn test_writer_tell() {
        let writer: Writer<Vec<&str>, ()> = Writer::<Vec<&str>, ()>::tell(vec!["logged"]);
        let ((), output) = writer.run();
        assert_eq!(output, vec!["logged"]);
    }

    #[test]
    fn test_writer_pure() {
        let writer: Writer<Vec<&str>, i32> = Writer::pure(42);
        let (value, output) = writer.run();
        assert_eq!(value, 42);
        assert_eq!(output, Vec::<&str>::new());
    }

    #[test]
    fn test_writer_map() {
        let writer: Writer<Vec<&str>, i32> = Writer::new(10, vec!["init"]);
        let mapped = writer.map(|x| x * 2);
        let (value, output) = mapped.run();
        assert_eq!(value, 20);
        assert_eq!(output, vec!["init"]);
    }

    #[test]
    fn test_writer_and_then() {
        let writer: Writer<Vec<String>, i32> = Writer::new(10, vec![String::from("first")]);
        let result = writer.and_then(|x| Writer::new(x * 2, vec![String::from("second")]));
        let (value, output) = result.run();
        assert_eq!(value, 20);
        assert_eq!(output, vec![String::from("first"), String::from("second")]);
    }

    #[test]
    fn test_writer_listen() {
        let writer: Writer<Vec<&str>, i32> = Writer::new(42, vec!["log"]);
        let listened = writer.listen();
        let ((value, log), output) = listened.run();
        assert_eq!(value, 42);
        assert_eq!(log, vec!["log"]);
        assert_eq!(output, vec!["log"]);
    }

    #[test]
    fn test_writer_censor() {
        let writer: Writer<String, i32> = Writer::new(42, String::from("hello"));
        let censored = writer.censor(|s| s.to_uppercase());
        let (value, output) = censored.run();
        assert_eq!(value, 42);
        assert_eq!(output, "HELLO");
    }

    #[test]
    fn test_writer_logging_example() {
        // Simulate logging computation
        fn log_step<T: Clone>(msg: &str, value: T) -> Writer<Vec<String>, T> {
            Writer::new(value, vec![msg.to_string()])
        }

        let computation = log_step("started", 1)
            .and_then(|x| log_step("doubled", x * 2))
            .and_then(|x| log_step("added 10", x + 10));

        let (result, log) = computation.run();
        assert_eq!(result, 12);
        assert_eq!(log, vec!["started", "doubled", "added 10"]);
    }

    #[test]
    fn test_writer_functor_identity() {
        let writer: Writer<String, i32> = Writer::new(42, String::from("test"));
        let (v1, o1) = writer.clone().run();
        let (v2, o2) = writer.map(|x| x).run();
        assert_eq!(v1, v2);
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_writer_functor_composition() {
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;

        let writer1: Writer<String, i32> = Writer::new(5, String::from("test"));
        let (left_v, left_o) = writer1.map(|x| f(g(x))).run();

        let writer2: Writer<String, i32> = Writer::new(5, String::from("test"));
        let (right_v, right_o) = writer2.map(g).map(f).run();

        assert_eq!(left_v, right_v);
        assert_eq!(left_o, right_o);
    }

    #[test]
    fn test_writer_monad_left_identity() {
        let f = |x: i32| Writer::new(x * 2, vec![String::from("doubled")]);
        let a = 5;

        let left = Writer::<Vec<String>, i32>::pure(a).and_then(f).run();
        let right = f(a).run();

        assert_eq!(left, right);
    }

    #[test]
    fn test_writer_monad_right_identity() {
        let m: Writer<String, i32> = Writer::new(42, String::from("log"));
        let (v1, o1) = m.clone().run();
        let (v2, o2) = m.and_then(Writer::pure).run();
        assert_eq!(v1, v2);
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_sum_monoid() {
        let a = Sum(5);
        let b = Sum(3);
        assert_eq!(a.mappend(&b), Sum(8));
        assert_eq!(Sum::<i32>::mempty(), Sum(0));
    }

    #[test]
    fn test_product_monoid() {
        let a = Product(5);
        let b = Product(3);
        assert_eq!(a.mappend(&b), Product(15));
        assert_eq!(Product::<i32>::mempty(), Product(1));
    }

    #[test]
    fn test_writer_t_new_run() {
        let writer: WriterT<Vec<&str>, (), i32> = WriterT::new(42, vec!["hello"]);
        let (value, output) = writer.run();
        assert_eq!(value, 42);
        assert_eq!(output, vec!["hello"]);
    }

    #[test]
    fn test_writer_t_tell() {
        let writer: WriterT<Vec<&str>, (), ()> = WriterT::<Vec<&str>, (), ()>::tell(vec!["logged"]);
        let ((), output) = writer.run();
        assert_eq!(output, vec!["logged"]);
    }

    #[test]
    fn test_writer_t_lift() {
        let writer: WriterT<Vec<&str>, (), i32> = WriterT::lift(42);
        let (value, output) = writer.run();
        assert_eq!(value, 42);
        assert_eq!(output, Vec::<&str>::new());
    }

    #[test]
    fn test_writer_t_map() {
        let writer: WriterT<Vec<&str>, (), i32> = WriterT::new(10, vec!["init"]);
        let mapped = writer.map(|x| x * 2);
        let (value, output) = mapped.run();
        assert_eq!(value, 20);
        assert_eq!(output, vec!["init"]);
    }

    #[test]
    fn test_writer_t_listen() {
        let writer: WriterT<Vec<&str>, (), i32> = WriterT::new(42, vec!["log"]);
        let listened = writer.listen();
        let ((value, log), output) = listened.run();
        assert_eq!(value, 42);
        assert_eq!(log, vec!["log"]);
        assert_eq!(output, vec!["log"]);
    }

    #[test]
    fn test_writer_t_censor() {
        let writer: WriterT<String, (), i32> = WriterT::new(42, String::from("hello"));
        let censored = writer.censor(|s| s.to_uppercase());
        let (value, output) = censored.run();
        assert_eq!(value, 42);
        assert_eq!(output, "HELLO");
    }
}
