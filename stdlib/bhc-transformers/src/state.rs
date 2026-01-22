//! State and StateT
//!
//! The State monad for computations with mutable state.
//! StateT is the transformer version that can be stacked with other monads.
//!
//! # Example
//!
//! ```ignore
//! let increment = State::new(|s: i32| (s, s + 1));
//! let (value, new_state) = increment.run(0);
//! assert_eq!(value, 0);
//! assert_eq!(new_state, 1);
//! ```

use crate::identity::Identity;
use std::fmt::{self, Debug};
use std::marker::PhantomData;

/// The State monad - computations with mutable state.
///
/// State encapsulates functions of the form `s -> (a, s)`, where `s` is the
/// state type that can be read and modified.
pub struct State<S, A> {
    run_fn: Box<dyn Fn(S) -> (A, S)>,
}

impl<S, A> State<S, A> {
    /// Create a new State from a function.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(S) -> (A, S) + 'static,
    {
        State { run_fn: Box::new(f) }
    }

    /// Run the state computation with an initial state.
    ///
    /// Returns (result, final_state).
    pub fn run(self, initial: S) -> (A, S) {
        (self.run_fn)(initial)
    }

    /// Run and return only the result, discarding the final state.
    pub fn eval(self, initial: S) -> A {
        self.run(initial).0
    }

    /// Run and return only the final state, discarding the result.
    pub fn exec(self, initial: S) -> S {
        self.run(initial).1
    }

    /// Get the current state.
    ///
    /// This is the `get` operation in Haskell.
    pub fn get() -> State<S, S>
    where
        S: Clone + 'static,
    {
        State::new(|s: S| (s.clone(), s))
    }

    /// Replace the state with a new value.
    ///
    /// This is the `put` operation in Haskell.
    pub fn put(new_state: S) -> State<S, ()>
    where
        S: Clone + 'static,
    {
        State::new(move |_: S| ((), new_state.clone()))
    }

    /// Modify the state with a function.
    ///
    /// This is the `modify` operation in Haskell.
    pub fn modify<F>(f: F) -> State<S, ()>
    where
        F: Fn(S) -> S + 'static,
    {
        State::new(move |s: S| ((), f(s)))
    }

    /// Modify the state with a strict function.
    ///
    /// Same as modify, but the new state is evaluated strictly.
    pub fn modify_strict<F>(f: F) -> State<S, ()>
    where
        F: Fn(S) -> S + 'static,
    {
        State::new(move |s: S| {
            let new_state = f(s);
            ((), new_state)
        })
    }

    /// Get a specific component of the state.
    ///
    /// This is the `gets` operation in Haskell.
    pub fn gets<B, F>(f: F) -> State<S, B>
    where
        F: Fn(&S) -> B + 'static,
        S: 'static,
    {
        State::new(move |s: S| {
            let b = f(&s);
            (b, s)
        })
    }

    /// Wrap a pure value in State (doesn't affect state).
    pub fn pure(value: A) -> Self
    where
        A: Clone + 'static,
    {
        State::new(move |s: S| (value.clone(), s))
    }

    /// Embed a simple state action.
    pub fn state<F>(f: F) -> Self
    where
        F: Fn(S) -> (A, S) + 'static,
    {
        State::new(f)
    }

    /// Map a function over the result.
    ///
    /// Functor instance.
    pub fn map<B, F>(self, f: F) -> State<S, B>
    where
        F: Fn(A) -> B + 'static,
        A: 'static,
        S: 'static,
    {
        State::new(move |s: S| {
            let (a, new_s) = (self.run_fn)(s);
            (f(a), new_s)
        })
    }

    /// Monadic bind.
    pub fn and_then<B, F>(self, f: F) -> State<S, B>
    where
        F: Fn(A) -> State<S, B> + 'static,
        A: 'static,
        S: 'static,
    {
        State::new(move |s: S| {
            let (a, s1) = (self.run_fn)(s);
            f(a).run(s1)
        })
    }

    /// Alias for and_then.
    pub fn bind<B, F>(self, f: F) -> State<S, B>
    where
        F: Fn(A) -> State<S, B> + 'static,
        A: 'static,
        S: 'static,
    {
        self.and_then(f)
    }

    /// Execute the state action and return a transformed state.
    pub fn with_state<F>(self, f: F) -> State<S, A>
    where
        F: Fn(S) -> S + 'static,
        A: 'static,
        S: 'static,
    {
        State::new(move |s: S| (self.run_fn)(f(s)))
    }
}

impl<S: 'static, A: Debug + 'static> Debug for State<S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "State(<function>)")
    }
}

/// StateT monad transformer.
///
/// StateT adds mutable state to any monad.
///
/// In Haskell: `newtype StateT s m a = StateT { runStateT :: s -> m (a, s) }`
///
/// Since Rust doesn't have higher-kinded types, we use a callback-based approach.
pub struct StateT<S, M, A> {
    run_fn: Box<dyn Fn(S) -> (A, S)>,
    _phantom: PhantomData<M>,
}

impl<S, M, A> StateT<S, M, A> {
    /// Create a new StateT from a function.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(S) -> (A, S) + 'static,
    {
        StateT {
            run_fn: Box::new(f),
            _phantom: PhantomData,
        }
    }

    /// Run the state computation.
    pub fn run(self, initial: S) -> (A, S) {
        (self.run_fn)(initial)
    }

    /// Run and return only the result.
    pub fn eval(self, initial: S) -> A {
        self.run(initial).0
    }

    /// Run and return only the final state.
    pub fn exec(self, initial: S) -> S {
        self.run(initial).1
    }

    /// Get the current state.
    pub fn get() -> StateT<S, M, S>
    where
        S: Clone + 'static,
    {
        StateT::new(|s: S| (s.clone(), s))
    }

    /// Replace the state.
    pub fn put(new_state: S) -> StateT<S, M, ()>
    where
        S: Clone + 'static,
    {
        StateT::new(move |_: S| ((), new_state.clone()))
    }

    /// Modify the state.
    pub fn modify<F>(f: F) -> StateT<S, M, ()>
    where
        F: Fn(S) -> S + 'static,
    {
        StateT::new(move |s: S| ((), f(s)))
    }

    /// Get a component of the state.
    pub fn gets<B, F>(f: F) -> StateT<S, M, B>
    where
        F: Fn(&S) -> B + 'static,
        S: 'static,
    {
        StateT::new(move |s: S| {
            let b = f(&s);
            (b, s)
        })
    }

    /// Map a function over the result.
    pub fn map<B, F>(self, f: F) -> StateT<S, M, B>
    where
        F: Fn(A) -> B + 'static,
        A: 'static,
        S: 'static,
    {
        StateT::new(move |s: S| {
            let (a, new_s) = (self.run_fn)(s);
            (f(a), new_s)
        })
    }

    /// Lift a monadic value into StateT.
    pub fn lift(ma: A) -> Self
    where
        A: Clone + 'static,
    {
        StateT::new(move |s: S| (ma.clone(), s))
    }
}

impl<S: 'static, M, A: Debug + 'static> Debug for StateT<S, M, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StateT(<function>)")
    }
}

/// Type alias for State using Identity as the base monad.
pub type StateI<S, A> = StateT<S, Identity<(A, S)>, A>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_new_run() {
        let state: State<i32, i32> = State::new(|s| (s * 2, s + 1));
        let (result, new_state) = state.run(5);
        assert_eq!(result, 10);
        assert_eq!(new_state, 6);
    }

    #[test]
    fn test_state_eval() {
        let state: State<i32, i32> = State::new(|s| (s * 2, s + 1));
        assert_eq!(state.eval(5), 10);
    }

    #[test]
    fn test_state_exec() {
        let state: State<i32, i32> = State::new(|s| (s * 2, s + 1));
        assert_eq!(state.exec(5), 6);
    }

    #[test]
    fn test_state_get() {
        let state: State<i32, i32> = State::<i32, i32>::get();
        let (result, new_state) = state.run(42);
        assert_eq!(result, 42);
        assert_eq!(new_state, 42);
    }

    #[test]
    fn test_state_put() {
        let state: State<i32, ()> = State::<i32, ()>::put(100);
        let ((), new_state) = state.run(42);
        assert_eq!(new_state, 100);
    }

    #[test]
    fn test_state_modify() {
        let state: State<i32, ()> = State::<i32, ()>::modify(|s| s * 2);
        let ((), new_state) = state.run(5);
        assert_eq!(new_state, 10);
    }

    #[test]
    fn test_state_gets() {
        #[derive(Clone)]
        struct Point {
            x: i32,
            y: i32,
        }

        let state: State<Point, i32> = State::<Point, i32>::gets(|p: &Point| p.x);
        let (result, _) = state.run(Point { x: 10, y: 20 });
        assert_eq!(result, 10);
    }

    #[test]
    fn test_state_pure() {
        let state: State<i32, String> = State::pure(String::from("hello"));
        let (result, new_state) = state.run(42);
        assert_eq!(result, "hello");
        assert_eq!(new_state, 42);
    }

    #[test]
    fn test_state_map() {
        let state: State<i32, i32> = State::new(|s| (s, s + 1));
        let mapped = state.map(|x| x * 2);
        let (result, new_state) = mapped.run(5);
        assert_eq!(result, 10); // 5 * 2
        assert_eq!(new_state, 6); // 5 + 1
    }

    #[test]
    fn test_state_and_then() {
        let state: State<i32, i32> = State::<i32, i32>::get();
        let result = state.and_then(|x| State::new(move |s| (x + s, s + 1)));
        let (value, new_state) = result.run(5);
        assert_eq!(value, 10); // 5 + 5
        assert_eq!(new_state, 6); // 5 + 1
    }

    #[test]
    fn test_state_counter() {
        // Simulate a counter using State monad
        let tick: State<i32, i32> = State::<i32, i32>::get().and_then(|n| {
            State::<i32, ()>::put(n + 1).and_then(move |()| State::pure(n))
        });

        // Run tick three times
        let (r1, s1) = tick.run(0);
        assert_eq!(r1, 0);
        assert_eq!(s1, 1);

        let tick2: State<i32, i32> = State::<i32, i32>::get().and_then(|n| {
            State::<i32, ()>::put(n + 1).and_then(move |()| State::pure(n))
        });
        let (r2, s2) = tick2.run(s1);
        assert_eq!(r2, 1);
        assert_eq!(s2, 2);
    }

    #[test]
    fn test_state_functor_identity() {
        let state: State<i32, i32> = State::new(|s| (s, s));
        let (r1, s1) = state.run(5);

        let state2: State<i32, i32> = State::new(|s| (s, s));
        let (r2, s2) = state2.map(|x| x).run(5);

        assert_eq!(r1, r2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_state_functor_composition() {
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;

        let state: State<i32, i32> = State::new(|s| (s, s));
        let (left_r, left_s) = state.map(move |x| f(g(x))).run(5);

        let state2: State<i32, i32> = State::new(|s| (s, s));
        let (right_r, right_s) = state2.map(g).map(f).run(5);

        assert_eq!(left_r, right_r);
        assert_eq!(left_s, right_s);
    }

    #[test]
    fn test_state_monad_left_identity() {
        let f = |x: i32| State::new(move |s: i32| (x * 2, s + 1));
        let a = 5;

        let left = State::pure(a).and_then(f).run(0);
        let right = f(a).run(0);

        assert_eq!(left, right);
    }

    #[test]
    fn test_state_monad_right_identity() {
        let m: State<i32, i32> = State::new(|s| (s, s + 1));
        let left = m.run(5);

        let m2: State<i32, i32> = State::new(|s| (s, s + 1));
        let right = m2.and_then(State::pure).run(5);

        assert_eq!(left, right);
    }

    #[test]
    fn test_state_t_new_run() {
        let state: StateT<i32, (), i32> = StateT::new(|s| (s * 2, s + 1));
        let (result, new_state) = state.run(5);
        assert_eq!(result, 10);
        assert_eq!(new_state, 6);
    }

    #[test]
    fn test_state_t_get() {
        let state: StateT<i32, (), i32> = StateT::<i32, (), i32>::get();
        let (result, new_state) = state.run(42);
        assert_eq!(result, 42);
        assert_eq!(new_state, 42);
    }

    #[test]
    fn test_state_t_put() {
        let state: StateT<i32, (), ()> = StateT::<i32, (), ()>::put(100);
        let ((), new_state) = state.run(42);
        assert_eq!(new_state, 100);
    }

    #[test]
    fn test_state_t_modify() {
        let state: StateT<i32, (), ()> = StateT::<i32, (), ()>::modify(|s| s * 2);
        let ((), new_state) = state.run(5);
        assert_eq!(new_state, 10);
    }

    #[test]
    fn test_state_t_map() {
        let state: StateT<i32, (), i32> = StateT::new(|s| (s, s + 1));
        let mapped = state.map(|x| x * 2);
        let (result, new_state) = mapped.run(5);
        assert_eq!(result, 10);
        assert_eq!(new_state, 6);
    }

    #[test]
    fn test_state_t_lift() {
        let state: StateT<i32, (), String> = StateT::lift(String::from("hello"));
        let (result, new_state) = state.run(42);
        assert_eq!(result, "hello");
        assert_eq!(new_state, 42);
    }
}
