//! Extended function combinators
//!
//! Additional function combinators corresponding to Haskell's Data.Function.
//!
//! # Example
//!
//! ```ignore
//! use bhc_base::function::*;
//!
//! // Using fix for recursive functions
//! let factorial = fix(|fac| Box::new(move |n| if n == 0 { 1 } else { n * fac(n - 1) }));
//! assert_eq!(factorial(5), 120);
//!
//! // Using on combinator
//! let compare_length = on(|a: usize, b: usize| a.cmp(&b), |s: &str| s.len());
//! ```

// ============================================================
// Core Function Combinators
// ============================================================

/// Identity function.
///
/// Returns its argument unchanged.
///
/// # Example
///
/// ```ignore
/// assert_eq!(id(42), 42);
/// assert_eq!(id("hello"), "hello");
/// ```
#[inline]
pub fn id<A>(x: A) -> A {
    x
}

/// Constant function.
///
/// Returns a function that always returns the first argument,
/// ignoring any second argument.
///
/// This is `const` in Haskell (renamed due to Rust keyword).
///
/// # Example
///
/// ```ignore
/// let always_five = constant(5);
/// assert_eq!(always_five(100), 5);
/// assert_eq!(always_five("ignored"), 5);
/// ```
#[inline]
pub fn constant<A: Clone, B>(x: A) -> impl Fn(B) -> A {
    move |_| x.clone()
}

/// Function composition.
///
/// Composes two functions: `compose(f, g)` is equivalent to `f . g` in Haskell.
/// That is, `compose(f, g)(x)` = `f(g(x))`.
///
/// # Example
///
/// ```ignore
/// let add_one = |x| x + 1;
/// let double = |x| x * 2;
/// let composed = compose(add_one, double);
/// assert_eq!(composed(5), 11); // (5 * 2) + 1
/// ```
#[inline]
pub fn compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(B) -> C,
    G: Fn(A) -> B,
{
    move |a| f(g(a))
}

/// Flip the arguments of a binary function.
///
/// # Example
///
/// ```ignore
/// let sub = |a, b| a - b;
/// let flipped = flip(sub);
/// assert_eq!(flipped(3, 10), 7); // 10 - 3
/// ```
#[inline]
pub fn flip<A, B, C, F>(f: F) -> impl Fn(B, A) -> C
where
    F: Fn(A, B) -> C,
{
    move |b, a| f(a, b)
}

/// Apply a function to an argument.
///
/// This is the `$` operator in Haskell.
/// In Rust this is trivial, but useful for consistency with Haskell patterns.
///
/// # Example
///
/// ```ignore
/// assert_eq!(apply(|x| x * 2, 5), 10);
/// ```
#[inline]
pub fn apply<A, B, F>(f: F, x: A) -> B
where
    F: FnOnce(A) -> B,
{
    f(x)
}

/// Reverse function application (flip of apply).
///
/// This is the `&` operator in Haskell.
///
/// # Example
///
/// ```ignore
/// assert_eq!(pipe(5, |x| x * 2), 10);
/// ```
#[inline]
pub fn pipe<A, B, F>(x: A, f: F) -> B
where
    F: FnOnce(A) -> B,
{
    f(x)
}

/// Apply a binary function with an intermediate projection.
///
/// `on(op, f)(x, y)` = `op(f(x), f(y))`
///
/// This is the `on` combinator from Data.Function.
///
/// # Example
///
/// ```ignore
/// // Compare strings by their length
/// let compare_length = on(|a: usize, b: usize| a.cmp(&b), |s: &str| s.len());
/// ```
#[inline]
pub fn on<A, B, C, F, G>(op: F, proj: G) -> impl Fn(A, A) -> C
where
    F: Fn(B, B) -> C,
    G: Fn(A) -> B,
{
    move |x, y| op(proj(x), proj(y))
}

// ============================================================
// Currying and Uncurrying
// ============================================================

/// Curry a binary function.
///
/// Transforms a function of two arguments into a function that takes
/// one argument and returns a function taking the second argument.
///
/// # Example
///
/// ```ignore
/// let add = |a, b| a + b;
/// let curried = curry(add);
/// let add_five = curried(5);
/// assert_eq!(add_five(3), 8);
/// ```
#[inline]
pub fn curry<A: Clone + 'static, B, C, F: Clone + 'static>(f: F) -> impl Fn(A) -> Box<dyn Fn(B) -> C>
where
    F: Fn(A, B) -> C,
{
    move |a: A| {
        let f = f.clone();
        let a = a.clone();
        Box::new(move |b: B| f(a.clone(), b))
    }
}

/// Uncurry a curried function.
///
/// Transforms a curried function back into a function of two arguments.
///
/// # Example
///
/// ```ignore
/// let curried = |a| move |b| a + b;
/// let uncurried = uncurry(curried);
/// assert_eq!(uncurried(5, 3), 8);
/// ```
#[inline]
pub fn uncurry<A, B, C, F, G>(f: F) -> impl Fn(A, B) -> C
where
    F: Fn(A) -> G,
    G: Fn(B) -> C,
{
    move |a, b| f(a)(b)
}

/// Curry a function of a tuple into a function of two arguments.
///
/// # Example
///
/// ```ignore
/// let add_pair = |(a, b)| a + b;
/// let curried = curry_pair(add_pair);
/// assert_eq!(curried(5, 3), 8);
/// ```
#[inline]
pub fn curry_pair<A, B, C, F>(f: F) -> impl Fn(A, B) -> C
where
    F: Fn((A, B)) -> C,
{
    move |a, b| f((a, b))
}

/// Uncurry a binary function into a function of a tuple.
///
/// # Example
///
/// ```ignore
/// let add = |a, b| a + b;
/// let uncurried = uncurry_pair(add);
/// assert_eq!(uncurried((5, 3)), 8);
/// ```
#[inline]
pub fn uncurry_pair<A, B, C, F>(f: F) -> impl Fn((A, B)) -> C
where
    F: Fn(A, B) -> C,
{
    move |(a, b)| f(a, b)
}

// ============================================================
// Fixed Point Combinator
// ============================================================

/// Fixed point combinator for recursive functions.
///
/// This allows defining recursive functions without explicit self-reference.
///
/// # Example
///
/// ```ignore
/// let factorial = fix(|fac| Box::new(move |n: u64| {
///     if n == 0 { 1 } else { n * fac(n - 1) }
/// }));
/// assert_eq!(factorial(5), 120);
/// ```
pub fn fix<A, B, F>(f: F) -> impl Fn(A) -> B
where
    F: Fn(&dyn Fn(A) -> B) -> Box<dyn Fn(A) -> B>,
{
    move |a| {
        // Create a recursive closure
        fn fix_inner<A, B, F>(f: &F, a: A) -> B
        where
            F: Fn(&dyn Fn(A) -> B) -> Box<dyn Fn(A) -> B>,
        {
            let rec = |x| fix_inner(f, x);
            f(&rec)(a)
        }
        fix_inner(&f, a)
    }
}

// ============================================================
// Tuple Operations
// ============================================================

/// Extract the first component of a pair.
#[inline]
pub fn fst<A, B>((a, _): (A, B)) -> A {
    a
}

/// Extract the second component of a pair.
#[inline]
pub fn snd<A, B>((_, b): (A, B)) -> B {
    b
}

/// Swap the components of a pair.
#[inline]
pub fn swap<A, B>((a, b): (A, B)) -> (B, A) {
    (b, a)
}

/// Apply a function to the first component of a pair.
#[inline]
pub fn first<A, B, C, F>(f: F, (a, b): (A, B)) -> (C, B)
where
    F: FnOnce(A) -> C,
{
    (f(a), b)
}

/// Apply a function to the second component of a pair.
#[inline]
pub fn second<A, B, C, F>(f: F, (a, b): (A, B)) -> (A, C)
where
    F: FnOnce(B) -> C,
{
    (a, f(b))
}

/// Apply two functions to the components of a pair.
#[inline]
pub fn both<A, B, C, D, F, G>(f: F, g: G, (a, b): (A, B)) -> (C, D)
where
    F: FnOnce(A) -> C,
    G: FnOnce(B) -> D,
{
    (f(a), g(b))
}

/// Duplicate a value into a pair.
#[inline]
pub fn dup<A: Clone>(a: A) -> (A, A) {
    (a.clone(), a)
}

// ============================================================
// Boolean Combinators
// ============================================================

/// Boolean `if-then-else` as a function.
///
/// # Example
///
/// ```ignore
/// assert_eq!(bool_case(true, 1, 2), 1);
/// assert_eq!(bool_case(false, 1, 2), 2);
/// ```
#[inline]
pub fn bool_case<A>(condition: bool, then_value: A, else_value: A) -> A {
    if condition {
        then_value
    } else {
        else_value
    }
}

/// Lazy boolean `if-then-else`.
///
/// Only evaluates the branch that is taken.
#[inline]
pub fn bool_case_lazy<A, F, G>(condition: bool, then_fn: F, else_fn: G) -> A
where
    F: FnOnce() -> A,
    G: FnOnce() -> A,
{
    if condition {
        then_fn()
    } else {
        else_fn()
    }
}

// ============================================================
// Utility Combinators
// ============================================================

/// Apply a function twice to a value.
///
/// # Example
///
/// ```ignore
/// let double = |x| x * 2;
/// assert_eq!(twice(double, 5), 20); // 5 * 2 * 2
/// ```
#[inline]
pub fn twice<A, F>(f: F, x: A) -> A
where
    F: Fn(A) -> A,
{
    f(f(x))
}

/// Apply a function n times to a value.
///
/// # Example
///
/// ```ignore
/// let add_one = |x| x + 1;
/// assert_eq!(iterate_n(add_one, 0, 5), 5);
/// ```
#[inline]
pub fn iterate_n<A, F>(f: F, mut x: A, n: usize) -> A
where
    F: Fn(A) -> A,
{
    for _ in 0..n {
        x = f(x);
    }
    x
}

/// Tap combinator - apply a side-effect function and return the original value.
///
/// Useful for debugging or logging in a pipeline.
///
/// # Example
///
/// ```ignore
/// let result = tap(5, |x| println!("Value: {}", x));
/// assert_eq!(result, 5);
/// ```
#[inline]
pub fn tap<A, F>(x: A, f: F) -> A
where
    F: FnOnce(&A),
{
    f(&x);
    x
}

/// Convert an unary function to work with references.
#[inline]
pub fn by_ref<A, B, F>(f: F) -> impl Fn(&A) -> B
where
    F: Fn(A) -> B,
    A: Clone,
{
    move |a| f(a.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id() {
        assert_eq!(id(42), 42);
        assert_eq!(id("hello"), "hello");
    }

    #[test]
    fn test_constant() {
        let always_five = constant(5);
        assert_eq!(always_five(100), 5);
        assert_eq!(always_five(200), 5);
    }

    #[test]
    fn test_compose() {
        let add_one = |x| x + 1;
        let double = |x| x * 2;
        let composed = compose(add_one, double);
        assert_eq!(composed(5), 11); // (5 * 2) + 1
    }

    #[test]
    fn test_flip() {
        let sub = |a: i32, b: i32| a - b;
        let flipped = flip(sub);
        assert_eq!(flipped(3, 10), 7); // 10 - 3
    }

    #[test]
    fn test_apply() {
        assert_eq!(apply(|x| x * 2, 5), 10);
    }

    #[test]
    fn test_pipe() {
        let result = pipe(5, |x| x * 2);
        assert_eq!(result, 10);

        // Chain multiple transformations
        let result = pipe(5, |x| pipe(x * 2, |y| y + 1));
        assert_eq!(result, 11);
    }

    #[test]
    fn test_on() {
        // Compare numbers by their absolute value
        let compare_abs = on(|a: i32, b: i32| a - b, |x: i32| x.abs());
        assert_eq!(compare_abs(3, -5), -2); // |3| - |-5| = 3 - 5 = -2
        assert_eq!(compare_abs(-3, 5), -2); // |-3| - |5| = 3 - 5 = -2
    }

    #[test]
    fn test_curry_pair() {
        let add_pair = |(a, b): (i32, i32)| a + b;
        let curried = curry_pair(add_pair);
        assert_eq!(curried(5, 3), 8);
    }

    #[test]
    fn test_uncurry_pair() {
        let add = |a: i32, b: i32| a + b;
        let uncurried = uncurry_pair(add);
        assert_eq!(uncurried((5, 3)), 8);
    }

    #[test]
    fn test_fst_snd() {
        assert_eq!(fst((1, 2)), 1);
        assert_eq!(snd((1, 2)), 2);
    }

    #[test]
    fn test_swap() {
        assert_eq!(swap((1, 2)), (2, 1));
    }

    #[test]
    fn test_first_second() {
        assert_eq!(first(|x| x * 2, (5, 10)), (10, 10));
        assert_eq!(second(|x| x * 2, (5, 10)), (5, 20));
    }

    #[test]
    fn test_both() {
        assert_eq!(both(|x| x * 2, |y| y + 1, (5, 10)), (10, 11));
    }

    #[test]
    fn test_dup() {
        assert_eq!(dup(5), (5, 5));
    }

    #[test]
    fn test_bool_case() {
        assert_eq!(bool_case(true, 1, 2), 1);
        assert_eq!(bool_case(false, 1, 2), 2);
    }

    #[test]
    fn test_bool_case_lazy() {
        let mut then_called = false;
        let mut else_called = false;

        let result = bool_case_lazy(
            true,
            || {
                then_called = true;
                1
            },
            || {
                else_called = true;
                2
            },
        );

        assert_eq!(result, 1);
        assert!(then_called);
        assert!(!else_called);
    }

    #[test]
    fn test_twice() {
        let double = |x| x * 2;
        assert_eq!(twice(double, 5), 20); // 5 * 2 * 2
    }

    #[test]
    fn test_iterate_n() {
        let add_one = |x| x + 1;
        assert_eq!(iterate_n(add_one, 0, 5), 5);
        assert_eq!(iterate_n(add_one, 0, 0), 0);
    }

    #[test]
    fn test_tap() {
        let mut logged = false;
        let result = tap(5, |_x| {
            logged = true;
        });
        assert_eq!(result, 5);
        assert!(logged);
    }

    #[test]
    fn test_by_ref() {
        let double = |x: i32| x * 2;
        let double_ref = by_ref(double);
        assert_eq!(double_ref(&5), 10);
    }

    // Note: The fix combinator is difficult to test in Rust due to lifetime
    // constraints. It's provided for completeness with Haskell's Data.Function
    // but recursive functions in Rust are better written using regular recursion.

    #[test]
    fn test_uncurry() {
        let curried = |a: i32| move |b: i32| a + b;
        let uncurried = uncurry(curried);
        assert_eq!(uncurried(5, 3), 8);
    }
}
