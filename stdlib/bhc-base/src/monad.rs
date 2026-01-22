//! Extended monad operations
//!
//! Additional monad operations corresponding to Haskell's Control.Monad.
//!
//! Since Rust doesn't have higher-kinded types, we provide these operations
//! as standalone functions for common monad-like types (Option, Result, Vec).
//!
//! # Example
//!
//! ```ignore
//! use bhc_base::monad::*;
//!
//! // Using guard with Option
//! let result = option_guard(5 > 3).and_then(|_| Some(42));
//! assert_eq!(result, Some(42));
//!
//! // Using when
//! let mut called = false;
//! option_when(true, || { called = true; Some(()) });
//! assert!(called);
//! ```


// ============================================================
// Guard and Conditional Operations
// ============================================================

/// Guard for Option - returns Some(()) if condition is true, None otherwise.
///
/// This is the `guard` operation from MonadPlus.
///
/// # Example
///
/// ```ignore
/// let result = option_guard(x > 0).and_then(|_| Some(x * 2));
/// ```
#[inline]
pub fn option_guard(condition: bool) -> Option<()> {
    if condition {
        Some(())
    } else {
        None
    }
}

/// Guard for Result - returns Ok(()) if condition is true, Err otherwise.
#[inline]
pub fn result_guard<E>(condition: bool, error: E) -> Result<(), E> {
    if condition {
        Ok(())
    } else {
        Err(error)
    }
}

/// Conditional execution for Option - executes action only if condition is true.
///
/// This is the `when` operation in Haskell.
#[inline]
pub fn option_when<A, F>(condition: bool, action: F) -> Option<()>
where
    F: FnOnce() -> Option<A>,
{
    if condition {
        action().map(|_| ())
    } else {
        Some(())
    }
}

/// Conditional execution for Result - executes action only if condition is true.
#[inline]
pub fn result_when<A, E, F>(condition: bool, action: F) -> Result<(), E>
where
    F: FnOnce() -> Result<A, E>,
{
    if condition {
        action().map(|_| ())
    } else {
        Ok(())
    }
}

/// Negated conditional - executes action only if condition is false.
///
/// This is the `unless` operation in Haskell.
#[inline]
pub fn option_unless<A, F>(condition: bool, action: F) -> Option<()>
where
    F: FnOnce() -> Option<A>,
{
    option_when(!condition, action)
}

/// Negated conditional for Result.
#[inline]
pub fn result_unless<A, E, F>(condition: bool, action: F) -> Result<(), E>
where
    F: FnOnce() -> Result<A, E>,
{
    result_when(!condition, action)
}

// ============================================================
// Join Operations (flatten nested monads)
// ============================================================

/// Flatten nested Options.
///
/// Equivalent to Haskell's `join`.
///
/// # Example
///
/// ```ignore
/// assert_eq!(option_join(Some(Some(42))), Some(42));
/// assert_eq!(option_join(Some(None::<i32>)), None);
/// assert_eq!(option_join(None::<Option<i32>>), None);
/// ```
#[inline]
pub fn option_join<A>(mma: Option<Option<A>>) -> Option<A> {
    mma.flatten()
}

/// Flatten nested Results (when both error types are the same).
#[inline]
pub fn result_join<A, E>(mma: Result<Result<A, E>, E>) -> Result<A, E> {
    mma.and_then(|x| x)
}

/// Flatten nested Vecs.
#[inline]
pub fn vec_join<A>(mma: Vec<Vec<A>>) -> Vec<A> {
    mma.into_iter().flatten().collect()
}

// ============================================================
// Filter Operations
// ============================================================

/// Filter with monadic predicate for Option.
///
/// This is `mfilter` in Haskell.
#[inline]
pub fn option_mfilter<A, F>(predicate: F, ma: Option<A>) -> Option<A>
where
    F: FnOnce(&A) -> bool,
{
    ma.filter(predicate)
}

/// Monadic filter - filter a list with a monadic predicate.
///
/// This is `filterM` in Haskell.
///
/// # Example
///
/// ```ignore
/// // Keep only positive numbers, but could fail
/// let items = vec![1, -2, 3, -4, 5];
/// let result = option_filter_m(|&x| if x > -10 { Some(x > 0) } else { None }, items);
/// assert_eq!(result, Some(vec![1, 3, 5]));
/// ```
pub fn option_filter_m<A, F>(predicate: F, items: Vec<A>) -> Option<Vec<A>>
where
    F: Fn(&A) -> Option<bool>,
{
    let mut result = Vec::new();
    for item in items {
        match predicate(&item) {
            Some(true) => result.push(item),
            Some(false) => {}
            None => return None,
        }
    }
    Some(result)
}

/// Monadic filter for Result.
pub fn result_filter_m<A, E, F>(predicate: F, items: Vec<A>) -> Result<Vec<A>, E>
where
    F: Fn(&A) -> Result<bool, E>,
{
    let mut result = Vec::new();
    for item in items {
        match predicate(&item) {
            Ok(true) => result.push(item),
            Ok(false) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(result)
}

// ============================================================
// Map and Sequence Operations
// ============================================================

/// Map a function over a list and collect Option results.
///
/// This is `mapM` for Option.
pub fn option_map_m<A, B, F>(f: F, items: Vec<A>) -> Option<Vec<B>>
where
    F: Fn(A) -> Option<B>,
{
    items.into_iter().map(f).collect()
}

/// Map a function over a list and collect Result results.
///
/// This is `mapM` for Result.
pub fn result_map_m<A, B, E, F>(f: F, items: Vec<A>) -> Result<Vec<B>, E>
where
    F: Fn(A) -> Result<B, E>,
{
    items.into_iter().map(f).collect()
}

/// Map a function for effect only (discarding results).
///
/// This is `mapM_` in Haskell.
pub fn option_map_m_<A, B, F>(f: F, items: Vec<A>) -> Option<()>
where
    F: Fn(A) -> Option<B>,
{
    for item in items {
        f(item)?;
    }
    Some(())
}

/// Map a function for effect only for Result.
pub fn result_map_m_<A, B, E, F>(f: F, items: Vec<A>) -> Result<(), E>
where
    F: Fn(A) -> Result<B, E>,
{
    for item in items {
        f(item)?;
    }
    Ok(())
}

/// Flipped mapM - `forM` in Haskell.
#[inline]
pub fn option_for_m<A, B, F>(items: Vec<A>, f: F) -> Option<Vec<B>>
where
    F: Fn(A) -> Option<B>,
{
    option_map_m(f, items)
}

/// Flipped mapM for Result.
#[inline]
pub fn result_for_m<A, B, E, F>(items: Vec<A>, f: F) -> Result<Vec<B>, E>
where
    F: Fn(A) -> Result<B, E>,
{
    result_map_m(f, items)
}

/// Flipped mapM_ - `forM_` in Haskell.
#[inline]
pub fn option_for_m_<A, B, F>(items: Vec<A>, f: F) -> Option<()>
where
    F: Fn(A) -> Option<B>,
{
    option_map_m_(f, items)
}

/// Flipped mapM_ for Result.
#[inline]
pub fn result_for_m_<A, B, E, F>(items: Vec<A>, f: F) -> Result<(), E>
where
    F: Fn(A) -> Result<B, E>,
{
    result_map_m_(f, items)
}

/// Sequence a list of Options into an Option of list.
///
/// This is `sequence` in Haskell.
///
/// # Example
///
/// ```ignore
/// assert_eq!(option_sequence(vec![Some(1), Some(2), Some(3)]), Some(vec![1, 2, 3]));
/// assert_eq!(option_sequence(vec![Some(1), None, Some(3)]), None);
/// ```
pub fn option_sequence<A>(items: Vec<Option<A>>) -> Option<Vec<A>> {
    items.into_iter().collect()
}

/// Sequence a list of Results into a Result of list.
pub fn result_sequence<A, E>(items: Vec<Result<A, E>>) -> Result<Vec<A>, E> {
    items.into_iter().collect()
}

/// Sequence for effect only.
///
/// This is `sequence_` in Haskell.
pub fn option_sequence_<A>(items: Vec<Option<A>>) -> Option<()> {
    for item in items {
        item?;
    }
    Some(())
}

/// Sequence for effect only for Result.
pub fn result_sequence_<A, E>(items: Vec<Result<A, E>>) -> Result<(), E> {
    for item in items {
        item?;
    }
    Ok(())
}

// ============================================================
// Fold Operations
// ============================================================

/// Monadic fold (left fold with a monadic operator).
///
/// This is `foldM` in Haskell.
///
/// # Example
///
/// ```ignore
/// let safe_div = |acc, x| if x == 0 { None } else { Some(acc / x) };
/// assert_eq!(option_fold_m(safe_div, 100, vec![2, 5]), Some(10));
/// assert_eq!(option_fold_m(safe_div, 100, vec![2, 0, 5]), None);
/// ```
pub fn option_fold_m<A, B, F>(f: F, init: A, items: Vec<B>) -> Option<A>
where
    F: Fn(A, B) -> Option<A>,
{
    let mut acc = init;
    for item in items {
        acc = f(acc, item)?;
    }
    Some(acc)
}

/// Monadic fold for Result.
pub fn result_fold_m<A, B, E, F>(f: F, init: A, items: Vec<B>) -> Result<A, E>
where
    F: Fn(A, B) -> Result<A, E>,
{
    let mut acc = init;
    for item in items {
        acc = f(acc, item)?;
    }
    Ok(acc)
}

/// Monadic fold for effect only.
///
/// This is `foldM_` in Haskell.
pub fn option_fold_m_<A, B, F>(f: F, init: A, items: Vec<B>) -> Option<()>
where
    F: Fn(A, B) -> Option<A>,
{
    option_fold_m(f, init, items).map(|_| ())
}

/// Monadic fold for effect only for Result.
pub fn result_fold_m_<A, B, E, F>(f: F, init: A, items: Vec<B>) -> Result<(), E>
where
    F: Fn(A, B) -> Result<A, E>,
{
    result_fold_m(f, init, items).map(|_| ())
}

// ============================================================
// Replicate Operations
// ============================================================

/// Repeat a monadic action n times and collect results.
///
/// This is `replicateM` in Haskell.
///
/// # Example
///
/// ```ignore
/// let mut counter = 0;
/// let action = || { counter += 1; Some(counter) };
/// let result = option_replicate_m(3, action);
/// assert_eq!(result, Some(vec![1, 2, 3]));
/// ```
pub fn option_replicate_m<A, F>(n: usize, mut f: F) -> Option<Vec<A>>
where
    F: FnMut() -> Option<A>,
{
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
        result.push(f()?);
    }
    Some(result)
}

/// Repeat a monadic action n times for Result.
pub fn result_replicate_m<A, E, F>(n: usize, mut f: F) -> Result<Vec<A>, E>
where
    F: FnMut() -> Result<A, E>,
{
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
        result.push(f()?);
    }
    Ok(result)
}

/// Repeat a monadic action n times for effect only.
///
/// This is `replicateM_` in Haskell.
pub fn option_replicate_m_<A, F>(n: usize, mut f: F) -> Option<()>
where
    F: FnMut() -> Option<A>,
{
    for _ in 0..n {
        f()?;
    }
    Some(())
}

/// Repeat a monadic action n times for effect only for Result.
pub fn result_replicate_m_<A, E, F>(n: usize, mut f: F) -> Result<(), E>
where
    F: FnMut() -> Result<A, E>,
{
    for _ in 0..n {
        f()?;
    }
    Ok(())
}

// ============================================================
// ZipWith Operations
// ============================================================

/// Zip two lists with a monadic function.
///
/// This is `zipWithM` in Haskell.
pub fn option_zip_with_m<A, B, C, F>(f: F, xs: Vec<A>, ys: Vec<B>) -> Option<Vec<C>>
where
    F: Fn(A, B) -> Option<C>,
{
    xs.into_iter()
        .zip(ys.into_iter())
        .map(|(x, y)| f(x, y))
        .collect()
}

/// Zip two lists with a monadic function for Result.
pub fn result_zip_with_m<A, B, C, E, F>(f: F, xs: Vec<A>, ys: Vec<B>) -> Result<Vec<C>, E>
where
    F: Fn(A, B) -> Result<C, E>,
{
    xs.into_iter()
        .zip(ys.into_iter())
        .map(|(x, y)| f(x, y))
        .collect()
}

/// Zip two lists with a monadic function for effect only.
///
/// This is `zipWithM_` in Haskell.
pub fn option_zip_with_m_<A, B, C, F>(f: F, xs: Vec<A>, ys: Vec<B>) -> Option<()>
where
    F: Fn(A, B) -> Option<C>,
{
    for (x, y) in xs.into_iter().zip(ys.into_iter()) {
        f(x, y)?;
    }
    Some(())
}

/// Zip two lists with a monadic function for effect only for Result.
pub fn result_zip_with_m_<A, B, C, E, F>(f: F, xs: Vec<A>, ys: Vec<B>) -> Result<(), E>
where
    F: Fn(A, B) -> Result<C, E>,
{
    for (x, y) in xs.into_iter().zip(ys.into_iter()) {
        f(x, y)?;
    }
    Ok(())
}

// ============================================================
// Map and Unzip Operations
// ============================================================

/// Map a function over a list and unzip the results.
///
/// This is `mapAndUnzipM` in Haskell.
pub fn option_map_and_unzip_m<A, B, C, F>(f: F, items: Vec<A>) -> Option<(Vec<B>, Vec<C>)>
where
    F: Fn(A) -> Option<(B, C)>,
{
    let mut bs = Vec::with_capacity(items.len());
    let mut cs = Vec::with_capacity(items.len());
    for item in items {
        let (b, c) = f(item)?;
        bs.push(b);
        cs.push(c);
    }
    Some((bs, cs))
}

/// Map a function over a list and unzip the results for Result.
pub fn result_map_and_unzip_m<A, B, C, E, F>(f: F, items: Vec<A>) -> Result<(Vec<B>, Vec<C>), E>
where
    F: Fn(A) -> Result<(B, C), E>,
{
    let mut bs = Vec::with_capacity(items.len());
    let mut cs = Vec::with_capacity(items.len());
    for item in items {
        let (b, c) = f(item)?;
        bs.push(b);
        cs.push(c);
    }
    Ok((bs, cs))
}

// ============================================================
// Lift Operations
// ============================================================

/// Lift a unary function into Option.
///
/// This is `liftM` in Haskell (same as fmap/map).
#[inline]
pub fn option_lift_m<A, B, F>(f: F, ma: Option<A>) -> Option<B>
where
    F: FnOnce(A) -> B,
{
    ma.map(f)
}

/// Lift a binary function into Option.
///
/// This is `liftM2` in Haskell.
#[inline]
pub fn option_lift_m2<A, B, C, F>(f: F, ma: Option<A>, mb: Option<B>) -> Option<C>
where
    F: FnOnce(A, B) -> C,
{
    match (ma, mb) {
        (Some(a), Some(b)) => Some(f(a, b)),
        _ => None,
    }
}

/// Lift a ternary function into Option.
///
/// This is `liftM3` in Haskell.
#[inline]
pub fn option_lift_m3<A, B, C, D, F>(
    f: F,
    ma: Option<A>,
    mb: Option<B>,
    mc: Option<C>,
) -> Option<D>
where
    F: FnOnce(A, B, C) -> D,
{
    match (ma, mb, mc) {
        (Some(a), Some(b), Some(c)) => Some(f(a, b, c)),
        _ => None,
    }
}

/// Lift a 4-ary function into Option.
///
/// This is `liftM4` in Haskell.
#[inline]
pub fn option_lift_m4<A, B, C, D, E, F>(
    f: F,
    ma: Option<A>,
    mb: Option<B>,
    mc: Option<C>,
    md: Option<D>,
) -> Option<E>
where
    F: FnOnce(A, B, C, D) -> E,
{
    match (ma, mb, mc, md) {
        (Some(a), Some(b), Some(c), Some(d)) => Some(f(a, b, c, d)),
        _ => None,
    }
}

/// Lift a 5-ary function into Option.
///
/// This is `liftM5` in Haskell.
#[inline]
pub fn option_lift_m5<A, B, C, D, E, G, F>(
    f: F,
    ma: Option<A>,
    mb: Option<B>,
    mc: Option<C>,
    md: Option<D>,
    me: Option<E>,
) -> Option<G>
where
    F: FnOnce(A, B, C, D, E) -> G,
{
    match (ma, mb, mc, md, me) {
        (Some(a), Some(b), Some(c), Some(d), Some(e)) => Some(f(a, b, c, d, e)),
        _ => None,
    }
}

/// Lift a unary function into Result.
#[inline]
pub fn result_lift_m<A, B, E, F>(f: F, ma: Result<A, E>) -> Result<B, E>
where
    F: FnOnce(A) -> B,
{
    ma.map(f)
}

/// Lift a binary function into Result.
#[inline]
pub fn result_lift_m2<A, B, C, E, F>(f: F, ma: Result<A, E>, mb: Result<B, E>) -> Result<C, E>
where
    F: FnOnce(A, B) -> C,
{
    match (ma, mb) {
        (Ok(a), Ok(b)) => Ok(f(a, b)),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

/// Lift a ternary function into Result.
#[inline]
pub fn result_lift_m3<A, B, C, D, E, F>(
    f: F,
    ma: Result<A, E>,
    mb: Result<B, E>,
    mc: Result<C, E>,
) -> Result<D, E>
where
    F: FnOnce(A, B, C) -> D,
{
    match (ma, mb, mc) {
        (Ok(a), Ok(b), Ok(c)) => Ok(f(a, b, c)),
        (Err(e), _, _) => Err(e),
        (_, Err(e), _) => Err(e),
        (_, _, Err(e)) => Err(e),
    }
}

// ============================================================
// Applicative-style Operations
// ============================================================

/// Apply a wrapped function to a wrapped value.
///
/// This is `ap` in Haskell (<*> from Applicative).
#[inline]
pub fn option_ap<A, B, F>(mf: Option<F>, ma: Option<A>) -> Option<B>
where
    F: FnOnce(A) -> B,
{
    match (mf, ma) {
        (Some(f), Some(a)) => Some(f(a)),
        _ => None,
    }
}

/// Apply a wrapped function to a wrapped value for Result.
#[inline]
pub fn result_ap<A, B, E, F>(mf: Result<F, E>, ma: Result<A, E>) -> Result<B, E>
where
    F: FnOnce(A) -> B,
{
    match (mf, ma) {
        (Ok(f), Ok(a)) => Ok(f(a)),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

// ============================================================
// Kleisli Composition
// ============================================================

/// Kleisli composition (left to right).
///
/// This is `>=>` (fish operator) in Haskell.
///
/// Composes two monadic functions: `(a -> m b) >=> (b -> m c)` gives `(a -> m c)`
#[inline]
pub fn option_kleisli<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> Option<C>
where
    F: Fn(A) -> Option<B>,
    G: Fn(B) -> Option<C>,
{
    move |a| f(a).and_then(&g)
}

/// Kleisli composition for Result.
#[inline]
pub fn result_kleisli<A, B, C, E, F, G>(f: F, g: G) -> impl Fn(A) -> Result<C, E>
where
    F: Fn(A) -> Result<B, E>,
    G: Fn(B) -> Result<C, E>,
{
    move |a| f(a).and_then(&g)
}

/// Reverse Kleisli composition (right to left).
///
/// This is `<=<` in Haskell.
#[inline]
pub fn option_kleisli_rev<A, B, C, F, G>(g: G, f: F) -> impl Fn(A) -> Option<C>
where
    F: Fn(A) -> Option<B>,
    G: Fn(B) -> Option<C>,
{
    option_kleisli(f, g)
}

/// Reverse Kleisli composition for Result.
#[inline]
pub fn result_kleisli_rev<A, B, C, E, F, G>(g: G, f: F) -> impl Fn(A) -> Result<C, E>
where
    F: Fn(A) -> Result<B, E>,
    G: Fn(B) -> Result<C, E>,
{
    result_kleisli(f, g)
}

// ============================================================
// MonadPlus Operations
// ============================================================

/// Sum of alternatives for Option (msum).
///
/// Returns the first Some value, or None if all are None.
pub fn option_msum<A>(items: Vec<Option<A>>) -> Option<A> {
    for item in items {
        if item.is_some() {
            return item;
        }
    }
    None
}

/// Sum of alternatives for Result (first Ok or last Err).
pub fn result_msum<A, E>(items: Vec<Result<A, E>>) -> Option<Result<A, E>> {
    let mut last_err = None;
    for item in items {
        match item {
            Ok(a) => return Some(Ok(a)),
            Err(e) => last_err = Some(Err(e)),
        }
    }
    last_err
}

/// Alternative choice for Option.
///
/// This is `mplus` in Haskell.
#[inline]
pub fn option_mplus<A>(ma: Option<A>, mb: Option<A>) -> Option<A> {
    ma.or(mb)
}

/// Alternative choice for Result (tries first, then second on error).
#[inline]
pub fn result_mplus<A, E>(ma: Result<A, E>, mb: Result<A, E>) -> Result<A, E> {
    ma.or(mb)
}

// ============================================================
// Void and Forever
// ============================================================

/// Discard the result, keeping only the effect.
///
/// This is `void` in Haskell.
#[inline]
pub fn option_void<A>(ma: Option<A>) -> Option<()> {
    ma.map(|_| ())
}

/// Discard the result for Result.
#[inline]
pub fn result_void<A, E>(ma: Result<A, E>) -> Result<(), E> {
    ma.map(|_| ())
}

/// Strict version of map for Option.
///
/// This is `<$!>` in Haskell (forces evaluation).
/// In Rust, this is the same as map since Rust is strict.
#[inline]
pub fn option_strict_map<A, B, F>(f: F, ma: Option<A>) -> Option<B>
where
    F: FnOnce(A) -> B,
{
    ma.map(f)
}

// ============================================================
// Utility Functions
// ============================================================

/// Concatenate a list of lists inside a monad.
///
/// This combines `concat` with `sequence`.
pub fn option_concat_map_m<A, B, F>(f: F, items: Vec<A>) -> Option<Vec<B>>
where
    F: Fn(A) -> Option<Vec<B>>,
{
    let nested = option_map_m(f, items)?;
    Some(nested.into_iter().flatten().collect())
}

/// Concatenate a list of lists inside a Result monad.
pub fn result_concat_map_m<A, B, E, F>(f: F, items: Vec<A>) -> Result<Vec<B>, E>
where
    F: Fn(A) -> Result<Vec<B>, E>,
{
    let nested = result_map_m(f, items)?;
    Ok(nested.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Guard tests
    #[test]
    fn test_option_guard_true() {
        assert_eq!(option_guard(true), Some(()));
    }

    #[test]
    fn test_option_guard_false() {
        assert_eq!(option_guard(false), None);
    }

    #[test]
    fn test_result_guard() {
        assert_eq!(result_guard(true, "error"), Ok(()));
        assert_eq!(result_guard(false, "error"), Err("error"));
    }

    // When/Unless tests
    #[test]
    fn test_option_when() {
        let mut called = false;
        let result = option_when(true, || {
            called = true;
            Some(42)
        });
        assert!(called);
        assert_eq!(result, Some(()));

        called = false;
        let result = option_when(false, || {
            called = true;
            Some(42)
        });
        assert!(!called);
        assert_eq!(result, Some(()));
    }

    #[test]
    fn test_option_unless() {
        let mut called = false;
        option_unless(false, || {
            called = true;
            Some(())
        });
        assert!(called);

        called = false;
        option_unless(true, || {
            called = true;
            Some(())
        });
        assert!(!called);
    }

    // Join tests
    #[test]
    fn test_option_join() {
        assert_eq!(option_join(Some(Some(42))), Some(42));
        assert_eq!(option_join(Some(None::<i32>)), None);
        assert_eq!(option_join(None::<Option<i32>>), None);
    }

    #[test]
    fn test_vec_join() {
        assert_eq!(vec_join(vec![vec![1, 2], vec![3, 4]]), vec![1, 2, 3, 4]);
        assert_eq!(vec_join(vec![vec![], vec![1]]), vec![1]);
    }

    // Filter tests
    #[test]
    fn test_option_mfilter() {
        assert_eq!(option_mfilter(|x| *x > 0, Some(5)), Some(5));
        assert_eq!(option_mfilter(|x| *x > 0, Some(-5)), None);
        assert_eq!(option_mfilter(|x: &i32| *x > 0, None), None);
    }

    #[test]
    fn test_option_filter_m() {
        let result = option_filter_m(|x| Some(*x > 0), vec![1, -2, 3, -4, 5]);
        assert_eq!(result, Some(vec![1, 3, 5]));

        let result = option_filter_m(|_| None::<bool>, vec![1, 2, 3]);
        assert_eq!(result, None);
    }

    // Map tests
    #[test]
    fn test_option_map_m() {
        let result = option_map_m(|x| Some(x * 2), vec![1, 2, 3]);
        assert_eq!(result, Some(vec![2, 4, 6]));

        let result = option_map_m(|x| if x > 2 { None } else { Some(x) }, vec![1, 2, 3]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_result_map_m() {
        let result: Result<Vec<i32>, &str> = result_map_m(|x| Ok(x * 2), vec![1, 2, 3]);
        assert_eq!(result, Ok(vec![2, 4, 6]));
    }

    // Sequence tests
    #[test]
    fn test_option_sequence() {
        assert_eq!(
            option_sequence(vec![Some(1), Some(2), Some(3)]),
            Some(vec![1, 2, 3])
        );
        assert_eq!(option_sequence(vec![Some(1), None, Some(3)]), None);
    }

    // Fold tests
    #[test]
    fn test_option_fold_m() {
        let safe_div = |acc: i32, x: i32| if x == 0 { None } else { Some(acc / x) };
        assert_eq!(option_fold_m(safe_div, 100, vec![2, 5]), Some(10));
        assert_eq!(option_fold_m(safe_div, 100, vec![2, 0, 5]), None);
    }

    // Replicate tests
    #[test]
    fn test_option_replicate_m() {
        let mut counter = 0;
        let result = option_replicate_m(3, || {
            counter += 1;
            Some(counter)
        });
        assert_eq!(result, Some(vec![1, 2, 3]));

        let mut counter = 0;
        let result = option_replicate_m(5, || {
            counter += 1;
            if counter > 3 {
                None
            } else {
                Some(counter)
            }
        });
        assert_eq!(result, None);
    }

    // ZipWith tests
    #[test]
    fn test_option_zip_with_m() {
        let result = option_zip_with_m(|x, y| Some(x + y), vec![1, 2, 3], vec![10, 20, 30]);
        assert_eq!(result, Some(vec![11, 22, 33]));

        let result = option_zip_with_m(
            |x, y| if y == 0 { None } else { Some(x / y) },
            vec![10, 20, 30],
            vec![2, 0, 5],
        );
        assert_eq!(result, None);
    }

    // Map and Unzip tests
    #[test]
    fn test_option_map_and_unzip_m() {
        let result = option_map_and_unzip_m(|x| Some((x, x * 2)), vec![1, 2, 3]);
        assert_eq!(result, Some((vec![1, 2, 3], vec![2, 4, 6])));
    }

    // Lift tests
    #[test]
    fn test_option_lift_m() {
        assert_eq!(option_lift_m(|x| x * 2, Some(5)), Some(10));
        assert_eq!(option_lift_m(|x: i32| x * 2, None), None);
    }

    #[test]
    fn test_option_lift_m2() {
        assert_eq!(option_lift_m2(|x, y| x + y, Some(1), Some(2)), Some(3));
        assert_eq!(option_lift_m2(|x: i32, y: i32| x + y, Some(1), None), None);
        assert_eq!(option_lift_m2(|x: i32, y: i32| x + y, None, Some(2)), None);
    }

    #[test]
    fn test_option_lift_m3() {
        assert_eq!(
            option_lift_m3(|x, y, z| x + y + z, Some(1), Some(2), Some(3)),
            Some(6)
        );
        assert_eq!(
            option_lift_m3(|x: i32, y: i32, z: i32| x + y + z, Some(1), None, Some(3)),
            None
        );
    }

    // Applicative tests
    #[test]
    fn test_option_ap() {
        let f: Option<fn(i32) -> i32> = Some(|x| x * 2);
        assert_eq!(option_ap(f, Some(5)), Some(10));
        assert_eq!(option_ap(None::<fn(i32) -> i32>, Some(5)), None);
    }

    // Kleisli tests
    #[test]
    fn test_option_kleisli() {
        let f = |x: i32| if x > 0 { Some(x * 2) } else { None };
        let g = |x: i32| if x < 100 { Some(x + 1) } else { None };
        let composed = option_kleisli(f, g);

        assert_eq!(composed(5), Some(11)); // 5 -> 10 -> 11
        assert_eq!(composed(-5), None); // fails at f
        assert_eq!(composed(50), None); // 50 -> 100, fails at g
    }

    // MonadPlus tests
    #[test]
    fn test_option_msum() {
        assert_eq!(option_msum(vec![None, Some(1), Some(2)]), Some(1));
        assert_eq!(option_msum(vec![None, None, None]), None::<i32>);
    }

    #[test]
    fn test_option_mplus() {
        assert_eq!(option_mplus(Some(1), Some(2)), Some(1));
        assert_eq!(option_mplus(None, Some(2)), Some(2));
        assert_eq!(option_mplus(Some(1), None), Some(1));
        assert_eq!(option_mplus(None::<i32>, None), None);
    }

    // Void test
    #[test]
    fn test_option_void() {
        assert_eq!(option_void(Some(42)), Some(()));
        assert_eq!(option_void(None::<i32>), None);
    }

    // ConcatMapM test
    #[test]
    fn test_option_concat_map_m() {
        let result = option_concat_map_m(|x| Some(vec![x, x * 2]), vec![1, 2, 3]);
        assert_eq!(result, Some(vec![1, 2, 2, 4, 3, 6]));
    }
}
