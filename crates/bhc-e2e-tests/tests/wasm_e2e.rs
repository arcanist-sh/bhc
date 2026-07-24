//! WASM backend E2E tests.
//!
//! These tests compile Haskell programs to WebAssembly and run them
//! using wasmtime with WASI support.

use bhc_e2e_tests::{format_failure_report, Backend, E2ERunner, Profile};

/// Run a single fixture test for WASM backend.
fn run_wasm_test(fixture_name: &str, profile: Profile) {
    let runner = E2ERunner::wasm(profile).keep_artifacts();
    let result = runner
        .run_fixture(fixture_name)
        .expect("Failed to run test");

    if !result.is_pass() && !result.is_skipped() {
        let fixture_path = bhc_e2e_tests::fixtures_dir().join(fixture_name);
        let test_case = bhc_e2e_tests::E2ETestCase::from_fixture(&fixture_path).unwrap();
        let report = format_failure_report(&test_case, Backend::Wasm, profile, &result, None);
        panic!("{}", report);
    }

    assert!(
        result.is_pass() || result.is_skipped(),
        "Test failed: {:?}",
        result
    );
}

// =============================================================================
// Tier 1: Simple Tests
// =============================================================================

#[test]
fn test_tier1_hello_wasm() {
    run_wasm_test("tier1_simple/hello", Profile::Default);
}

#[test]
fn test_tier1_arithmetic_wasm() {
    run_wasm_test("tier1_simple/arithmetic", Profile::Default);
}

// Tier 1: let bindings

#[test]
fn test_tier1_let_binding_wasm() {
    run_wasm_test("tier1_simple/let_binding", Profile::Default);
}

// Tier 1: list range (fused sum/enumFromTo)

#[test]
fn test_tier1_list_range_wasm() {
    run_wasm_test("tier1_simple/list_range", Profile::Default);
}

// =============================================================================
// Tier 2: Function Tests
// =============================================================================

#[test]
fn test_tier2_fibonacci_wasm() {
    run_wasm_test("tier2_functions/fibonacci", Profile::Default);
}

#[test]
fn test_tier2_factorial_wasm() {
    run_wasm_test("tier2_functions/factorial", Profile::Default);
}

#[test]
fn test_tier2_guards_wasm() {
    run_wasm_test("tier2_functions/guards", Profile::Default);
}

#[test]
fn test_tier2_pattern_match_wasm() {
    run_wasm_test("tier2_functions/pattern_match", Profile::Default);
}

#[test]
fn test_tier2_where_bindings_wasm() {
    run_wasm_test("tier2_functions/where_bindings", Profile::Default);
}

#[test]
fn test_tier2_mutual_recursion_wasm() {
    run_wasm_test("tier2_functions/mutual_recursion", Profile::Default);
}

// =============================================================================
// Tier 3: IO Tests
// =============================================================================

#[test]
fn test_tier3_print_sequence_wasm() {
    run_wasm_test("tier3_io/print_sequence", Profile::Default);
}

#[test]
fn test_tier3_multi_bind_wasm() {
    run_wasm_test("tier3_io/multi_bind", Profile::Default);
}

#[test]
fn test_tier3_catch_test_wasm() {
    run_wasm_test("tier3_io/catch_test", Profile::Default);
}

// stdin input: `getLine` reads a line as a String (newline stripped) via the
// WASI `fd_read` runtime. The fixture supplies a `stdin.txt`.
#[test]
fn test_tier3_stdin_echo_wasm() {
    run_wasm_test("tier3_io/stdin_echo", Profile::Default);
}

// stdin input: `readLn` reads a line and parses it as an Int (handles a leading
// sign), composing with `getLine` and arithmetic.
#[test]
fn test_tier3_stdin_readln_wasm() {
    run_wasm_test("tier3_io/stdin_readln", Profile::Default);
}

// stdin input: `getContents` reads all of stdin to EOF as one String via the
// chunked `read_all` runtime.
#[test]
fn test_tier3_stdin_getcontents_wasm() {
    run_wasm_test("tier3_io/stdin_getcontents", Profile::Default);
}

// stdin input: `interact f` reads all of stdin, applies the (lambda) function,
// and writes the result to stdout.
#[test]
fn test_tier3_stdin_interact_wasm() {
    run_wasm_test("tier3_io/stdin_interact", Profile::Default);
}

// =============================================================================
// Tier 5: Benchmark Tests
// =============================================================================

#[test]
fn test_tier5_sum_list_wasm() {
    run_wasm_test("tier5_benchmark/sum_list", Profile::Default);
}

// =============================================================================
// Edge Profile Tests (minimal runtime)
// =============================================================================

#[test]
fn test_tier1_hello_wasm_edge() {
    run_wasm_test("tier1_simple/hello", Profile::Edge);
}

#[test]
fn test_tier1_arithmetic_wasm_edge() {
    run_wasm_test("tier1_simple/arithmetic", Profile::Edge);
}

#[test]
fn test_tier1_let_binding_wasm_edge() {
    run_wasm_test("tier1_simple/let_binding", Profile::Edge);
}

#[test]
fn test_tier2_factorial_wasm_edge() {
    run_wasm_test("tier2_functions/factorial", Profile::Edge);
}

// =============================================================================
// Numeric Profile Tests (SIMD, fusion)
// =============================================================================

#[test]
fn test_tier1_arithmetic_wasm_numeric() {
    run_wasm_test("tier1_simple/arithmetic", Profile::Numeric);
}

// =============================================================================
// Tier 2: Additional language coverage
// =============================================================================

#[test]
fn test_tier2_lambda_wasm() {
    run_wasm_test("tier2_functions/lambda", Profile::Default);
}

// User-defined ADTs: constructor tags come from module metadata, not just
// the well-known set. Regression test for nullary constructors all lowering
// to tag 0 (Red/Green/Blue → 1,1,1 instead of 1,2,3).
#[test]
fn test_tier2_custom_adt_wasm() {
    run_wasm_test("tier2_functions/custom_adt", Profile::Default);
}

// Dynamic strings: `putStrLn (if cond then "yes" else "no")`. The `if` yields a
// length-prefixed string pointer that carries its own length, so it prints
// directly. Also exercises `==`/`/=` on a derived-Eq ADT.
#[test]
fn test_tier2_derive_eq_wasm() {
    run_wasm_test("tier2_functions/derive_eq", Profile::Default);
}

// User-defined Monad: do-notation desugars to a dictionary-specialized
// `>>=` applied to a closure. The WASM backend has no first-class closures,
// so the call is inlined/beta-reduced at lowering time. Also exercises
// `show` on an Int.
#[test]
fn test_tier2_user_monad_wasm() {
    run_wasm_test("tier2_functions/user_monad", Profile::Default);
}

// =============================================================================
// First-class closures (call_indirect)
// =============================================================================

// Lambdas passed to higher-order functions. The worker/wrapper pass forces the
// function argument with `case f of x -> x arg`, so `f` must be a real closure
// value and the application must go through `call_indirect`.
#[test]
fn test_tier2_higher_order_wasm() {
    run_wasm_test("tier2_functions/higher_order", Profile::Default);
}

// Recursive higher-order function: the closure is a genuine runtime parameter
// threaded through recursion (cannot be inlined away), so this exercises
// `call_indirect` directly.
#[test]
fn test_tier2_closure_map_wasm() {
    run_wasm_test("tier2_functions/closure_map", Profile::Default);
}

// Closure capturing a free variable from its environment.
#[test]
fn test_tier2_closure_capture_wasm() {
    run_wasm_test("tier2_functions/closure_capture", Profile::Default);
}

// =============================================================================
// Partial application of top-level functions
// =============================================================================

// A top-level function applied to fewer arguments than its arity eta-expands
// to a closure that captures the supplied args. Covers partial application
// passed to a HOF, a multi-argument partial, and a partial stored in a CAF
// then over-applied.
#[test]
fn test_tier2_partial_app_wasm() {
    run_wasm_test("tier2_functions/partial_app", Profile::Default);
}

// =============================================================================
// Type-directed show / print
// =============================================================================

// `show`/`print` beyond Int: nullary constructors render as their name
// (`True`, `Red`, ...), and boolean-valued operators render as `True`/`False`
// from the runtime tag. Types are erased, so this is inferred structurally.
#[test]
fn test_tier2_show_types_wasm() {
    run_wasm_test("tier2_functions/show_types", Profile::Default);
}

// The BHC prelude `showInt`/`showBool`/`showChar` builtins: `showInt` renders an
// Int via the integer formatter, `showBool` from the runtime tag, and `showChar`
// wraps a one-char string in single quotes.
#[test]
fn test_tier3_show_types_wasm() {
    run_wasm_test("tier3_io/show_types", Profile::Default);
}

// =============================================================================
// Floating point (boxed f64)
// =============================================================================

// Double literals, runtime arithmetic (+,-,*,/), negation, comparison, and the
// double formatter (round to 6 decimals, strip trailing zeros). Doubles are
// represented as boxed f64 values.
#[test]
fn test_tier2_floats_wasm() {
    run_wasm_test("tier2_functions/floats", Profile::Default);
}

// =============================================================================
// Strings (length-prefixed) and concatenation
// =============================================================================

// `++` on strings: each String value is a length-prefixed block, so dynamic
// strings carry their own length and concatenate/print at runtime.
#[test]
fn test_tier2_string_concat_wasm() {
    run_wasm_test("tier2_functions/string_concat", Profile::Default);
}

// `show` produces a String value, so it composes with `++`
// (`"value = " ++ show n`) and works for Int, Double, and Bool.
#[test]
fn test_tier2_show_string_wasm() {
    run_wasm_test("tier2_functions/show_string", Profile::Default);
}

// Recursive show: statically-known lists, tuples, and constructors-with-fields
// expand into ++ chains, recursing on each element/field (with parens for
// nested compound values, e.g. `Just (Just 7)`).
#[test]
fn test_tier2_show_recursive_wasm() {
    run_wasm_test("tier2_functions/show_recursive", Profile::Default);
}

// A top-level function used as a value (passed to a recursive higher-order
// function) eta-expands to a closure, so `applyN n inc x` works with the
// bare function name `inc`.
#[test]
fn test_tier2_func_value_wasm() {
    run_wasm_test("tier2_functions/func_value", Profile::Default);
}

// =============================================================================
// List operations (injected prelude)
// =============================================================================

// map/filter/foldr/foldl/length are synthesized as Core when referenced and
// not user-defined. Operator sections like `(+)` become closures; runtime
// lists with a concrete element type print via a runtime walk.
#[test]
fn test_tier2_list_ops_wasm() {
    run_wasm_test("tier2_functions/list_ops", Profile::Default);
}

// `[a..b]` ranges build real cons lists via an injected enumFromTo, so they
// compose with map/filter/foldl/length; `sum [1..n]` still uses the fast
// enumFromTo fusion.
#[test]
fn test_tier2_ranges_wasm() {
    run_wasm_test("tier2_functions/ranges", Profile::Default);
}

// Floating-point math: sqrt, abs, and Double->Int conversions
// (truncate/floor/ceiling/round). Nested float arithmetic is detected
// structurally even when intermediate types are erased.
#[test]
fn test_tier2_float_math_wasm() {
    run_wasm_test("tier2_functions/float_math", Profile::Default);
}

// fromIntegral (Int -> Double) at float-consumption sites: the target type
// is erased, but under sqrt or mixed with a Double the conversion is known,
// so the Int is converted with F64ConvertI32S instead of being unboxed.
#[test]
fn test_tier2_from_integral_wasm() {
    run_wasm_test("tier2_functions/from_integral", Profile::Default);
}

// A broader batch of injected list-prelude functions: take, drop, replicate,
// null, head, tail, product, zipWith, zip, all, any, and, or.
#[test]
fn test_tier2_list_prelude_wasm() {
    run_wasm_test("tier2_functions/list_prelude", Profile::Default);
}

// takeWhile / dropWhile (predicate-driven list prefix/suffix).
#[test]
fn test_tier2_take_while_wasm() {
    run_wasm_test("tier2_functions/take_while", Profile::Default);
}

// Rational (Ratio Int) as a normalized [num|den] pair: `%` construction with
// gcd reduction, +/-/*// arithmetic, negate/abs/signum, ==, numerator/
// denominator, and `show` as "n % d".
#[test]
fn test_tier2_rational_wasm() {
    run_wasm_test("tier2_functions/rational", Profile::Default);
}

// =============================================================================
// Multimodule compilation
// =============================================================================

// Several modules merged into one WASM module: a cross-module function and a
// cross-module ADT (type + constructors defined in Helper, used in Main).
#[test]
fn test_tier2_multimodule_wasm() {
    run_wasm_test("tier2_functions/multimodule", Profile::Default);
}

// Same top-level name (`go`) defined in two modules: name qualification keeps
// them distinct so each module's references resolve to its own definition.
#[test]
fn test_tier2_multimodule_shadow_wasm() {
    run_wasm_test("tier2_functions/multimodule_shadow", Profile::Default);
}

// A transitive import chain (Main -> Middle -> Base): inlining a callee across
// modules alpha-renames its body, since VarIds are not unique across modules
// (otherwise the substitution aliases and inlining loops).
#[test]
fn test_tier2_multimodule_chain_wasm() {
    run_wasm_test("tier2_functions/multimodule_chain", Profile::Default);
}

// List `++` and `reverse` over cons cells, dispatched away from string concat
// by operand kind (literal list / concrete non-Char list type / list-returning
// function). String `++` is unaffected.
#[test]
fn test_tier2_list_append_wasm() {
    run_wasm_test("tier2_functions/list_append", Profile::Default);
}

// Grow-on-demand allocation: cumulative allocation (~585 KB, no GC) exceeds the
// Edge profile's 256 KB initial memory, so the allocator must grow linear
// memory (up to the 4 MB max) for the program to complete.
#[test]
fn test_tier2_heap_grow_wasm() {
    run_wasm_test("tier2_functions/heap_grow", Profile::Edge);
}

// Tail-call optimization: self-tail-recursive functions compile to a WASM
// loop, so they run in constant stack (100K iterations would overflow the
// call stack without TCO).
#[test]
fn test_tier2_tco_wasm() {
    run_wasm_test("tier2_functions/tco", Profile::Default);
}

// Dictionary passing: a recursive function polymorphic over a user class. The
// dictionary is passed at runtime (selected per instance via `$sel_N`) and
// threaded through recursion — it cannot be inlined/specialized away.
#[test]
fn test_tier3_dict_passing_wasm() {
    run_wasm_test("tier3_io/dict_passing", Profile::Default);
}

// A constrained function passed as a value to a higher-order function with a
// concrete parameter type: its dictionary is resolved from that expected type
// and applied as a partial application, then called via call_indirect.
#[test]
fn test_tier3_dict_higher_order_wasm() {
    run_wasm_test("tier3_io/dict_higher_order", Profile::Default);
}

// `print` of a user-defined ADT value renders via its derived Show.
#[test]
fn test_tier3_print_adt_wasm() {
    run_wasm_test("tier3_io/print_adt", Profile::Default);
}

// `show` of a user ADT inside a compound recurses into the element.
#[test]
fn test_tier3_show_adt_nested_wasm() {
    run_wasm_test("tier3_io/show_adt_nested", Profile::Default);
}

// Injected list/prelude functions: maximum/minimum, unzip, splitAt/span/break,
// findIndex/elemIndex/isPrefixOf, fst/snd, fromMaybe, even/odd, divMod/quotRem;
// plus rem/quot and floored div/mod primitives, and show of a runtime list.
#[test]
fn test_tier3_max_min_and_or_wasm() {
    run_wasm_test("tier3_io/max_min_and_or", Profile::Default);
}

#[test]
fn test_tier3_list_unzip_wasm() {
    run_wasm_test("tier3_io/list_unzip", Profile::Default);
}

#[test]
fn test_tier3_list_split_span_wasm() {
    run_wasm_test("tier3_io/list_split_span", Profile::Default);
}

#[test]
fn test_tier3_divmod_wasm() {
    run_wasm_test("tier3_io/divmod", Profile::Default);
}

// min/max/subtract, foldl1/foldr1, flip/const, and Char predicates.
#[test]
fn test_tier3_fold_misc_wasm() {
    run_wasm_test("tier3_io/fold_misc", Profile::Default);
}

#[test]
fn test_tier3_flip_test_wasm() {
    run_wasm_test("tier3_io/flip_test", Profile::Default);
}

#[test]
fn test_tier3_any_all_wasm() {
    run_wasm_test("tier3_io/any_all", Profile::Default);
}

// mapM_/mapM (IO over a list), intersect, and scans (scanl/scanl1/scanr).
#[test]
fn test_tier3_mapm_basic_wasm() {
    run_wasm_test("tier3_io/mapm_basic", Profile::Default);
}

#[test]
fn test_tier3_intersect_basic_wasm() {
    run_wasm_test("tier3_io/intersect_basic", Profile::Default);
}

#[test]
fn test_tier3_scanr_basic_wasm() {
    run_wasm_test("tier3_io/scanr_basic", Profile::Default);
}

// reverse, tails/inits, lookup, concat/concatMap, maybeToList, stripPrefix,
// isSuffixOf/isInfixOf, not, otherwise.
#[test]
fn test_tier3_elem_index_prefix_wasm() {
    run_wasm_test("tier3_io/elem_index_prefix", Profile::Default);
}

#[test]
fn test_tier3_tails_inits_wasm() {
    run_wasm_test("tier3_io/tails_inits", Profile::Default);
}

#[test]
fn test_tier3_pattern_guards_wasm() {
    run_wasm_test("tier3_io/pattern_guards", Profile::Default);
}

// Harder list fns: sortOn, nubBy, groupBy, deleteBy, unionBy, intersectBy,
// insert, mapAccumL/mapAccumR.
#[test]
fn test_tier3_list_by_ops_wasm() {
    run_wasm_test("tier3_io/list_by_ops", Profile::Default);
}

// Derived Ord: compare on enums + Ordering (LT/EQ/GT) constructors and show,
// plus maximumBy/minimumBy.
#[test]
fn test_tier3_derive_ord_wasm() {
    run_wasm_test("tier3_io/derive_ord", Profile::Default);
}

#[test]
fn test_tier3_ordering_basic_wasm() {
    run_wasm_test("tier3_io/ordering_basic", Profile::Default);
}

// Derived Enum/Bounded: fromEnum/toEnum/succ/pred as tag arithmetic,
// minBound/maxBound via the single-user-enum heuristic, and show of a computed
// enum value rendering the constructor name from its tag.
#[test]
fn test_tier3_derive_enum_wasm() {
    run_wasm_test("tier3_io/derive_enum", Profile::Default);
}

// Derived Functor: fmap routed to the generated $derived_fmap_<Type> for user
// ADTs (Box/Pair), plus the Maybe and list (map) arms.
#[test]
fn test_tier3_derive_functor_wasm() {
    run_wasm_test("tier3_io/derive_functor", Profile::Default);
}

// Derived Foldable: foldr routed to $derived_foldr_<Type> (Box/Pair/Maybe2),
// including (:) used as a first-class function value.
#[test]
fn test_tier3_derive_foldable_wasm() {
    run_wasm_test("tier3_io/derive_foldable", Profile::Default);
}

// Derived Read for an enum: `read "Green" :: Color` resolves to the matching
// constructor tag via the single-user-enum heuristic.
#[test]
fn test_tier3_derive_read_wasm() {
    run_wasm_test("tier3_io/derive_read", Profile::Default);
}

// Derived Traversable: traverse/mapM reduce to the derived Functor (fmap)/map
// in the eager IO model — the effect runs as each element is evaluated.
#[test]
fn test_tier3_derive_traversable_wasm() {
    run_wasm_test("tier3_io/derive_traversable", Profile::Default);
}

// Derived Generic + Control.DeepSeq: force/id are identities and seq/deepseq
// evaluate-then-return-second in the strict runtime.
#[test]
fn test_tier3_derive_generic_wasm() {
    run_wasm_test("tier3_io/derive_generic", Profile::Default);
}

// GHC.Generics from/to as runtime identities: `to (from x)` roundtrips, and
// `show` sees through the wrappers.
#[test]
fn test_tier3_generic_from_to_wasm() {
    run_wasm_test("tier3_io/generic_from_to", Profile::Default);
}

// Pattern matching on a derived-Generic enum's Rep: a bare `from d` builds the
// real outer sum (M1 identity wrapper over L1/R1), split at ceil(n/2) by tag.
#[test]
fn test_tier3_generic_pattern_wasm() {
    run_wasm_test("tier3_io/generic_pattern", Profile::Default);
}

// Prelude/combinator additions: zip3/zipWith3, unfoldr, gcd/lcm, when/unless.
#[test]
fn test_tier3_zip3_basic_wasm() {
    run_wasm_test("tier3_io/zip3_basic", Profile::Default);
}

#[test]
fn test_tier3_unfoldr_basic_wasm() {
    run_wasm_test("tier3_io/unfoldr_basic", Profile::Default);
}

#[test]
fn test_tier3_numeric_ops_wasm() {
    run_wasm_test("tier3_io/numeric_ops", Profile::Default);
}

#[test]
fn test_tier3_when_unless_wasm() {
    run_wasm_test("tier3_io/when_unless", Profile::Default);
}

// Monadic combinators in the eager-IO model: filterM/foldM/foldM_/zipWithM/
// zipWithM_ reduce to pure folds/maps; replicateM/replicateM_ unroll a
// statically-counted action so its effects repeat.
#[test]
fn test_tier3_monadic_combinators_wasm() {
    run_wasm_test("tier3_io/monadic_combinators", Profile::Default);
}

#[test]
fn test_tier3_zipwithm_basic_wasm() {
    run_wasm_test("tier3_io/zipwithm_basic", Profile::Default);
}

// Data.Either / Data.Maybe combinators, list-monad guard, runtime-list sum
// (OverloadedLists), and showDouble.
#[test]
fn test_tier3_data_either_wasm() {
    run_wasm_test("tier3_io/data_either", Profile::Default);
}

#[test]
fn test_tier3_data_maybe_wasm() {
    run_wasm_test("tier3_io/data_maybe", Profile::Default);
}

#[test]
fn test_tier3_guard_basic_wasm() {
    run_wasm_test("tier3_io/guard_basic", Profile::Default);
}

#[test]
fn test_tier3_overloaded_lists_wasm() {
    run_wasm_test("tier3_io/overloaded_lists", Profile::Default);
}

#[test]
fn test_tier3_show_double_wasm() {
    run_wasm_test("tier3_io/show_double", Profile::Default);
}

// IORef as a one-slot heap cell (new/read/write/modify).
#[test]
fn test_tier3_ioref_basic_wasm() {
    run_wasm_test("tier3_io/ioref_basic", Profile::Default);
}

// OverloadedStrings: fromString is the identity for String.
#[test]
fn test_tier3_overloaded_strings_wasm() {
    run_wasm_test("tier3_io/overloaded_strings", Profile::Default);
}

// ScopedTypeVariables: `show` of a `[a]`-typed result walks the list.
#[test]
fn test_tier3_scoped_tyvars_list_wasm() {
    run_wasm_test("tier3_io/scoped_tyvars_list", Profile::Default);
}

// GeneralizedNewtypeDeriving / DerivingStrategies: a newtype constructor is
// identity at runtime, so derived Num/Eq/Ord operate on the underlying value.
#[test]
fn test_tier3_gnd_basic_wasm() {
    run_wasm_test("tier3_io/gnd_basic", Profile::Default);
}

#[test]
fn test_tier3_gnd_newtype_erasure_wasm() {
    run_wasm_test("tier3_io/gnd_newtype_erasure", Profile::Default);
}

#[test]
fn test_tier3_deriving_strategies_wasm() {
    run_wasm_test("tier3_io/deriving_strategies", Profile::Default);
}

// Data.Map/Set/IntMap/IntSet over assoc-list / list representations.
#[test]
fn test_tier3_map_basic_wasm() {
    run_wasm_test("tier3_io/map_basic", Profile::Default);
}

#[test]
fn test_tier3_set_basic_wasm() {
    run_wasm_test("tier3_io/set_basic", Profile::Default);
}

#[test]
fn test_tier3_intmap_intset_wasm() {
    run_wasm_test("tier3_io/intmap_intset", Profile::Default);
}

#[test]
fn test_tier3_map_maybe_wasm() {
    run_wasm_test("tier3_io/map_maybe", Profile::Default);
}

#[test]
fn test_tier3_map_complete_wasm() {
    run_wasm_test("tier3_io/map_complete", Profile::Default);
}

// Data.Sequence over a list representation.
#[test]
fn test_tier3_sequence_basic_wasm() {
    run_wasm_test("tier3_io/sequence_basic", Profile::Default);
}

// bracket (happy path): acquire, use, release run in order via `const`.
#[test]
fn test_tier3_bracket_io_wasm() {
    run_wasm_test("tier3_io/bracket_io", Profile::Default);
}

// Exception model (eager IO): `throwIO`/`error`/`readFile` set a pending-exception
// flag, effectful ops no-op while it is set, and `catch`/`finally`/`onException`
// run the handler / cleanup. `readFile` always raises (no filesystem under WASI).
#[test]
fn test_tier3_catch_file_error_wasm() {
    run_wasm_test("tier3_io/catch_file_error", Profile::Default);
}

#[test]
fn test_tier3_exception_hierarchy_wasm() {
    run_wasm_test("tier3_io/exception_hierarchy", Profile::Default);
}

#[test]
fn test_tier3_exception_test_wasm() {
    run_wasm_test("tier3_io/exception_test", Profile::Default);
}

// Lazy bottom let-bindings (`let x = error …`): the strict backend must not
// evaluate the RHS at the binding site; forcing is deferred to (conditional) use.
#[test]
fn test_tier3_lazy_let_wasm() {
    run_wasm_test("tier3_io/lazy_let", Profile::Default);
}

#[test]
fn test_tier3_lazy_let_basic_wasm() {
    run_wasm_test("tier3_io/lazy_let_basic", Profile::Default);
}

#[test]
fn test_tier3_lazy_let_conditional_wasm() {
    run_wasm_test("tier3_io/lazy_let_conditional", Profile::Default);
}

// A user-defined monad with its own bind/return.
#[test]
fn test_tier3_user_monad_wasm() {
    run_wasm_test("tier3_io/user_monad", Profile::Default);
}

// Monad transformers, rewritten to eager representations: ReaderT (\r->a),
// StateT (\s->(a,s)), WriterT ((a,w)), ExceptT (Either e a).
#[test]
fn test_tier3_reader_t_wasm() {
    run_wasm_test("tier3_io/reader_t", Profile::Default);
}

#[test]
fn test_tier3_state_t_wasm() {
    run_wasm_test("tier3_io/state_t", Profile::Default);
}

#[test]
fn test_tier3_state_t_case_wasm() {
    run_wasm_test("tier3_io/state_t_case", Profile::Default);
}

#[test]
fn test_tier3_state_t_string_wasm() {
    run_wasm_test("tier3_io/state_t_string", Profile::Default);
}

#[test]
fn test_tier3_writer_t_wasm() {
    run_wasm_test("tier3_io/writer_t", Profile::Default);
}

#[test]
fn test_tier3_except_t_wasm() {
    run_wasm_test("tier3_io/except_t", Profile::Default);
}

#[test]
fn test_tier3_monad_error_wasm() {
    run_wasm_test("tier3_io/monad_error", Profile::Default);
}

// Two-layer transformer stacks (data outer over context inner), rewritten via
// the inner monad's bind/return: ExceptT/WriterT over StateT/ReaderT.
#[test]
fn test_tier3_cross_except_state_wasm() {
    run_wasm_test("tier3_io/cross_except_state", Profile::Default);
}

#[test]
fn test_tier3_cross_writer_reader_wasm() {
    run_wasm_test("tier3_io/cross_writer_reader", Profile::Default);
}

#[test]
fn test_tier3_cross_writer_state_wasm() {
    run_wasm_test("tier3_io/cross_writer_state", Profile::Default);
}

// Two-context Reader+State stacks (mtl-auto-lifted); outer layer inferred from
// the program's eliminator nesting.
#[test]
fn test_tier3_cross_state_reader_wasm() {
    run_wasm_test("tier3_io/cross_state_reader", Profile::Default);
}

#[test]
fn test_tier3_cross_reader_state_wasm() {
    run_wasm_test("tier3_io/cross_reader_state", Profile::Default);
}

#[test]
fn test_tier3_cross_except_reader_wasm() {
    run_wasm_test("tier3_io/cross_except_reader", Profile::Default);
}

// String-representation discriminator: putStrLn/length handle both a marked
// length-prefixed pstr and a runtime cons-[Char].
#[test]
fn test_tier3_bytestring_basic_wasm() {
    run_wasm_test("tier3_io/bytestring_basic", Profile::Default);
}

#[test]
fn test_tier3_char_ranges_wasm() {
    run_wasm_test("tier3_io/char_ranges", Profile::Default);
}

// Data.Text (+ .Lazy/.Encoding) over the pstr representation; Data.ByteString
// .Lazy over the list representation.
#[test]
fn test_tier3_text_basic_wasm() {
    run_wasm_test("tier3_io/text_basic", Profile::Default);
}

#[test]
fn test_tier3_text_encoding_wasm() {
    run_wasm_test("tier3_io/text_encoding", Profile::Default);
}

// Data.Text.IO write/read/append over the in-memory file table (no real
// filesystem under WASI): writeFile then readFile round-trips, appendFile
// concatenates.
#[test]
fn test_tier3_text_io_wasm() {
    run_wasm_test("tier3_io/text_io", Profile::Default);
}

#[test]
fn test_tier3_lazy_text_basic_wasm() {
    run_wasm_test("tier3_io/lazy_text_basic", Profile::Default);
}

#[test]
fn test_tier3_lazy_bytestring_basic_wasm() {
    run_wasm_test("tier3_io/lazy_bytestring_basic", Profile::Default);
}

// ByteString.Builder rendered to a pstr (intDec/stringUtf8/singleton/append).
#[test]
fn test_tier3_builder_basic_wasm() {
    run_wasm_test("tier3_io/builder_basic", Profile::Default);
}

// System.FilePath operations on the pstr path (split on / and .).
#[test]
fn test_tier3_filepath_basic_wasm() {
    run_wasm_test("tier3_io/filepath_basic", Profile::Default);
}

// Tuple combinators (swap/curry/uncurry) and Data.Function (&), plus
// succ/pred as first-class function values.
#[test]
fn test_tier3_tuple_functions_wasm() {
    run_wasm_test("tier3_io/tuple_functions", Profile::Default);
}

#[test]
fn test_tier3_data_function_wasm() {
    run_wasm_test("tier3_io/data_function", Profile::Default);
}

// Char predicates/conversions as first-class fns over String args: filter/map/
// any/all convert a pstr to a cons-[Char]; ord/chr/isAscii/isLetter/digitToInt.
#[test]
fn test_tier3_char_first_class_wasm() {
    run_wasm_test("tier3_io/char_first_class", Profile::Default);
}

// Stepped char enums: enumFromThenTo (ascending and descending) + take of
// infinite enumFromThen/enumFrom, printed as Strings.
#[test]
fn test_tier3_char_enum_step_wasm() {
    run_wasm_test("tier3_io/char_enum_step", Profile::Default);
}

// foreign import ccall math: sqrt (f64 instruction), sin/cos (Taylor series).
#[test]
fn test_tier3_foreign_import_wasm() {
    run_wasm_test("tier3_io/foreign_import", Profile::Default);
}

// take k of an infinite producer (iterate/repeat/cycle/enumFrom/enumFromThen)
// fuses to a finite list; enumFromThenTo and until synthesize directly.
#[test]
fn test_tier3_take_iterate_wasm() {
    run_wasm_test("tier3_io/take_iterate", Profile::Default);
}

#[test]
fn test_tier3_enum_functions_wasm() {
    run_wasm_test("tier3_io/enum_functions", Profile::Default);
}

// read / readMaybe of Int (parse_int), and runtime-Maybe show (Just n/Nothing).
#[test]
fn test_tier3_string_read_wasm() {
    run_wasm_test("tier3_io/string_read", Profile::Default);
}

// Arbitrary-precision Integer: `show` of a closed integer constant that
// overflows the i32 runtime (e.g. `factorial 50`) is evaluated exactly at
// compile time and emitted as a decimal string.
#[test]
fn test_tier3_integer_large_wasm() {
    run_wasm_test("tier3_io/integer_large", Profile::Default);
}

// A larger JSON-ish key/value parser (recursive descent over Strings). Exercises
// the pstr->cons normalization at cons/nil case scrutinees (so user functions
// can pattern-match string literals as [Char]) and short-circuiting &&/||.
#[test]
fn test_tier3_milestone_e_json_wasm() {
    run_wasm_test("tier3_io/milestone_e_json", Profile::Default);
}

// A StateT-based CSV parser. Exercises transitive monad-kind inference (the
// do-blocks of sub-parsers thread State without naming get/put themselves) and
// `++` over cons-[Char] strings built char-by-char.
#[test]
fn test_tier3_milestone_d_csv_parser_wasm() {
    run_wasm_test("tier3_io/milestone_d_csv_parser", Profile::Default);
}

// A 4-module markdown->HTML renderer. Exercises cross-module constructor tags
// (Parser builds Block values, Render matches them) and the guard that keeps a
// top-level function whose VarId collides with a local from being applied as a
// closure.
#[test]
fn test_tier3_milestone_c_markdown_wasm() {
    run_wasm_test("tier3_io/milestone_c_markdown", Profile::Default);
}

// data family / data instance: the instance constructors are harvested from
// their patterns (the frontend does not surface them into Core), and an
// accessor's erased Bool result is recovered from its binding signature so
// `show` renders True/False.
#[test]
fn test_tier3_data_family_basic_wasm() {
    run_wasm_test("tier3_io/data_family_basic", Profile::Default);
}

// Char predicates with showBool / showChar (one-char quoted string).
#[test]
fn test_tier3_char_predicates_wasm() {
    run_wasm_test("tier3_io/char_predicates", Profile::Default);
}

// `print` of a String shows it quoted and escaped via the show_string runtime.
#[test]
fn test_tier3_print_string_wasm() {
    run_wasm_test("tier3_io/print_string", Profile::Default);
}

// A constrained function passed to a polymorphic higher-order function, where
// the instantiation is pinned by sibling value arguments. The dictionary is
// resolved from the whole call's argument types.
#[test]
fn test_tier3_dict_sibling_inferred_wasm() {
    run_wasm_test("tier3_io/dict_sibling_inferred", Profile::Default);
}

// =============================================================================
// Tier 4: Fusion (Numeric Profile)
// =============================================================================

// `sum (map dbl xs)` with a named `dbl :: Int -> Int` fuses to a strict
// `foldl'` (typed Core IR, spec/BHC-BRIEF-0002) and stays correct on wasm — the
// fusion is a shared Core->Core pass, so the wasm backend must codegen the
// fused form correctly too.
#[test]
fn test_tier4_sum_map_named_wasm() {
    run_wasm_test("tier4_fusion/sum_map_named", Profile::Numeric);
}
