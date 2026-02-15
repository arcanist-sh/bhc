# Optimization Pass Guidelines

**Rule ID:** BHC-RULE-013
**Applies to:** Core IR simplifier, pattern compilation, strictness analysis, dictionary specialization
**Inspired by:** HBC (Haskell B. Compiler) by Lennart Augustsson, via HCT translation

---

## Principles

1. **Correctness over speed** — Every transformation must preserve semantics
2. **Iterate to fixpoint** — Run the simplifier until no more changes occur
3. **Measure before optimizing** — Profile before adding passes
4. **Transparency** — Dump IR before/after each pass for debugging (`-ddump-core-after-*`)
5. **Profile-aware** — Default profile preserves laziness; Numeric profile is strict-by-default

---

## Core Simplifier

BHC MUST implement a Core IR simplifier. Without it, every compiled program carries
redundant bindings, unsimplified beta-redexes, and unexploited known-constructor
information. LLVM cannot compensate — it operates below the level of algebraic data
types, closures, and thunks.

### Required Transformations (Priority Order)

#### 1. Beta Reduction

Substitute the argument into the body when a lambda is immediately applied:

```haskell
-- Before
(\x -> x + 1) 42

-- After
42 + 1
```

Implementation: when encountering `App (Lam x body) arg`, substitute `arg` for `x`
in `body`. Guard against size explosion — only inline if `arg` is atomic (variable,
literal) or used exactly once.

#### 2. Case-of-Known-Constructor

When the scrutinee of a case expression is a known constructor application,
select the matching alternative directly:

```haskell
-- Before
case Just 42 of
  Nothing -> 0
  Just x  -> x + 1

-- After
42 + 1
```

This is the single most impactful optimization for functional code. It eliminates
allocation of the constructor and the entire case dispatch.

#### 3. Dead Binding Elimination

Remove let-bindings whose bound variable is never referenced:

```haskell
-- Before
let x = expensive_computation
in 42

-- After
42
```

Use reference counting (as in HBC's simplifier) or a simple free-variable check.

#### 4. Constant Folding

Evaluate arithmetic on known literals at compile time:

```haskell
-- Before
1 + 2

-- After
3
```

Applies to: integer arithmetic, boolean operators, string concatenation of
literals, comparison of literals.

#### 5. Case-of-Case

When one case expression is nested inside another's scrutinee, push the outer
case into each alternative of the inner case:

```haskell
-- Before
case (case x of { True -> A; False -> B }) of
  A -> e1
  B -> e2

-- After
case x of
  True  -> e1  -- case A of { A -> e1; B -> e2 } simplified
  False -> e2
```

This duplicates code but eliminates intermediate data. Use a size budget to avoid
explosion.

#### 6. Inlining

Replace a variable reference with its definition when profitable:

- **Always inline:** Wrappers, single-use bindings, trivial expressions (variables, literals)
- **Consider inlining:** Small functions (body size < threshold), functions marked `INLINE`
- **Never inline:** Recursive functions, large functions without `INLINE`

HBC uses reference counting to decide: if a binding is used exactly once, always
substitute. If used multiple times, only substitute if the body is small.

```haskell
-- Inline threshold: body has fewer than N nodes (suggested: 20)
shouldInline :: Binding -> Bool
shouldInline (Bind name expr usage) =
  usage == Once || (usage == Many && exprSize expr < inlineThreshold)
```

#### 7. Let-Floating (Float-Out / Float-In)

- **Float-out:** Move let-bindings outward to increase sharing
- **Float-in:** Move let-bindings inward to avoid computing unused branches

```haskell
-- Float-in: move binding into the branch that uses it
let x = expensive in
case b of
  True  -> x + 1
  False -> 0

-- After float-in:
case b of
  True  -> let x = expensive in x + 1
  False -> 0
```

### Simplifier Architecture

The simplifier should be structured as a single recursive traversal that applies
all transformations in one pass, then iterates until no changes occur:

```rust
fn simplify(expr: &Expr) -> Expr {
    let mut current = expr.clone();
    loop {
        let simplified = simplify_pass(&current);
        if simplified == current { break; }
        current = simplified;
    }
    current
}
```

**Pass ordering within one iteration:**
1. Inline small/single-use bindings
2. Beta-reduce
3. Case-of-known-constructor
4. Case-of-case (with size budget)
5. Constant fold
6. Dead binding elimination

**Iteration limit:** Cap at 10 iterations to prevent pathological cases.

### Key Files

```
crates/bhc-core/src/
├── simplify.rs          # Core simplifier (NEW)
├── simplify/
│   ├── beta.rs          # Beta reduction
│   ├── case.rs          # Case-of-known-constructor, case-of-case
│   ├── dead.rs          # Dead binding elimination
│   ├── fold.rs          # Constant folding
│   ├── inline.rs        # Inlining decisions
│   └── float.rs         # Let-floating
```

---

## Pattern Match Compilation

BHC MUST compile pattern matches to efficient decision trees with full
exhaustiveness and overlap checking. The current equation-by-equation approach
generates redundant checks and provides no warnings.

### Algorithm: Column-Based Match Compilation

Use the Augustsson/Sestoft algorithm (as implemented in HBC). The core idea:

1. **Select a column** — Choose the scrutinee column with the most information
   (constructors > variables)
2. **Group by constructor** — Partition equations by the head constructor of
   the selected column
3. **Generate case dispatch** — Emit a `case` on the selected scrutinee with
   one alternative per constructor group
4. **Recurse** — For each constructor group, compile the remaining columns
5. **Handle defaults** — Variable/wildcard patterns become the default branch

```
Input equations:
  f []     ys     = e1
  f (x:xs) []     = e2
  f (x:xs) (y:ys) = e3

Step 1: Select column 0 (has constructors [] and :)
Step 2: Group by constructor
  Group []:  { ([], ys) -> e1 }
  Group (:): { ((x:xs), []) -> e2, ((x:xs), (y:ys)) -> e3 }

Step 3: Generate case on arg0
  case arg0 of
    []    -> e1[ys=arg1]
    x:xs  -> case arg1 of          -- recurse on column 1
               []   -> e2
               y:ys -> e3
```

### Exhaustiveness Checking

After compiling match equations, check whether all constructors of each
scrutinee type are covered. If not, emit a warning:

```
Warning: Pattern match(es) are non-exhaustive
In an equation for 'f':
    Patterns not matched: [] (_:_)
```

Implementation: track which constructors appear in each case expression.
If the type has N constructors and fewer than N alternatives appear (with no
default), the match is non-exhaustive.

### Overlap Detection

Detect when a pattern is shadowed by an earlier one:

```
Warning: Pattern match is redundant
In an equation for 'f':
    f _ _ = ...  -- This equation is never reached
```

Implementation: when compiling equations, if an equation's pattern is a
strict subset of an earlier equation's pattern, flag it as redundant.

### Guard Compilation

Pattern guards should desugar to nested case expressions with fallthrough:

```haskell
-- Source
f x | x > 0    = "positive"
    | x == 0   = "zero"
    | otherwise = "negative"

-- Compiled
case x > 0 of
  True  -> "positive"
  False -> case x == 0 of
    True  -> "zero"
    False -> "negative"
```

### Key Files

```
crates/bhc-hir-to-core/src/
├── pattern.rs           # Pattern compilation (REFACTOR)
├── pattern/
│   ├── decision.rs      # Decision tree generation
│   ├── exhaustive.rs    # Exhaustiveness checker
│   └── overlap.rs       # Overlap/redundancy detection
```

---

## Strictness Analysis

For the Default profile (lazy evaluation), BHC SHOULD implement demand analysis
to identify strict arguments and avoid unnecessary thunk allocation.

### Boolean Tree Approach (HBC-Style)

Represent the demand on each argument as a boolean tree:

- `T` (top) — definitely demanded (strict)
- `F` (bottom) — not demanded (lazy)
- `And [d1, d2, ...]` — demanded if ALL sub-demands are demanded
- `Or [d1, d2, ...]` — demanded if ANY sub-demand is demanded

```
-- f x y = x + y
-- Demand on x: T (always evaluated by +)
-- Demand on y: T (always evaluated by +)

-- g x y = if x then y else 0
-- Demand on x: T (always evaluated by if)
-- Demand on y: Or[T, F] = depends on x
```

### Analysis Steps

1. **Compute abstract semantics** — Traverse each binding, computing demand
   signatures for its arguments
2. **Find fixpoint** — For recursive bindings, iterate until the demand
   signatures stabilize
3. **Annotate bindings** — Mark strict arguments with bang patterns
4. **Worker/wrapper** — Split functions into a strict "worker" (unboxed args)
   and a lazy "wrapper" (forces args then calls worker)

### Worker/Wrapper Transformation

When an argument is always demanded, split the function:

```haskell
-- Before (strict in both args after analysis)
f :: Int -> Int -> Int
f x y = x + y

-- After worker/wrapper
f :: Int -> Int -> Int
f x y = case x of I# x# -> case y of I# y# -> f_worker x# y#

f_worker :: Int# -> Int# -> Int
f_worker x# y# = I# (x# +# y#)
```

This avoids thunk allocation for strict arguments and enables unboxed
arithmetic.

### Profile Interaction

| Profile | Strictness Behavior |
|---------|-------------------|
| **Default** | Lazy by default; demand analysis identifies strict args |
| **Numeric** | Strict by default; analysis is a no-op (everything strict) |
| **Server** | Same as Default |
| **Edge** | Same as Default, but skip worker/wrapper to minimize code size |

### Key Files

```
crates/bhc-core/src/
├── demand.rs            # Demand analysis (NEW)
├── demand/
│   ├── tree.rs          # Boolean tree type and operations
│   ├── analyze.rs       # Abstract interpretation
│   └── annotate.rs      # Apply strictness annotations
├── worker_wrapper.rs    # Worker/wrapper transformation (NEW)
```

---

## Dictionary Specialization

When a typeclass-polymorphic function is called at a known concrete type,
BHC SHOULD specialize it to avoid dictionary passing overhead.

### When to Specialize

- **Always:** `SPECIALIZE` pragma present
- **Consider:** Function is called at a concrete type and body is small
- **Never:** Function is exported and no call sites are visible

### Direct Method Selection

When the dictionary argument to a method selection (`$sel_N dict`) is a
known dictionary constructor, reduce directly:

```haskell
-- Before
let dict = (method1, method2, method3)
in $sel_1 dict arg

-- After
method2 arg
```

This eliminates both the dictionary allocation and the selection overhead.

### Specialization Strategy

1. During simplification, track which concrete types flow into dictionary parameters
2. When a polymorphic function is called with a known instance, create a
   specialized copy with the dictionary inlined
3. Replace the call site with a call to the specialized version
4. Dead code elimination removes the now-unused generic version (if all
   call sites were specialized)

### Key Files

```
crates/bhc-core/src/
├── specialize.rs        # Dictionary specialization (NEW)
```

---

## Pass Pipeline

The full optimization pipeline for Core IR:

```
Core IR (from HIR-to-Core lowering)
    │
    ▼
┌──────────────────────┐
│  1. Simplifier       │  Beta, case-of-known, inline, dead code
│     (iterate to FP)  │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  2. Demand Analysis  │  Compute strictness signatures (Default only)
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  3. Worker/Wrapper   │  Split strict functions (Default only)
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  4. Specialization   │  Monomorphize dictionary-passing functions
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  5. Simplifier       │  Clean up after worker/wrapper and specialization
│     (second round)   │
└──────────────────────┘
    │
    ▼
  Codegen (LLVM)
```

### Dump Flags

```bash
bhc -ddump-core-initial          # Core before optimization
bhc -ddump-core-after-simpl      # After simplifier pass 1
bhc -ddump-core-after-demand     # After demand analysis
bhc -ddump-core-after-ww         # After worker/wrapper
bhc -ddump-core-after-spec       # After specialization
bhc -ddump-core-final            # Final Core before codegen
bhc -ddump-core-stats            # Simplifier statistics (rules fired, inlines, etc.)
```

---

## Testing Optimization Passes

### Correctness Tests

Every optimization pass MUST have tests proving semantic preservation:

```rust
#[test]
fn beta_reduction_preserves_semantics() {
    let before = parse_core("(\\x -> x + 1) 42");
    let after = simplify(before);
    assert_eq!(eval(before), eval(after));
}
```

### Optimization Effect Tests

Test that expected optimizations actually fire:

```rust
#[test]
fn case_of_known_constructor_fires() {
    let input = parse_core("case Just 42 of { Nothing -> 0; Just x -> x }");
    let output = simplify(input);
    // Should not contain any case expression
    assert!(!contains_case(&output));
    assert_eq!(eval(&output), Value::Int(42));
}

#[test]
fn exhaustiveness_warning_emitted() {
    let source = "f True = 1";  // Missing False case
    let warnings = compile_and_collect_warnings(source);
    assert!(warnings.iter().any(|w| w.contains("non-exhaustive")));
}
```

### Performance Regression Tests

```rust
#[test]
fn simplifier_reduces_code_size() {
    let input = compile_to_core(LARGE_PROGRAM);
    let output = simplify(input);
    assert!(core_size(&output) < core_size(&input));
}
```

---

## Historical Reference

The optimization strategies in this document draw from the Haskell B. Compiler
(HBC, 1982-1999) by Lennart Augustsson and Thomas Johnsson, as translated to
Haskell 98 by Anthony Travers (HCT, 2018). Key reference files:

| HCT File | BHC Equivalent | Technique |
|----------|----------------|-----------|
| `Simpl/Simpl.hs` | `bhc-core/src/simplify.rs` | Reference-counting simplifier |
| `Simpl/Casetr.hs` | `bhc-core/src/simplify/case.rs` | Case transformations |
| `Transform/Match.hs` | `bhc-hir-to-core/src/pattern.rs` | Augustsson pattern compilation |
| `Transform/Case.hs` | `bhc-hir-to-core/src/pattern/decision.rs` | Decision tree generation |
| `Strict/Calcstrict.hs` | `bhc-core/src/demand.rs` | Boolean-tree demand analysis |
| `Strict/Strict.hs` | `bhc-core/src/demand/tree.rs` | Boolean tree data type |
| `ExprE/Classtrans.hs` | `bhc-core/src/specialize.rs` | Dictionary method inlining |

The HCT source is available at:
`https://gitlab.haskell.org/-/project/1/uploads/2cc126c8b6a5b51fb18dd56fec129f2f/hct-2018-05-02-PRE-ALPHA.src.tar.gz`
(SHA256: `8a9bfbe9de10d3cbb7593716abb5ae1275a24d3fb5b2c0dec289cf8953500ea4`)
