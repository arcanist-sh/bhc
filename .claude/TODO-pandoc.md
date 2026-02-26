# Road to Pandoc: BHC Compilation TODO

**Document ID:** BHC-TODO-PANDOC
**Status:** In Progress
**Created:** 2026-01-30
**Updated:** 2026-02-26

---

## Goal

Compile and run [Pandoc](https://github.com/jgm/pandoc), a ~60k LOC Haskell
document converter with ~80 transitive package dependencies. This serves as the
north-star integration target for BHC's real-world Haskell compatibility.

---

## Current State

BHC compiles real Haskell programs to native executables via LLVM:
- **185 native E2E tests** passing (including monad transformers, file IO, markdown parser, JSON parser, GADTs, type extensions, lazy Text/ByteString, GHC.Generics from/to roundtrip, foreign import ccall, deriving Read)
- All intermediate milestones A–E.70 done, plus full GHC.Generics and Data.Sequence
- **Separate compilation pipeline**: `-c` mode, `.bhi` interface generation and consumption, `--odir`/`--hidir`/`--numeric-version`/`--package-db` flags
- **hx package manager integration** wired — hx has .cabal parsing, Hackage fetch, dependency solver, BHC backend crate with correct CLI flags, filesystem-based package DB, and BHC builtin package mapping

### Standard Library & IO (E.5–E.31)
- Monad transformers: StateT, ReaderT, ExceptT, WriterT all working
- Nested transformer stacks: all cross-transformer combinations working (E.55–E.57)
- MTL typeclasses: MonadReader, MonadState, MonadError, MonadWriter
- Exception handling: catch, bracket, finally, onException (E.5)
- Multi-package support with import paths (E.6)
- Data.Text: packed UTF-8 with 25+ operations (E.7)
- Data.ByteString: 24 RTS functions, Data.Text.Encoding bridge (E.8)
- Data.Char predicates + Char Enum ranges, first-class predicates (E.9, E.36, E.37)
- Data.Text.IO: native Text file/handle I/O (E.10)
- Show for compound/nested types via recursive ShowTypeDesc (E.11, E.31)
- Numeric ops: even/odd, gcd/lcm, divMod/quotRem, fromIntegral + IORef (E.12)
- Data.Maybe, Data.Either, Control.Monad combinators (E.13, E.14, E.18)
- Extensive Data.List: 70+ operations (E.15, E.16, E.26)
- Ordering ADT (LT/EQ/GT), compare returning Ordering (E.17)
- System.FilePath + System.Directory (E.19)
- Data.Map/Set/IntMap/IntSet: full operation sets (E.21, E.22, E.29)
- Stock deriving: Eq, Show, Ord for user-defined ADTs (E.23, E.24)
- Arithmetic, Enum, Folds, Higher-order, IO Input builtins (E.25–E.30)

### Language Extensions (E.32–E.64)
- OverloadedStrings + IsString typeclass (E.32)
- Record syntax: named fields, accessors, construction, update, RecordWildCards, NamedFieldPuns (E.33)
- ViewPatterns codegen with fallthrough (E.34)
- TupleSections + MultiWayIf (E.35)
- Manual typeclass instances with Show dispatch (E.38)
- User-defined typeclasses: dictionary-passing, higher-kinded, default methods, superclasses (E.39–E.41)
- DeriveAnyClass for user-defined typeclasses (E.42)
- Word types (Word8/16/32/64), Integer arbitrary precision, lazy let-bindings (E.43–E.45)
- ScopedTypeVariables (E.46)
- GeneralizedNewtypeDeriving with newtype erasure (E.47)
- FlexibleInstances, FlexibleContexts, instance context propagation (E.48)
- MultiParamTypeClasses (E.49)
- FunctionalDependencies (E.50)
- DeriveFunctor, DeriveFoldable, DeriveTraversable (E.51–E.53)
- DeriveEnum + DeriveBounded (E.54)
- Cross-transformer codegen: ReaderT/StateT, ExceptT/StateT+ReaderT, WriterT/StateT+ReaderT (E.55–E.57)
- Full lazy let-bindings for Haskell semantics (E.58)
- EmptyDataDecls + strict field annotations (E.59)
- GADTs with type refinement (E.60)
- TypeOperators for infix type syntax (E.61)
- StandaloneDeriving + PatternSynonyms + nested pattern fallthrough (E.62)
- Full GHC.Generics: Rep type algebra (V1/U1/K1/M1/:+:/:*:), from/to, pattern matching + NFData/DeepSeq stubs (E.63)
- EmptyCase, StrictData, DefaultSignatures, OverloadedLists (E.64)

### Gap to Pandoc

**Completed (previously missing, now done):**
1. ~~OverloadedStrings + IsString~~ — Done (E.32)
2. ~~Record syntax~~ — Done (E.33): named fields, accessors, construction, update, RecordWildCards
3. ~~ViewPatterns codegen~~ — Done (E.34)
4. ~~TupleSections + MultiWayIf~~ — Done (E.35)
5. ~~GeneralizedNewtypeDeriving~~ — Done (E.47): newtype erasure lifting instances
6. ~~GHC.Generics~~ — Done: full Rep type algebra (V1/U1/K1/M1/:+:/:*:) + working from/to + pattern matching
7. ~~CPP preprocessing~~ — Done (E.67)
8. ~~Lazy Text/ByteString~~ — Done
9. ~~Exception hierarchy~~ — Done
10. ~~Data.Sequence~~ — Done: Vec-backed RTS with full codegen pipeline
11. ~~`foreign import ccall`~~ — Done: parsing (safe/unsafe/interruptible), HIR-to-Core lowering with proper types, LLVM codegen with C function declaration + BHC wrapper (unbox args → call C → box result), E2E test with sin/cos/sqrt

**Still missing — Compiler completeness (before libraries):**
1. **General monadic `>>=`/`>>`/`return` via dictionary dispatch** — Currently hardcoded to IO/StateT/ReaderT/ExceptT/WriterT. Any user-defined monad or library monad (parsec, conduit) breaks.
3. ~~**Type Applications (`f @Int`)**~~ ✅ Full pipeline: parser → AST → HIR → typeck (forall instantiation) → Core → codegen (TyApp erasure). E2E test passing.
4. ~~**`DerivingStrategies`/`DerivingVia`**~~ ✅ Full pipeline: parser matches keyword tokens (stock/newtype/anyclass/via) → DerivingClause AST → HIR → strategy-aware dispatch. E2E test passing.
5. **`strict`/`lazy`/etc. reserved as keywords** — Lexer treats valid Haskell identifiers as keywords. Breaks any code using these as variable names.
6. ~~**Record field access type checking**~~ ✅ FieldAccess resolves accessor type; RecordUpdate verifies field existence and type compatibility via constructor scheme instantiation.
7. ~~**`import Foo (pattern X)` syntax**~~ ✅ Pattern synonym imports and exports now parsed and lowered.
8. ~~**`deriving Read`**~~ ✅ Core IR deriving infrastructure + inline LLVM codegen for read dispatch. E2E test: show/read roundtrip for nullary ADT constructors.
9. **`mask`/`uninterruptibleMask`** — RTS stubs execute action without masking. Correctness issue for async exception safety.
10. **`Rational` type** — Faked as `(Int, Int)` tuple, wrong semantics.
11. **Qualified record construction** — `Module.Con { field = val }` not parsed.
12. **`.hs-boot` mutual module recursion** — Not supported.
13. **Extensions silently ignored** — Many LANGUAGE pragmas accepted but have no semantic effect.

**Still missing — Libraries (after compiler is complete):**
1. **Full package system** — End-to-end testing with real Hackage packages, conditional deps
2. **Template Haskell** — Required for aeson JSON deriving (alternative: GHC.Generics now works)
3. **parsec/megaparsec** — Pandoc depends on parsec for some formats
4. **aeson** — JSON serialization with ToJSON/FromJSON (now unblocked by full Generics)
5. **process/time/network-uri** — External dependency packages

---

## Tier 0 — Compiler Completeness

These are compiler-level gaps that must be resolved before libraries can be compiled
from Hackage source. No amount of stdlib work can compensate for these — they affect
the compiler's ability to process valid Haskell code.

### 0.1 `foreign import ccall` / `foreign export ccall`

**Status:** ✅ `foreign import` complete (parsing, lowering, codegen, E2E test)
**Scope:** Medium
**Impact:** Blocker — nearly every real-world Haskell package uses C FFI

- [x] Parse `foreign import ccall [safe|unsafe|interruptible] "c_name" hs_name :: Type`
- [ ] Parse `foreign export ccall "c_name" hs_name :: Type`
- [x] Lower foreign imports to Core IR with proper types (ForeignImport struct in CoreModule)
- [x] Codegen: emit LLVM `declare` for the C function
- [x] Codegen: generate BHC wrapper that unboxes args → calls C → boxes result
- [x] Handle `safe` vs `unsafe` calling convention (parsed and stored, both generate same LLVM for now)
- [x] Support Double (f64), Float (f32), Int (i64), Ptr (opaque pointer) marshalling
- [x] E2E test: `sin`, `cos`, `sqrt` with correct Double show formatting
- [ ] Support `CInt`, `CString`, `FunPtr` marshalling
- [ ] `foreign export` codegen

**Key files:**
- `crates/bhc-parser/src/decl.rs` — `parse_foreign_decl_with_doc()`
- `crates/bhc-hir-to-core/src/context.rs` — foreign import lowering with pre-registered vars
- `crates/bhc-codegen/src/llvm/lower.rs` — `declare_foreign_imports()`, wrapper generation
- `crates/bhc-core/src/lib.rs` — `ForeignImport` struct, `foreign_imports` field on `CoreModule`

### 0.2 General Monadic `>>=`/`>>`/`return`/`pure` via Dictionary Dispatch

**Status:** ❌ Hardcoded to builtin monads only
**Scope:** Medium-Large
**Impact:** Blocker — do-notation only works for IO/StateT/ReaderT/ExceptT/WriterT

`do`-notation desugars to `>>=`/`>>` calls correctly, but codegen dispatches only to
a fixed set of builtin monads. Any user-defined monad, or library monad like `ParsecT`,
`Conduit`, `Free`, etc., will fail at codegen time.

- [ ] Route `>>=`/`>>`/`return`/`pure` through dictionary-passing infrastructure
- [ ] When the monad type is known and has a dictionary, select the method from the dict
- [ ] When the monad type is a builtin, fall through to existing hardcoded fast paths
- [ ] Handle `Applicative`'s `<*>` via dictionary dispatch (needed for `ApplicativeDo`)
- [ ] E2E test: user-defined monad with `do`-notation (e.g., simple `Identity` or `Writer`)
- [ ] E2E test: `Monad` instance for a custom newtype

**Key files:**
- `crates/bhc-lower/src/desugar.rs` — do-notation desugaring (correct)
- `crates/bhc-codegen/src/llvm/lower.rs` — `>>=`/`>>` dispatch (hardcoded, ~line 4095)
- `crates/bhc-hir-to-core/src/` — dictionary construction for Monad instances

### 0.3 Type Applications (`f @Int x`) ✅ COMPLETE

**Status:** 🟢 Full pipeline implemented
**Scope:** Small
**Impact:** High — very common in modern Haskell

Implemented: parser produces `TypeApp` AST nodes, lowering passes them to HIR,
type checker instantiates forall binders with explicit type arguments (handles
nested type apps like `f @Int @Bool`), codegen erases `TyApp` at runtime.
Also fixed codegen builtin/primop/constructor detection to skip through `TyApp`.

- [x] `TypeApp` variant in AST `Expr` enum
- [x] Parse `@Type` in application expressions
- [x] Lower `TypeApp` from AST to HIR
- [x] Type inference: instantiate forall binders with explicit type args
- [x] Codegen: skip `TyApp` in `is_saturated_builtin`, `is_saturated_primop`,
  `is_saturated_constructor`, and `lower_application` arg collection
- [x] E2E test: `id @Int 42`, `const @Int @String 10 "hello"`

**Key files:**
- `crates/bhc-ast/src/lib.rs` — `TypeApp` variant
- `crates/bhc-parser/src/expr.rs` — parse `@Type` in `parse_app_expr`
- `crates/bhc-lower/src/lower.rs` — lower `TypeApp` to HIR
- `crates/bhc-typeck/src/infer.rs` — forall instantiation with type args
- `crates/bhc-codegen/src/llvm/lower.rs` — TyApp peeling in saturation checks

### 0.4 `DerivingStrategies` / `DerivingVia` ✅ COMPLETE

**Status:** ✅ Full pipeline implemented
**Scope:** Medium
**Impact:** High — ubiquitous in modern Hackage code

- [x] `DerivingStrategy` enum (Stock/Newtype/Anyclass/Via) + `DerivingClause` struct in AST and HIR
- [x] Parser fixed: matches keyword tokens (`TokenKind::Stock`/`Newtype`/`Anyclass`/`Via`) instead of `eat_ident`
- [x] AST → HIR lowering propagates strategy (including `Via(Type)` lowering)
- [x] Strategy-aware dispatch: `stock` → stock deriving, `newtype` → GND (empty instance), `anyclass` → DeriveAnyClass, `via` → empty instance
- [x] Type checker updated for `DerivingClause`
- [x] E2E test: `deriving stock (Show, Eq)` on data, `deriving newtype (Show, Eq)` on newtype

**Note:** True coerce-based `DerivingVia` (generating methods via `coerce` through
the via-type) is deferred. Current `Via` strategy uses empty instance with defaults,
which covers the common case (NFData, ToJSON via Generics). Stock Show deriving on
newtypes has a pre-existing codegen issue (newtype erasure vs pattern match) — use
`deriving newtype (Show)` for newtypes instead.

**Key files:**
- `crates/bhc-ast/src/lib.rs` — `DerivingStrategy`, `DerivingClause`
- `crates/bhc-parser/src/decl.rs` — `parse_single_deriving()`
- `crates/bhc-hir/src/lib.rs` — HIR `DerivingStrategy`, `DerivingClause`
- `crates/bhc-lower/src/lower.rs` — `lower_deriving_strategy()`
- `crates/bhc-hir-to-core/src/context.rs` — strategy-aware dispatch
- `crates/bhc-hir-to-core/src/deriving.rs` — `derive_empty_instance` (pub)

### 0.5 Context-Sensitive Keywords (Lexer Fix) ✅

**Status:** ✅ Complete
**Scope:** Small
**Impact:** Medium-high — breaks any code using these as identifiers

All 10 context-sensitive words now lex as normal `Ident` tokens. Parser call sites
updated to use `check_ident_str`/`expect_ident_str`/`eat_ident_str` for context-aware
matching. `lazy` only triggers H26 lazy-expression parsing when followed by `{`.

- [x] Remove `strict`, `lazy`, `linear`, `tensor` from keyword list
- [x] Make `family`, `role`, `stock`, `anyclass`, `via` context-sensitive
- [x] Keep `pattern` context-sensitive (only keyword at top-level before ConId)
- [x] E2E test: `context_keywords` — all 8 words used as variable names

**Key files:**
- `crates/bhc-lexer/src/token.rs` — 10 `TokenKind` variants removed
- `crates/bhc-parser/src/lib.rs` — `check_ident_str`, `expect_ident_str` added
- `crates/bhc-parser/src/decl.rs` — 9 call sites updated (family×4, pattern×2, stock, anyclass, via)
- `crates/bhc-parser/src/expr.rs` — 3 call sites updated (lazy)

### 0.6 Record Field Type Checking ✅

**Status:** 🟢 Complete
**Scope:** Medium
**Impact:** Medium-high — incorrect types inferred for field access

- [x] Look up field type from the record's data declaration during type inference
- [x] Verify field exists on the record type in record updates
- [x] Verify new value type matches field type in record updates
- [x] Handle polymorphic record fields
- [x] E2E test: record_field_types (Person + Box a with updates)

**Implementation:**
- Added `field_name_to_con: FxHashMap<Symbol, Vec<(DefId, DefId)>>` to `TyCtxt` — maps field name to (constructor DefId, accessor DefId) for FieldAccess resolution
- Added `type_to_data_cons: FxHashMap<Symbol, Vec<DefId>>` to `TyCtxt` — maps type constructor name to its data constructor DefIds for RecordUpdate lookup
- `FieldAccess`: instantiates accessor function type, unifies with record type to extract field type
- `RecordUpdate`: resolves record type constructor, finds constructors, instantiates constructor scheme, unifies field values against expected types
- Maps populated in `register_data_type`, `register_newtype`, and early imported-constructor registration

**Key files:**
- `crates/bhc-typeck/src/context.rs` — new maps + population during registration
- `crates/bhc-typeck/src/infer.rs` — `FieldAccess` (~line 331), `RecordUpdate` (~line 341), `extract_type_con_name` helper

### 0.7 `import Foo (pattern X)` Syntax ✅

**Status:** 🟢 Complete
**Scope:** Small
**Impact:** Medium — needed for importing pattern synonyms

- [x] Parse `pattern` keyword prefix in import item lists
- [x] Parse `pattern` keyword prefix in export item lists
- [x] Route pattern imports/exports through AST → HIR lowering
- [x] Handle pattern imports in module loader (Only and Hiding specs)
- [x] E2E test: pattern_import (export + use pattern synonyms with `pattern` prefix in export list)

**Key files:**
- `crates/bhc-ast/src/lib.rs` — `Import::Pattern` and `Export::Pattern` variants
- `crates/bhc-parser/src/decl.rs` — `parse_import_item` and `parse_export` with `check_ident_str("pattern")`
- `crates/bhc-lower/src/lower.rs` — `lower_import` and `lower_export` pattern arms
- `crates/bhc-lower/src/loader.rs` — `filter_imports` pattern arms (Only + Hiding)

### 0.8 `deriving Read`

**Status:** ✅ Complete
**Scope:** Medium
**Impact:** Medium — used in many packages for serialization/parsing

- [x] Implement `derive_read_data` and `derive_read_newtype` in deriving.rs
- [x] Make `read` polymorphic (`String -> a`) in typeck context.rs
- [x] Add Read to builtin instances for Int/Float/Double/Char/String/Integer
- [x] Inline LLVM codegen (`lower_read_adt_inline`) for char-list comparison against constructor names
- [x] E2E test: `show (read "Red" :: Color) == "Red"` roundtrip for nullary ADT constructors

**Key files:**
- `crates/bhc-hir-to-core/src/deriving.rs` — `derive_read_data`, `derive_read_newtype`
- `crates/bhc-typeck/src/context.rs` — polymorphic `read`/`readMaybe`, Read builtin instances
- `crates/bhc-codegen/src/llvm/lower.rs` — `lower_read_adt_inline`, `infer_read_target_type`, `derived_read_fns` map

### 0.9 `mask` / `uninterruptibleMask` (Async Exception Safety)

**Status:** 🟡 RTS stubs execute action without masking
**Scope:** Medium
**Impact:** Medium — correctness issue for `bracket` and resource-safe code

The RTS stubs just run the action directly. For correctness, `mask` should set a
thread-local flag that defers asynchronous exceptions until the next safe point.

- [ ] Add `masked` flag to RTS thread state
- [ ] `mask`: set flag, run action with `restore` callback, clear flag
- [ ] `uninterruptibleMask`: stronger version, block even at safe points
- [ ] Deferred exceptions delivered when mask is lifted
- [ ] E2E test: `mask` prevents async exception during critical section

**Key files:**
- `rts/bhc-rts/src/ffi.rs` — `bhc_mask`, `bhc_unmask` stubs (~line 2409)

### 0.10 `Rational` Type

**Status:** ❌ Faked as `(Int, Int)` tuple
**Scope:** Medium
**Impact:** Medium — wrong semantics for numeric conversions

- [ ] Implement proper `Rational` type as `Ratio Integer` (numerator/denominator pair)
- [ ] `(%)` operator for construction with GCD normalization
- [ ] `numerator`, `denominator` accessors
- [ ] `fromRational` / `toRational` conversions
- [ ] `Num`/`Fractional`/`Real` instances
- [ ] E2E test: `1 % 3 + 1 % 6 == 1 % 2`

**Key files:**
- `crates/bhc-typeck/src/builtins.rs` — Rational type registration (~line 2428)

### 0.11 Qualified Record Construction

**Status:** ❌ TODO in parser
**Scope:** Small
**Impact:** Low-medium — used in multi-module code

- [ ] Parse `Module.Constructor { field = val }` syntax
- [ ] Resolve the constructor through the module qualifier
- [ ] E2E test: qualified record construction across modules

**Key files:**
- `crates/bhc-parser/src/expr.rs` — record construction (~line 254)

### 0.12 `.hs-boot` Mutual Module Recursion

**Status:** ❌ Not supported
**Scope:** Medium
**Impact:** Medium — required for packages with circular module deps

- [ ] Parse `.hs-boot` files (subset of `.hs`: type signatures, data declarations)
- [ ] Generate preliminary `.bhi` from `.hs-boot`
- [ ] Use boot interface to break circular dependency during compilation
- [ ] Verify boot interface matches actual module after compilation

**Key files:**
- `crates/bhc-driver/src/lib.rs` — module dependency resolution

### 0.13 Silently Ignored Extensions

**Status:** 🟡 Many LANGUAGE pragmas accepted with no semantic effect
**Scope:** Varies
**Impact:** Variable — subtle silent correctness bugs

Extensions that are accepted but ignored should either be implemented or produce a
clear warning. Key extensions to audit:

- [ ] `RankNTypes` — higher-rank polymorphism (may partially work via existing forall)
- [ ] `ExistentialQuantification` — existential types in data constructors
- [ ] `ConstraintKinds` — use constraints as types
- [ ] `QuantifiedConstraints` — `forall a. C a => ...` in constraints
- [ ] `AllowAmbiguousTypes` — suppress ambiguity errors (interacts with TypeApplications)
- [ ] `UndecidableInstances` — relax instance termination check
- [ ] `NumericUnderscores` — `1_000_000` literal syntax
- [ ] `BinaryLiterals` — `0b1010` literal syntax
- [ ] Emit warning for unimplemented but accepted extensions

---

## Tier 1 — Showstoppers (Libraries & Ecosystem)

These must be resolved before real-world Haskell libraries can be compiled.

### 1.1 Package System Integration

**Status:** ✅ Compile-only pipeline complete (E.66), hx build pipeline wired
**Scope:** Medium (remaining: end-to-end testing, CPP, conditional deps)

Multi-package support with `-I` import paths is working (E.6). BHC now has a
full separate compilation pipeline: `-c` mode compiles single modules to `.o` +
`.bhi`, with `--odir`/`--hidir`/`--package-db` flags. The `TypeConverter` bridge
converts `.bhi` interface types to internal `Ty`/`Scheme` for cross-module type
checking. The [hx](https://github.com/raskell-io/hx) package manager has full
`.cabal` parsing, Hackage integration, dependency resolution, and its BHC backend
crate (`hx-bhc`) is now fully wired: CLI flags match BHC's actual clap-based CLI,
package DB uses filesystem operations (no `bhc-pkg`), and BHC builtin packages
(base, text, containers, etc.) are mapped to skip compilation.

- [x] Wire package resolution into `bhc-driver` compilation pipeline (basic import paths)
- [x] `--numeric-version` flag (hx-bhc's `BhcCompilerConfig::detect_with_path()` calls this)
- [x] `-c` (compile-only) mode: compile to `.o` without linking
- [x] `--odir`/`--hidir` flags: output directories for `.o` and `.bhi` files
- [x] `-package-db` flag: package database location (repeatable)
- [x] `-package-id` flag: expose a dependency by package ID (repeatable)
- [x] Generate `.bhi` interface files from compiled modules (`bhc-interface/generate.rs`)
- [x] Consume `.bhi` interface files during type checking (`TypeConverter` + `DefInfo.type_scheme`)
- [x] Load `.bhi` files for imported modules from hidir and package-db directories
- [x] Wire `hx build --backend bhc` through full pipeline (hx repo) — flags fixed, 45 hx-bhc tests pass
- [x] Filesystem-based package DB in hx-bhc (no bhc-pkg, just directory scan of `.conf` files)
- [x] BHC builtin package mapping (base→bhc-base, text→bhc-text, etc.) in hx-bhc
- [ ] Parse `.cabal` files (hx-solver already has this — needs testing with BHC)
- [ ] Resolve transitive dependency graph (hx-solver already has this)
- [ ] Fetch packages from Hackage (hx-solver already has this)
- [ ] Support `PackageImports` extension for disambiguating modules
- [ ] Handle conditional dependencies (flags, OS checks, impl checks)
- [ ] Cache compiled packages to avoid recompilation
- [ ] Handle mutual module recursion (`.hs-boot` files)
- [ ] Incremental recompilation (check timestamps / hashes)

**Key files:**
- `crates/bhc/src/main.rs` — CLI flags (`-c`, `--odir`, `--hidir`, etc.)
- `crates/bhc-session/src/lib.rs` — `Options` struct with compile-only fields
- `crates/bhc-driver/src/lib.rs` — `compile_module_only()`, `load_interfaces_for_imports()`
- `crates/bhc-interface/src/convert.rs` — `TypeConverter` (.bhi → internal types)
- `crates/bhc-interface/src/generate.rs` — `.bhi` generation from compiled modules
- `crates/bhc-lower/src/context.rs` — `DefInfo.type_scheme`, `define_with_type()`
- `crates/bhc-typeck/src/context.rs` — Interface type scheme consumption in `register_lowered_builtins()`

### 1.2 Data.Text and Data.ByteString

**Status:** ✅ Core APIs complete (E.7 + E.8), Text.IO complete (E.10), Lazy variants complete
**Scope:** Small (remaining: decodeUtf8', SIMD)

Data.Text (E.7): packed UTF-8 with 25+ operations. Data.ByteString (E.8): 24
RTS functions with identical memory layout. Data.Text.Encoding (E.8): zero-copy
encodeUtf8/decodeUtf8 bridge.

- [x] Implement packed UTF-8 `Text` representation (not `[Char]`)
- [x] Core Text API: pack, unpack, append, cons, snoc, head, tail, length,
      null, map, take, drop, toLower, toUpper, toCaseFold, toTitle,
      isPrefixOf, isSuffixOf, isInfixOf, eq, compare, singleton, empty,
      filter, foldl', concat, intercalate, strip, words, lines, splitOn, replace
- [x] Text.IO: readFile, writeFile, appendFile, hGetContents, hGetLine, hPutStr, hPutStrLn, putStr, putStrLn, getLine, getContents
- [x] Text.Encoding: encodeUtf8, decodeUtf8
- [ ] Text.Encoding: decodeUtf8' (with Either error handling)
- [x] Lazy Text: Data.Text.Lazy (14 functions: empty, fromStrict, toStrict, pack, unpack, null, length, append, fromChunks, toChunks, head, tail, take, drop)
- [x] ByteString: packed byte array type (identical layout to Text)
- [x] ByteString API (24 functions): pack, unpack, empty, singleton, append,
      cons, snoc, head, last, tail, init, length, null, take, drop, reverse,
      elem, index, eq, compare, isPrefixOf, isSuffixOf, readFile, writeFile
- [x] ByteString.Lazy (20 functions): empty, fromStrict, toStrict, fromChunks, toChunks, null, length, pack, append, head, tail, take, drop, filter, isPrefixOf, readFile, writeFile, putStr, hPutStr, hGetContents
- [x] ByteString.Lazy.Char8 (6 functions): unpack, lines, unlines, take, dropWhile, cons
- [x] Data.Text.Lazy.Encoding: encodeUtf8, decodeUtf8
- [x] ByteString.Builder (16 RTS functions + codegen dispatch, E2E test)
- [ ] SIMD-optimized operations where applicable (memchr, memcmp, etc.)

**Key files:**
- `stdlib/bhc-text/src/text.rs` — Text RTS (25+ FFI functions)
- `stdlib/bhc-text/src/bytestring.rs` — ByteString RTS (24 FFI functions)
- `crates/bhc-typeck/src/builtins.rs` — type registrations
- `stdlib/bhc-text/src/lazy_text.rs` — Lazy Text RTS (14 FFI functions)
- `stdlib/bhc-text/src/lazy_bytestring.rs` — Lazy ByteString + Char8 + Encoding RTS (28 FFI functions)
- `crates/bhc-codegen/src/llvm/lower.rs` — VarIds 1000200-1000477

### 1.3 Full IO and Exception Handling

**Status:** ✅ Core exception handling complete (E.5), exception hierarchy complete, file IO working, directory ops complete (E.19)
**Scope:** Small (remaining: async exceptions stubs, temp files)

Exception handling (catch, bracket, finally, onException) is working (E.5).
File IO (readFile, writeFile, openFile, hClose) is working. System ops
(getArgs, getEnv, exitWith) are working. Directory operations (E.19) complete.

- [x] Handle abstraction: `Handle`, `IOMode`
- [x] File operations: `openFile`, `hClose`, `hFlush`
- [x] Reading: `hGetLine`, `hGetContents`, `hIsEOF`
- [x] Writing: `hPutStr`, `hPutStrLn`
- [x] Standard handles: `stdin`, `stdout`, `stderr`
- [x] File-level: `readFile`, `writeFile`, `appendFile`
- [x] Exception types: `SomeException`, `IOException`, `ErrorCall`
- [x] Exception primitives: `throw`, `throwIO`, `catch`, `try`
- [x] Resource management: `bracket`, `bracket_`, `finally`, `onException`
- [x] Exception hierarchy: `Exception` typeclass with `toException`/`fromException`
- [ ] Asynchronous exceptions: `mask`, `uninterruptibleMask` (at least stubs)
- [x] System operations: `getArgs`, `getProgName`, `getEnv`, `lookupEnv`
- [x] Exit: `exitSuccess`, `exitFailure`, `exitWith`
- [x] Directory: `doesFileExist`, `doesDirectoryExist`, `createDirectory`,
      `removeFile`, `removeDirectory`, `getCurrentDirectory`, `setCurrentDirectory`,
      `renameFile`, `copyFile`, `listDirectory` (E.19)
- [ ] Temporary files: `withTempFile`, `withTempDirectory`

**Key files:**
- `stdlib/bhc-system/` — system/IO crate
- `rts/bhc-rts/` — runtime entry points
- `crates/bhc-codegen/src/llvm/lower.rs` — codegen handlers

### 1.4 Template Haskell

**Status:** Syntax parsed, no evaluation
**Scope:** Large

Pandoc's dependencies (aeson, lens, generic-deriving) use TH for deriving
instances, generating boilerplate, and compile-time code generation.

- [ ] TH expression AST (`Language.Haskell.TH.Syntax`)
- [ ] Quotation brackets: `[| expr |]`, `[d| decls |]`, `[t| type |]`, `[p| pat |]`
- [ ] Splice evaluation: `$(expr)` at compile time
- [ ] Name lookup: `'name` and `''TypeName` quotes
- [ ] Reification: `reify`, `reifyInstances`, `reifyType`
- [ ] TH monad: `Q` monad with fresh name generation, module info, etc.
- [ ] Cross-stage persistence for spliced values
- [ ] `DeriveLift` and `Lift` class

**Alternative approach:** Full GHC.Generics is now implemented (from/to with
Rep type algebra). This unblocks aeson/yaml generic deriving without TH.
Combined with DerivingVia (Tier 0.4), this covers the most common TH use
cases. TH can be deferred until a package genuinely requires splice evaluation.

---

## Tier 2 — Major Gaps

Required for Pandoc but solvable without architectural changes.

### 2.1 GADT and Type Family Completion

**Status:** ✅ GADTs working (E.60), type families working (open, closed, associated), data families working
**Scope:** Small (remaining: kind inference improvements)

- [x] GADT type checking: refine types in branches based on constructor (E.60)
- [x] Type family reduction during type checking
- [x] Closed type families with overlapping equations (first-match semantics)
- [x] Open type families with standalone instances
- [x] Associated type families (class-associated, with defaults)
- [x] Data families (standalone data families with data instances)
- [ ] Kind inference improvements (currently requires manual signatures)

### 2.2 Multi-Module Compilation

**Status:** ✅ Core workflow complete including `.bhi` interface files (E.66)
**Scope:** Small (remaining: `.hs-boot` mutual recursion, incremental recompilation)

- [x] Compile multiple modules in dependency order (BFS discovery + Kahn's toposort)
- [x] Cross-module type info via ModuleRegistry (types flow between modules)
- [x] Separate compilation: each module to `.o`, link at end
- [x] Module-qualified symbol mangling (no link-time collisions)
- [x] Generate `.bhi` interface files from compiled modules (E.66)
- [x] Read `.bhi` interface files during type checking with correct type schemes (E.66)
- [x] `-c` compile-only mode: produce `.o` + `.bhi` without linking (E.66)
- [ ] Handle mutual module recursion (`.hs-boot` files)
- [ ] Incremental recompilation (check timestamps / hashes)

### 2.3 Missing Standard Libraries

Each of these is a Pandoc dependency that must exist in BHC's stdlib or be
compiled from Hackage source.

#### containers (Data.Map, Data.Set, Data.Sequence, Data.IntMap, Data.IntSet)
- [x] Data.Map — RTS-backed BTreeMap (basic ops + WithKey variants done)
- [x] Data.Set — RTS-backed BTreeSet (full type support + unions/partition, E.22)
- [x] Data.IntMap — shares Map RTS (full type support, E.22)
- [x] Data.IntSet — shares Set RTS (full type support + filter/foldr, E.22)
- [x] Data.Sequence — Vec-backed RTS with full codegen pipeline
- [x] Data.Map.update, Data.Map.alter, Data.Map.unions, Data.Map.keysSet (E.21)
- [ ] Data.Graph, Data.Tree (used by some Pandoc deps)

#### mtl / transformers
- [x] `runReaderT`, `runStateT`, `runExceptT`, `runWriterT` — all working
- [x] `ask`, `local`, `get`, `put`, `modify`, `throwError`, `catchError` — all working
- [x] `lift`, `liftIO` — working for single-layer transformers
- [x] MonadReader, MonadState, MonadError, MonadWriter classes — registered in type system
- [x] Codegen for nested transformer stacks: `StateT s (ReaderT r IO)` working
- [x] Codegen for nested transformer stacks: `ReaderT r (StateT s IO)` working (E.55)
- [x] ExceptT cross-transformer: ExceptT over StateT, ExceptT over ReaderT (E.56)
- [x] WriterT cross-transformer: WriterT over StateT, WriterT over ReaderT (E.57)

#### parsec / megaparsec
- [ ] Pandoc has its own parsers but depends on parsec for some formats
- [ ] Either port parsec source or implement a compatible API
- [ ] `ParsecT` monad transformer, combinators (`many`, `try`, `<|>`, etc.)

#### aeson (JSON)
- [ ] JSON value type (`Value`: Object, Array, String, Number, Bool, Null)
- [ ] `encode`, `decode`, `eitherDecode`
- [ ] `ToJSON` / `FromJSON` type classes
- [ ] Generic deriving for JSON instances (requires TH or GHC.Generics)

#### yaml
- [ ] YAML parsing (wraps libyaml via FFI or pure Haskell)
- [ ] `decodeFileEither`, `encodeFile`
- [ ] Pandoc uses YAML for document metadata

#### Other dependencies
- [x] `filepath` — file path manipulation (`</>`, `takeExtension`, etc.) (E.19)
- [x] `directory` — filesystem operations (E.19)
- [ ] `process` — spawn subprocesses
- [ ] `time` — date/time types
- [ ] `network-uri` — URI parsing
- [ ] `http-client` — HTTP requests (optional, for URL fetching)
- [ ] `skylighting` — syntax highlighting (large dep)
- [ ] `doctemplates` — Pandoc's template system
- [ ] `texmath` — TeX math parsing
- [ ] `xml-conduit` or `xml` — XML parsing
- [ ] `zip-archive` — ZIP file handling (for EPUB, DOCX)

### 2.4 Deriving Infrastructure

**Status:** ✅ Extensive — 9 stock derivable classes + DeriveAnyClass + GND + DerivingStrategies + full GHC.Generics
**Scope:** Small (remaining: Read, Ix)

- [x] `GHC.Generics` — Full implementation: V1, U1, K1, M1, :+:, :*: rep types + working from/to + pattern matching
- [x] Generic representations: `V1`, `U1`, `K1`, `M1`, `:+:`, `:*:` (DefIds 12400-12415)
- [x] `from` / `to` methods with balanced binary tree sum/product encoding
- [x] Derive `Generic` for user-defined types (enums, products, multi-constructor, newtypes)
- [x] Stock deriving: `Eq`, `Show` for simple enums and ADTs with fields (E.23)
- [x] Stock deriving: `Ord` for simple enums and ADTs with fields (E.24)
- [x] Stock deriving: `Enum`, `Bounded` for enums (E.54)
- [x] Stock deriving: `Functor` (E.51)
- [x] Stock deriving: `Foldable` (E.52)
- [x] Stock deriving: `Traversable` (E.53)
- [x] Stock deriving: `Read` (Tier 0.8 ✅)
- [ ] Stock deriving: `Ix`
- [x] `DeriveAnyClass` for type classes with default method implementations (E.42)
- [x] `GeneralizedNewtypeDeriving` for lifting instances through newtypes (E.47)
- [x] `StandaloneDeriving` (E.62)
- [x] `DerivingStrategies`/`DerivingVia` — strategy-aware dispatch (Tier 0.4 ✅)

---

## Tier 3 — Solvable with Current Architecture

### 3.1 Remaining Codegen Builtins

**Status:** ~500+ of 587 builtins lowered (E.13–E.31 added ~90+ functions + derived dispatches)
**Scope:** Small-Medium (ongoing)

- [ ] Monadic codegen: general `>>=`, `>>`, `return` via dictionary dispatch
- [x] `mapM_` (E.14), `when`, `unless` (E.14), `guard` (E.13)
- [x] `mapM`, `forM`, `forM_`, `sequence`, `sequence_`, `void`
- [x] `filterM`, `foldM`, `foldM_`, `replicateM`, `replicateM_`, `zipWithM`, `zipWithM_` (E.18)
- [x] `foldMap` (delegates to concatMap, E.16)
- [ ] Foldable/Traversable: `traverse`, `sequenceA`, `toList`
- [x] Data.Maybe: fromMaybe, maybe, listToMaybe, maybeToList, catMaybes, mapMaybe (E.13)
- [x] Data.Either: either, fromLeft, fromRight, lefts, rights, partitionEithers (E.13)
- [x] Data.List: any, all (E.14), scanr, scanl1, scanr1, unfoldr, intersect, zip3, zipWith3 (E.15)
- [x] Data.List: iterate, repeat, cycle (take-fused, E.15)
- [x] Data.List: elemIndex, findIndex, isPrefixOf, isSuffixOf, isInfixOf, tails, inits (E.16)
- [x] maximumBy, minimumBy (E.16), compare returns Ordering ADT (E.17)
- [x] Fixed stubs: maximum, minimum, and, or, Data.Map.notMember (E.16)
- [x] Data.Map.update, Data.Map.alter, Data.Map.unions, Data.Map.keysSet (E.21)
- [x] Data.Set.unions, Data.Set.partition (E.22)
- [x] Data.IntSet.filter, Data.IntSet.foldr (reuse Set implementations, E.22)
- [x] Full typeck/context.rs type entries for Data.Set (30), Data.IntMap (25), Data.IntSet (15) (E.22)
- [x] Fixed VarId suffix bug in Set/IntSet binary/predicate/extremum dispatches (E.22)
- [x] `show` dispatch for Bool-returning container builtins (Data.Map.member/null, Data.Set.member/null, etc.) (E.21)
- [x] `compare` returns Ordering ADT (LT/EQ/GT) with proper show support (E.17)

### 3.2 Numeric and Conversion Operations

- [x] `show` for standard types: showInt, showBool, showChar, showFloat (type-specialized, E.9)
- [x] `show` for compound types: String, [a], Maybe, Either, (a,b), () (E.11)
- [x] `show` for Double/Float literals (E.29)
- [x] `show` for nested compound types via recursive ShowTypeDesc (E.31)
- [x] `read` for Int (RTS bhc_read_int, E.25)
- [x] `readMaybe` for Int (RTS bhc_try_read_int, E.25)
- [x] `fromString` (identity, E.25)
- [ ] `reads` for parsing (general)
- [x] `fromIntegral`, `toInteger`, `fromInteger` (identity pass-through, E.12)
- [x] `even`, `odd` (inline LLVM srem, E.12)
- [x] `gcd`, `lcm` (RTS functions, E.12)
- [x] `divMod`, `quotRem` (floor-division / truncation, returns tuple, E.12)
- [x] `IORef`: newIORef, readIORef, writeIORef, modifyIORef (E.12)
- [ ] `realToFrac`
- [ ] `Rational` type and operations
- [x] `Data.Char` predicates: isAlpha, isDigit, isUpper, isLower, isAlphaNum, isSpace, isPunctuation, toUpper, toLower, ord, chr, digitToInt, intToDigit (E.9)
- [ ] `Data.Char` full Unicode categories (currently ASCII-only)

### 3.3 Performance (Core IR Optimization Pipeline)

**Status:** Phases O.1-O.3 complete (Core Simplifier E.68-E.69, Pattern Match E.70, Demand Analysis + Worker/Wrapper O.3)
**Scope:** Large (foundational infrastructure)
**Reference:** `rules/013-optimization.md`, HBC/HCT simplifier architecture

BHC has a Core IR simplifier (E.68-E.69) with full local and top-level transforms:
constant folding, beta reduction, case-of-known-constructor, case-of-case (with size
budget), local and top-level dead binding elimination, local let inlining, and
top-level cheap inlining (Var aliases, Lit constants). Top-level inlining skips
protected names (`$derived_*`, `main`, `bhc_*`, etc.) for codegen safety. Top-level
dead elimination is export-aware (respects module export lists). Pattern match
compilation (E.70) uses Augustsson/Sestoft decision trees with exhaustiveness and
overlap checking. Demand analysis + worker/wrapper (O.3) computes boolean-tree
strictness signatures with fixpoint iteration for recursive groups, and splits
strict-arg functions into worker/wrapper pairs. Wired into the driver pipeline
after the first simplifier pass, gated on lazy profiles (Default/Server/Realtime)
and opt_level != None. Dictionary specialization remains to be implemented.

#### Phase O.1: Core Simplifier ✅ COMPLETE (E.68-E.69)
- [x] Beta reduction: `(\x -> body) arg` → `body[x := arg]` (cheap args only)
- [x] Case-of-known-constructor: `case Just 42 of { Just x -> x }` → `42`
- [x] Case-of-case: push outer case into inner case alternatives (with size budget)
- [x] Dead binding elimination: local lets + top-level (export-aware, cheap RHS only)
- [x] Constant folding: `1 + 2` → `3` for Int/Double literals
- [x] Local let inlining: substitute cheap single-use local bindings
- [x] Top-level inlining: cheap-only (Var aliases, Lit constants); protected names skipped
- [x] Top-level dead binding elimination: export-aware; non-exported cheap bindings removed
- [x] Occurrence analysis: Dead/Once/OnceInLam/Many reference counting
- [x] Capture-avoiding substitution with alpha-renaming
- [x] Iterate to fixpoint (cap at 10 iterations)
- [ ] `-ddump-core-after-simpl` dump flag

#### Phase O.2: Pattern Match Compilation (HIGH — correctness + quality) ✅ COMPLETE (E.70)
- [x] Replace equation-by-equation compilation with Augustsson decision trees
- [x] Exhaustiveness checking with non-exhaustive pattern warnings
- [x] Overlap/redundancy detection with shadowed pattern warnings
- [x] Guard compilation via nested case fallthrough (linear fallback)

#### Phase O.3: Demand Analysis + Worker/Wrapper (MEDIUM — Default profile perf) ✅ COMPLETE (O.3)
- [x] Boolean-tree demand analysis for strictness signatures
- [x] Fixpoint iteration for recursive binding groups
- [x] Annotate strict arguments
- [x] Worker/wrapper split for strict-arg functions (unboxed workers)
- [ ] `-ddump-core-after-demand` dump flag

#### Phase O.4: Dictionary Specialization ✅
- [x] Direct method selection when dictionary is known constructor
- [x] Cleanup simplifier pass after specialization
- [ ] Monomorphize polymorphic functions at concrete call sites (future)
- [ ] SPECIALIZE pragma support (future)

#### Key Files
```
crates/bhc-core/src/
├── simplify/
│   ├── mod.rs               # ✅ Main simplifier loop, config, stats
│   ├── expr_util.rs         # ✅ Shared utilities (is_cheap, expr_size, free_var_ids)
│   ├── subst.rs             # ✅ Capture-avoiding substitution
│   ├── occurrence.rs        # ✅ Occurrence analysis (Dead/Once/OnceInLam/Many)
│   ├── beta.rs              # ✅ Beta reduction
│   ├── case.rs              # ✅ Case-of-known-constructor
│   ├── dead.rs              # ✅ Dead binding elimination
│   ├── fold.rs              # ✅ Constant folding
│   └── inline.rs            # ✅ Inlining decisions
├── demand.rs                # ✅ Demand analysis (boolean-tree strictness, fixpoint iteration)
├── worker_wrapper.rs        # ✅ Worker/wrapper transformation (case-wrapping strict args)
└── specialize.rs            # ✅ Dictionary specialization (O.4)
```

---

## Intermediate Milestones

Rather than jumping straight to Pandoc, build toward it incrementally:

### Milestone A: Multi-Module Program ✅
- [x] Compile a 3-file Haskell program with imports between modules
- [x] Verify type checking works across module boundaries
- [x] Verify codegen produces correct linked executable

### Milestone B: File Processing Utility ✅
- [x] Compile a program that reads a file, transforms content, writes output
- [x] Requires: File IO, String operations, basic error handling
- [x] Example: word count, line reversal, simple grep

### Milestone C: Simple Markdown Parser ✅
- [x] Compile a ~500 LOC Markdown-to-HTML converter
- [x] Requires: Text processing, Data.Map for link references, File IO
- [x] No external dependencies — self-contained

### Milestone D: StateT-Based Parser ✅
- [x] Compile a program using StateT for structured input parsing
- [x] Requires: Monad transformers working with String state
- [x] Example: CSV parser using `StateT String IO`
- [x] E2E test: `tier3_io/milestone_d_csv_parser` passes

### Milestone E: JSON/YAML Processing ✅
- [x] Compile a program that parses JSON, extracts fields, writes output
- [x] Self-contained JSON parser without external dependencies
- [x] E2E test: `tier3_io/milestone_e_json` passes (outputs "Alice" and "30" from `{"name": "Alice", "age": 30}`)

### Milestone E.5: Exception Handling ✅
- [x] Implement `throw`, `catch`, `try` for IO exceptions
- [x] Implement `bracket`, `finally`, `onException` for resource management
- [x] Exception hierarchy: `SomeException`, `IOException`, `ErrorCall`
- [x] E2E tests: `bracket_io`, `catch_file_error`, `exception_test`, `handle_io`

### Milestone E.6: Multi-Package Program ✅
- [x] Wire import paths into `bhc-driver` via `-I` flag
- [x] Compile programs that import from external package directories
- [x] E2E test: `tier3_io/package_import` passes

### Milestone E.7: Data.Text Foundation ✅
- [x] Implement packed UTF-8 `Text` type (not `[Char]`)
- [x] Core API: pack, unpack, append, length, null, take, drop, toUpper, toLower
- [x] RTS-backed implementation in bhc-text with UTF-8 encoding
- [x] E2E test: `tier3_io/text_basic` passes (pack, unpack, append, toUpper, take, drop)

### Milestone E.8: Data.ByteString + Text Completion ✅
- [x] ByteString RTS: 24 FFI functions with same memory layout as Text
- [x] ByteString type system: `bytestring_con`/`bytestring_ty` + 23 PrimOps
- [x] ByteString codegen: VarIds 1000400-1000423
- [x] Text.Encoding: `encodeUtf8` (zero-copy), `decodeUtf8` (validates UTF-8)
- [x] Additional Text ops: filter, foldl', concat, intercalate, strip, words, lines, splitOn, replace
- [x] E2E tests: `tier3_io/bytestring_basic` and `tier3_io/text_encoding` pass
- [x] 43 total E2E tests pass, 66 bhc-text unit tests pass

### Milestone E.9: Data.Char + Type-Specialized Show ✅
- [x] Data.Char predicates: isAlpha, isDigit, isUpper, isLower, isAlphaNum, isSpace, isPunctuation
- [x] Data.Char conversions: toUpper, toLower, ord, chr, digitToInt, intToDigit
- [x] Type-specialized show: showInt, showBool, showChar, showFloat
- [x] Proper ADT boolean returns from char predicates (tag 0=False, 1=True)
- [x] bhc-base linked, VarId bug fixes
- [x] E2E tests: `tier3_io/char_predicates`, `tier3_io/show_types`
- [x] 45 total E2E tests pass

### Milestone E.10: Data.Text.IO ✅
- [x] 7 RTS functions in bhc-text: readFile, writeFile, appendFile, hGetContents, hGetLine, hPutStr, hPutStrLn
- [x] 4 codegen-composed convenience functions: putStr, putStrLn, getLine, getContents
- [x] Handle functions use sentinel-pointer pattern (1=stdin, 2=stdout, 3=stderr)
- [x] Fixed import shadowing bug in register_standard_module_exports
- [x] E2E test: `tier3_io/text_io`
- [x] 46 total E2E tests pass

### Milestone E.11: Show Compound Types ✅
- [x] 6 RTS show functions: bhc_show_string, bhc_show_list, bhc_show_maybe, bhc_show_either, bhc_show_tuple2, bhc_show_unit
- [x] Expression-based type inference (infer_show_from_expr) since expr.ty() returns Error in Core IR
- [x] ShowCoerce extended: StringList, List, MaybeOf, EitherOf, Tuple2Of, Unit
- [x] RTS type tags: 0=Int, 1=Double, 2=Float, 3=Bool, 4=Char, 5=String for element formatting
- [x] VarIds 1000092-1000097, DefIds 10105-10110
- [x] E2E tests: show_string, show_list, show_maybe, show_either, show_tuple, show_unit
- [x] 52 total E2E tests pass

### Milestone E.12: Numeric Conversions + IORef ✅
- [x] `fromIntegral`/`toInteger`/`fromInteger` as identity pass-through
- [x] `even`/`odd` via inline LLVM srem, returning proper Bool ADT
- [x] `gcd`/`lcm` via RTS functions (VarIds 1000500-1000501)
- [x] `divMod` with floor-division semantics (sign adjustment for negative dividends)
- [x] `quotRem` with truncation-toward-zero (LLVM sdiv/srem)
- [x] IORef: newIORef, readIORef, writeIORef, modifyIORef (VarIds 1000502-1000504, DefIds 10400-10404)
- [x] Fixed DefIds 10500-10507 for numeric ops (bypasses 30-entry sequential array misalignment)
- [x] Show inference for Bool/Int-returning functions (expr_returns_bool, expr_returns_int)
- [x] E2E tests: numeric_ops, divmod, ioref_basic
- [x] 55 total E2E tests pass

### Milestone E.13: Data.Maybe + Data.Either + guard ✅
- [x] Data.Maybe: fromMaybe, maybe, listToMaybe, maybeToList, catMaybes, mapMaybe
- [x] Data.Either: either, fromLeft, fromRight, lefts, rights, partitionEithers
- [x] Control.Monad: guard
- [x] 13 pure LLVM codegen functions, no RTS needed
- [x] Fixed DefIds 10600-10622
- [x] Shared helper: `build_inline_reverse()` for catMaybes, lefts, rights, mapMaybe, partitionEithers
- [x] E2E tests: data_maybe, data_either, guard_basic
- [x] 58 total E2E tests pass

### Milestone E.14: when/unless + any/all + Closure Wrapping ✅
- [x] Fix when/unless Bool bug: use `extract_adt_tag()` not `ptr_to_int()` for Bool ADT
- [x] Implement `any`/`all` with loop + predicate closure + short-circuit
- [x] Add `even`/`odd` to `lower_builtin_direct` for first-class closure wrapping
- [x] E2E tests: when_unless, mapm_basic, any_all
- [x] 61 total E2E tests pass

### Milestone E.15: Data.List Completions ✅
- [x] Finite ops: scanr, scanl1, scanr1, unfoldr, intersect, zip3, zipWith3
- [x] Infinite generators (take-fused): iterate, repeat, cycle
- [x] Fixed DefIds 10700-10706
- [x] E2E tests: scanr_basic, unfoldr_basic, zip3_basic, take_iterate, intersect_basic
- [x] 66 total E2E tests pass

### Milestone E.16: Fix Broken Stubs + List Operations ✅
- [x] Fixed 5 broken stubs: maximum (accumulator loop), minimum, and (Bool tag short-circuit), or, Data.Map.notMember (XOR inversion)
- [x] 10 new functions: elemIndex, findIndex, isPrefixOf, isSuffixOf, isInfixOf, tails, inits, maximumBy, minimumBy, foldMap
- [x] Fixed DefIds 10800-10809
- [x] E2E tests: max_min_and_or, elem_index_prefix, tails_inits
- [x] 69 total E2E tests pass
- [x] Known limitation: `show` doesn't dispatch correctly for Bool-returning builtins (and/or/isPrefixOf); `compare` returns Int not Ordering ADT

### Milestone E.17: Ordering ADT + compare ✅
- [x] Ordering ADT (LT/EQ/GT) as zero-field ADT (tag 0=LT, 1=EQ, 2=GT)
- [x] `compare` returns Ordering instead of Int (fixed DefId 10900)
- [x] ShowCoerce::Ordering + RTS `bhc_show_ordering`
- [x] Fixed flat calling convention in maximumBy/minimumBy
- [x] E2E test: ordering_basic
- [x] 70 total E2E tests pass

### Milestone E.18: Monadic Combinators ✅
- [x] 7 pure LLVM codegen functions: filterM, foldM, foldM_, replicateM, replicateM_, zipWithM, zipWithM_
- [x] Fixed DefIds 11000-11006
- [x] Key pitfall: replicateM/replicateM_ re-lower action_expr each iteration, creating new blocks
- [x] E2E tests: monadic_combinators, zipwithm_basic
- [x] 70 total E2E tests pass (68+2 new, 4 pre-existing text/exception failures)

### Milestone E.19: System.FilePath + System.Directory ✅
- [x] System.FilePath: takeFileName, takeDirectory, takeExtension, dropExtension, takeBaseName, replaceExtension, isAbsolute, isRelative, hasExtension, splitExtension, </>
- [x] System.Directory: setCurrentDirectory, removeDirectory, renameFile, copyFile
- [x] 14 RTS FFI functions + 1 codegen-composed (splitExtension)
- [x] Fixed DefIds 11100-11115, VarIds 1000520-1000534
- [x] Key pitfall: `typeck/context.rs` type match must include ALL builtin types — `builtins.rs` `register_value()` alone is insufficient
- [x] Also fixed missing types for createDirectory, removeFile in typeck match
- [x] E2E tests: filepath_basic, directory_ops
- [x] 72 total E2E tests pass (70+2 new, 4 pre-existing text/exception failures)

### Milestone E.20: Fix DefId Misalignment for Text/ByteString/Exceptions ✅
- [x] Fixed DefId misalignment for Data.Text (38 funcs), Data.ByteString (24 funcs), Data.Text.Encoding (2 funcs)
- [x] Fixed DefIds 11200-11273 for all Text/ByteString/Encoding functions
- [x] Added typeck/context.rs match entries for throwIO/throw/try/evaluate
- [x] Added typeck/context.rs match entries for all Data.Text, Data.ByteString, Data.Map functions
- [x] 74 total E2E tests pass (72 existing + 4 previously-broken text/exception tests fixed)

### Milestone E.21: Data.Map Completion ✅
- [x] Linked bhc-containers in driver (1-line change)
- [x] Implemented 4 stubbed codegen functions: `unions` (cons-list fold), `keysSet` (iterate + set insert), `update` (lookup + Maybe closure + delete/insert), `alter` (build input Maybe + closure + delete/insert)
- [x] Fixed Bool ADT returns: container predicates (member, null, isSubmapOf, set_member, set_null) now use `allocate_bool_adt()` instead of `int_to_ptr()`
- [x] Fixed show inference: `expr_returns_bool()` now recognizes qualified container names (Data.Map.member, Data.Set.null, etc.)
- [x] Fixed type signatures for `update` (`b -> Maybe b`) and `alter` (`Maybe b -> Maybe b`) in builtins.rs + typeck/context.rs
- [x] E2E tests: map_basic (un-ignored), map_complete (new: update/alter/unions)
- [x] 76 total E2E tests pass (74 existing + 2 new, 0 failures)

### Milestone E.22: Data.Set/IntMap/IntSet Type Completion ✅
- [x] Added ~70 type match entries to typeck/context.rs for Data.Set (30), Data.IntMap (25), Data.IntSet (15)
- [x] Replaced 4 stub dispatches: Set.unions (cons-list walk + union accumulator), Set.partition (dual-accumulator with predicate + tuple return), IntSet.filter (reuse Set.filter), IntSet.foldr (reuse Set.foldr)
- [x] Fixed pre-existing VarId suffix bug: Set/IntSet binary/predicate/extremum dispatches passed suffix (e.g., 1127) instead of full VarId (1000127) — 13 dispatch sites fixed
- [x] E2E tests: set_basic (new: fromList/size/member/insert/delete/union/intersection/difference/filter/foldr), intmap_intset (new: IntSet size/member/filter/foldr + IntMap size/member/insert/delete)
- [x] 78 total E2E tests pass (76 existing + 2 new, 0 failures)

### Milestone E.23: Stock Deriving — Eq, Show for User ADTs ✅
- [x] Fixed `fresh_var` off-by-one bug in `deriving.rs`: name used counter N but VarId used N+1
- [x] Shared `DerivingContext` across all data types in a module (was recreated per type, causing VarId collision)
- [x] `fresh_counter` starts at 50000 to avoid collision with fixed DefId ranges (10000-11273)
- [x] Added `type_name: Option<String>` to `ConstructorMeta` for ADT type tracking
- [x] Pre-pass `detect_derived_instance_methods`: scans `$derived_show_*`/`$derived_eq_*` bindings
- [x] `strip_deriving_counter_suffix`: extracts clean type name from `Color_50000` → `Color`
- [x] `tag_constructors_with_type`: walks derived binding bodies, tags constructors with their type name
- [x] `infer_adt_type_from_expr`: checks constructor names in metadata for type dispatch
- [x] `lower_builtin_show` dispatches to derived show via indirect call `fn(env_ptr, value) -> string_ptr`
- [x] `PrimOp::Eq` dispatches to derived eq via indirect call `fn(env_ptr, lhs, rhs) -> Bool ADT` → `extract_adt_tag()`
- [x] Fixed `register_constructor` to preserve existing `type_name` when re-registering
- [x] E2E tests: derive_show (enum + ADT with fields), derive_eq (enum equality + inequality)
- [x] 80 total E2E tests pass (78 existing + 2 new, 0 failures)

### Milestone E.24: Stock Deriving — Ord for User ADTs ✅
- [x] Added `derived_compare_fns` dispatch table mirroring `derived_show_fns`/`derived_eq_fns` pattern
- [x] Extended `detect_derived_instance_methods` to detect `$derived_compare_*` bindings
- [x] `lower_builtin_compare` dispatches to derived compare for user ADTs before falling through to Int comparison
- [x] `PrimOp::Lt/Le/Gt/Ge` dispatch through derived compare: call → extract Ordering tag → compare tag
- [x] Made `compare` polymorphic (`a -> a -> Ordering`) in builtins.rs + fixed DefId block
- [x] Made `<`/`<=`/`>`/`>=` polymorphic (`a -> a -> Bool`) in builtins.rs + typeck/context.rs `cmp_binop()`
- [x] E2E test: derive_ord (compare on enums + comparison operators + multiple types)
- [x] 81 total E2E tests pass (80 existing + 1 new, 0 failures)

### Milestone E.25: String Type Class Methods ✅
- [x] `fromString` as identity pass-through
- [x] `read` (String→Int) via RTS `bhc_read_int`
- [x] `readMaybe` (String→Maybe Int) via RTS `bhc_try_read_int`
- [x] Fixed DefIds 11300-11302, VarIds 1000540-1000541
- [x] Show inference: readMaybe recognized as Maybe-returning
- [x] E2E test: string_read
- [x] 82 total E2E tests pass

### Milestone E.26: More List Operations ✅
- [x] 10 RTS functions: sortOn, nubBy, groupBy, deleteBy, unionBy, intersectBy, stripPrefix, insert, mapAccumL, mapAccumR
- [x] Internal helpers: extract_bool (dual Bool representation), call_eq_closure, alloc_nothing/just/tuple
- [x] Fixed DefIds 11400-11409, VarIds 1000550-1000559
- [x] Show inference: `expr_looks_like_list` integrated into `infer_show_from_expr` App case
- [x] E2E test: list_by_ops (13 assertions covering all 10 functions)
- [x] 83 total E2E tests pass

### Milestone E.27: Data.Function + Data.Tuple Builtins ✅
- [x] `succ`/`pred` via inline LLVM add/sub (Int → Int)
- [x] `(&)` reverse application operator (a → (a → b) → b)
- [x] `swap` for tuples: extract fields, allocate reversed tuple via `allocate_ptr_pair_tuple()`
- [x] `curry`: allocate tuple from (x, y), call f(tuple) via 1-arg closure
- [x] `uncurry`: extract fst/snd from pair, flat 3-arg call fn(env, fst, snd)
- [x] `fst`/`snd` added to `lower_builtin_direct` for first-class closure use (e.g., `map fst pairs`)
- [x] `succ`/`pred`/`swap` added to `lower_builtin_direct` for first-class closure use
- [x] Fixed DefIds 11500-11505, arity + dispatch entries
- [x] Show inference: succ/pred added to `expr_returns_int()`
- [x] Key pitfall: BHC compiles multi-arg functions as FLAT `fn(env, x, y)`, not curried — uncurry must use 3-arg call
- [x] E2E tests: data_function (succ/pred/(&)/map succ/map pred), tuple_functions (fst/snd/swap/curry/uncurry/map fst/map snd/map swap)
- [x] 85 total E2E tests pass (83 existing + 2 new, 0 failures)

### Milestone E.28: Arithmetic, Enum, Folds, Higher-Order, IO Input ✅
- [x] Arithmetic: `min`, `max`, `subtract` (inline LLVM compare+select / sub)
- [x] Enumeration: `enumFrom`, `enumFromThen`, `enumFromThenTo` (list-building loops with step)
- [x] Folds: `foldl1` (head as init, foldl tail), `foldr1` (reverse, head as init, foldl with flipped args)
- [x] Higher-order: `comparing` (call f on both args, inline compare), `until` (loop with predicate + transform)
- [x] IO Input: `getChar`, `isEOF`, `getContents` (3 RTS functions), `interact` (codegen-composed)
- [x] Partial builtin application: `create_partial_builtin_closure()` enables `map (min 5) xs` for codegen-only builtins
- [x] Fixed DefIds 11600-11613, VarIds 1000560-1000562
- [x] E2E tests: enum_functions (min/max/subtract/enum/foldl1/foldr1/until), fold_misc (foldl1/foldr1 with user fn, map with partial min/max/subtract)
- [x] 87 total E2E tests pass (85 existing + 2 new, 0 failures)

### Milestone E.29: flip + show Double + Data.Map.mapMaybe ✅
- [x] Fix flip calling convention: was using curried 2-step calls (segfault), fixed to flat 3-arg `fn(env, arg2, arg1)`
- [x] Show Double/Float literals: `ShowCoerce::Double` handles `FloatValue` directly (fpext f32→f64)
- [x] `expr_returns_double()`: recognizes unary (sqrt/sin/cos/...) and binary (/, **) Double-returning functions
- [x] `expr_looks_like_list`: Added Data.Map.toList/keys/elems/assocs, Data.Set.toList/elems
- [x] Data.Map.mapMaybe/mapMaybeWithKey: Fixed DefIds 11700-11701
- [x] flip/const added to `lower_builtin_direct` for first-class use
- [x] E2E tests: flip_test, show_double, map_maybe
- [x] 90 total E2E tests pass

### Milestone E.30: Unified Bool Extraction (extract_bool_tag) ✅
- [x] Two Bool representations: tagged-int-as-pointer (0/1) vs Bool ADT (heap struct)
- [x] `extract_bool_tag()`: checks `ptr_to_int <= 1` — if so, raw value; else loads ADT tag
- [x] Applied to: filter, takeWhile, dropWhile, span, break, find, partition
- [x] Phi predecessor pitfall: creates 3 new blocks; phi nodes must reference `bool_merge` block
- [x] E2E tests: filter_bool, list_predicate_ops, partition_test
- [x] 93 total E2E tests pass

### Milestone E.31: Show Nested Compound Types ✅
- [x] `ShowTypeDesc` struct: `#[repr(C)]` with `tag: i64`, `child1/child2: *const ShowTypeDesc`
- [x] Tags: 0-7 primitives (Int/Double/Float/Bool/Char/String/Unit/Ordering), 10-13 compounds (List/Maybe/Tuple2/Either)
- [x] `show_any()`: recursive dispatch in RTS, handles nested types at any depth
- [x] `show_any_prec()`: precedence-aware parens for constructor apps (Just x, Left x, Right x)
- [x] LLVM global descriptor trees built at compile time from expression structure analysis
- [x] `bhc_show_with_desc(ptr, desc)` FFI entry point (VarId 1000099)
- [x] Backward compatible: primitive ShowCoerce variants unchanged, compound types use new path
- [x] E2E tests: show_nested (list of tuples), show_nested_maybe (Maybe of list), show_nested_list (list of lists)
- [x] 96 total E2E tests pass

### Milestone E.32+: Road to Pandoc

#### E.32: OverloadedStrings + IsString ✅
- [x] `IsString` typeclass with `fromString :: String -> a` method
- [x] `OverloadedStrings` extension: string literals desugar to `fromString "..."` calls
- [x] `IsString` instances for Text, ByteString (via pack)
- [x] Identity instance for String

#### E.33: Record Syntax ✅
- [x] Named field declarations in data types
- [x] Field accessor functions (auto-generated from field names)
- [x] Record construction syntax `Foo { bar = 1, baz = "x" }`
- [x] Record update syntax `r { field = newVal }`
- [x] `RecordWildCards` extension (`Foo{..}` brings fields into scope)
- [x] `NamedFieldPuns` extension

#### E.34: ViewPatterns Codegen ✅
- [x] Lower `f -> pat` patterns to `let tmp = f arg in case tmp of pat -> ...`
- [x] Handle in case expressions and function argument patterns
- [x] Fallthrough semantics for non-matching patterns

#### E.35: TupleSections + MultiWayIf ✅
- [x] `TupleSections`: `(,x)` as partial tuple constructors, `(x,,z)` etc. — parser desugars to lambda
- [x] `MultiWayIf`: `if | cond1 -> e1 | cond2 -> e2 | otherwise -> e3` — parser desugars to nested if-then-else
- [x] Added `otherwise` to `lower_builtin_direct` for first-class use

#### E.36: Char Enum Ranges ✅
- [x] Polymorphic enum functions for Char type
- [x] Char range syntax: `['a'..'z']`

#### E.37: Char First-Class Predicates ✅
- [x] Data.Char predicates usable as first-class functions
- [x] Fix print Bool ADT dispatch

#### E.38: Manual Typeclass Instances ✅
- [x] Three-layer approach: lower → HIR-to-Core rename → codegen detect
- [x] `$instance_show_`/`$instance_==_`/`$instance_compare_` prefix dispatch

#### E.39: Dictionary-Passing for User-Defined Typeclasses ✅
- [x] Full dictionary-passing pipeline for user-defined typeclasses
- [x] ClassRegistry, DictContext, dict construction, `$sel_N` selectors

#### E.40: Higher-Kinded Dictionary Passing ✅
- [x] Dictionary passing for higher-kinded type variables (e.g., `Functor f`)

#### E.41: Default Methods + Superclass Constraints ✅
- [x] Default method implementations in typeclass declarations
- [x] Superclass constraint propagation

#### E.42: DeriveAnyClass ✅
- [x] Derive instances for user-defined typeclasses with default methods

#### E.43–E.45: Word Types + Integer + Lazy Let ✅
- [x] Word8/Word16/Word32/Word64 types with conversion operations
- [x] Integer arbitrary precision via `num-bigint` RTS (19 FFI functions)
- [x] Lazy let-bindings (initial support)

#### E.46: ScopedTypeVariables ✅
- [x] `ScopedTypeVariables` extension enabling type variable scoping

#### E.47: GeneralizedNewtypeDeriving ✅
- [x] Lift typeclass instances through `newtype` wrappers via newtype erasure
- [x] Support in `deriving` clause

#### E.48: FlexibleInstances + FlexibleContexts ✅
- [x] `FlexibleInstances` — instances on concrete types, nested types
- [x] `FlexibleContexts` — non-variable constraints in contexts
- [x] Instance context propagation

#### E.49: MultiParamTypeClasses ✅
- [x] Multiple type parameters in class declarations

#### E.50: FunctionalDependencies ✅
- [x] `| a -> b` functional dependency syntax in class declarations

#### E.51: DeriveFunctor ✅
- [x] Automatic `fmap` derivation for pure types

#### E.52: DeriveFoldable ✅
- [x] Automatic `foldr` derivation for user ADTs

#### E.53: DeriveTraversable ✅
- [x] Automatic `traverse` derivation for user ADTs

#### E.54: DeriveEnum + DeriveBounded ✅
- [x] Enum instances (toEnum/fromEnum) for simple enums
- [x] Bounded instances (minBound/maxBound) for simple enums

#### E.55: ReaderT-over-StateT Cross-Transformer ✅
- [x] `ReaderT r (StateT s IO)` nested codegen with 3-arg closures

#### E.56: ExceptT Cross-Transformer ✅
- [x] ExceptT over StateT and ExceptT over ReaderT codegen

#### E.57: WriterT Cross-Transformer ✅
- [x] WriterT over StateT and WriterT over ReaderT codegen

#### E.58: Full Lazy Let-Bindings ✅
- [x] Full Haskell-semantics lazy let-bindings

#### E.59: EmptyDataDecls + Strict Fields ✅
- [x] `EmptyDataDecls` extension (data types with no constructors)
- [x] `Type::Bang` for strict field annotations

#### E.60: GADTs ✅
- [x] GADT syntax with type refinement in pattern matches
- [x] Bool field extraction fix

#### E.61: TypeOperators ✅
- [x] Infix type syntax (e.g., `a :+: b`)

#### E.62: StandaloneDeriving + PatternSynonyms ✅
- [x] `deriving instance Eq Foo` standalone syntax
- [x] `pattern P x = Constructor x` bidirectional pattern synonyms
- [x] Nested pattern fallthrough fix

#### E.63: DeriveGeneric + NFData/DeepSeq Stubs ✅
- [x] `DeriveGeneric` generates stub Generic instances
- [x] NFData/DeepSeq stubs for Pandoc compatibility

#### E.64: EmptyCase + StrictData + DefaultSignatures + OverloadedLists ✅
- [x] `EmptyCase` — case with no alternatives
- [x] `StrictData` — all fields strict by default in module
- [x] `DefaultSignatures` — default method type signatures in classes
- [x] `OverloadedLists` — list literal desugaring via `fromList`

#### E.65: Layout Rule Verification ✅
- [x] Verified layout rule is fully complete for Haskell 2010
- [x] All 162 E2E tests use indentation-based layout (only 3 use explicit braces)
- [x] 39 lexer unit tests cover edge cases
- [x] Comprehensive layout-focused E2E test validates all layout contexts

#### E.66: Separate Compilation Pipeline + hx Integration ✅
- [x] `--numeric-version` flag for hx-bhc detection
- [x] `-c` (compile-only) mode: compile to `.o` without linking
- [x] `--odir`/`--hidir` flags for output directories
- [x] `-package-db`/`-package-id` flags for package database
- [x] `compile_module_only()` in driver: parse → lower → typecheck → codegen → write `.o` + `.bhi`
- [x] Interface generation: `bhc-interface/generate.rs` walks AST/typed module to extract exports
- [x] `TypeConverter` bridge: `bhc-interface/convert.rs` converts interface types → internal `Ty`/`Scheme`
- [x] `.bhi` consumption in type checker: `DefInfo.type_scheme` carries types through lowering pipeline
- [x] `load_interfaces_for_imports()`: searches hidir + package-db directories for `.bhi` files
- [x] `register_lowered_builtins()` uses interface type schemes instead of fresh variables
- [x] Integration tests verify type roundtrip through `.bhi` files

#### E.67+: Remaining Road to Pandoc (Proposed)

### Milestone F: Pandoc (Minimal)
- [ ] Compile Pandoc with a subset of readers/writers (e.g., Markdown → HTML only)
- [ ] Skip optional dependencies (skylighting, texmath, etc.)
- [ ] Requires: All Tier 1 and Tier 2 items

### Milestone G: Pandoc (Full)
- [ ] Compile full Pandoc with all readers/writers
- [ ] Pass Pandoc's test suite
- [ ] Performance within 2x of GHC-compiled Pandoc

---

## Key Files Reference

| File | Role |
|------|------|
| `crates/bhc/src/main.rs` | CLI flags (`-c`, `--odir`, `--hidir`, `--numeric-version`, etc.) |
| `crates/bhc-session/src/lib.rs` | Session options (compile_only, odir, hidir, package_dbs) |
| `crates/bhc-driver/src/lib.rs` | Compilation orchestration, `compile_module_only()`, `.bhi` loading |
| `crates/bhc-codegen/src/llvm/lower.rs` | LLVM lowering — add builtin handlers here |
| `crates/bhc-typeck/src/builtins.rs` | Type signatures for all builtins |
| `crates/bhc-typeck/src/context.rs` | Type checker context, `register_lowered_builtins()` |
| `crates/bhc-lower/src/context.rs` | Lowering context, `DefInfo.type_scheme`, `define_with_type()` |
| `crates/bhc-interface/src/lib.rs` | Module interface types and serialization |
| `crates/bhc-interface/src/convert.rs` | `TypeConverter` (.bhi → internal types) |
| `crates/bhc-interface/src/generate.rs` | `.bhi` generation from compiled modules |
| `crates/bhc-core/src/eval/mod.rs` | Evaluator implementations |
| `crates/bhc-package/` | Package management (to be superseded by hx) |
| `stdlib/bhc-base/` | Base library RTS functions, Data.Char predicates |
| `stdlib/bhc-text/src/text.rs` | Text RTS (25+ FFI functions, E.7+E.8) |
| `stdlib/bhc-text/src/text_io.rs` | Text.IO RTS (7 FFI functions, E.10) |
| `stdlib/bhc-text/src/bytestring.rs` | ByteString RTS (24 FFI functions, E.8) |
| `stdlib/bhc-system/` | System/IO operations |
| `rts/bhc-rts/src/ffi.rs` | FFI functions for FilePath/Directory (E.19) |
| `stdlib/bhc-containers/` | Container data structures |
| `stdlib/bhc-transformers/` | Monad transformers |
| `rts/bhc-rts/` | Core runtime system |
| `crates/bhc-e2e-tests/fixtures/tier3_io/` | Transformer and IO test fixtures |

---

## Recent Progress

### 2026-02-24: ByteString.Builder

Implemented `Data.ByteString.Builder` with 16 new RTS functions and ~45 codegen
dispatch entries. Builder uses the same chunk-list representation as lazy ByteString
(`Empty | Chunk !BS LazyBS`), so `toLazyByteString` and `lazyByteString` are identity
operations. Many functions (empty, append, byteString, hPutBuilder) reuse existing
lazy ByteString RTS functions via codegen dispatch.

1. **RTS** (`builder.rs`): 16 functions (VarIds 1000478-1000493) — singleton, charUtf8,
   stringUtf8, intDec, char7/8, string7/8, word16/32/64 BE/LE, wordHex, word8HexFixed
2. **Codegen**: 16 LLVM declarations + ~45 arity/dispatch entries + composed helpers
   for float/double encoding (inline LLVM fptrunc/bitcast) and hex fixed-width
3. **Type system**: `builder_con`/`builder_ty` + ~62 function type signatures
4. **Lowering**: Fixed DefIds 11450-11498 + 11510-11521 (avoiding E.27 collision at 11500-11505)
5. **E2E test**: `builder_basic` — empty/singleton/intDec/append/stringUtf8 with length verification

175 total E2E tests pass (174 passed + 1 ignored). 88 bhc-text unit tests pass.

**Known limitation**: `hPutBuilder` expects PointerValue handle but integer sentinel
(2=stdout) is IntValue — needs int-to-ptr coercion (deferred).

### 2026-02-24: Lazy Text + Lazy ByteString

Implemented lazy Text and lazy ByteString as chunk-lists (matching GHC's
`Empty | Chunk !StrictVariant LazyVariant` representation) with 42 RTS functions
covering the Pandoc-essential subset.

1. **RTS** (`lazy_text.rs`): 14 functions (VarIds 1000270-1000283) — empty, fromStrict,
   toStrict, pack, unpack, null, length, append, fromChunks, toChunks, head, tail, take, drop
2. **RTS** (`lazy_bytestring.rs`): 28 functions (VarIds 1000440-1000477) — 20 core BS.Lazy
   functions + 6 Char8 functions (unpack, lines, unlines, take, dropWhile, cons) + 2
   encoding functions (encodeUtf8, decodeUtf8)
3. **Codegen**: 42 LLVM declarations + arity entries + dispatch entries in lower.rs
4. **Type system**: LazyText + LazyByteString type constructors + 42 function signatures
5. **Lowering**: Fixed DefIds 11380-11441 across 4 modules (Data.Text.Lazy,
   Data.ByteString.Lazy, Data.ByteString.Lazy.Char8, Data.Text.Lazy.Encoding)
6. **E2E tests**: `lazy_text_basic` and `lazy_bytestring_basic` (fromStrict/toStrict
   roundtrip, length, head, append, pack/unpack)

175 total E2E tests pass (173 existing + 2 new, 0 failures). 79 bhc-text unit tests pass.

**Bug found**: BHC's lexer treats `strict` as a keyword (it is NOT a Haskell keyword).
Using `let strict = ...` silently drops `main` from the AST. Workaround: use different
variable names. Parser fix deferred.

### 2026-02-24: Exception Hierarchy (SomeException / IOException / ErrorCall)

Added tagged `SomeException` struct to the RTS enabling type-filtered `catch`. Three
exception types: `SomeException` (tag 0, catch-all), `IOException` (tag 1), `ErrorCall`
(tag 2). Key changes:

1. **RTS** (`ffi.rs`): `bhc_make_some_exception`, `bhc_exc_get_tag`, `bhc_exc_get_payload`,
   `bhc_show_exception`, `bhc_catch_typed` (5-arg with type_tag). `bhc_catch` delegates to
   `bhc_catch_typed(tag=0)`. `bhc_readFile` wraps errors in `IOException`. 4 new unit tests.
2. **Codegen** (`lower.rs`): `lower_builtin_error` now throws catchable `ErrorCall` instead of
   `panic!`. `lower_builtin_throw` wraps in `SomeException`. `lower_builtin_catch` calls
   `bhc_catch_typed` with inferred type tag from handler parameter type.
3. **Type system**: `toException`, `fromException`, `displayException`, `userError`, `ioError`
   registered in builtins + context + Control.Exception exports.
4. **E2E test**: `exception_hierarchy` — catch-all, catchable `error`, IO error catch.

173 total E2E tests passed at time of that commit (172 existing + 1 new, 0 failures). 53 RTS unit tests pass.

### 2026-02-22: hx Build Pipeline Wiring

Wired the hx package manager's BHC backend (`hx-bhc`) to generate correct CLI
flags matching BHC's actual clap-based CLI. Six changes across the hx repo:

1. **compile.rs**: Fixed 7 flag formats (`--hidir`, `--odir`, `--import-path`,
   `--package-db`, `-O 2`, `--Wall`, `--Werror`) + `.hi` → `.bhi` extension
2. **native.rs**: Fixed `--package-db` and `--package` flags
3. **package_db.rs**: Replaced `bhc-pkg` subprocess calls with filesystem ops
   (directory scan of `.conf` files via `tokio::fs`)
4. **build.rs (CLI)**: Removed `--make`, fixed flag formats for direct BHC invocation
5. **builtin_packages.rs** (new): Maps 12 Haskell packages (base, text, containers,
   transformers, etc.) to BHC builtins with synthetic package IDs
6. **full_native.rs**: Integrated builtin package check in dependency builder

All 45 hx-bhc tests pass, full hx workspace builds cleanly, 162 BHC E2E tests unaffected.

### 2026-02-21: Separate Compilation + hx Integration (E.65–E.66)

**E.65: Layout Rule Verification** — Verified Haskell 2010 layout rule is fully complete.
All 162 E2E tests use indentation-based layout. 39 lexer unit tests cover edge cases.

**E.66: Separate Compilation Pipeline** — Two commits implementing the full GHC-compatible
compilation protocol needed by hx-bhc:

1. **Compile-only mode** (`e81cdac`): Added `--numeric-version`, `-c`, `--odir`/`--hidir`,
   `-package-db`/`-package-id` CLI flags. `compile_module_only()` runs parse → lower →
   typecheck → codegen, writes `.o` to odir and `.bhi` to hidir. Interface generation via
   `bhc-interface/generate.rs` walks AST/typed module to extract exported values, types,
   classes, instances, and constructors.

2. **Interface consumption** (`87b91a2`): `TypeConverter` bridge (`bhc-interface/convert.rs`)
   converts `.bhi` interface types to internal `Ty`/`Scheme`. Added `DefInfo.type_scheme`
   to carry type schemes from driver → lowering → type checker. `register_lowered_builtins()`
   now uses interface type schemes instead of fresh variables. `load_interfaces_for_imports()`
   searches hidir + package-db directories for `.bhi` files of imported modules.

**Impact on Pandoc gaps:** The BHC side of the package compilation protocol is now complete.
The hx package manager (separate repo) already has .cabal parsing, Hackage integration,
dependency resolution, and a BHC backend crate (`hx-bhc`). Remaining work is in the hx
repo: wire `hx build --backend bhc` through the full pipeline, create filesystem-based
package DB, and add BHC builtin package mapping.

### 2026-02-21: Roadmap Assessment (E.32–E.64)

33 milestones completed since last update, adding 66+ E2E tests. Major areas:

**Language Extensions (E.32–E.35):**
- OverloadedStrings + IsString (E.32), Record syntax with wildcards/puns (E.33)
- ViewPatterns codegen (E.34), TupleSections + MultiWayIf (E.35)

**Typeclass Revolution (E.38–E.42):**
- Manual instances (E.38), dictionary-passing (E.39), higher-kinded (E.40)
- Default methods + superclasses (E.41), DeriveAnyClass (E.42)

**Type System Extensions (E.43–E.50):**
- Word types + Integer + lazy let (E.43–E.45), ScopedTypeVariables (E.46)
- GeneralizedNewtypeDeriving (E.47), FlexibleInstances/Contexts (E.48)
- MultiParamTypeClasses (E.49), FunctionalDependencies (E.50)

**Deriving Infrastructure (E.51–E.54):**
- DeriveFunctor (E.51), DeriveFoldable (E.52), DeriveTraversable (E.53)
- DeriveEnum + DeriveBounded (E.54)

**Monad Transformers (E.55–E.57):**
- ReaderT-over-StateT (E.55), ExceptT cross-transformer (E.56), WriterT cross-transformer (E.57)

**Advanced Features (E.58–E.64):**
- Full lazy let-bindings (E.58), EmptyDataDecls + strict fields (E.59)
- GADTs with type refinement (E.60), TypeOperators (E.61)
- StandaloneDeriving + PatternSynonyms (E.62), DeriveGeneric + NFData stubs (E.63)
- EmptyCase + StrictData + DefaultSignatures + OverloadedLists (E.64)

**Impact on Pandoc gaps:** Items #1-5 from the original "Missing for Pandoc" list are now complete
(OverloadedStrings, Records, ViewPatterns, TupleSections/MultiWayIf, GeneralizedNewtypeDeriving).
Remaining blockers: package system, CPP preprocessing, full GHC.Generics.

### 2026-02-12: Milestones E.25–E.31
- E.25: String read/readMaybe/fromString (82 tests)
- E.26: 10 RTS list functions: sortOn, nubBy, groupBy, etc. (83 tests)
- E.27: succ/pred/(&)/swap/curry/uncurry (85 tests)
- E.28: 14 builtins (min/max/subtract/enum/folds/comparing/until/IO input), partial builtin application (87 tests)
- E.29: flip fix (flat 3-arg), show Double/Float, Data.Map.mapMaybe (90 tests)
- E.30: Unified Bool extraction (extract_bool_tag) for 7 list functions (93 tests)
- E.31: Recursive ShowTypeDesc for nested compound show (96 tests)

### 2026-02-11: Milestones E.20–E.24
- E.20: Fixed DefId misalignment for Text/ByteString/exceptions (74 tests)
- E.21: Data.Map completion (update/alter/unions), Bool ADT fixes (76 tests)
- E.22: Data.Set/IntMap/IntSet type completion, VarId suffix bug fix (78 tests)
- E.23: Stock deriving Eq/Show for user ADTs (80 tests)
- E.24: Stock deriving Ord, polymorphic compare (81 tests)

### 2026-02-09: Milestones E.15–E.19
- E.15: Data.List completions (scanr, unfoldr, zip3, iterate, repeat, cycle) (66 tests)
- E.16: Fix broken stubs + 10 new list operations (69 tests)
- E.17: Ordering ADT with compare (70 tests)
- E.18: 7 monadic combinators (70 tests)
- E.19: System.FilePath + System.Directory (72 tests)

### 2026-02-07: Milestones E.11–E.14
- E.11: Show compound types (52 tests)
- E.12: Numeric conversions + IORef (55 tests)
- E.13: Data.Maybe + Data.Either + guard (58 tests)
- E.14: when/unless + any/all + closure wrapping (61 tests)

### 2026-02-05–07: Milestones E.7–E.10
- E.7: Data.Text packed UTF-8 (43 tests)
- E.8: Data.ByteString + Text.Encoding (43 tests)
- E.9: Data.Char predicates + type-specialized show (45 tests)
- E.10: Data.Text.IO (46 tests)

### 2026-02-05: Milestones A–E (Foundations)
- Milestone A: Multi-module compilation
- Milestone B: File processing (word count, transform)
- Milestone C: Markdown parser (~500 LOC)
- Milestone D: StateT-based CSV parser
- Milestone E: JSON parser
- Nested transformer codegen (StateT over ReaderT)
- Exception handling (catch, bracket, finally)
