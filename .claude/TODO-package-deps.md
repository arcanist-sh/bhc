# Feature brief: Package dependency resolution for `bhc check` / compile

**Status:** Not started. Self-contained brief for a fresh session.
**Why:** When checking a real package, most modules come back **SKIPPED**, not
failed — they import modules from *other packages* (dependencies) that aren't
on disk in the package's own `src/`, so the driver can't satisfy the import and
skips the module. Resolving transitive deps is the gate to real end-to-end
Hackage compilation. (Measured this session: e.g. optparse-applicative checked
as 0 passed / 1 failed / **16 skipped of 17**; vector-algorithms 0/0/**11 of 11**.)

---

## Goal

Given a package (or `.cabal` / a module that imports external packages), make
`bhc check` (and ideally `bhc -c` compile) **fetch/locate, build, and register
the transitive dependencies** so their modules are available for import
resolution — turning today's "skipped" modules into checked ones.

Minimal first milestone: `bhc check <pkg>` resolves deps that are *already
available locally* (a provided package set / dirs), so skipped→checked without
network. Then layer on hx-driven fetch.

---

## How it works today (the mechanism to extend)

All in `crates/bhc-driver/src/lib.rs`:

- **Entry:** `check_with_discovery(paths)` (~4231) expands dirs to `.hs` files,
  calls `check_files_ordered(files)` (~3936).
- **Skip logic:** `check_files_ordered` builds `module_info: Vec<(path, name,
  imports)>`, then `builtin_module_set()` (~3544, the stdlib/builtin module
  names) and `local_names` (the modules in the file set). A module is
  **satisfiable** iff every import is in `local_names` OR `builtins`
  (~4010-4017). Unsatisfiable modules are effectively skipped.
- **Ordering:** `topological_sort` (~3478) orders satisfiable modules deps-first;
  cyclic groups get a pre-registration pass (AST-only export extraction via
  `build_module_exports_from_ast`, ~1785).
- **Per-module check:** `check_unit_for_multimodule` (~1933) → `lower_with_registry`
  (~2097) seeds a `ModuleCache` from the `ModuleRegistry` (each
  `CompiledModuleInfo` carries `exports`, remapped to fresh DefIds). Type-checks,
  then builds exports for downstream modules (`build_module_exports_from_hir`).
- **Builtin stubs:** modules in `builtin_module_set` are NOT read from disk —
  the lowering stub mechanism (`LowerContext::with_builtins`) resolves their
  qualified names as stubs (warning: "stub function ... used"). This is why
  `T.pack`/`Map.insert` resolve even though Data.Text has no source. NOTE: stubs
  cover VALUES but not external ADT **constructors** (e.g. pandoc-types `Div`),
  which is the separate Pandoc cascade — see [[project_pandoc_check_state]].
- **Import search:** `lower_with_registry` resolves on-disk imports via
  `search_paths` = `session.options.import_paths` + `stdlib_path` +
  `BHC_STDLIB_PATH` (it does NOT include the directories passed to `check`).
- **Separate compilation already exists:** `bhc -c` produces `.bhi` interface
  files; flags `--odir`/`--hidir`/`--package-db`/`--numeric-version`. So the
  cross-module *interface* machinery is in place — the gap is **acquiring and
  wiring the dependency modules**.
- **hx is the package manager** (separate tool, see .claude/CLAUDE.md): it has
  `.cabal` parsing, a dependency solver, Hackage fetch, a filesystem-based
  package DB, a `bhc` backend crate (`hx-bhc`) that generates the right BHC CLI
  flags, and a map of standard packages → BHC builtins. End-to-end testing is
  listed as the remaining gap (CLAUDE.md Phase 8.3 / Phase 9.8).

---

## What to build (suggested increments)

1. **Dependency module roots in the check path (no network).** Let `bhc check`
   take dependency source roots (e.g. `--package-dir <dir>` repeatable, or read
   a package DB) and include their modules in discovery so imports resolve.
   Concretely: thread extra source roots into `check_with_discovery` /
   `check_files_ordered` so `local_names` includes dep modules and `search_paths`
   includes their dirs. Re-measure a package's skipped count.
   - Quick validation harness without hx: `bhc check <pkg>/src <dep1>/src <dep2>/src …`
     should already pull deps in as "local" (multi-path discovery). Try this
     first to confirm the registry/import path works before adding flags — it
     may "just work" for vendored deps and quantify the upside.
2. **Consume hx's package DB.** Have `bhc check`/compile read hx's
   filesystem package DB (compiled `.bhi` + artifacts) so a `cabal`-resolved dep
   set is used. hx already builds this; wire BHC to consult it for imports.
3. **hx-driven fetch + build of transitive deps**, then check/compile the target
   against them. Reuse hx-solver (`.cabal` parse, Hackage fetch, solver) + the
   `-c`/`.bhi` pipeline. This is the real end-to-end Hackage milestone.

---

## Gotchas

- **Builtin shadowing:** a module that is BOTH in `builtin_module_set` and
  provided as a dep source — decide precedence. For real deps you likely want
  on-disk source to win (so real constructors resolve), but many "builtin"
  modules (Data.Map etc.) are intentionally stubbed and faster left stubbed.
  Make it explicit.
- **DefId remapping:** `lower_with_registry` already remaps each module's export
  DefIds to fresh ids to avoid cross-module collisions — preserve that when
  adding dep modules.
- **VarId non-uniqueness across modules** has bitten codegen before (cross-module
  constructor-tag bug, alpha-rename-on-inline) — watch for it when compiling deps.
- **TH / unsupported extensions** in deps (aeson, pandoc-types) will still fail
  to compile from source — those need TH or stubs regardless (orthogonal).
- **Cycles:** `.hs-boot` mutual recursion is unsupported (TODO-pandoc.md 0.12);
  the current cycle handling is a best-effort AST-export pre-pass.

## How to test

- `bhc check <pkg>/src <dep>/src` (multi-path) — does the skipped count drop?
- Pick a small pure dependency chain (e.g. `split` depends only on base) and a
  consumer; verify the consumer's imports resolve once `split` is provided.
- Watch for cascade: as deps resolve, previously-skipped modules may surface new
  real bugs — that's the point (it's how the parser-bug mining worked).

See also: [[project_pandoc_check_state]] (pandoc-types cascade), the
`bhc check` mechanics above, and `crates/bhc-driver/src/lib.rs`.
