# Feature brief: Package dependency resolution for `bhc check` / compile

**Status:** Milestones 1 & 2 done (no-network dependency roots + package DB
`.bhi` consumption). Milestone 3 (hx-driven fetch/build of transitive deps) not
started. Self-contained brief.

**Milestone 2 (landed):** `bhc check` resolves imports from a package DB of
compiled `.bhi` interfaces (separate compilation), and the compile side actually
produces such a DB.
- Read side: the satisfiability filter in `check_files_ordered_reporting` now
  treats an import as satisfiable if a `.bhi` for it exists in any
  `--package-db <dir>` or the `--hidir` (predicate `package_provides`), so the
  module is checked instead of skipped; lowering loads the interface via the
  existing `load_interfaces_for_imports`. `check_file` (single-file path) now
  uses `lower_with_registry` (empty registry) so it loads interfaces too.
  `--package-db` is global (works before/after the subcommand) and is wired into
  the `check` CLI builder.
- Write side: single-file `bhc -c Foo.hs --hidir <db> --odir <o>` now routes to
  `compile_module_only` (was falling through to the executable path, ignoring
  both dirs and emitting no `.bhi`). Two fixes made this work: (a)
  `compile_files_ordered` honors `compile_only` in its ≤1-file branch; (b)
  `compile_module_only` writes the interface using the **AST module name**
  (`Data.Split` → `<db>/Data/Split.bhi`), not the file stem. `compile_module_only`
  also uses `lower_with_registry` so a dep that imports an already-built dep
  resolves from the DB. The `build` subcommand path (`compile_files`) now also
  wires `-c`/`--odir`/`--hidir`/`--package-db`.
- Validated end-to-end: build `Data.Split` into a DB with `-c`, then check a
  consumer (and a 2-module Helper/Main package, source of the dep absent) against
  `--package-db` → resolves OK. Driver tests
  `test_package_db_bhi_roundtrip_resolves_check` and
  `test_package_dir_deps_resolve_and_are_unreported`.

**Milestone 1 (landed):** `bhc check` accepts `--package-dir <DIR>` (repeatable,
global so it works before or after the subcommand). Each dir's `.hs` modules are
parsed/ordered/checked alongside the target so the target's imports resolve, but
the dependency modules are **not** reported in the results — only modules reached
from the target paths are. Implementation: `Compiler::check_files_ordered_reporting`
(driver) takes a `report_only: Option<&FxHashSet<Utf8PathBuf>>` filter; the public
`check_with_discovery_with_deps(paths, dep_roots)` collects dep `.hs` files,
prepends them to the file set, and passes the target set as the report filter.
The registry/`ModuleCache` (topological order) already makes dep exports available
to importers; package dirs are also added to `import_paths` for on-disk fallback.
Validated: a 2-module target importing a `Data.Split` dep goes from 2 skipped →
2 OK with the dep unreported (driver test
`test_package_dir_deps_resolve_and_are_unreported`).
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

1. **Dependency module roots in the check path (no network). — DONE.**
   Implemented as `--package-dir <dir>` (repeatable, global). See the
   Milestone 1 note at the top. The multi-path harness
   (`bhc check <pkg>/src <dep>/src`) also works and was confirmed — but it
   *reports* the dep modules too; `--package-dir` is the version that keeps the
   target's result set clean. Next: re-measure a real Hackage package's skipped
   count by pointing `--package-dir` at its (vendored/already-fetched) deps.
2. **Consume hx's package DB. — DONE (incl. hx `.conf` layout).** `bhc check`/`-c`
   read a package DB of `.bhi` interfaces (`--package-db`, global) and `-c`
   produces them. Both layouts are supported (driver `package_interface_dirs`):
   the flat `<db>/Module/Path.bhi` layout from `bhc -c --hidir <db>`, AND hx's
   layout where the DB dir holds `<package-id>.conf` files and the actual `.bhi`
   live in each conf's `import-dirs:` (parsed by `parse_conf_import_dirs`). `-c`
   now also writes the object nested by module path (`<odir>/Data/Split.o`) to
   match hx's `compile_module` expectation. Verified against a synthetic `.conf`
   matching hx `generate_registration_file` exactly (driver test
   `test_package_db_conf_import_dirs_resolve_check`).
   (a) Real hx-built DB end-to-end — DONE. Drove hx's actual
   `hx_bhc::package_build::build_package` on a local cabal package with this BHC
   binary (per-module `bhc -c`, static lib, `.conf` generation) → then
   `bhc check --package-db <db>` resolves a consumer with the dep source absent.
   This surfaced and fixed a real hx bug: `build_package` left `.bhi` in
   `<build>/hi` while the `.conf` declares `import-dirs: <install>/lib` and never
   installed them — fixed in hx (`fix/bhc-install-interfaces`, new
   `install_interfaces`). Gated test (BHC_PATH): hx-bhc
   `test_build_package_then_check_consumer`.
   (b) exposed-modules gating + (c) --package-id scoping — DONE. Driver
   `collect_package_interfaces` returns flat dirs (hidir + db dirs, any `.bhi`
   resolves — backward compat) and parsed `.conf` packages
   (`ConfPackage`/`parse_conf_package`: id, exposed-modules, import-dirs).
   `resolve_interface_in` only resolves a module from a package if it is in that
   package's `exposed-modules`; when `--package-id` flags are given only matching
   package ids are visible (empty ⇒ all visible). `--package-id` is now global and
   wired into `check`. hx reconciled to pass `--package-id` (double dash) in
   compile/native_builder/full_native (was `-package-id`). Tests: driver
   `test_package_db_scoping_and_exposed_modules`, `test_parse_conf_package`,
   `test_resolve_interface_in_honors_exposed_modules`; hx e2e
   `test_build_package_then_check_consumer` now asserts positive + negative
   `--package-id` scoping.
   `depends:` transitive visibility — DONE. `collect_package_interfaces` parses
   every `.conf` in the DBs, and when `--package-id` ids are given the visible set
   is the transitive closure over `depends:` (selecting P also exposes P's deps;
   selecting a dependency does not expose its dependents). `ConfPackage.depends`
   parsed from the `.conf`; hx's `generate_registration_file` already emits
   `depends:`. Test: driver `test_package_db_depends_transitive_visibility`.
   REPL flag reconcile — DONE. hx's repl.rs ran the non-existent `bhc
   --interactive` with GHC-style `-package-db=`/`-i<dir>`/`--tensor-fusion`; it
   now invokes `bhc repl` with BHC spellings (`--profile`, `--import-path`,
   `--package-db`). `bhc repl` (`start_repl`) forwards `--package-db`/
   `--package-id`/`--import-path` to bhci; bhci now accepts those flags, resolves
   `:load` targets against `--import-path`, and lists them under `:show packages`.
   (bhci's evaluator is still stubbed and does not consume `.bhi` for import
   *resolution* — that's the separate REPL-eval gap, not this flag reconcile.)
   Tests: hx `test_repl_args_construction`/`test_repl_args_minimal`; verified
   `:load Foo.hs` resolves via `--import-path` end-to-end.
   `--tensor-fusion` flag mismatch — DONE. `bhc` now defines `--tensor-fusion`
   (global; `Options.tensor_fusion`, builder setter, wired into both compile
   paths). It OR's into the `compile_unit` tensor-pipeline gate
   (`profile == Numeric || tensor_fusion`); on the `-c` per-module path it is
   simply accepted (that path has no tensor block). hx already passes the
   double-dash spelling, so no hx change needed. Verified: hx-style
   `bhc -c --tensor-fusion`, `bhc check --tensor-fusion`, and a full default-
   profile build with `--tensor-fusion` all succeed (exe runs).
3. **hx-driven build of transitive deps**, then check/compile the target against
   them — core DONE (non-network). hx's `BhcFullNativeBuilder::build_project`
   (`hx/crates/hx-bhc/src/full_native.rs`) already walks the resolved build plan
   in topological order: builds each package with `build_package`, registers its
   `.conf` in a `BhcPackageDb`, then builds the local project with `--package-db`
   + `--package-id` per dep. Fixed a real bug: `build_dependency` now threads the
   in-progress package DB into each dependency's `--package-db`, so a dep can
   resolve already-built deps (it previously only passed user DBs). Verified
   end-to-end over a local `Data.Base <- Data.Mid <- app` chain (gated test
   `test_transitive_dependency_build_and_check`): libbase then libmid (libmid's
   compile resolves Data.Base via the DB), then `bhc check app --package-db <db>
   --package-id <libmid>` resolves the whole `depends:` closure.
   CLI wiring — DONE. `hx build --backend bhc --native` (`run_bhc_native_build`)
   now calls `try_bhc_full_native_build`: when the project has external deps and
   a cached resolution (`hx lock`), it resolves → fetches → plans → drives
   `BhcFullNativeBuilder::build_project` into a `BhcPackageDb`, then links the
   project against it. BHC builtins are marked pre-installed (from
   `builtin_packages()`) so they aren't fetched/built. It falls back to the
   local-only build when there are no deps, no lockfile, or fetch fails (so the
   no-network/no-lock case is unchanged). `BhcCompilerConfig` gained `Clone`.
   Remaining (network only): an actual run against real Hackage — the
   solver→fetch→extract path needs network + a populated index, so the
   dependency branch is compile-checked and fallback-verified here but not
   executed end-to-end. Run `hx lock && hx build --backend bhc --native` on a
   machine with network to exercise it.

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
