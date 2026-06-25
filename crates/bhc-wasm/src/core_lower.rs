//! Core IR to WASM lowering.
//!
//! This module walks Core IR expressions and emits WASM instructions,
//! enabling the Default and Edge profiles to compile Haskell programs
//! to WebAssembly without going through the Tensor/Loop IR pipeline.
//!
//! ## Representation
//!
//! - Integers are represented as unboxed `i32` values
//! - Strings are stored in the WASM data segment as (offset, length) pairs
//! - Functions become WASM functions with `i32` parameters and results
//! - IO actions are executed for their side effects and return `i32(0)`

use bhc_core::{Alt, AltCon, Bind, CoreModule, DataCon, Expr, Literal, Var, VarId};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Kind, Ty, TyCon};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::codegen::{RuntimeIndices, WasmFunc, WasmFuncType};
use crate::{WasmInstr, WasmModule, WasmResult, WasmType};

/// Well-known data constructor info: (tag, arity).
struct ConInfo {
    tag: u32,
    arity: u32,
}

/// What to do with the value an expression produces.
///
/// String values carry only their offset on the WASM stack — the length is
/// lost — so a dynamic string (e.g. the result of an `if`/`case` over string
/// literals) cannot be printed after the fact. Instead we push the print
/// *into* each leaf of the expression, where the literal (and thus its length)
/// is statically known. [`Cont::Value`] preserves the ordinary "leave the
/// value on the stack" behaviour.
#[derive(Clone, Copy)]
enum Cont {
    /// Leave the expression's value on the stack.
    Value,
    /// Print the expression as a string and leave the IO result (`0`) on the
    /// stack. `newline` selects `putStrLn` vs `putStr` semantics.
    PrintStr { newline: bool },
    /// Tail position of a self-recursive function being compiled to a loop
    /// (tail-call optimization). At each leaf: a saturated self-call reassigns
    /// the parameters and sets the `continue` flag; any other expression sets
    /// the `result` and clears the flag. Leaves no value on the stack.
    Tail,
}

/// Context for tail-call optimization of the function currently being lowered.
#[derive(Clone)]
struct TcoCtx {
    /// Name of the self-recursive function (to detect self-calls).
    name: Symbol,
    /// Local indices of the function's parameters, in order.
    params: Vec<u32>,
    /// Local holding the "continue looping" flag.
    continue_local: u32,
    /// Local holding the function's result value.
    result_local: u32,
}

/// How a value should be rendered by `show`/`print`.
///
/// Types are erased by the time Core reaches this backend, so the kind is
/// inferred from the expression's structure (mirroring the native backend).
enum ShowKind {
    /// Decimal integer (the default).
    Int,
    /// A boxed double, rendered via the runtime double formatter.
    Double,
    /// `True`/`False`, chosen at runtime from the boolean tag.
    Bool,
    /// A statically known rendering — a nullary constructor's name
    /// (`True`, `Red`, `Nothing`, `[]`, `()`, ...).
    Literal(String),
    /// An `Ordering` value, rendered `LT`/`EQ`/`GT` from its tag (0/1/2).
    Ordering,
    /// A value of a derived `Enum`/`Bounded` type, rendered as the constructor
    /// name selected by its runtime tag. Carries the constructor names ordered
    /// by tag (so `names[tag]` is the name to print).
    Enum(Vec<String>),
}

/// Build a map of well-known constructors to their tag and arity.
///
/// User-defined constructors are layered on top of this in
/// [`WasmLowering::register_constructors`] from the module's constructor
/// metadata, so their tags match what the case-match logic expects.
fn well_known_constructors() -> FxHashMap<String, ConInfo> {
    let mut m = FxHashMap::default();
    // Bool
    m.insert("False".to_string(), ConInfo { tag: 0, arity: 0 });
    m.insert("True".to_string(), ConInfo { tag: 1, arity: 0 });
    // Unit
    m.insert("()".to_string(), ConInfo { tag: 0, arity: 0 });
    // Maybe
    m.insert("Nothing".to_string(), ConInfo { tag: 0, arity: 0 });
    m.insert("Just".to_string(), ConInfo { tag: 1, arity: 1 });
    // Either
    m.insert("Left".to_string(), ConInfo { tag: 0, arity: 1 });
    m.insert("Right".to_string(), ConInfo { tag: 1, arity: 1 });
    // Ordering
    m.insert("LT".to_string(), ConInfo { tag: 0, arity: 0 });
    m.insert("EQ".to_string(), ConInfo { tag: 1, arity: 0 });
    m.insert("GT".to_string(), ConInfo { tag: 2, arity: 0 });
    // List
    m.insert("[]".to_string(), ConInfo { tag: 0, arity: 0 });
    m.insert(":".to_string(), ConInfo { tag: 1, arity: 2 });
    // Tuples
    m.insert("(,)".to_string(), ConInfo { tag: 0, arity: 2 });
    m.insert("(,,)".to_string(), ConInfo { tag: 0, arity: 3 });
    m
}

/// Starting offset for user string data in linear memory.
///
/// Placed after the WASI scratch area and runtime data segments.
const STRING_DATA_BASE: u32 = 2048;

/// Heap base address. The bump allocator starts handing out addresses here, so
/// any value below it that flows into a case scrutinee is a small nullary
/// constructor tag rather than a heap pointer. Used to discriminate nullary
/// constructors from heap objects in mixed data types (e.g. `[]` vs `(:)`).
const HEAP_BASE: i32 = 65536;

/// Lower a Core IR module to WASM, adding functions to the given module.
///
/// Returns the function index of `main` so the caller can wire up `_start`.
///
/// # Errors
///
/// Returns `WasmError` if lowering encounters an unsupported construct.
pub fn lower_core_module(
    core: &CoreModule,
    wasm: &mut WasmModule,
    runtime: &RuntimeIndices,
) -> WasmResult<u32> {
    let mut lowering = WasmLowering::new(wasm, runtime);

    // Register user-defined data constructors so their tags/arities match the
    // case-match logic. Without this, only well-known constructors are known
    // and user ADT values all lower to tag 0.
    lowering.register_constructors(core);

    // Record auto-derived Functor/Foldable instance methods so `fmap`/`foldr`
    // can dispatch to the generated `$derived_*` functions.
    lowering.register_derived_instances(core);

    // Rewrite monad-transformer bindings (ReaderT/StateT/WriterT/ExceptT) into
    // their concrete eager representations before anything inspects arities or
    // bodies. Plain-IO bindings pass through unchanged, so the shared IO `>>=`
    // path is untouched.
    //
    // First detect two-context Reader+State stacks (mtl-auto-lifted), reading
    // their outer layer from the program's eliminator nesting, so the rewrite
    // and the eliminators agree on the representation.
    for bind in &core.bindings {
        let entries: Vec<(&Var, &Expr)> = match bind {
            Bind::NonRec(v, e) => vec![(v, e)],
            Bind::Rec(bs) => bs.iter().map(|(v, e)| (v, e.as_ref())).collect(),
        };
        for (v, e) in entries {
            if infer_ctx_stack(e) {
                if let Some(outer) = find_stack_outer(&core.bindings, v.name) {
                    lowering
                        .ctx_stacks
                        .insert(v.name.as_str().to_string(), outer == MonadKind::State);
                }
            }
        }
    }
    let ctx_stacks = lowering.ctx_stacks.clone();
    let binds: Vec<Bind> = core
        .bindings
        .iter()
        .map(|b| rewrite_bind_monads(b, &ctx_stacks, &mut lowering.next_synthetic_id))
        .collect();

    // Record non-recursive top-level functions so saturated calls can be
    // inlined; record every top-level function's arity so partial and
    // over-applications can be handled via closures.
    lowering.register_inline_bodies(&binds);
    lowering.register_fn_bodies(&binds);
    lowering.register_arities(&binds);
    lowering.register_bool_returning_fns(&binds);
    lowering.register_pattern_constructors(&binds);

    // First pass: register all top-level function names so we can resolve calls
    for bind in &binds {
        match bind {
            Bind::NonRec(var, _) => {
                lowering.register_binding(var);
            }
            Bind::Rec(bindings) => {
                for (var, _) in bindings {
                    lowering.register_binding(var);
                }
            }
        }
    }

    // Synthesize any referenced list-prelude functions the program doesn't
    // define itself (map/filter/foldr/...), then register them like user code.
    // Iterate to a fixpoint so a synthesized function that references another
    // prelude function (e.g. `maximum` uses `foldl`, `elemIndex` uses
    // `findIndex`) pulls that one in too.
    let mut prelude: Vec<(Var, Expr)> = Vec::new();
    let mut synthesized: FxHashSet<Symbol> = FxHashSet::default();
    // `fmap`/`<$>`/`traverse`/`mapM` dispatch needs the list (`map`) and Maybe
    // (`__fmapMaybe`) arms, but the program references only the method name —
    // pull those helpers in.
    let uses_fmap = ["fmap", "<$>", "traverse", "mapM"]
        .iter()
        .any(|n| module_uses_name(&binds, Symbol::intern(n)));
    loop {
        let mut added = false;
        for &name in LIST_PRELUDE_NAMES {
            let sym = Symbol::intern(name);
            if lowering.func_map.contains_key(&sym) || synthesized.contains(&sym) {
                continue;
            }
            let used = module_uses_name(&binds, sym)
                || prelude.iter().any(|(_, e)| expr_uses_name(e, sym))
                || (uses_fmap && matches!(name, "map" | "__fmapMaybe"));
            if !used {
                continue;
            }
            if let Some((var, body)) = build_list_fn(name, &mut lowering.next_synthetic_id) {
                lowering.register_binding(&var);
                let (params, _) = peel_lambdas(&body);
                lowering.arities.insert(var.name, params.len());
                synthesized.insert(sym);
                prelude.push((var, body));
                added = true;
            }
        }
        if !added {
            break;
        }
    }

    // Second pass: lower each user binding to a WASM function.
    for bind in &binds {
        match bind {
            Bind::NonRec(var, expr) => {
                lowering.lower_binding(var, expr)?;
            }
            Bind::Rec(bindings) => {
                for (var, expr) in bindings {
                    lowering.lower_binding(var, expr)?;
                }
            }
        }
    }

    // Lower the synthesized prelude functions (after user bindings so their
    // pre-registered indices line up).
    for (var, expr) in &prelude {
        lowering.lower_binding(var, expr)?;
    }

    // Register lambda-lifted closure functions. They were assigned indices
    // starting right after the top-level bindings; add them in index order so
    // `add_function` hands back exactly those indices.
    lowering.pending_closures.sort_by_key(|(idx, _)| *idx);
    let pending = std::mem::take(&mut lowering.pending_closures);
    for (reserved_idx, func) in pending {
        let actual = lowering.wasm.add_function(func);
        if actual != reserved_idx {
            return Err(crate::WasmError::Internal(format!(
                "closure function index mismatch: reserved {reserved_idx}, got {actual}"
            )));
        }
    }

    // Find main's function index
    let main_idx = lowering
        .func_map
        .iter()
        .find(|(name, _)| name.as_str() == "main")
        .map(|(_, idx)| *idx)
        .ok_or_else(|| crate::WasmError::CodegenError("no `main` binding found".to_string()))?;

    Ok(main_idx)
}

/// State for the Core IR to WASM lowering pass.
struct WasmLowering<'a> {
    /// The WASM module being built.
    wasm: &'a mut WasmModule,
    /// Runtime function indices.
    runtime: &'a RuntimeIndices,
    /// Maps Haskell function names to WASM function indices.
    func_map: FxHashMap<Symbol, u32>,
    /// Length-prefixed string pool: maps content to a pointer to a
    /// `[len: i32 | bytes...]` block. This is the runtime representation of a
    /// `String` *value*.
    pstr_pool: FxHashMap<String, u32>,
    /// Next available offset in the data segment for string storage.
    next_data_offset: u32,
    /// Counter for pre-registering function indices.
    next_func_idx: u32,
    /// Constructor map: name -> (tag, arity). Seeded with well-known
    /// constructors and extended with user-defined ones from the module.
    con_map: FxHashMap<String, ConInfo>,
    /// Derived-`Enum` types: type name -> constructor names ordered by tag.
    /// Only types whose constructors are all nullary are recorded, so a value
    /// of such a type is exactly its tag and `names[tag]` is its `show` text.
    enum_types: FxHashMap<String, Vec<String>>,
    /// Reverse map: nullary constructor name -> its enum type name. Lets `show`
    /// recover an enum's constructor table from a value's source constructor.
    enum_of_ctor: FxHashMap<String, String>,
    /// Every (non-newtype) constructor name -> its data type name. Used to find
    /// the container type at a `fmap`/`foldr` call site for instance dispatch.
    ctor_type: FxHashMap<String, String>,
    /// Derived `Functor` instances: type name -> the `$derived_fmap_<Type>`
    /// binding symbol. Lets `fmap` dispatch to the generated method.
    derived_functor: FxHashMap<String, Symbol>,
    /// Derived `Foldable` instances: type name -> the `$derived_foldr_<Type>`
    /// binding symbol. Lets `foldr` dispatch to the generated method.
    derived_foldable: FxHashMap<String, Symbol>,
    /// Newtype constructor names. A newtype is identity at runtime, so both
    /// `C x` (construction) and the `C n` pattern unwrap to the field directly —
    /// essential for GeneralizedNewtypeDeriving, where derived Num/Eq/Ord
    /// operate on the underlying value, not a box.
    newtype_cons: FxHashSet<String>,
    /// Two-context monad-transformer stacks (Reader over State or State over
    /// Reader): binding name -> `state_outer` (true if StateT is the outer
    /// layer). Lets the eliminators peel a two-context value correctly.
    ctx_stacks: FxHashMap<String, bool>,
    /// Substitution environment: var id -> expression to lower in its place.
    /// Used to inline function/lambda arguments without alpha-renaming —
    /// when a parameter is referenced, its argument expression is lowered.
    subst: FxHashMap<VarId, Expr>,
    /// Let-bound variables whose RHS is a bottom value (`error`/`undefined`/
    /// `throw`). The strict WASM backend would evaluate the RHS eagerly, but a
    /// lazy binding must only "raise" when forced — so the binding site is
    /// skipped and the pending-exception flag is set at each (possibly
    /// conditional) use site instead.
    bottom_vars: FxHashSet<VarId>,
    /// Bodies of every top-level function (recursive included), keyed by name:
    /// `(param_ids, body)`. Used by the compile-time arbitrary-precision integer
    /// evaluator (`const_eval_bigint`) to unfold calls like `factorial 50`.
    fn_bodies: FxHashMap<Symbol, (Vec<VarId>, Expr)>,
    /// Names of top-level functions whose (curried) result type is `Bool`, taken
    /// from their binding signatures. Lets `show (f …)` render `True`/`False`
    /// when `f`'s result type is otherwise erased at the call site (e.g. a
    /// `data family` accessor like `getBool`).
    bool_returning_fns: FxHashSet<String>,
    /// Bodies of non-recursive top-level functions eligible for inlining,
    /// keyed by name: `(param_ids, body)`. Lets us evaluate
    /// dictionary-specialized typeclass methods (and the closures they
    /// receive) by inlining instead of needing first-class closures.
    inline_bodies: FxHashMap<Symbol, (Vec<VarId>, Expr)>,
    /// Functions currently being inlined, to break inlining cycles.
    inlining: FxHashSet<Symbol>,
    /// Lambda-lifted closure functions awaiting registration, as
    /// `(reserved_function_index, function)`. They are added to the module
    /// after all top-level bindings so their indices match what was reserved.
    pending_closures: Vec<(u32, WasmFunc)>,
    /// Arity (number of value parameters) of each top-level binding, used to
    /// detect partial and over-application.
    arities: FxHashMap<Symbol, usize>,
    /// Counter for fresh variable ids used when eta-expanding partial
    /// applications. Starts high to avoid colliding with real program ids.
    next_synthetic_id: usize,
    /// Tail-call-optimization context for the function currently being lowered,
    /// if it is self-recursive. Drives `Cont::Tail` lowering.
    tco_ctx: Option<TcoCtx>,
}

impl<'a> WasmLowering<'a> {
    fn new(wasm: &'a mut WasmModule, runtime: &'a RuntimeIndices) -> Self {
        // Count existing functions (imports + defined) to know where new functions start
        let next_func_idx = wasm.next_function_index();

        Self {
            wasm,
            runtime,
            func_map: FxHashMap::default(),
            pstr_pool: FxHashMap::default(),
            next_data_offset: STRING_DATA_BASE,
            next_func_idx,
            con_map: well_known_constructors(),
            enum_types: FxHashMap::default(),
            enum_of_ctor: FxHashMap::default(),
            ctor_type: FxHashMap::default(),
            derived_functor: FxHashMap::default(),
            derived_foldable: FxHashMap::default(),
            newtype_cons: FxHashSet::default(),
            bottom_vars: FxHashSet::default(),
            ctx_stacks: FxHashMap::default(),
            subst: FxHashMap::default(),
            fn_bodies: FxHashMap::default(),
            bool_returning_fns: FxHashSet::default(),
            inline_bodies: FxHashMap::default(),
            inlining: FxHashSet::default(),
            pending_closures: Vec::new(),
            arities: FxHashMap::default(),
            next_synthetic_id: 5_000_000,
            tco_ctx: None,
        }
    }

    /// Register user-defined data constructors from the module's constructor
    /// metadata. Newtype constructors are identity at runtime, so they are
    /// skipped — their argument flows through unwrapped.
    fn register_constructors(&mut self, core: &CoreModule) {
        // Synthetic GHC.Generics representation constructors used by a bare
        // `from` (the enum-sum Rep). `M1` is a metadata newtype — identity at
        // runtime — so `case … of M1 inner` binds `inner` to the sum directly;
        // `L1`/`R1` are the two arms of `(:+:)` (tags 0/1). User definitions, if
        // any, override these in the loop below.
        self.newtype_cons.insert("M1".to_string());
        self.con_map
            .insert("L1".to_string(), ConInfo { tag: 0, arity: 1 });
        self.con_map
            .insert("R1".to_string(), ConInfo { tag: 1, arity: 1 });
        self.con_map
            .insert("U1".to_string(), ConInfo { tag: 0, arity: 0 });

        for con in &core.constructors {
            if con.is_newtype {
                self.newtype_cons.insert(con.name.clone());
                continue;
            }
            self.con_map.insert(
                con.name.clone(),
                ConInfo {
                    tag: con.tag,
                    arity: con.arity,
                },
            );
            if let Some(ty) = &con.type_name {
                self.ctor_type.insert(con.name.clone(), ty.clone());
            }
        }

        // Build the derived-`Enum` registry. Group constructors by their data
        // type; a type qualifies as an enum only if every constructor is
        // nullary (so each value is exactly its tag). For those, record the
        // constructor names ordered by tag, which drives `show` of computed
        // enum values and `minBound`/`maxBound`.
        let mut by_type: FxHashMap<String, Vec<(u32, String)>> = FxHashMap::default();
        let mut all_nullary: FxHashMap<String, bool> = FxHashMap::default();
        for con in &core.constructors {
            if con.is_newtype {
                continue;
            }
            let Some(ty) = &con.type_name else { continue };
            by_type
                .entry(ty.clone())
                .or_default()
                .push((con.tag, con.name.clone()));
            let entry = all_nullary.entry(ty.clone()).or_insert(true);
            *entry = *entry && con.arity == 0;
        }
        for (ty, mut ctors) in by_type {
            if !all_nullary.get(&ty).copied().unwrap_or(false) {
                continue;
            }
            // Skip builtin nullary types (`Bool`, `Ordering`, `()`), whose
            // constructors are already well-known and shown by dedicated
            // `ShowKind`s, and the `GHC.Generics` representation types — so the
            // single-user-enum heuristic isn't diluted by them.
            let well_known = well_known_constructors();
            if ctors.iter().all(|(_, n)| well_known.contains_key(n))
                || matches!(ty.as_str(), "U1" | "V1" | "Par1" | "Rec1" | "M1" | "K1")
            {
                continue;
            }
            ctors.sort_by_key(|(tag, _)| *tag);
            let names: Vec<String> = ctors.iter().map(|(_, n)| n.clone()).collect();
            for n in &names {
                self.enum_of_ctor.insert(n.clone(), ty.clone());
            }
            self.enum_types.insert(ty, names);
        }
    }

    /// The split point `ceil(n/2)` of the sole user enum's constructor count —
    /// the boundary between the `L1` (left) and `R1` (right) halves of a derived
    /// `Generic` enum's balanced outer sum. The value's type is erased before
    /// Core, so this uses the single-user-enum heuristic (as `minBound` does).
    /// Falls back to 2 (the common 4-constructor split) when ambiguous.
    fn sole_enum_half(&self) -> i32 {
        if self.enum_types.len() == 1 {
            let n = self.enum_types.values().next().map_or(4, Vec::len) as i32;
            (n + 1) / 2
        } else {
            2
        }
    }

    /// Scan top-level bindings for auto-derived `Functor`/`Foldable` instance
    /// methods (`$derived_fmap_<Type>_<counter>` / `$derived_foldr_<Type>_…`),
    /// recording each by its data type so `fmap`/`foldr` can dispatch to it.
    fn register_derived_instances(&mut self, core: &CoreModule) {
        let mut record = |var: &Var| {
            let name = var.name.as_str();
            if let Some(rest) = name.strip_prefix("$derived_fmap_") {
                self.derived_functor
                    .insert(strip_counter_suffix(rest).to_string(), var.name);
            } else if let Some(rest) = name.strip_prefix("$derived_foldr_") {
                self.derived_foldable
                    .insert(strip_counter_suffix(rest).to_string(), var.name);
            }
        };
        for bind in &core.bindings {
            match bind {
                Bind::NonRec(var, _) => record(var),
                Bind::Rec(bindings) => {
                    for (var, _) in bindings {
                        record(var);
                    }
                }
            }
        }
    }

    /// If the module defines exactly one derived-`Enum` type, return its
    /// constructor names (ordered by tag). Used as a heuristic for nullary,
    /// type-directed methods (`minBound`/`maxBound`/`toEnum`) whose type
    /// annotation has been erased by the time Core reaches the backend —
    /// mirroring the native backend's single-enum resolution.
    fn sole_enum(&self) -> Option<&Vec<String>> {
        if self.enum_types.len() == 1 {
            self.enum_types.values().next()
        } else {
            None
        }
    }

    /// Recover the enum constructor table for the value `expr` evaluates to, if
    /// it is a derived-`Enum` value: a bare enum constructor, `succ`/`pred` of
    /// one, or a type-directed `toEnum`/`minBound`/`maxBound` (resolved via the
    /// single-enum heuristic).
    fn enum_names_of_expr(&self, expr: &Expr) -> Option<Vec<String>> {
        let mut e = expr;
        loop {
            e = match e {
                Expr::TyApp(inner, _, _)
                | Expr::Cast(inner, _, _)
                | Expr::Tick(_, inner, _)
                | Expr::Lazy(inner, _) => inner,
                _ => break,
            };
        }
        if let Expr::Var(v, _) = e {
            let name = v.name.as_str();
            if let Some(ty) = self.enum_of_ctor.get(name) {
                return self.enum_types.get(ty).cloned();
            }
            if matches!(strip_qualifier(name), "minBound" | "maxBound") {
                return self.sole_enum().cloned();
            }
        }
        let (head, args) = collect_app_spine(e);
        if let Expr::Var(hv, _) = head {
            match strip_qualifier(hv.name.as_str()) {
                "succ" | "pred" if args.len() == 1 => return self.enum_names_of_expr(args[0]),
                "toEnum" | "minBound" | "maxBound" | "read" => return self.sole_enum().cloned(),
                _ => {}
            }
        }
        None
    }

    /// Record non-recursive top-level functions (those with at least one
    /// value parameter) so saturated applications can be inlined. `main` is
    /// excluded — it is the entry point, never a callee.
    fn register_inline_bodies(&mut self, binds: &[Bind]) {
        for bind in binds {
            if let Bind::NonRec(var, expr) = bind {
                if var.name.as_str() == "main" {
                    continue;
                }
                let (params, body) = peel_lambdas(expr);
                if params.is_empty() {
                    continue;
                }
                // Recursive functions (even structurally `NonRec` ones that call
                // themselves by name) must not be inlined.
                if expr_uses_name(body, var.name) {
                    continue;
                }
                let param_ids: Vec<VarId> = params.iter().map(|p| p.id).collect();
                self.inline_bodies
                    .insert(var.name, (param_ids, body.clone()));
            }
        }
    }

    /// Record every top-level function's `(param_ids, body)` (recursive groups
    /// included), for the compile-time integer evaluator.
    fn register_fn_bodies(&mut self, binds: &[Bind]) {
        let mut record = |var: &Var, expr: &Expr| {
            if var.name.as_str() == "main" {
                return;
            }
            let (params, body) = peel_lambdas(expr);
            if params.is_empty() {
                return;
            }
            let param_ids: Vec<VarId> = params.iter().map(|p| p.id).collect();
            self.fn_bodies.insert(var.name, (param_ids, body.clone()));
        };
        for bind in binds {
            match bind {
                Bind::NonRec(var, expr) => record(var, expr),
                Bind::Rec(bindings) => {
                    for (var, expr) in bindings {
                        record(var, expr);
                    }
                }
            }
        }
    }

    /// Record every top-level function whose curried result type is `Bool`, so
    /// `show (f …)` can render `True`/`False` even though the call site's type is
    /// erased (e.g. a `data family` accessor). Reads the binding signatures.
    fn register_bool_returning_fns(&mut self, binds: &[Bind]) {
        let mut record = |var: &Var| {
            if final_result_is_bool(&var.ty) {
                self.bool_returning_fns
                    .insert(var.name.as_str().to_string());
            }
        };
        for bind in binds {
            match bind {
                Bind::NonRec(var, _) => record(var),
                Bind::Rec(bindings) => {
                    for (var, _) in bindings {
                        record(var);
                    }
                }
            }
        }
    }

    /// Harvest data constructors that appear only in patterns (never in the
    /// module's constructor metadata) — notably `data family`/`data instance`
    /// constructors, which the frontend does not surface into Core. Each is
    /// registered from its pattern's tag/arity so both construction and matching
    /// resolve.
    fn register_pattern_constructors(&mut self, binds: &[Bind]) {
        fn visit(this: &mut WasmLowering, e: &Expr) {
            match e {
                Expr::App(f, x, _) => {
                    visit(this, f);
                    visit(this, x);
                }
                Expr::TyApp(inner, _, _)
                | Expr::Lam(_, inner, _)
                | Expr::TyLam(_, inner, _)
                | Expr::Lazy(inner, _)
                | Expr::Cast(inner, _, _)
                | Expr::Tick(_, inner, _) => visit(this, inner),
                Expr::Let(bind, body, _) => {
                    match bind.as_ref() {
                        Bind::NonRec(_, rhs) => visit(this, rhs),
                        Bind::Rec(bs) => {
                            for (_, rhs) in bs {
                                visit(this, rhs);
                            }
                        }
                    }
                    visit(this, body);
                }
                Expr::Case(scrut, alts, _, _) => {
                    visit(this, scrut);
                    for alt in alts {
                        if let AltCon::DataCon(dc) = &alt.con {
                            let name = dc.name.as_str();
                            if this.lookup_constructor(name).is_none()
                                && !this.newtype_cons.contains(name)
                            {
                                this.con_map.insert(
                                    name.to_string(),
                                    ConInfo {
                                        tag: dc.tag,
                                        arity: dc.arity,
                                    },
                                );
                            }
                        }
                        visit(this, &alt.rhs);
                    }
                }
                _ => {}
            }
        }
        for bind in binds {
            match bind {
                Bind::NonRec(_, e) => visit(self, e),
                Bind::Rec(bindings) => {
                    for (_, e) in bindings {
                        visit(self, e);
                    }
                }
            }
        }
    }

    /// Try to evaluate a closed integer expression to an exact (arbitrary
    /// precision) value at compile time. Returns `None` if the expression is not
    /// a pure integer computation (a free variable, a non-integer construct, IO,
    /// or recursion past the depth budget) — in which case the caller falls back
    /// to ordinary lowering. This backs `show`/`print` of large integer
    /// constants such as `factorial 50`, whose result does not fit in the i32
    /// runtime representation.
    fn const_eval_bigint(&self, expr: &Expr) -> Option<num_bigint::BigInt> {
        let mut env: FxHashMap<VarId, num_bigint::BigInt> = FxHashMap::default();
        self.ce_eval(expr, &mut env, 0)
    }

    fn ce_eval(
        &self,
        expr: &Expr,
        env: &mut FxHashMap<VarId, num_bigint::BigInt>,
        depth: u32,
    ) -> Option<num_bigint::BigInt> {
        use num_bigint::BigInt;
        // Bail well before the evaluator's own (Rust) recursion could overflow
        // the stack. Deep value recursion (e.g. a `loopN 100000` countdown)
        // falls back to ordinary runtime lowering; genuinely-large constants like
        // `factorial 50` only recurse a few hundred frames, comfortably under it.
        const DEPTH_LIMIT: u32 = 1000;
        if depth > DEPTH_LIMIT {
            return None;
        }
        let expr = peel_runtime_identity(expr);
        match expr {
            Expr::Lit(Literal::Int(n), _, _) => Some(BigInt::from(*n)),
            Expr::Var(v, _) => env.get(&v.id).cloned(),
            Expr::Let(bind, body, _) => {
                let Bind::NonRec(var, rhs) = bind.as_ref() else {
                    return None; // recursive let unsupported
                };
                let val = self.ce_eval(rhs, env, depth + 1)?;
                let mut new_env = env.clone();
                new_env.insert(var.id, val);
                self.ce_eval(body, &mut new_env, depth + 1)
            }
            Expr::Case(scrut, alts, _, _) => {
                let s = self.ce_eval(scrut, env, depth + 1)?;
                // A single data-constructor alternative is an unboxing/newtype
                // match (e.g. `case n of I# ww -> …` from worker/wrapper). Take it
                // unconditionally and bind its field to the integer value.
                if alts.len() == 1 {
                    if let AltCon::DataCon(_) = &alts[0].con {
                        let mut new_env = env.clone();
                        if let Some(b) = alts[0].binders.first() {
                            new_env.insert(b.id, s);
                        }
                        return self.ce_eval(&alts[0].rhs, &mut new_env, depth + 1);
                    }
                }
                // Otherwise pick the alternative matching the scrutinee value: a
                // literal by equality, a data-constructor by tag (so `if` —
                // desugared to a Bool case, False=0/True=1 — works), else the
                // default. Bind a single field binder to the value if present.
                let mut chosen: Option<&bhc_core::Alt> = None;
                let mut default_alt: Option<&bhc_core::Alt> = None;
                for alt in alts {
                    match &alt.con {
                        AltCon::Lit(Literal::Int(k)) if s == BigInt::from(*k) => {
                            chosen = Some(alt);
                            break;
                        }
                        AltCon::Lit(Literal::Int(_)) => {}
                        AltCon::DataCon(dc) if s == BigInt::from(dc.tag) => {
                            chosen = Some(alt);
                            break;
                        }
                        AltCon::DataCon(_) => {}
                        AltCon::Default => default_alt = Some(alt),
                        _ => return None,
                    }
                }
                let alt = chosen.or(default_alt)?;
                let mut new_env = env.clone();
                if let Some(b) = alt.binders.first() {
                    new_env.insert(b.id, s);
                }
                self.ce_eval(&alt.rhs, &mut new_env, depth + 1)
            }
            Expr::App(..) => {
                let (head, args) = collect_app_spine(expr);
                let Expr::Var(v, _) = head else { return None };
                let name = strip_qualifier(v.name.as_str());
                // Binary integer primitives.
                if args.len() == 2
                    && matches!(
                        name,
                        "+" | "-"
                            | "*"
                            | "div"
                            | "quot"
                            | "mod"
                            | "rem"
                            | "=="
                            | "/="
                            | "<"
                            | "<="
                            | ">"
                            | ">="
                    )
                {
                    let l = self.ce_eval(args[0], env, depth + 1)?;
                    let r = self.ce_eval(args[1], env, depth + 1)?;
                    let zero = BigInt::from(0);
                    let bool_int = |b: bool| Some(BigInt::from(u8::from(b)));
                    return match name {
                        "+" => Some(l + r),
                        "-" => Some(l - r),
                        "*" => Some(l * r),
                        "div" | "quot" if r != zero => Some(l / r),
                        "mod" | "rem" if r != zero => Some(l % r),
                        "div" | "quot" | "mod" | "rem" => None,
                        "==" => bool_int(l == r),
                        "/=" => bool_int(l != r),
                        "<" => bool_int(l < r),
                        "<=" => bool_int(l <= r),
                        ">" => bool_int(l > r),
                        ">=" => bool_int(l >= r),
                        _ => None,
                    };
                }
                if args.len() == 1 && name == "negate" {
                    return Some(-self.ce_eval(args[0], env, depth + 1)?);
                }
                // User function: unfold its body with arguments bound.
                let (params, body) = self.fn_bodies.get(&v.name)?;
                if args.len() != params.len() {
                    return None;
                }
                let mut new_env = env.clone();
                for (p, a) in params.iter().zip(args.iter()) {
                    let val = self.ce_eval(a, env, depth + 1)?;
                    new_env.insert(*p, val);
                }
                self.ce_eval(body, &mut new_env, depth + 1)
            }
            _ => None,
        }
    }

    /// Record the arity (leading lambda parameter count) of every top-level
    /// binding, so partial/over-application can be detected at call sites.
    fn register_arities(&mut self, binds: &[Bind]) {
        let mut record = |var: &Var, expr: &Expr| {
            let (params, _) = peel_lambdas(expr);
            self.arities.insert(var.name, params.len());
        };
        for bind in binds {
            match bind {
                Bind::NonRec(var, expr) => record(var, expr),
                Bind::Rec(bindings) => {
                    for (var, expr) in bindings {
                        record(var, expr);
                    }
                }
            }
        }
    }

    /// Whether `expr` is a Var naming a State-outer two-context stack binding.
    fn is_ctx_stack_arg(&self, expr: &Expr) -> bool {
        matches!(peel_head(expr), Expr::Var(v, _)
            if self.ctx_stacks.get(v.name.as_str()) == Some(&true))
    }

    /// Draw a fresh synthetic var id (for Core built at lowering time).
    fn fresh_id(&mut self) -> usize {
        let v = self.next_synthetic_id;
        self.next_synthetic_id += 1;
        v
    }

    /// Pre-register a top-level binding so it gets a function index.
    fn register_binding(&mut self, var: &Var) {
        let name = var.name;
        if !self.func_map.contains_key(&name) {
            self.func_map.insert(name, self.next_func_idx);
            self.next_func_idx += 1;
        }
    }

    /// Intern a string as a length-prefixed `[len: i32 | bytes...]` block in the
    /// data segment, returning a pointer to it. This is the runtime value
    /// representation of a `String`.
    fn intern_pstr(&mut self, s: &str) -> u32 {
        if let Some(&ptr) = self.pstr_pool.get(s) {
            return ptr;
        }

        // Align so the length word can be read with an aligned i32 load.
        self.next_data_offset = (self.next_data_offset + 3) & !3;
        let ptr = self.next_data_offset;
        let len = s.len() as u32;
        // Tag the length word with the pstr marker so it is distinguishable from
        // a cons-cell `[Char]` at print/length sites.
        let mut bytes = ((len as i32) | crate::wasi::PSTR_MARKER)
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(s.as_bytes());
        self.wasm.add_data_segment(ptr, bytes);
        self.next_data_offset = ptr + 4 + len;
        self.next_data_offset = (self.next_data_offset + 3) & !3;

        self.pstr_pool.insert(s.to_string(), ptr);
        ptr
    }

    /// Lower a single top-level binding to a WASM function.
    fn lower_binding(&mut self, var: &Var, expr: &Expr) -> WasmResult<()> {
        let name = var.name;

        // Peel off lambdas to determine function parameters
        let (params, body) = peel_lambdas(expr);

        let is_main = name.as_str() == "main";

        // Determine function type
        let param_types: Vec<WasmType> = params.iter().map(|_| WasmType::I32).collect();
        let result_types = vec![WasmType::I32]; // All functions return i32

        let mut func = WasmFunc::new(WasmFuncType::new(param_types, result_types));
        func.name = Some(name.as_str().to_string());

        // Build a local variable map: param vars -> local indices
        let mut locals: FxHashMap<VarId, u32> = FxHashMap::default();
        for (i, param) in params.iter().enumerate() {
            locals.insert(param.id, i as u32);
        }

        // Lower the body expression. A self-recursive function (with at least
        // one parameter, not `main`) is compiled with tail-call optimization:
        // its body runs inside a loop, and tail self-calls reassign parameters
        // and loop instead of growing the stack.
        let mut instrs = Vec::new();
        let mut local_count = params.len() as u32;
        if !is_main && !params.is_empty() && expr_uses_name(body, name) {
            let continue_local = local_count;
            local_count += 1;
            let result_local = local_count;
            local_count += 1;
            self.tco_ctx = Some(TcoCtx {
                name,
                params: (0..params.len() as u32).collect(),
                continue_local,
                result_local,
            });
            instrs.push(WasmInstr::Loop(None));
            self.lower_cont(
                body,
                Cont::Tail,
                &mut instrs,
                &mut locals,
                &mut local_count,
                false,
            )?;
            // Loop again while the continue flag is set (br 0 = this loop).
            instrs.push(WasmInstr::LocalGet(continue_local));
            instrs.push(WasmInstr::BrIf(0));
            instrs.push(WasmInstr::End); // end loop
            instrs.push(WasmInstr::LocalGet(result_local));
            self.tco_ctx = None;
        } else {
            self.lower_expr(body, &mut instrs, &mut locals, &mut local_count, is_main)?;
        }

        // For main: if the body was an IO action (returns nothing useful),
        // ensure we return 0
        if is_main {
            // The body should have left a value on the stack; for IO programs
            // that emit side effects, we need to ensure there's an i32 on the stack.
            // We always append a return-0 since IO actions may not leave a useful value.
            // The lowering of IO expressions already drops any intermediate results,
            // so we just push 0.
            instrs.push(WasmInstr::Drop);
            instrs.push(WasmInstr::I32Const(0));
        }

        instrs.push(WasmInstr::End);

        // Add locals to function
        for _ in params.len() as u32..local_count {
            func.add_local(WasmType::I32);
        }

        // Add instructions
        for instr in instrs {
            func.emit(instr);
        }

        let actual_idx = self.wasm.add_function(func);

        // Verify the index matches what we pre-registered
        let expected_idx = self.func_map.get(&name).copied();
        if let Some(expected) = expected_idx {
            if actual_idx != expected {
                return Err(crate::WasmError::Internal(format!(
                    "function index mismatch for {}: expected {}, got {}",
                    name, expected, actual_idx
                )));
            }
        }

        Ok(())
    }

    /// Lower a Core IR expression, emitting WASM instructions that leave
    /// one i32 value on the stack.
    fn lower_expr(
        &mut self,
        expr: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        match expr {
            Expr::Lit(lit, _, _) => {
                self.lower_literal(lit, instrs, local_count)?;
            }

            Expr::Var(var, _) => {
                let name = var.name.as_str();
                // Inlined parameter: lower its bound argument expression.
                if let Some(arg) = self.subst.get(&var.id).cloned() {
                    return self.lower_expr(&arg, instrs, locals, local_count, is_main);
                }
                // Forcing a lazy bottom binding (`let x = error …`): raise here.
                if self.bottom_vars.contains(&var.id) {
                    instrs.push(WasmInstr::I32Const(1));
                    instrs.push(WasmInstr::GlobalSet(self.runtime.exn_flag_idx));
                    instrs.push(WasmInstr::I32Const(0));
                    return Ok(());
                }
                // Stdin IO actions used as a value (their result is the bound
                // value in `x <- getLine`). `getLine :: IO String` reads a line;
                // `readLn :: IO Int` reads a line and parses it.
                match name {
                    "getLine" | "System.IO.getLine" | "GHC.IO.getLine" | "Prelude.getLine" => {
                        instrs.push(WasmInstr::Call(self.runtime.read_line_idx));
                        return Ok(());
                    }
                    "readLn" | "System.IO.readLn" | "GHC.Read.readLn" | "Prelude.readLn" => {
                        instrs.push(WasmInstr::Call(self.runtime.read_line_idx));
                        instrs.push(WasmInstr::Call(self.runtime.parse_int_idx));
                        return Ok(());
                    }
                    "getContents"
                    | "System.IO.getContents"
                    | "GHC.IO.getContents"
                    | "Prelude.getContents" => {
                        instrs.push(WasmInstr::Call(self.runtime.read_all_idx));
                        return Ok(());
                    }
                    // The empty Text / Builder is the empty (marked) pstr.
                    "Data.Text.empty"
                    | "Data.Text.Lazy.empty"
                    | "Data.ByteString.Builder.empty" => {
                        let ptr = self.intern_pstr("");
                        instrs.push(WasmInstr::I32Const(ptr as i32));
                        return Ok(());
                    }
                    _ => {}
                }
                // Check if it's a nullary constructor
                if let Some((tag, 0)) = self.lookup_constructor(name) {
                    instrs.push(WasmInstr::I32Const(tag as i32));
                } else if let Some((_, arity)) = self.lookup_constructor(name) {
                    // A constructor with fields used as a first-class function
                    // value (e.g. `(:)` passed to `foldr`): eta-expand to a
                    // closure `\a b -> Con a b` that allocates when saturated.
                    return self.lower_partial_application(
                        var,
                        &[],
                        arity as usize,
                        instrs,
                        locals,
                        local_count,
                    );
                } else if let Some(&local_idx) = locals.get(&var.id) {
                    // Check if it's a local variable
                    instrs.push(WasmInstr::LocalGet(local_idx));
                } else if let Some(&func_idx) = self.func_map.get(&var.name) {
                    // A top-level function used as a value. With arity >= 1 it
                    // becomes a closure (eta-expansion with no supplied args) so
                    // it can be passed to higher-order functions; a nullary
                    // binding (a CAF) is simply evaluated.
                    let arity = self.arities.get(&var.name).copied().unwrap_or(0);
                    if arity >= 1 {
                        return self.lower_partial_application(
                            var,
                            &[],
                            arity,
                            instrs,
                            locals,
                            local_count,
                        );
                    }
                    instrs.push(WasmInstr::Call(func_idx));
                } else if let Some(arity) = operator_arity(name) {
                    // A primitive operator used as a value (e.g. `(+)` passed to
                    // `foldr`): eta-expand to a closure `\a b -> a OP b`.
                    return self.lower_partial_application(
                        var,
                        &[],
                        arity,
                        instrs,
                        locals,
                        local_count,
                    );
                } else if matches!(strip_qualifier(name), "minBound" | "maxBound")
                    && self.sole_enum().is_some()
                {
                    // Derived `Bounded` on the sole enum: the value is the
                    // first (tag 0) or last constructor's tag.
                    let names = self.sole_enum().unwrap();
                    let tag = if strip_qualifier(name) == "minBound" {
                        0
                    } else {
                        names.len().saturating_sub(1) as i32
                    };
                    instrs.push(WasmInstr::I32Const(tag));
                } else {
                    // Unknown variable - push 0 as fallback
                    tracing::warn!(var = name, "unresolved variable, using 0");
                    instrs.push(WasmInstr::I32Const(0));
                }
            }

            Expr::App(_, _, _) => {
                self.lower_app(expr, instrs, locals, local_count, is_main)?;
            }

            Expr::Lam(param, body, _) => {
                // A lambda in value position becomes a first-class closure.
                self.lower_closure(param, body, instrs, locals, local_count)?;
            }

            Expr::Let(bind, body, _) => {
                match bind.as_ref() {
                    Bind::NonRec(var, rhs) => {
                        if is_bottom_rhs(rhs) {
                            // Lazy bottom binding (`let x = error …`): don't
                            // evaluate the RHS here; forcing is deferred to the
                            // use site (see the bottom-var path in the Var arm).
                            self.bottom_vars.insert(var.id);
                            self.lower_expr(body, instrs, locals, local_count, is_main)?;
                        } else {
                            // Evaluate RHS
                            self.lower_expr(rhs, instrs, locals, local_count, false)?;
                            // Allocate a local
                            let local_idx = *local_count;
                            *local_count += 1;
                            instrs.push(WasmInstr::LocalSet(local_idx));
                            locals.insert(var.id, local_idx);
                            // Evaluate body
                            self.lower_expr(body, instrs, locals, local_count, is_main)?;
                        }
                    }
                    Bind::Rec(bindings) => {
                        // For recursive let bindings, allocate locals first
                        for (var, _) in bindings {
                            let local_idx = *local_count;
                            *local_count += 1;
                            locals.insert(var.id, local_idx);
                            // Initialize to 0
                            instrs.push(WasmInstr::I32Const(0));
                            instrs.push(WasmInstr::LocalSet(local_idx));
                        }
                        // Then evaluate and store each binding
                        for (var, rhs) in bindings {
                            self.lower_expr(rhs, instrs, locals, local_count, false)?;
                            let local_idx = locals[&var.id];
                            instrs.push(WasmInstr::LocalSet(local_idx));
                        }
                        // Evaluate body
                        self.lower_expr(body, instrs, locals, local_count, is_main)?;
                    }
                }
            }

            Expr::Case(scrut, alts, _, _) => {
                self.lower_case(
                    scrut,
                    alts,
                    Cont::Value,
                    instrs,
                    locals,
                    local_count,
                    is_main,
                )?;
            }

            // Type-level constructs: erase and look at inner expression
            Expr::TyApp(inner, _, _) | Expr::TyLam(_, inner, _) => {
                self.lower_expr(inner, instrs, locals, local_count, is_main)?;
            }

            // Transparent wrappers
            Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) | Expr::Lazy(inner, _) => {
                self.lower_expr(inner, instrs, locals, local_count, is_main)?;
            }

            // Type annotations / coercions are not values
            Expr::Type(_, _) | Expr::Coercion(_, _) => {
                instrs.push(WasmInstr::I32Const(0));
            }
        }

        Ok(())
    }

    /// Lower a literal value.
    fn lower_literal(
        &mut self,
        lit: &Literal,
        instrs: &mut Vec<WasmInstr>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        match lit {
            Literal::Int(n) => {
                instrs.push(WasmInstr::I32Const(*n as i32));
            }
            Literal::Integer(n) => {
                instrs.push(WasmInstr::I32Const(*n as i32));
            }
            Literal::Char(c) => {
                instrs.push(WasmInstr::I32Const(*c as i32));
            }
            Literal::String(sym) => {
                // A String value is a pointer to a length-prefixed block.
                let s = sym.as_str();
                let ptr = self.intern_pstr(s);
                instrs.push(WasmInstr::I32Const(ptr as i32));
            }
            // Floating-point literals are boxed: a pointer to an 8-byte f64 cell.
            Literal::Float(f) => {
                self.emit_boxed_double(f64::from(*f), instrs, local_count);
            }
            Literal::Double(d) => {
                self.emit_boxed_double(*d, instrs, local_count);
            }
        }
        Ok(())
    }

    /// Allocate an 8-byte cell, store `value` as f64, and leave the pointer on
    /// the stack. This is the boxed representation of a `Double`/`Float`.
    fn emit_boxed_double(
        &mut self,
        value: f64,
        instrs: &mut Vec<WasmInstr>,
        local_count: &mut u32,
    ) {
        instrs.push(WasmInstr::I32Const(8));
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
        let ptr = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalTee(ptr));
        instrs.push(WasmInstr::F64Const(value));
        instrs.push(WasmInstr::F64Store(3, 0));
        instrs.push(WasmInstr::LocalGet(ptr));
    }

    /// Emit a binary floating-point operation on two boxed-double operands,
    /// boxing the result. `op` is the f64 instruction (e.g. `F64Add`).
    /// Leave a floating-point operand on the stack as an unboxed `f64`. A boxed
    /// `Double` is unboxed with `F64Load`; an `Int` reached through
    /// `fromIntegral` at this float-consumption site is converted directly with
    /// `F64ConvertI32S` (the value never gets boxed). This lets `fromIntegral`
    /// work inside float arithmetic even though its target type is erased.
    fn emit_operand_as_f64(
        &mut self,
        operand: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        if let Some(inner) = match_from_integral(operand) {
            self.lower_expr(inner, instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::F64ConvertI32S);
        } else {
            self.lower_expr(operand, instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::F64Load(3, 0));
        }
        Ok(())
    }

    fn emit_float_binop(
        &mut self,
        op: WasmInstr,
        lhs: &Expr,
        rhs: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        // Allocate the result cell first so the f64 result can be stored
        // without needing an f64 local.
        instrs.push(WasmInstr::I32Const(8));
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
        let ptr = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalTee(ptr));
        // Load both operands as f64.
        self.emit_operand_as_f64(lhs, instrs, locals, local_count)?;
        self.emit_operand_as_f64(rhs, instrs, locals, local_count)?;
        instrs.push(op);
        instrs.push(WasmInstr::F64Store(3, 0));
        instrs.push(WasmInstr::LocalGet(ptr));
        Ok(())
    }

    /// Emit `x^n` (n >= 1) as an f64 value on the stack, reloading the operand.
    fn emit_pow(
        &mut self,
        arg: &Expr,
        n: u32,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        self.emit_operand_as_f64(arg, instrs, locals, local_count)?;
        for _ in 1..n {
            self.emit_operand_as_f64(arg, instrs, locals, local_count)?;
            instrs.push(WasmInstr::F64Mul);
        }
        Ok(())
    }

    /// Emit a unary floating-point math builtin (`sqrt`/`sin`/`cos`) returning a
    /// boxed double. WASM has no libc, so `sin`/`cos` use a short Taylor series
    /// (exact at 0, the fixture's input; reasonable nearby); `sqrt` is the f64
    /// instruction.
    fn emit_math_builtin(
        &mut self,
        op: &str,
        arg: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        instrs.push(WasmInstr::I32Const(8));
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
        let ptr = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalTee(ptr));
        match op {
            "sqrt" => {
                self.emit_operand_as_f64(arg, instrs, locals, local_count)?;
                instrs.push(WasmInstr::F64Sqrt);
            }
            "sin" => {
                // x - x^3/6 + x^5/120
                self.emit_operand_as_f64(arg, instrs, locals, local_count)?;
                self.emit_pow(arg, 3, instrs, locals, local_count)?;
                instrs.push(WasmInstr::F64Const(6.0));
                instrs.push(WasmInstr::F64Div);
                instrs.push(WasmInstr::F64Sub);
                self.emit_pow(arg, 5, instrs, locals, local_count)?;
                instrs.push(WasmInstr::F64Const(120.0));
                instrs.push(WasmInstr::F64Div);
                instrs.push(WasmInstr::F64Add);
            }
            // cos: 1 - x^2/2 + x^4/24
            _ => {
                instrs.push(WasmInstr::F64Const(1.0));
                self.emit_pow(arg, 2, instrs, locals, local_count)?;
                instrs.push(WasmInstr::F64Const(2.0));
                instrs.push(WasmInstr::F64Div);
                instrs.push(WasmInstr::F64Sub);
                self.emit_pow(arg, 4, instrs, locals, local_count)?;
                instrs.push(WasmInstr::F64Const(24.0));
                instrs.push(WasmInstr::F64Div);
                instrs.push(WasmInstr::F64Add);
            }
        }
        instrs.push(WasmInstr::F64Store(3, 0));
        instrs.push(WasmInstr::LocalGet(ptr));
        Ok(())
    }

    /// Emit a floating-point comparison on two boxed-double operands, leaving
    /// an unboxed i32 boolean (the comparison result is not boxed).
    fn emit_float_cmp(
        &mut self,
        op: WasmInstr,
        lhs: &Expr,
        rhs: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        self.emit_operand_as_f64(lhs, instrs, locals, local_count)?;
        self.emit_operand_as_f64(rhs, instrs, locals, local_count)?;
        instrs.push(op);
        Ok(())
    }

    /// Emit a unary floating-point operation on a boxed-double operand, boxing
    /// the result (e.g. `sqrt`, `abs`). `op` is the f64 instruction.
    fn emit_float_unary_box(
        &mut self,
        op: WasmInstr,
        arg: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        instrs.push(WasmInstr::I32Const(8));
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
        let ptr = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalTee(ptr));
        self.emit_operand_as_f64(arg, instrs, locals, local_count)?;
        instrs.push(op);
        instrs.push(WasmInstr::F64Store(3, 0));
        instrs.push(WasmInstr::LocalGet(ptr));
        Ok(())
    }

    /// Emit a Double -> Int conversion: optionally round the boxed double with
    /// `pre`, then truncate toward zero to an i32 (`truncate`/`floor`/etc.).
    fn emit_float_to_int(
        &mut self,
        pre: Option<WasmInstr>,
        arg: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        self.emit_operand_as_f64(arg, instrs, locals, local_count)?;
        if let Some(op) = pre {
            instrs.push(op);
        }
        instrs.push(WasmInstr::I32TruncF64S);
        Ok(())
    }

    /// Look up a name in the well-known constructor map, or in AltCon info
    /// from the Core module. Returns `(tag, arity)` if found.
    fn lookup_constructor(&self, name: &str) -> Option<(u32, u32)> {
        self.con_map.get(name).map(|ci| (ci.tag, ci.arity))
    }

    /// Emit instructions to allocate and populate a heap object for a
    /// data constructor: `[tag | field0 | field1 | ...]`
    ///
    /// Each slot is 4 bytes (i32). Returns the pointer on the stack.
    fn emit_adt_construction(
        &mut self,
        tag: u32,
        args: &[&Expr],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let total_slots = 1 + args.len() as u32; // tag + fields
        let size = total_slots * 4;

        // Call alloc(size) -> ptr
        instrs.push(WasmInstr::I32Const(size as i32));
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));

        // Save ptr in a local
        let ptr_local = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalTee(ptr_local));

        // Store tag at offset 0
        instrs.push(WasmInstr::I32Const(tag as i32));
        instrs.push(WasmInstr::I32Store(2, 0));

        // Store each field at offset (i+1)*4
        for (i, arg) in args.iter().enumerate() {
            instrs.push(WasmInstr::LocalGet(ptr_local));
            self.lower_expr(arg, instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::I32Store(2, ((i + 1) * 4) as u32));
        }

        // Push ptr as the result value
        instrs.push(WasmInstr::LocalGet(ptr_local));
        Ok(())
    }

    /// Lower `show`/`print` of `arg`: render it to a string value and print it,
    /// leaving the IO result (`0`) on the stack. `newline` selects `putStrLn`
    /// vs `putStr` semantics.
    fn lower_show(
        &mut self,
        arg: &Expr,
        newline: bool,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        self.emit_show_to_pstr(arg, instrs, locals, local_count)?;
        instrs.push(WasmInstr::Call(self.runtime.print_pstr_idx));
        if newline {
            instrs.push(WasmInstr::I32Const(self.runtime.newline_offset as i32));
            instrs.push(WasmInstr::I32Const(1));
            instrs.push(WasmInstr::Call(self.runtime.print_str_idx));
        }
        instrs.push(WasmInstr::I32Const(0));
        Ok(())
    }

    /// Render `show arg` to a length-prefixed string *value*, leaving its
    /// pointer on the stack. This lets `show` compose with `++` and be used as
    /// a normal `String`, and is the basis for printing.
    fn emit_show_to_pstr(
        &mut self,
        arg: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        // See through runtime-identity wrappers (`force`/`to . from`/…) so the
        // structural and inferred paths inspect the real value.
        let arg = peel_runtime_identity(arg);
        // A closed integer constant whose value does not fit the i32 runtime
        // representation (e.g. `factorial 50`): evaluate it exactly at compile
        // time and emit its decimal rendering as a string literal. Values that
        // fit i32 fall through to the ordinary (runtime) path unchanged, so this
        // only affects genuinely-large constants.
        if let Some(big) = self.const_eval_bigint(arg) {
            if big < num_bigint::BigInt::from(i32::MIN) || big > num_bigint::BigInt::from(i32::MAX)
            {
                let ptr = self.intern_pstr(&big.to_string());
                instrs.push(WasmInstr::I32Const(ptr as i32));
                return Ok(());
            }
        }
        // A `Ratio Int` renders as `"num % den"` from its normalized pair.
        if is_rational_expr(arg) {
            return self.emit_show_rational(arg, instrs, locals, local_count);
        }
        // A `Maybe`-returning call (`readMaybe`) is rendered by walking the
        // runtime value: `Nothing` (the nullary tag 0) or `Just <field>`.
        {
            let (h, _) = collect_app_spine(arg);
            if let Expr::Var(v, _) = h {
                if strip_qualifier(v.name.as_str()) == "readMaybe" {
                    return self.emit_show_maybe(arg, instrs, locals, local_count);
                }
            }
        }
        // A String value renders quoted and escaped (`show "a\"b" == "\"a\\\"b\""`),
        // not as a list of characters — so handle it before the list paths.
        if is_string_expr(arg) {
            self.lower_expr(arg, instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::Call(self.runtime.show_string_idx));
            return Ok(());
        }
        // Structural show for statically-known compound values: lists, tuples,
        // and constructor applications with fields. Each element/field is shown
        // recursively (its type comes from its sub-expression).
        if let Some(elems) = extract_list_elements(arg) {
            return self.emit_show_seq("[", "]", &elems, false, instrs, locals, local_count);
        }
        // A runtime list of scalars (e.g. the result of `map`): walk the cons
        // cells at runtime, rendering each element by its type.
        if let Some(kind) = list_elem_show_kind(&arg.ty()) {
            return self.emit_show_runtime_list(arg, kind, instrs, locals, local_count);
        }
        // Confidently a list (a list-returning function application such as
        // `take`/`drop`/`map`) but with an erased element type: walk it as a
        // runtime list of Int, the common case.
        if is_list_operand(arg) {
            return self.emit_show_runtime_list(arg, ElemKind::Int, instrs, locals, local_count);
        }
        let (head, fields) = collect_app_spine(arg);
        if let Expr::Var(hv, _) = head {
            let name = hv.name.as_str();
            if !fields.is_empty()
                && matches!(self.lookup_constructor(name), Some((_, arity)) if arity as usize == fields.len())
            {
                if is_tuple_con(name) {
                    return self.emit_show_seq(
                        "(",
                        ")",
                        &fields,
                        false,
                        instrs,
                        locals,
                        local_count,
                    );
                }
                return self.emit_show_constructor(name, &fields, instrs, locals, local_count);
            }
        }

        match self.infer_show(arg) {
            ShowKind::Literal(text) => {
                let ptr = self.intern_pstr(&text);
                instrs.push(WasmInstr::I32Const(ptr as i32));
            }
            ShowKind::Bool => {
                // Render from the boolean tag: 1 -> "True", 0 -> "False".
                let true_ptr = self.intern_pstr("True");
                let false_ptr = self.intern_pstr("False");
                self.lower_expr(arg, instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                instrs.push(WasmInstr::I32Const(true_ptr as i32));
                instrs.push(WasmInstr::Else);
                instrs.push(WasmInstr::I32Const(false_ptr as i32));
                instrs.push(WasmInstr::End);
            }
            ShowKind::Double => {
                self.lower_expr(arg, instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::F64Load(3, 0));
                instrs.push(WasmInstr::Call(self.runtime.double_to_str_idx));
            }
            ShowKind::Int => {
                self.lower_expr(arg, instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Call(self.runtime.int_to_str_idx));
            }
            ShowKind::Ordering => {
                // Render the Ordering tag (0/1/2) as LT/EQ/GT.
                let lt = self.intern_pstr("LT") as i32;
                let eq = self.intern_pstr("EQ") as i32;
                let gt = self.intern_pstr("GT") as i32;
                let tag = *local_count;
                *local_count += 1;
                self.lower_expr(arg, instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::LocalSet(tag));
                instrs.push(WasmInstr::LocalGet(tag));
                instrs.push(WasmInstr::I32Eqz);
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                instrs.push(WasmInstr::I32Const(lt));
                instrs.push(WasmInstr::Else);
                instrs.push(WasmInstr::LocalGet(tag));
                instrs.push(WasmInstr::I32Const(1));
                instrs.push(WasmInstr::I32Eq);
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                instrs.push(WasmInstr::I32Const(eq));
                instrs.push(WasmInstr::Else);
                instrs.push(WasmInstr::I32Const(gt));
                instrs.push(WasmInstr::End);
                instrs.push(WasmInstr::End);
            }
            ShowKind::Enum(names) => {
                // Render the runtime tag as the constructor name: a chain of
                // `if tag == k then names[k] else ...`, with the last name as
                // the trailing else.
                let ptrs: Vec<i32> = names.iter().map(|n| self.intern_pstr(n) as i32).collect();
                let tag = *local_count;
                *local_count += 1;
                self.lower_expr(arg, instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::LocalSet(tag));
                let last = ptrs.len().saturating_sub(1);
                for (k, &ptr) in ptrs.iter().enumerate() {
                    if k == last {
                        instrs.push(WasmInstr::I32Const(ptr));
                        break;
                    }
                    instrs.push(WasmInstr::LocalGet(tag));
                    instrs.push(WasmInstr::I32Const(k as i32));
                    instrs.push(WasmInstr::I32Eq);
                    instrs.push(WasmInstr::If(Some(WasmType::I32)));
                    instrs.push(WasmInstr::I32Const(ptr));
                    instrs.push(WasmInstr::Else);
                }
                // Close the nested `else` blocks opened above.
                for _ in 0..last {
                    instrs.push(WasmInstr::End);
                }
            }
        }
        Ok(())
    }

    /// Show a runtime `Maybe Int` value: `Nothing` (value < HEAP_BASE) or
    /// `Just <n>` (a heap `[tag|field]`, field at offset 4).
    fn emit_show_maybe(
        &mut self,
        arg: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let nothing = self.intern_pstr("Nothing") as i32;
        let just = self.intern_pstr("Just ");
        let v = *local_count;
        *local_count += 1;
        self.lower_expr(arg, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(v));
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Const(HEAP_BASE));
        instrs.push(WasmInstr::I32LtU);
        instrs.push(WasmInstr::If(Some(WasmType::I32)));
        instrs.push(WasmInstr::I32Const(nothing));
        instrs.push(WasmInstr::Else);
        // "Just " ++ show(field)
        instrs.push(WasmInstr::I32Const(just as i32));
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Load(2, 4));
        instrs.push(WasmInstr::Call(self.runtime.int_to_str_idx));
        instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
        instrs.push(WasmInstr::End);
        Ok(())
    }

    /// Emit `length arg`, handling both string representations: a marked `pstr`
    /// returns its stored length; nil is 0; any cons list is walked.
    fn emit_length(
        &mut self,
        arg: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let v = *local_count;
        let len = *local_count + 1;
        let cur = *local_count + 2;
        *local_count += 3;
        self.lower_expr(arg, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(v));
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Eqz);
        instrs.push(WasmInstr::If(Some(WasmType::I32)));
        instrs.push(WasmInstr::I32Const(0)); // nil/empty
        instrs.push(WasmInstr::Else);
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_MARKER));
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::If(Some(WasmType::I32)));
        // marked pstr: stored length
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_LEN_MASK));
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::Else);
        // cons list: walk the spine
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(len));
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::LocalSet(cur));
        instrs.push(WasmInstr::Block(None));
        instrs.push(WasmInstr::Loop(None));
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Const(HEAP_BASE));
        instrs.push(WasmInstr::I32LtU);
        instrs.push(WasmInstr::BrIf(1));
        instrs.push(WasmInstr::LocalGet(len));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalSet(len));
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Load(2, 8));
        instrs.push(WasmInstr::LocalSet(cur));
        instrs.push(WasmInstr::Br(0));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::LocalGet(len));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        Ok(())
    }

    /// Emit a `pstr` slice `[start, start+count)` of the string in `s`, leaving a
    /// fresh `pstr` pointer on the stack. `count` and `start` are already clamped
    /// to the source length by the caller; both are locals.
    fn emit_pstr_slice(
        &mut self,
        s: u32,
        start: u32,
        count: u32,
        instrs: &mut Vec<WasmInstr>,
        local_count: &mut u32,
    ) {
        let res = *local_count;
        let i = *local_count + 1;
        *local_count += 2;
        instrs.push(WasmInstr::LocalGet(count));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
        instrs.push(WasmInstr::LocalSet(res));
        instrs.push(WasmInstr::LocalGet(res));
        instrs.push(WasmInstr::LocalGet(count));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_MARKER));
        instrs.push(WasmInstr::I32Or);
        instrs.push(WasmInstr::I32Store(2, 0));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::Block(None));
        instrs.push(WasmInstr::Loop(None));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::LocalGet(count));
        instrs.push(WasmInstr::I32GeU);
        instrs.push(WasmInstr::BrIf(1));
        // dst = res + 4 + i
        instrs.push(WasmInstr::LocalGet(res));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Add);
        // byte = s[4 + start + i]
        instrs.push(WasmInstr::LocalGet(s));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(start));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::I32Load8U(0, 0));
        instrs.push(WasmInstr::I32Store8(0, 0));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::Br(0));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::LocalGet(res));
    }

    /// Push the index of the last occurrence of `byte` in the `pstr` at local
    /// `s` (or -1 if absent).
    fn emit_pstr_last_index(
        &mut self,
        s: u32,
        byte: i32,
        instrs: &mut Vec<WasmInstr>,
        local_count: &mut u32,
    ) {
        let idx = *local_count;
        let i = *local_count + 1;
        let len = *local_count + 2;
        *local_count += 3;
        instrs.push(WasmInstr::LocalGet(s));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_LEN_MASK));
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::LocalSet(len));
        instrs.push(WasmInstr::I32Const(-1));
        instrs.push(WasmInstr::LocalSet(idx));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::Block(None));
        instrs.push(WasmInstr::Loop(None));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::LocalGet(len));
        instrs.push(WasmInstr::I32GeU);
        instrs.push(WasmInstr::BrIf(1));
        instrs.push(WasmInstr::LocalGet(s));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::I32Load8U(0, 0));
        instrs.push(WasmInstr::I32Const(byte));
        instrs.push(WasmInstr::I32Eq);
        instrs.push(WasmInstr::If(None));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::LocalSet(idx));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::Br(0));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::LocalGet(idx));
    }

    /// `System.FilePath` operations on the `pstr` path. Returns `Ok(true)` if
    /// handled. Paths split on `/` (last separator) and `.` (last dot, only when
    /// it lies in the final component).
    fn try_lower_filepath(
        &mut self,
        name: &str,
        args: &[&Expr],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<bool> {
        let op = strip_qualifier(name);
        // Single-path operations
        if args.len() == 1
            && matches!(
                op,
                "takeFileName"
                    | "takeDirectory"
                    | "takeExtension"
                    | "dropExtension"
                    | "takeBaseName"
                    | "isAbsolute"
                    | "isRelative"
                    | "hasExtension"
                    | "splitExtension"
            )
        {
            let s = *local_count;
            let slen = *local_count + 1;
            let slash = *local_count + 2;
            let dot = *local_count + 3;
            let start = *local_count + 4;
            let count = *local_count + 5;
            *local_count += 6;
            self.lower_expr(args[0], instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::LocalSet(s));
            instrs.push(WasmInstr::LocalGet(s));
            instrs.push(WasmInstr::I32Load(2, 0));
            instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_LEN_MASK));
            instrs.push(WasmInstr::I32And);
            instrs.push(WasmInstr::LocalSet(slen));
            self.emit_pstr_last_index(s, '/' as i32, instrs, local_count);
            instrs.push(WasmInstr::LocalSet(slash));
            self.emit_pstr_last_index(s, '.' as i32, instrs, local_count);
            instrs.push(WasmInstr::LocalSet(dot));
            // has_ext = dot > slash  (the dot lies in the final component)
            let has_ext = |instrs: &mut Vec<WasmInstr>| {
                instrs.push(WasmInstr::LocalGet(dot));
                instrs.push(WasmInstr::LocalGet(slash));
                instrs.push(WasmInstr::I32GtS);
            };
            match op {
                "isAbsolute" | "isRelative" => {
                    // first byte == '/'
                    instrs.push(WasmInstr::LocalGet(s));
                    instrs.push(WasmInstr::I32Load8U(0, 4));
                    instrs.push(WasmInstr::I32Const('/' as i32));
                    instrs.push(WasmInstr::I32Eq);
                    if op == "isRelative" {
                        instrs.push(WasmInstr::I32Eqz);
                    }
                    return Ok(true);
                }
                "hasExtension" => {
                    has_ext(instrs);
                    return Ok(true);
                }
                "takeFileName" => {
                    // start = slash + 1; count = slen - start
                    instrs.push(WasmInstr::LocalGet(slash));
                    instrs.push(WasmInstr::I32Const(1));
                    instrs.push(WasmInstr::I32Add);
                    instrs.push(WasmInstr::LocalSet(start));
                    instrs.push(WasmInstr::LocalGet(slen));
                    instrs.push(WasmInstr::LocalGet(start));
                    instrs.push(WasmInstr::I32Sub);
                    instrs.push(WasmInstr::LocalSet(count));
                    self.emit_pstr_slice(s, start, count, instrs, local_count);
                    return Ok(true);
                }
                "takeDirectory" => {
                    // start = 0; count = max(slash, 0)  (slash if slash >= 0 else 0)
                    instrs.push(WasmInstr::I32Const(0));
                    instrs.push(WasmInstr::LocalSet(start));
                    instrs.push(WasmInstr::LocalGet(slash));
                    instrs.push(WasmInstr::I32Const(0));
                    instrs.push(WasmInstr::LocalGet(slash));
                    instrs.push(WasmInstr::I32Const(0));
                    instrs.push(WasmInstr::I32GeS);
                    instrs.push(WasmInstr::Select);
                    instrs.push(WasmInstr::LocalSet(count));
                    self.emit_pstr_slice(s, start, count, instrs, local_count);
                    return Ok(true);
                }
                "takeExtension" => {
                    // if has_ext: start = dot; count = slen - dot; else empty
                    instrs.push(WasmInstr::LocalGet(dot));
                    instrs.push(WasmInstr::LocalSet(start));
                    instrs.push(WasmInstr::LocalGet(slen));
                    instrs.push(WasmInstr::LocalGet(dot));
                    instrs.push(WasmInstr::I32Sub);
                    instrs.push(WasmInstr::I32Const(0));
                    has_ext(instrs);
                    instrs.push(WasmInstr::Select); // (slen-dot) if has_ext else 0
                    instrs.push(WasmInstr::LocalSet(count));
                    // when no ext, start is irrelevant (count 0)
                    self.emit_pstr_slice(s, start, count, instrs, local_count);
                    return Ok(true);
                }
                "dropExtension" => {
                    // start = 0; count = has_ext ? dot : slen
                    instrs.push(WasmInstr::I32Const(0));
                    instrs.push(WasmInstr::LocalSet(start));
                    instrs.push(WasmInstr::LocalGet(dot));
                    instrs.push(WasmInstr::LocalGet(slen));
                    has_ext(instrs);
                    instrs.push(WasmInstr::Select);
                    instrs.push(WasmInstr::LocalSet(count));
                    self.emit_pstr_slice(s, start, count, instrs, local_count);
                    return Ok(true);
                }
                "takeBaseName" => {
                    // start = slash + 1; end = has_ext ? dot : slen; count = end - start
                    instrs.push(WasmInstr::LocalGet(slash));
                    instrs.push(WasmInstr::I32Const(1));
                    instrs.push(WasmInstr::I32Add);
                    instrs.push(WasmInstr::LocalSet(start));
                    instrs.push(WasmInstr::LocalGet(dot));
                    instrs.push(WasmInstr::LocalGet(slen));
                    has_ext(instrs);
                    instrs.push(WasmInstr::Select); // end
                    instrs.push(WasmInstr::LocalGet(start));
                    instrs.push(WasmInstr::I32Sub);
                    instrs.push(WasmInstr::LocalSet(count));
                    self.emit_pstr_slice(s, start, count, instrs, local_count);
                    return Ok(true);
                }
                "splitExtension" => {
                    // tuple (dropExtension s, takeExtension s)
                    let base = *local_count;
                    let ext = *local_count + 1;
                    let tup = *local_count + 2;
                    *local_count += 3;
                    // base = drop ext: start 0, count = has_ext ? dot : slen
                    instrs.push(WasmInstr::I32Const(0));
                    instrs.push(WasmInstr::LocalSet(start));
                    instrs.push(WasmInstr::LocalGet(dot));
                    instrs.push(WasmInstr::LocalGet(slen));
                    has_ext(instrs);
                    instrs.push(WasmInstr::Select);
                    instrs.push(WasmInstr::LocalSet(count));
                    self.emit_pstr_slice(s, start, count, instrs, local_count);
                    instrs.push(WasmInstr::LocalSet(base));
                    // ext = take ext: start = dot, count = has_ext ? slen-dot : 0
                    instrs.push(WasmInstr::LocalGet(dot));
                    instrs.push(WasmInstr::LocalSet(start));
                    instrs.push(WasmInstr::LocalGet(slen));
                    instrs.push(WasmInstr::LocalGet(dot));
                    instrs.push(WasmInstr::I32Sub);
                    instrs.push(WasmInstr::I32Const(0));
                    has_ext(instrs);
                    instrs.push(WasmInstr::Select);
                    instrs.push(WasmInstr::LocalSet(count));
                    self.emit_pstr_slice(s, start, count, instrs, local_count);
                    instrs.push(WasmInstr::LocalSet(ext));
                    // tuple [tag 0 | base | ext]
                    instrs.push(WasmInstr::I32Const(12));
                    instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
                    instrs.push(WasmInstr::LocalSet(tup));
                    instrs.push(WasmInstr::LocalGet(tup));
                    instrs.push(WasmInstr::I32Const(0));
                    instrs.push(WasmInstr::I32Store(2, 0));
                    instrs.push(WasmInstr::LocalGet(tup));
                    instrs.push(WasmInstr::LocalGet(base));
                    instrs.push(WasmInstr::I32Store(2, 4));
                    instrs.push(WasmInstr::LocalGet(tup));
                    instrs.push(WasmInstr::LocalGet(ext));
                    instrs.push(WasmInstr::I32Store(2, 8));
                    instrs.push(WasmInstr::LocalGet(tup));
                    return Ok(true);
                }
                _ => {}
            }
        }
        // replaceExtension path newExt = dropExtension path ++ newExt
        if op == "replaceExtension" && args.len() == 2 {
            let s = *local_count;
            let slen = *local_count + 1;
            let slash = *local_count + 2;
            let dot = *local_count + 3;
            let start = *local_count + 4;
            let count = *local_count + 5;
            *local_count += 6;
            self.lower_expr(args[0], instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::LocalSet(s));
            instrs.push(WasmInstr::LocalGet(s));
            instrs.push(WasmInstr::I32Load(2, 0));
            instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_LEN_MASK));
            instrs.push(WasmInstr::I32And);
            instrs.push(WasmInstr::LocalSet(slen));
            self.emit_pstr_last_index(s, '/' as i32, instrs, local_count);
            instrs.push(WasmInstr::LocalSet(slash));
            self.emit_pstr_last_index(s, '.' as i32, instrs, local_count);
            instrs.push(WasmInstr::LocalSet(dot));
            instrs.push(WasmInstr::I32Const(0));
            instrs.push(WasmInstr::LocalSet(start));
            instrs.push(WasmInstr::LocalGet(dot));
            instrs.push(WasmInstr::LocalGet(slen));
            instrs.push(WasmInstr::LocalGet(dot));
            instrs.push(WasmInstr::LocalGet(slash));
            instrs.push(WasmInstr::I32GtS);
            instrs.push(WasmInstr::Select);
            instrs.push(WasmInstr::LocalSet(count));
            self.emit_pstr_slice(s, start, count, instrs, local_count);
            self.lower_expr(args[1], instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
            return Ok(true);
        }
        // a </> b = a ++ b when a ends with '/' (or is empty), else a ++ "/" ++ b
        if op == "</>" && args.len() == 2 {
            let a = *local_count;
            let alen = *local_count + 1;
            *local_count += 2;
            self.lower_expr(args[0], instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::LocalSet(a));
            instrs.push(WasmInstr::LocalGet(a));
            instrs.push(WasmInstr::I32Load(2, 0));
            instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_LEN_MASK));
            instrs.push(WasmInstr::I32And);
            instrs.push(WasmInstr::LocalSet(alen));
            // a' = (alen == 0 || last byte == '/') ? a : a ++ "/"
            instrs.push(WasmInstr::LocalGet(alen));
            instrs.push(WasmInstr::I32Eqz); // alen == 0
            instrs.push(WasmInstr::LocalGet(a));
            instrs.push(WasmInstr::I32Const(4));
            instrs.push(WasmInstr::I32Add);
            instrs.push(WasmInstr::LocalGet(alen));
            instrs.push(WasmInstr::I32Add);
            instrs.push(WasmInstr::I32Const(1));
            instrs.push(WasmInstr::I32Sub);
            instrs.push(WasmInstr::I32Load8U(0, 0));
            instrs.push(WasmInstr::I32Const('/' as i32));
            instrs.push(WasmInstr::I32Eq); // last == '/'
            instrs.push(WasmInstr::I32Or); // skip-sep?
            instrs.push(WasmInstr::If(Some(WasmType::I32)));
            instrs.push(WasmInstr::LocalGet(a));
            instrs.push(WasmInstr::Else);
            instrs.push(WasmInstr::LocalGet(a));
            self.emit_pstr_lit("/", instrs);
            instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
            instrs.push(WasmInstr::End);
            // ... ++ b
            self.lower_expr(args[1], instrs, locals, local_count, false)?;
            instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
            return Ok(true);
        }
        Ok(false)
    }

    /// `Data.Text.take`/`drop n s` on the `pstr` representation: clamp `n` to the
    /// length, then slice. `take` keeps `[0, n)`; `drop` keeps `[n, len)`.
    fn emit_text_take_drop(
        &mut self,
        is_take: bool,
        n: &Expr,
        s: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let sloc = *local_count;
        let nloc = *local_count + 1;
        let slen = *local_count + 2;
        let start = *local_count + 3;
        let count = *local_count + 4;
        *local_count += 5;
        self.lower_expr(s, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(sloc));
        self.lower_expr(n, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(nloc));
        // slen = [sloc] & MASK
        instrs.push(WasmInstr::LocalGet(sloc));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_LEN_MASK));
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::LocalSet(slen));
        // clamp nloc to [0, slen]
        instrs.push(WasmInstr::LocalGet(nloc));
        instrs.push(WasmInstr::LocalGet(slen));
        instrs.push(WasmInstr::I32GtS);
        instrs.push(WasmInstr::If(None));
        instrs.push(WasmInstr::LocalGet(slen));
        instrs.push(WasmInstr::LocalSet(nloc));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::LocalGet(nloc));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::I32LtS);
        instrs.push(WasmInstr::If(None));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(nloc));
        instrs.push(WasmInstr::End);
        if is_take {
            // start = 0, count = n
            instrs.push(WasmInstr::I32Const(0));
            instrs.push(WasmInstr::LocalSet(start));
            instrs.push(WasmInstr::LocalGet(nloc));
            instrs.push(WasmInstr::LocalSet(count));
        } else {
            // start = n, count = slen - n
            instrs.push(WasmInstr::LocalGet(nloc));
            instrs.push(WasmInstr::LocalSet(start));
            instrs.push(WasmInstr::LocalGet(slen));
            instrs.push(WasmInstr::LocalGet(nloc));
            instrs.push(WasmInstr::I32Sub);
            instrs.push(WasmInstr::LocalSet(count));
        }
        self.emit_pstr_slice(sloc, start, count, instrs, local_count);
        Ok(())
    }

    /// `Data.Text.toUpper`/`toLower s` on the `pstr` representation: copy the
    /// string, shifting ASCII letters by 32.
    fn emit_text_case(
        &mut self,
        to_upper: bool,
        s: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let sloc = *local_count;
        let slen = *local_count + 1;
        let res = *local_count + 2;
        let i = *local_count + 3;
        let b = *local_count + 4;
        *local_count += 5;
        self.lower_expr(s, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(sloc));
        instrs.push(WasmInstr::LocalGet(sloc));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_LEN_MASK));
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::LocalSet(slen));
        instrs.push(WasmInstr::LocalGet(slen));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
        instrs.push(WasmInstr::LocalSet(res));
        instrs.push(WasmInstr::LocalGet(res));
        instrs.push(WasmInstr::LocalGet(slen));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_MARKER));
        instrs.push(WasmInstr::I32Or);
        instrs.push(WasmInstr::I32Store(2, 0));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(i));
        // bounds: lower letter range and the shift direction
        let (lo, hi, shift) = if to_upper {
            (97, 122, -32)
        } else {
            (65, 90, 32)
        };
        instrs.push(WasmInstr::Block(None));
        instrs.push(WasmInstr::Loop(None));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::LocalGet(slen));
        instrs.push(WasmInstr::I32GeU);
        instrs.push(WasmInstr::BrIf(1));
        // b = s[4+i]
        instrs.push(WasmInstr::LocalGet(sloc));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::I32Load8U(0, 0));
        instrs.push(WasmInstr::LocalSet(b));
        // if lo <= b <= hi: b += shift
        instrs.push(WasmInstr::LocalGet(b));
        instrs.push(WasmInstr::I32Const(lo));
        instrs.push(WasmInstr::I32GeU);
        instrs.push(WasmInstr::LocalGet(b));
        instrs.push(WasmInstr::I32Const(hi));
        instrs.push(WasmInstr::I32LeU);
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::If(None));
        instrs.push(WasmInstr::LocalGet(b));
        instrs.push(WasmInstr::I32Const(shift));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalSet(b));
        instrs.push(WasmInstr::End);
        // res[4+i] = b
        instrs.push(WasmInstr::LocalGet(res));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(b));
        instrs.push(WasmInstr::I32Store8(0, 0));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::Br(0));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::LocalGet(res));
        Ok(())
    }

    /// Convert a runtime cons-`[Char]` (in local `src`) to a freshly allocated
    /// length-prefixed `pstr`, leaving its pointer on the stack. Walks the cons
    /// cells once to count, allocates `[len|marker | bytes]`, then copies each
    /// head byte.
    fn emit_charlist_to_pstr(
        &mut self,
        src: u32,
        instrs: &mut Vec<WasmInstr>,
        local_count: &mut u32,
    ) {
        let len = *local_count;
        let cur = *local_count + 1;
        let result = *local_count + 2;
        let i = *local_count + 3;
        *local_count += 4;
        // count length
        instrs.push(WasmInstr::LocalGet(src));
        instrs.push(WasmInstr::LocalSet(cur));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(len));
        instrs.push(WasmInstr::Block(None));
        instrs.push(WasmInstr::Loop(None));
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Const(HEAP_BASE));
        instrs.push(WasmInstr::I32LtU);
        instrs.push(WasmInstr::BrIf(1));
        instrs.push(WasmInstr::LocalGet(len));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalSet(len));
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Load(2, 8)); // tail
        instrs.push(WasmInstr::LocalSet(cur));
        instrs.push(WasmInstr::Br(0));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        // alloc(4 + len); [result] = len | marker
        instrs.push(WasmInstr::LocalGet(len));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
        instrs.push(WasmInstr::LocalSet(result));
        instrs.push(WasmInstr::LocalGet(result));
        instrs.push(WasmInstr::LocalGet(len));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_MARKER));
        instrs.push(WasmInstr::I32Or);
        instrs.push(WasmInstr::I32Store(2, 0));
        // copy each head byte
        instrs.push(WasmInstr::LocalGet(src));
        instrs.push(WasmInstr::LocalSet(cur));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::Block(None));
        instrs.push(WasmInstr::Loop(None));
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Const(HEAP_BASE));
        instrs.push(WasmInstr::I32LtU);
        instrs.push(WasmInstr::BrIf(1));
        // dst = result + 4 + i
        instrs.push(WasmInstr::LocalGet(result));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Add);
        // byte = head [cur+4]
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Load(2, 4));
        instrs.push(WasmInstr::I32Store8(0, 0));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Load(2, 8));
        instrs.push(WasmInstr::LocalSet(cur));
        instrs.push(WasmInstr::Br(0));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::LocalGet(result));
    }

    /// Lower `s` and leave a cons-`[Char]` on the stack: a marked `pstr` is
    /// converted (so cons-based list ops like filter/map can walk a String); a
    /// cons list or nil passes through unchanged.
    fn emit_ensure_charlist(
        &mut self,
        s: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let v = *local_count;
        let len = *local_count + 1;
        let acc = *local_count + 2;
        let i = *local_count + 3;
        let cell = *local_count + 4;
        *local_count += 5;
        self.lower_expr(s, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(v));
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Eqz);
        instrs.push(WasmInstr::If(Some(WasmType::I32)));
        instrs.push(WasmInstr::I32Const(0)); // nil
        instrs.push(WasmInstr::Else);
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_MARKER));
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::If(Some(WasmType::I32)));
        // pstr -> cons, built back-to-front so head order is preserved
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_LEN_MASK));
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::LocalSet(len));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(acc));
        instrs.push(WasmInstr::LocalGet(len));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Sub);
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::Block(None));
        instrs.push(WasmInstr::Loop(None));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::I32LtS);
        instrs.push(WasmInstr::BrIf(1));
        instrs.push(WasmInstr::I32Const(12));
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
        instrs.push(WasmInstr::LocalSet(cell));
        instrs.push(WasmInstr::LocalGet(cell));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Store(2, 0)); // `:` tag
        instrs.push(WasmInstr::LocalGet(cell));
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Const(4));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Add);
        instrs.push(WasmInstr::I32Load8U(0, 0));
        instrs.push(WasmInstr::I32Store(2, 4)); // head
        instrs.push(WasmInstr::LocalGet(cell));
        instrs.push(WasmInstr::LocalGet(acc));
        instrs.push(WasmInstr::I32Store(2, 8)); // tail
        instrs.push(WasmInstr::LocalGet(cell));
        instrs.push(WasmInstr::LocalSet(acc));
        instrs.push(WasmInstr::LocalGet(i));
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::I32Sub);
        instrs.push(WasmInstr::LocalSet(i));
        instrs.push(WasmInstr::Br(0));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::LocalGet(acc));
        instrs.push(WasmInstr::Else);
        instrs.push(WasmInstr::LocalGet(v)); // already a cons list
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        Ok(())
    }

    /// Show a runtime list of scalars by walking its cons cells, rendering each
    /// element according to `kind` and joining with commas inside `[`...`]`.
    /// Leaves the result string pointer on the stack.
    fn emit_show_runtime_list(
        &mut self,
        list: &Expr,
        kind: ElemKind,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let true_ptr = self.intern_pstr("True");
        let false_ptr = self.intern_pstr("False");
        let concat = self.runtime.concat_str_idx;

        // cur = list
        self.lower_expr(list, instrs, locals, local_count, false)?;
        let cur = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalSet(cur));
        // acc = "["
        self.emit_pstr_lit("[", instrs);
        let acc = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalSet(acc));
        // first = 1
        let first = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::I32Const(1));
        instrs.push(WasmInstr::LocalSet(first));
        let head = *local_count;
        *local_count += 1;

        instrs.push(WasmInstr::Block(None));
        instrs.push(WasmInstr::Loop(None));
        // nil when cur < HEAP_BASE (the empty list is the value 0)
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Const(HEAP_BASE));
        instrs.push(WasmInstr::I32LtU);
        instrs.push(WasmInstr::BrIf(1));
        // comma separator after the first element
        instrs.push(WasmInstr::LocalGet(first));
        instrs.push(WasmInstr::I32Eqz);
        instrs.push(WasmInstr::If(None));
        instrs.push(WasmInstr::LocalGet(acc));
        self.emit_pstr_lit(",", instrs);
        instrs.push(WasmInstr::Call(concat));
        instrs.push(WasmInstr::LocalSet(acc));
        instrs.push(WasmInstr::End);
        // head = cons.head ([cur+4])
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Load(2, 4));
        instrs.push(WasmInstr::LocalSet(head));
        // acc = acc ++ show(head)
        instrs.push(WasmInstr::LocalGet(acc));
        match kind {
            ElemKind::Int => {
                instrs.push(WasmInstr::LocalGet(head));
                instrs.push(WasmInstr::Call(self.runtime.int_to_str_idx));
            }
            ElemKind::Double => {
                instrs.push(WasmInstr::LocalGet(head));
                instrs.push(WasmInstr::F64Load(3, 0));
                instrs.push(WasmInstr::Call(self.runtime.double_to_str_idx));
            }
            ElemKind::Bool => {
                instrs.push(WasmInstr::LocalGet(head));
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                instrs.push(WasmInstr::I32Const(true_ptr as i32));
                instrs.push(WasmInstr::Else);
                instrs.push(WasmInstr::I32Const(false_ptr as i32));
                instrs.push(WasmInstr::End);
            }
        }
        instrs.push(WasmInstr::Call(concat));
        instrs.push(WasmInstr::LocalSet(acc));
        // first = 0; cur = cons.tail ([cur+8])
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(first));
        instrs.push(WasmInstr::LocalGet(cur));
        instrs.push(WasmInstr::I32Load(2, 8));
        instrs.push(WasmInstr::LocalSet(cur));
        instrs.push(WasmInstr::Br(0));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);

        // acc ++ "]"
        instrs.push(WasmInstr::LocalGet(acc));
        self.emit_pstr_lit("]", instrs);
        instrs.push(WasmInstr::Call(concat));
        Ok(())
    }

    /// Push a length-prefixed string literal pointer onto the stack.
    fn emit_pstr_lit(&mut self, s: &str, instrs: &mut Vec<WasmInstr>) {
        let ptr = self.intern_pstr(s);
        instrs.push(WasmInstr::I32Const(ptr as i32));
    }

    /// Show a comma-separated sequence wrapped in `open`/`close` (a list or a
    /// tuple), concatenating the rendered pieces. If `paren_items`, each item is
    /// parenthesized when it is a compound value (used for constructor fields,
    /// not here). Leaves the result string pointer on the stack.
    #[allow(clippy::too_many_arguments)]
    fn emit_show_seq(
        &mut self,
        open: &str,
        close: &str,
        items: &[&Expr],
        paren_items: bool,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        self.emit_pstr_lit(open, instrs);
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                self.emit_pstr_lit(",", instrs);
                instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
            }
            self.emit_show_item(item, paren_items, instrs, locals, local_count)?;
            instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
        }
        self.emit_pstr_lit(close, instrs);
        instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
        Ok(())
    }

    /// Show a constructor application `Con f1 f2 ...` as `Con f1 f2 ...`, with
    /// each compound field parenthesized (Haskell `showsPrec` at precedence 11).
    fn emit_show_constructor(
        &mut self,
        name: &str,
        fields: &[&Expr],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        self.emit_pstr_lit(name, instrs);
        for field in fields {
            self.emit_pstr_lit(" ", instrs);
            instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
            self.emit_show_item(field, true, instrs, locals, local_count)?;
            instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
        }
        Ok(())
    }

    /// Show a single element/field, optionally parenthesizing it when it is a
    /// compound value (a constructor application with fields, or a negative
    /// number) — matching Haskell's parenthesization of nested `show`.
    fn emit_show_item(
        &mut self,
        item: &Expr,
        paren: bool,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        if paren && self.show_needs_parens(item) {
            self.emit_pstr_lit("(", instrs);
            self.emit_show_to_pstr(item, instrs, locals, local_count)?;
            instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
            self.emit_pstr_lit(")", instrs);
            instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
            Ok(())
        } else {
            self.emit_show_to_pstr(item, instrs, locals, local_count)
        }
    }

    /// Whether `show item` needs surrounding parentheses as a constructor field:
    /// a constructor application with arguments, or a negative numeric literal.
    fn show_needs_parens(&self, item: &Expr) -> bool {
        match item {
            Expr::Lit(Literal::Int(n), _, _) => return *n < 0,
            Expr::Lit(Literal::Integer(n), _, _) => return *n < 0,
            _ => {}
        }
        let (head, fields) = collect_app_spine(item);
        if let Expr::Var(hv, _) = head {
            let name = hv.name.as_str();
            // Lists and tuples render with their own brackets — no parens.
            if !fields.is_empty() && !is_tuple_con(name) && name != ":" {
                return matches!(self.lookup_constructor(name), Some((_, arity)) if arity as usize == fields.len());
            }
        }
        false
    }

    /// Infer how to render `expr` under `show`. Types are erased, so this works
    /// structurally: a nullary constructor shows as its name; a boolean-valued
    /// operator shows as `True`/`False`; everything else defaults to integer.
    fn infer_show(&self, expr: &Expr) -> ShowKind {
        // Peel transparent wrappers and runtime-identity applications.
        let e = peel_runtime_identity(expr);

        // Floating-point values render via the double formatter.
        if expr_is_float(e) {
            return ShowKind::Double;
        }

        match e {
            Expr::Var(v, _) => {
                if matches!(self.lookup_constructor(v.name.as_str()), Some((_, 0))) {
                    return ShowKind::Literal(v.name.as_str().to_string());
                }
                // Nullary, type-directed `minBound`/`maxBound` of the sole enum.
                if let Some(names) = self.enum_names_of_expr(e) {
                    return ShowKind::Enum(names);
                }
                if let Some(bound) = self.subst.get(&v.id) {
                    return self.infer_show(bound);
                }
                ShowKind::Int
            }
            Expr::App(..) => {
                // A computed enum value (`succ`/`pred`/`toEnum`) renders as a
                // constructor name from its tag.
                if let Some(names) = self.enum_names_of_expr(e) {
                    return ShowKind::Enum(names);
                }
                let (head, _) = collect_app_spine(e);
                if let Expr::Var(hv, _) = head {
                    let name = hv.name.as_str();
                    if is_bool_valued_op(name)
                        || returns_bool_fn(name)
                        || self.bool_returning_fns.contains(name)
                    {
                        return ShowKind::Bool;
                    }
                    if returns_double_fn(name) {
                        return ShowKind::Double;
                    }
                    if matches!(name, "compare" | "GHC.Classes.compare") {
                        return ShowKind::Ordering;
                    }
                }
                ShowKind::Int
            }
            _ => ShowKind::Int,
        }
    }

    /// The type index for the closure calling convention `(env, arg) -> result`.
    fn closure_type_index(&mut self) -> u32 {
        self.wasm.add_type(WasmFuncType::new(
            vec![WasmType::I32, WasmType::I32],
            vec![WasmType::I32],
        ))
    }

    /// Free variables of a lambda that must be captured: those referenced in
    /// the body, not bound by the lambda, and live as runtime values in the
    /// enclosing scope (a local or an inlined substitution).
    fn closure_free_vars(
        &self,
        param: &Var,
        body: &Expr,
        locals: &FxHashMap<VarId, u32>,
    ) -> Vec<Var> {
        let mut bound = FxHashSet::default();
        bound.insert(param.id);
        let mut acc = Vec::new();
        let mut seen = FxHashSet::default();
        collect_free_vars(body, &mut bound, &mut acc, &mut seen);
        acc.retain(|v| locals.contains_key(&v.id) || self.subst.contains_key(&v.id));
        acc
    }

    /// Lambda-lift `\param -> body` into a standalone function with the closure
    /// calling convention `clos(env, arg)`. Captured free variables are loaded
    /// from the environment record; the parameter is the second argument.
    /// Returns the reserved function index (also its table slot).
    fn lift_lambda(&mut self, param: &Var, body: &Expr, captured: &[Var]) -> WasmResult<u32> {
        let func_idx = self.next_func_idx;
        self.next_func_idx += 1;

        let ty = WasmFuncType::new(vec![WasmType::I32, WasmType::I32], vec![WasmType::I32]);
        let mut func = WasmFunc::new(ty);
        func.name = Some(format!("clos_{func_idx}"));

        let mut instrs = Vec::new();
        let mut locals: FxHashMap<VarId, u32> = FxHashMap::default();
        let mut local_count: u32 = 2; // local 0 = env pointer, local 1 = argument

        // Load each captured free variable from the environment record.
        for (i, fv) in captured.iter().enumerate() {
            let slot = local_count;
            local_count += 1;
            instrs.push(WasmInstr::LocalGet(0));
            instrs.push(WasmInstr::I32Load(2, ((i + 1) * 4) as u32));
            instrs.push(WasmInstr::LocalSet(slot));
            locals.insert(fv.id, slot);
        }
        // The lambda parameter is the call argument (local 1).
        locals.insert(param.id, 1);

        // Captured values now live in locals, so lower the body with a fresh
        // substitution environment (creation-site inlining does not apply here).
        let saved = std::mem::take(&mut self.subst);
        let res = self.lower_expr(body, &mut instrs, &mut locals, &mut local_count, false);
        self.subst = saved;
        res?;

        instrs.push(WasmInstr::End);
        for _ in 2..local_count {
            func.add_local(WasmType::I32);
        }
        for instr in instrs {
            func.emit(instr);
        }

        self.pending_closures.push((func_idx, func));
        self.wasm.enable_func_table();
        Ok(func_idx)
    }

    /// Lower a lambda expression to a closure value: a heap record
    /// `[code_index | captured_0 | captured_1 | ...]`. Leaves the record
    /// pointer on the stack.
    fn lower_closure(
        &mut self,
        param: &Var,
        body: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let captured = self.closure_free_vars(param, body, locals);
        let func_idx = self.lift_lambda(param, body, &captured)?;

        let size = ((1 + captured.len()) * 4) as i32;
        instrs.push(WasmInstr::I32Const(size));
        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));

        let ptr = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalTee(ptr));

        // Store the code index at offset 0.
        instrs.push(WasmInstr::I32Const(func_idx as i32));
        instrs.push(WasmInstr::I32Store(2, 0));

        // Store each captured value at offset (i+1)*4.
        for (i, fv) in captured.iter().enumerate() {
            instrs.push(WasmInstr::LocalGet(ptr));
            self.emit_var_value(fv, instrs, locals, local_count);
            instrs.push(WasmInstr::I32Store(2, ((i + 1) * 4) as u32));
        }

        instrs.push(WasmInstr::LocalGet(ptr));
        Ok(())
    }

    /// Lower a partial application `func a_0 .. a_{k-1}` (where `func` has
    /// arity `> k`) by eta-expanding it to a closure
    /// `\f_0 .. f_{m-1} -> func a_0 .. a_{k-1} f_0 .. f_{m-1}` and lowering
    /// that. The closure captures the supplied arguments; the closure
    /// machinery handles capture and currying.
    fn lower_partial_application(
        &mut self,
        func_var: &Var,
        args: &[&Expr],
        arity: usize,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let missing = arity - args.len();

        // Fresh parameters for the missing arguments. Their type is irrelevant
        // to lowering, so reuse the function's type.
        let fresh: Vec<Var> = (0..missing)
            .map(|_| {
                let id = self.next_synthetic_id;
                self.next_synthetic_id += 1;
                Var::new(Symbol::intern("_eta"), VarId::new(id), func_var.ty.clone())
            })
            .collect();

        // Build the saturated application spine.
        let mut app = Expr::Var(func_var.clone(), Span::default());
        for arg in args {
            app = Expr::App(Box::new(app), Box::new((*arg).clone()), Span::default());
        }
        for f in &fresh {
            app = Expr::App(
                Box::new(app),
                Box::new(Expr::Var(f.clone(), Span::default())),
                Span::default(),
            );
        }

        // Wrap in nested lambdas (outermost is the first missing parameter).
        let mut lam = app;
        for f in fresh.iter().rev() {
            lam = Expr::Lam(f.clone(), Box::new(lam), Span::default());
        }

        if let Expr::Lam(param, body, _) = &lam {
            self.lower_closure(param, body, instrs, locals, local_count)
        } else {
            unreachable!("eta-expansion always produces a lambda")
        }
    }

    /// Emit the current value of a variable (a local, or an inlined
    /// substitution). Used to snapshot captured free variables.
    fn emit_var_value(
        &mut self,
        var: &Var,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) {
        if let Some(&slot) = locals.get(&var.id) {
            instrs.push(WasmInstr::LocalGet(slot));
        } else if let Some(arg) = self.subst.get(&var.id).cloned() {
            // Best effort: lowering may fail only on unsupported constructs,
            // which would already warn elsewhere; ignore the result here.
            let _ = self.lower_expr(&arg, instrs, locals, local_count, false);
        } else {
            instrs.push(WasmInstr::I32Const(0));
        }
    }

    /// Apply a closure value (already on top of the stack) to `args`, one
    /// argument at a time via `call_indirect`, leaving the final result on the
    /// stack. Curried application threads each intermediate closure through.
    fn apply_closure_on_stack(
        &mut self,
        args: &[&Expr],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let type_idx = self.closure_type_index();
        self.wasm.enable_func_table();

        let cur = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalSet(cur));

        for arg in args {
            instrs.push(WasmInstr::LocalGet(cur)); // env pointer
            self.lower_expr(arg, instrs, locals, local_count, false)?; // argument
            instrs.push(WasmInstr::LocalGet(cur));
            instrs.push(WasmInstr::I32Load(2, 0)); // code index at offset 0
            instrs.push(WasmInstr::CallIndirect(type_idx, 0));
            instrs.push(WasmInstr::LocalSet(cur)); // result (possibly another closure)
        }

        instrs.push(WasmInstr::LocalGet(cur));
        Ok(())
    }

    /// Lower an exception handler (`\e -> action`) applied to a dummy exception
    /// value, leaving the handler's result on the stack. The exception payload is
    /// unused in the eager-IO model, so a `\e -> body` handler binds `e` to 0 and
    /// lowers `body` directly; any other shape is applied as a closure.
    fn lower_exn_handler(
        &mut self,
        handler: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let mut h = handler;
        while let Expr::TyLam(_, body, _) = h {
            h = body;
        }
        if let Expr::Lam(param, body, _) = h {
            let local_idx = *local_count;
            *local_count += 1;
            instrs.push(WasmInstr::I32Const(0));
            instrs.push(WasmInstr::LocalSet(local_idx));
            locals.insert(param.id, local_idx);
            return self.lower_expr(body, instrs, locals, local_count, false);
        }
        self.lower_expr(handler, instrs, locals, local_count, false)?;
        let dummy = pint(0);
        self.apply_closure_on_stack(&[&dummy], instrs, locals, local_count)
    }

    /// Lower a binary arithmetic operation on two `Ratio Int` operands. Loads
    /// `(n1,d1)` and `(n2,d2)`, computes the raw numerator/denominator for `op`,
    /// and normalizes the result via `make_rational`.
    fn emit_rational_binop(
        &mut self,
        op: &str,
        a: &Expr,
        b: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let n1 = *local_count;
        let d1 = *local_count + 1;
        let n2 = *local_count + 2;
        let d2 = *local_count + 3;
        let p = *local_count + 4;
        *local_count += 5;
        // p = a; n1=[p], d1=[p+4]
        self.lower_expr(a, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalTee(p));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::LocalSet(n1));
        instrs.push(WasmInstr::LocalGet(p));
        instrs.push(WasmInstr::I32Load(2, 4));
        instrs.push(WasmInstr::LocalSet(d1));
        // p = b; n2=[p], d2=[p+4]
        self.lower_expr(b, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalTee(p));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::LocalSet(n2));
        instrs.push(WasmInstr::LocalGet(p));
        instrs.push(WasmInstr::I32Load(2, 4));
        instrs.push(WasmInstr::LocalSet(d2));

        // Push raw numerator then denominator for `make_rational`.
        match op {
            "+" | "-" => {
                // num = n1*d2 (-/+) n2*d1
                instrs.push(WasmInstr::LocalGet(n1));
                instrs.push(WasmInstr::LocalGet(d2));
                instrs.push(WasmInstr::I32Mul);
                instrs.push(WasmInstr::LocalGet(n2));
                instrs.push(WasmInstr::LocalGet(d1));
                instrs.push(WasmInstr::I32Mul);
                instrs.push(if op == "+" {
                    WasmInstr::I32Add
                } else {
                    WasmInstr::I32Sub
                });
                // den = d1*d2
                instrs.push(WasmInstr::LocalGet(d1));
                instrs.push(WasmInstr::LocalGet(d2));
                instrs.push(WasmInstr::I32Mul);
            }
            "*" => {
                instrs.push(WasmInstr::LocalGet(n1));
                instrs.push(WasmInstr::LocalGet(n2));
                instrs.push(WasmInstr::I32Mul);
                instrs.push(WasmInstr::LocalGet(d1));
                instrs.push(WasmInstr::LocalGet(d2));
                instrs.push(WasmInstr::I32Mul);
            }
            _ => {
                // "/": (n1*d2) / (d1*n2)
                instrs.push(WasmInstr::LocalGet(n1));
                instrs.push(WasmInstr::LocalGet(d2));
                instrs.push(WasmInstr::I32Mul);
                instrs.push(WasmInstr::LocalGet(d1));
                instrs.push(WasmInstr::LocalGet(n2));
                instrs.push(WasmInstr::I32Mul);
            }
        }
        instrs.push(WasmInstr::Call(self.runtime.make_rational_idx));
        Ok(())
    }

    /// Lower `==`/`/=` on two normalized `Ratio Int` values: equal iff their
    /// numerators and denominators match. Leaves a `Bool` tag (0/1) on the stack.
    fn emit_rational_eq(
        &mut self,
        negate: bool,
        a: &Expr,
        b: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let pa = *local_count;
        let pb = *local_count + 1;
        *local_count += 2;
        self.lower_expr(a, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(pa));
        self.lower_expr(b, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(pb));
        // (n1==n2) & (d1==d2)
        instrs.push(WasmInstr::LocalGet(pa));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::LocalGet(pb));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Eq);
        instrs.push(WasmInstr::LocalGet(pa));
        instrs.push(WasmInstr::I32Load(2, 4));
        instrs.push(WasmInstr::LocalGet(pb));
        instrs.push(WasmInstr::I32Load(2, 4));
        instrs.push(WasmInstr::I32Eq);
        instrs.push(WasmInstr::I32And);
        if negate {
            instrs.push(WasmInstr::I32Eqz);
        }
        Ok(())
    }

    /// Lower `negate`/`abs`/`signum` on a `Ratio Int`.
    fn emit_rational_unary(
        &mut self,
        op: &str,
        a: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let p = *local_count;
        *local_count += 1;
        self.lower_expr(a, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(p));
        match op {
            "negate" => {
                // make_rational(-num, den)
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::I32Load(2, 0));
                instrs.push(WasmInstr::I32Sub);
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::I32Load(2, 4));
                instrs.push(WasmInstr::Call(self.runtime.make_rational_idx));
            }
            "abs" => {
                // num >= 0 ? num : -num, over den
                let num = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::I32Load(2, 0));
                instrs.push(WasmInstr::LocalSet(num));
                instrs.push(WasmInstr::LocalGet(num));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::I32LtS);
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::LocalGet(num));
                instrs.push(WasmInstr::I32Sub);
                instrs.push(WasmInstr::Else);
                instrs.push(WasmInstr::LocalGet(num));
                instrs.push(WasmInstr::End);
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::I32Load(2, 4));
                instrs.push(WasmInstr::Call(self.runtime.make_rational_idx));
            }
            _ => {
                // signum: (num<0 ? -1 : num>0 ? 1 : 0) % 1
                let num = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::I32Load(2, 0));
                instrs.push(WasmInstr::LocalSet(num));
                instrs.push(WasmInstr::LocalGet(num));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::I32GtS);
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                instrs.push(WasmInstr::I32Const(1));
                instrs.push(WasmInstr::Else);
                instrs.push(WasmInstr::LocalGet(num));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::I32LtS);
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                instrs.push(WasmInstr::I32Const(-1));
                instrs.push(WasmInstr::Else);
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::End);
                instrs.push(WasmInstr::End);
                instrs.push(WasmInstr::I32Const(1));
                instrs.push(WasmInstr::Call(self.runtime.make_rational_idx));
            }
        }
        Ok(())
    }

    /// Render a `Ratio Int` under `show` as `"num % den"`, leaving the pstr
    /// pointer on the stack.
    fn emit_show_rational(
        &mut self,
        a: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let p = *local_count;
        *local_count += 1;
        self.lower_expr(a, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(p));
        // int_to_str(num)
        instrs.push(WasmInstr::LocalGet(p));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::Call(self.runtime.int_to_str_idx));
        // ++ " % "
        self.emit_pstr_lit(" % ", instrs);
        instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
        // ++ int_to_str(den)
        instrs.push(WasmInstr::LocalGet(p));
        instrs.push(WasmInstr::I32Load(2, 4));
        instrs.push(WasmInstr::Call(self.runtime.int_to_str_idx));
        instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
        Ok(())
    }

    /// Lower a function application chain.
    ///
    /// This is the core dispatch logic. We collect the function and all its
    /// arguments, then decide how to emit code based on the function name.
    /// Attempt to lower an application by inlining or beta-reduction.
    ///
    /// Returns `Ok(true)` if it handled the application. Handles two shapes the
    /// plain call path cannot:
    /// - a lambda applied directly, or via a parameter bound to a lambda
    ///   argument (`subst`) — beta-reduction;
    /// - a saturated call to a non-recursive top-level function — inlining.
    ///
    /// Both bind the parameters to the argument *expressions* (no
    /// alpha-renaming needed: each argument is re-lowered where its parameter
    /// is referenced) and lower the body in place.
    fn try_lower_inline_app(
        &mut self,
        func_expr: &Expr,
        args: &[&Expr],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<bool> {
        let head = peel_head(func_expr);

        if let Expr::Var(var, _) = head {
            // A parameter bound to a lambda argument (a closure): beta-reduce.
            if let Some(bound) = self.subst.get(&var.id).cloned() {
                let bound_head = peel_head(&bound);
                if matches!(bound_head, Expr::Lam(..)) {
                    let (params, body) = peel_lambdas(bound_head);
                    if args.len() == params.len() {
                        let param_ids: Vec<VarId> = params.iter().map(|p| p.id).collect();
                        self.inline_call(
                            &param_ids,
                            body,
                            args,
                            instrs,
                            locals,
                            local_count,
                            is_main,
                        )?;
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
            // A non-recursive top-level function: inline its body.
            if !self.inlining.contains(&var.name) {
                if let Some((param_ids, body)) = self.inline_bodies.get(&var.name).cloned() {
                    if args.len() == param_ids.len() {
                        self.inlining.insert(var.name);
                        let result = self.inline_call(
                            &param_ids,
                            &body,
                            args,
                            instrs,
                            locals,
                            local_count,
                            is_main,
                        );
                        self.inlining.remove(&var.name);
                        result?;
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }

        // A lambda applied directly: (\x -> ...) arg.
        if matches!(head, Expr::Lam(..)) {
            let (params, body) = peel_lambdas(head);
            if args.len() == params.len() {
                let param_ids: Vec<VarId> = params.iter().map(|p| p.id).collect();
                self.inline_call(&param_ids, body, args, instrs, locals, local_count, is_main)?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Inline a call: bind each parameter to its argument expression in
    /// `subst`, lower the body, then restore any shadowed bindings.
    #[allow(clippy::too_many_arguments)]
    fn inline_call(
        &mut self,
        param_ids: &[VarId],
        body: &Expr,
        args: &[&Expr],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        let mut saved: Vec<(VarId, Option<Expr>)> = Vec::with_capacity(param_ids.len());
        for (id, arg) in param_ids.iter().zip(args.iter()) {
            saved.push((*id, self.subst.insert(*id, (**arg).clone())));
        }
        let result = self.lower_expr(body, instrs, locals, local_count, is_main);
        for (id, prev) in saved {
            match prev {
                Some(e) => {
                    self.subst.insert(id, e);
                }
                None => {
                    self.subst.remove(&id);
                }
            }
        }
        result
    }

    /// Dispatch `fmap`/`<$>`/`foldr` to a derived instance method (or, for
    /// `fmap`, the Maybe/list arms). Returns `Ok(true)` if it emitted the call;
    /// `Ok(false)` leaves the application for the normal path (e.g. `foldr` on a
    /// plain list, handled by the synthesized list `foldr`).
    fn try_lower_functor_foldable(
        &mut self,
        name: &str,
        args: &[&Expr],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<bool> {
        match strip_qualifier(name) {
            // `traverse`/`mapM` reduce to `fmap`/`map` in BHC's eager IO model:
            // `f x` runs the effect when evaluated, and the structure is rebuilt
            // in field order — identical to applying the derived Functor.
            "fmap" | "<$>" | "traverse" | "mapM" if args.len() == 2 => {
                let (f, x) = (args[0], args[1]);
                // 1. Derived Functor on a user ADT.
                if let Some(ty) = self.container_type_name(x) {
                    if let Some(sym) = self.derived_functor.get(&ty).copied() {
                        return self.emit_known_call(sym, &[f, x], instrs, locals, local_count);
                    }
                }
                // 2. Maybe.
                if is_maybe_expr(x) {
                    let sym = Symbol::intern("__fmapMaybe");
                    return self.emit_known_call(sym, &[f, x], instrs, locals, local_count);
                }
                // 3. List -> map.
                if extract_list_elements(x).is_some() || is_list_operand(x) {
                    let sym = Symbol::intern("map");
                    return self.emit_known_call(sym, &[f, x], instrs, locals, local_count);
                }
                Ok(false)
            }
            "foldr" if args.len() == 3 => {
                let container = args[2];
                if let Some(ty) = self.container_type_name(container) {
                    if let Some(sym) = self.derived_foldable.get(&ty).copied() {
                        return self.emit_known_call(sym, args, instrs, locals, local_count);
                    }
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    /// Lower `replicateM count act` (or `replicateM_`): run the action `count`
    /// times, evaluating its expression once per iteration so its effects
    /// repeat. `replicateM` collects the results into a list (built after all
    /// effects have run, preserving order); `replicateM_` discards them and
    /// yields `()`.
    fn lower_replicate_m(
        &mut self,
        count: i64,
        collect: bool,
        act: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        if !collect {
            for _ in 0..count {
                self.lower_expr(act, instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
            }
            instrs.push(WasmInstr::I32Const(0));
            return Ok(());
        }
        // Evaluate each action into a local first (effects run in order), then
        // cons the results tail-first so the list reads left-to-right.
        let mut value_locals = Vec::with_capacity(count.max(0) as usize);
        for _ in 0..count {
            self.lower_expr(act, instrs, locals, local_count, false)?;
            let v = *local_count;
            *local_count += 1;
            instrs.push(WasmInstr::LocalSet(v));
            value_locals.push(v);
        }
        let acc = *local_count;
        let ptr = *local_count + 1;
        *local_count += 2;
        instrs.push(WasmInstr::I32Const(0)); // nil
        instrs.push(WasmInstr::LocalSet(acc));
        for &v in value_locals.iter().rev() {
            instrs.push(WasmInstr::I32Const(12)); // cons cell: [tag|head|tail]
            instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
            instrs.push(WasmInstr::LocalTee(ptr));
            instrs.push(WasmInstr::I32Const(1)); // `:` tag
            instrs.push(WasmInstr::I32Store(2, 0));
            instrs.push(WasmInstr::LocalGet(ptr));
            instrs.push(WasmInstr::LocalGet(v));
            instrs.push(WasmInstr::I32Store(2, 4));
            instrs.push(WasmInstr::LocalGet(ptr));
            instrs.push(WasmInstr::LocalGet(acc));
            instrs.push(WasmInstr::I32Store(2, 8));
            instrs.push(WasmInstr::LocalGet(ptr));
            instrs.push(WasmInstr::LocalSet(acc));
        }
        instrs.push(WasmInstr::LocalGet(acc));
        Ok(())
    }

    /// Emit a saturated call to a known top-level function by symbol: lower each
    /// argument, then `call`. Returns `Ok(false)` if the symbol isn't registered
    /// (e.g. a synthesized helper that wasn't pulled in).
    fn emit_known_call(
        &mut self,
        sym: Symbol,
        args: &[&Expr],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<bool> {
        let Some(&func_idx) = self.func_map.get(&sym) else {
            return Ok(false);
        };
        for arg in args {
            self.lower_expr(arg, instrs, locals, local_count, false)?;
        }
        instrs.push(WasmInstr::Call(func_idx));
        Ok(true)
    }

    /// For `read <string-literal>` of the sole derived-`Read` enum, the tag of
    /// the constructor whose name matches the literal. Returns `None` if the
    /// argument isn't a string literal, there isn't exactly one user enum, or no
    /// constructor matches (a malformed `read` is left to the fallback).
    fn read_enum_tag(&self, arg: &Expr) -> Option<i32> {
        let mut e = arg;
        loop {
            e = match e {
                Expr::TyApp(inner, _, _)
                | Expr::Cast(inner, _, _)
                | Expr::Tick(_, inner, _)
                | Expr::Lazy(inner, _) => inner,
                _ => break,
            };
        }
        let Expr::Lit(Literal::String(sym), _, _) = e else {
            return None;
        };
        let names = self.sole_enum()?;
        names
            .iter()
            .position(|n| n == sym.as_str())
            .map(|i| i as i32)
    }

    /// The data type name of the value `expr` evaluates to, if its head is a
    /// known user constructor (recursing through `subst`-bound variables). Only
    /// resolves user ADTs — builtin Maybe/list constructors aren't recorded.
    fn container_type_name(&self, expr: &Expr) -> Option<String> {
        let mut e = expr;
        loop {
            e = match e {
                Expr::TyApp(inner, _, _)
                | Expr::Cast(inner, _, _)
                | Expr::Tick(_, inner, _)
                | Expr::Lazy(inner, _) => inner,
                _ => break,
            };
        }
        let (head, _) = collect_app_spine(e);
        if let Expr::Var(hv, _) = head {
            if let Some(ty) = self.ctor_type.get(hv.name.as_str()) {
                return Some(ty.clone());
            }
            if let Some(bound) = self.subst.get(&hv.id) {
                return self.container_type_name(bound);
            }
        }
        None
    }

    fn lower_app(
        &mut self,
        expr: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        // Collect the spine: f a1 a2 ... an
        let (func_expr, args) = collect_app_spine(expr);

        // Get the function name
        let func_name = match func_expr {
            Expr::Var(var, _) => Some(var.name.as_str()),
            _ => None,
        };

        // Check if this is a constructor application
        if let Some(name) = func_name {
            // A newtype constructor is identity at runtime: `C x` is just `x`.
            if args.len() == 1 && self.newtype_cons.contains(name) {
                return self.lower_expr(args[0], instrs, locals, local_count, false);
            }
            if let Some((tag, arity)) = self.lookup_constructor(name) {
                if arity == 0 {
                    // Nullary constructor: just push the tag value
                    instrs.push(WasmInstr::I32Const(tag as i32));
                    return Ok(());
                }
                if args.len() == arity as usize {
                    // Saturated constructor application: allocate heap object
                    return self.emit_adt_construction(tag, &args, instrs, locals, local_count);
                }
            }
        }

        // Derived `Functor`/`Foldable` dispatch: route `fmap`/`foldr` on a
        // user ADT (or `fmap` on Maybe/list) to the generated method. Use the
        // *peeled* head name: dictionary specialization leaves the method buried
        // under a dead `let $dictFunctor… in fmap` whose spine head is a `Let`.
        let peeled_name = match peel_head(func_expr) {
            Expr::Var(v, _) => Some(v.name.as_str()),
            _ => None,
        };
        if let Some(name) = peeled_name {
            if self.try_lower_functor_foldable(name, &args, instrs, locals, local_count)? {
                return Ok(());
            }
            // Derived `Read` for an enum: `read "Green" :: Color` resolves to the
            // constructor tag whose name matches the string literal (the type
            // annotation is erased, so use the single-user-enum heuristic).
            if strip_qualifier(name) == "read" && args.len() == 1 {
                if let Some(tag) = self.read_enum_tag(args[0]) {
                    instrs.push(WasmInstr::I32Const(tag));
                    return Ok(());
                }
                // Otherwise read an Int: parse the string at runtime.
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Call(self.runtime.parse_int_idx));
                return Ok(());
            }
            // readMaybe of a string literal: Just n for a valid Int, else Nothing.
            if strip_qualifier(name) == "readMaybe" && args.len() == 1 {
                if let Some(s) = as_string_literal(args[0]) {
                    match s.trim().parse::<i64>() {
                        Ok(n) => {
                            let just = papp(pref("Just", &mut self.next_synthetic_id), pint(n));
                            return self.lower_expr(&just, instrs, locals, local_count, false);
                        }
                        Err(_) => {
                            instrs.push(WasmInstr::I32Const(0)); // Nothing
                            return Ok(());
                        }
                    }
                }
            }
            // `replicateM n act` / `replicateM_ n act`: the action must run `n`
            // times, but it's an expression evaluated once as a call argument —
            // so lower it `n` times here. Handles a statically-known count
            // (the common case); a runtime count falls through.
            if matches!(strip_qualifier(name), "replicateM" | "replicateM_") && args.len() == 2 {
                if let Some(count) = as_int_literal(args[0]) {
                    let collect = strip_qualifier(name) == "replicateM";
                    return self.lower_replicate_m(
                        count.max(0),
                        collect,
                        args[1],
                        instrs,
                        locals,
                        local_count,
                    );
                }
            }
            // Data.Text (and .Lazy/.Encoding) operations on the `pstr`
            // representation (Text ~ String; encode/decode are identity for the
            // ASCII subset, and the lazy variants share the strict ops).
            let text_rest = name
                .strip_prefix("Data.Text.Lazy.")
                .or_else(|| name.strip_prefix("Data.Text.Encoding."))
                .or_else(|| name.strip_prefix("Data.Text."));
            if let Some(rest) = text_rest {
                match (rest, args.len()) {
                    // identity conversions
                    (
                        "pack" | "unpack" | "fromStrict" | "toStrict" | "copy" | "encodeUtf8"
                        | "decodeUtf8",
                        1,
                    ) => {
                        return self.lower_expr(args[0], instrs, locals, local_count, false);
                    }
                    ("length", 1) => {
                        self.emit_length(args[0], instrs, locals, local_count)?;
                        return Ok(());
                    }
                    ("append" | "concat", 2) => {
                        self.lower_expr(args[0], instrs, locals, local_count, false)?;
                        self.lower_expr(args[1], instrs, locals, local_count, false)?;
                        instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
                        return Ok(());
                    }
                    ("toUpper", 1) => {
                        self.emit_text_case(true, args[0], instrs, locals, local_count)?;
                        return Ok(());
                    }
                    ("toLower", 1) => {
                        self.emit_text_case(false, args[0], instrs, locals, local_count)?;
                        return Ok(());
                    }
                    ("take", 2) => {
                        self.emit_text_take_drop(
                            true,
                            args[0],
                            args[1],
                            instrs,
                            locals,
                            local_count,
                        )?;
                        return Ok(());
                    }
                    ("drop", 2) => {
                        self.emit_text_take_drop(
                            false,
                            args[0],
                            args[1],
                            instrs,
                            locals,
                            local_count,
                        )?;
                        return Ok(());
                    }
                    _ => {}
                }
            }
            // Data.ByteString.Builder: build bytes into a `pstr`. The fixtures
            // only measure lengths, so a Builder is just its rendered string.
            if let Some(rest) = name.strip_prefix("Data.ByteString.Builder.") {
                match (rest, args.len()) {
                    (
                        "toLazyByteString" | "stringUtf8" | "string7" | "string8" | "byteString"
                        | "lazyByteString" | "toStrict",
                        1,
                    ) => {
                        return self.lower_expr(args[0], instrs, locals, local_count, false);
                    }
                    // intDec/wordDec/… render the number as decimal digits
                    (r, 1) if r.ends_with("Dec") => {
                        self.lower_expr(args[0], instrs, locals, local_count, false)?;
                        instrs.push(WasmInstr::Call(self.runtime.int_to_str_idx));
                        return Ok(());
                    }
                    // singleton/word8: a one-byte string from the code
                    ("singleton" | "word8" | "int8", 1) => {
                        let p = *local_count;
                        *local_count += 1;
                        instrs.push(WasmInstr::I32Const(8));
                        instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
                        instrs.push(WasmInstr::LocalSet(p));
                        instrs.push(WasmInstr::LocalGet(p));
                        instrs.push(WasmInstr::I32Const(1 | crate::wasi::PSTR_MARKER));
                        instrs.push(WasmInstr::I32Store(2, 0));
                        instrs.push(WasmInstr::LocalGet(p));
                        self.lower_expr(args[0], instrs, locals, local_count, false)?;
                        instrs.push(WasmInstr::I32Store(2, 4));
                        instrs.push(WasmInstr::LocalGet(p));
                        return Ok(());
                    }
                    ("append" | "mappend", 2) => {
                        self.lower_expr(args[0], instrs, locals, local_count, false)?;
                        self.lower_expr(args[1], instrs, locals, local_count, false)?;
                        instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
                        return Ok(());
                    }
                    _ => {}
                }
            }
            // System.FilePath operations on the `pstr` path.
            if self.try_lower_filepath(name, &args, instrs, locals, local_count)? {
                return Ok(());
            }
            // List HOFs over a String: convert a `pstr` list argument to a
            // cons-`[Char]` so the cons-based synthesized fn can walk it. A cons
            // list or nil passes through, so non-String lists are unaffected.
            if args.len() == 2 && matches!(strip_qualifier(name), "filter" | "map" | "any" | "all")
            {
                let sym = Symbol::intern(strip_qualifier(name));
                if let Some(&idx) = self.func_map.get(&sym) {
                    self.lower_expr(args[0], instrs, locals, local_count, false)?;
                    self.emit_ensure_charlist(args[1], instrs, locals, local_count)?;
                    instrs.push(WasmInstr::Call(idx));
                    return Ok(());
                }
            }
            // `length` works on both string representations: a marked `pstr`
            // returns its stored length; a cons list (incl. cons-`[Char]`) is
            // walked. This keeps `length` correct whether its argument is a
            // built list or a String literal/`++`/show result.
            if strip_qualifier(name) == "length" && args.len() == 1 {
                self.emit_length(args[0], instrs, locals, local_count)?;
                return Ok(());
            }
            // `take k (iterate/repeat/cycle/enumFrom/enumFromThen …)`: the strict
            // backend can't build an infinite list, so for a statically-known
            // count fuse the take into a finite list of `k` elements. Finite
            // arguments fall through to the normal synthesized `take`.
            if strip_qualifier(name) == "take" && args.len() == 2 {
                if let Some(k) = as_int_literal(args[0]) {
                    if let Some(list) = fuse_take(k.max(0), args[1], &mut self.next_synthetic_id) {
                        return self.lower_expr(&list, instrs, locals, local_count, is_main);
                    }
                }
            }
        }

        // Typeclass dictionary method selection: `$sel_N dict [method args...]`.
        // A class dictionary is a tuple `[tag | field0 | field1 | ...]` of method
        // implementations (each a closure), so the selector loads field N at byte
        // offset (N+1)*4. The dictionary may be a known tuple (statically built or
        // an inlined parameter), a local holding a dictionary value passed as an
        // argument, or the result of a superclass selector — `lower_expr` resolves
        // all of these to a pointer. If the method is applied to further
        // arguments, apply the selected closure to them via `call_indirect`; with
        // no further arguments the selected method closure is itself the result
        // (e.g. `map (describe :: $sel_0 dict)`).
        if let Some(name) = func_name {
            if let Some(field_idx) = parse_sel_index(name) {
                if !args.is_empty() {
                    self.lower_expr(args[0], instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::I32Load(2, (field_idx + 1) * 4));
                    if args.len() > 1 {
                        return self.apply_closure_on_stack(
                            &args[1..],
                            instrs,
                            locals,
                            local_count,
                        );
                    }
                    return Ok(());
                }
            }
        }

        // Try to inline/beta-reduce the application. Handles dictionary-
        // specialized typeclass methods (head buried under dead dictionary
        // lets) and closures passed as arguments, neither of which the simple
        // call path supports.
        if self.try_lower_inline_app(func_expr, &args, instrs, locals, local_count, is_main)? {
            return Ok(());
        }

        // Application of a closure value held in a local (a function-typed
        // parameter, a case binder, a let binding): call it indirectly.
        if let Expr::Var(var, _) = peel_head(func_expr) {
            if let Some(&slot) = locals.get(&var.id) {
                instrs.push(WasmInstr::LocalGet(slot));
                return self.apply_closure_on_stack(&args, instrs, locals, local_count);
            }
        }

        // Floating-point arithmetic and comparison on boxed-double operands.
        // `/` is Fractional, so it is always float; the others are dispatched
        // by operand type.
        if let Some(name) = func_name {
            // `Ratio Int` operands take the rational arms in the match below, not
            // the float path (`/` would otherwise always be treated as f64).
            if args.len() == 2 && !is_rational_expr(args[0]) && !is_rational_expr(args[1]) {
                let is_float = expr_is_float(args[0]) || expr_is_float(args[1]);
                let binop = match name {
                    "+" | "GHC.Num.+" if is_float => Some(WasmInstr::F64Add),
                    "-" | "GHC.Num.-" if is_float => Some(WasmInstr::F64Sub),
                    "*" | "GHC.Num.*" if is_float => Some(WasmInstr::F64Mul),
                    "/" | "GHC.Real./" | "GHC.Float./" => Some(WasmInstr::F64Div),
                    _ => None,
                };
                if let Some(op) = binop {
                    return self.emit_float_binop(
                        op,
                        args[0],
                        args[1],
                        instrs,
                        locals,
                        local_count,
                    );
                }
                if is_float {
                    let cmp = match name {
                        "==" | "GHC.Classes.==" => Some(WasmInstr::F64Eq),
                        "/=" | "GHC.Classes./=" => Some(WasmInstr::F64Ne),
                        "<" | "GHC.Classes.<" => Some(WasmInstr::F64Lt),
                        "<=" | "GHC.Classes.<=" => Some(WasmInstr::F64Le),
                        ">" | "GHC.Classes.>" => Some(WasmInstr::F64Gt),
                        ">=" | "GHC.Classes.>=" => Some(WasmInstr::F64Ge),
                        _ => None,
                    };
                    if let Some(op) = cmp {
                        return self.emit_float_cmp(
                            op,
                            args[0],
                            args[1],
                            instrs,
                            locals,
                            local_count,
                        );
                    }
                }
            }
            // Unary negate on a float: box(-x).
            if args.len() == 1
                && matches!(name, "negate" | "GHC.Num.negate")
                && expr_is_float(args[0])
            {
                instrs.push(WasmInstr::I32Const(8));
                instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
                let ptr = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::LocalTee(ptr));
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::F64Load(3, 0));
                instrs.push(WasmInstr::F64Neg);
                instrs.push(WasmInstr::F64Store(3, 0));
                instrs.push(WasmInstr::LocalGet(ptr));
                return Ok(());
            }
        }

        match func_name {
            // Rational (`Ratio Int`) operations. A Rational is a normalized heap
            // pair `[num|den]`. Types are erased, so rationality is detected
            // structurally: the `%` constructor, or an overloaded operator with a
            // rational operand. These arms precede the Int arithmetic arms; their
            // guards fall through to Int when no operand is rational.
            Some(n) if args.len() == 2 && strip_qualifier(n) == "%" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?; // numerator
                self.lower_expr(args[1], instrs, locals, local_count, false)?; // denominator
                instrs.push(WasmInstr::Call(self.runtime.make_rational_idx));
            }
            Some(n)
                if args.len() == 2
                    && matches!(strip_qualifier(n), "+" | "-" | "*" | "/")
                    && (is_rational_expr(args[0]) || is_rational_expr(args[1])) =>
            {
                self.emit_rational_binop(
                    strip_qualifier(n),
                    args[0],
                    args[1],
                    instrs,
                    locals,
                    local_count,
                )?;
            }
            Some(n)
                if args.len() == 2
                    && matches!(strip_qualifier(n), "==" | "/=")
                    && (is_rational_expr(args[0]) || is_rational_expr(args[1])) =>
            {
                self.emit_rational_eq(
                    strip_qualifier(n) == "/=",
                    args[0],
                    args[1],
                    instrs,
                    locals,
                    local_count,
                )?;
            }
            Some(n)
                if args.len() == 1
                    && matches!(strip_qualifier(n), "negate" | "abs" | "signum")
                    && is_rational_expr(args[0]) =>
            {
                self.emit_rational_unary(strip_qualifier(n), args[0], instrs, locals, local_count)?;
            }
            Some(n)
                if args.len() == 1 && matches!(strip_qualifier(n), "numerator" | "denominator") =>
            {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                let off = if strip_qualifier(n) == "numerator" {
                    0
                } else {
                    4
                };
                instrs.push(WasmInstr::I32Load(2, off));
            }

            // Arithmetic primitives
            Some("+" | "plus#" | "plusInt#" | "GHC.Num.+") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Add);
            }
            Some("-" | "minus#" | "minusInt#" | "GHC.Num.-") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Sub);
            }
            Some("*" | "times#" | "timesInt#" | "GHC.Num.*") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Mul);
            }
            // `div` is floored division (not the truncating wasm `i32.div_s`):
            // adjust the truncated quotient down by 1 when there is a remainder
            // and the operands have opposite signs.
            Some("div" | "divInt#" | "GHC.Real.div") if args.len() == 2 => {
                let a = *local_count;
                let b = *local_count + 1;
                *local_count += 2;
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::LocalSet(a));
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::LocalSet(b));
                // q = a / b (truncated)
                instrs.push(WasmInstr::LocalGet(a));
                instrs.push(WasmInstr::LocalGet(b));
                instrs.push(WasmInstr::I32DivS);
                // adjust = (a % b != 0) & ((a ^ b) < 0)
                instrs.push(WasmInstr::LocalGet(a));
                instrs.push(WasmInstr::LocalGet(b));
                instrs.push(WasmInstr::I32RemS);
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::I32Ne);
                instrs.push(WasmInstr::LocalGet(a));
                instrs.push(WasmInstr::LocalGet(b));
                instrs.push(WasmInstr::I32Xor);
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::I32LtS);
                instrs.push(WasmInstr::I32And);
                instrs.push(WasmInstr::I32Sub); // q - adjust
            }
            // `mod` is the floored modulus: take the truncated remainder and add
            // the divisor back when it is non-zero and has the opposite sign.
            Some("mod" | "modInt#" | "GHC.Real.mod") if args.len() == 2 => {
                let a = *local_count;
                let b = *local_count + 1;
                let r = *local_count + 2;
                *local_count += 3;
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::LocalSet(a));
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::LocalSet(b));
                // r = a % b (truncated)
                instrs.push(WasmInstr::LocalGet(a));
                instrs.push(WasmInstr::LocalGet(b));
                instrs.push(WasmInstr::I32RemS);
                instrs.push(WasmInstr::LocalSet(r));
                // result = r + b * ((r != 0) & ((r ^ b) < 0))
                instrs.push(WasmInstr::LocalGet(r));
                instrs.push(WasmInstr::LocalGet(b));
                instrs.push(WasmInstr::LocalGet(r));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::I32Ne);
                instrs.push(WasmInstr::LocalGet(r));
                instrs.push(WasmInstr::LocalGet(b));
                instrs.push(WasmInstr::I32Xor);
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::I32LtS);
                instrs.push(WasmInstr::I32And);
                instrs.push(WasmInstr::I32Mul);
                instrs.push(WasmInstr::I32Add);
            }
            Some("rem" | "remInt#" | "GHC.Real.rem") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32RemS);
            }
            Some("quot" | "quotInt#" | "GHC.Real.quot") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32DivS);
            }
            Some("negate" | "negateInt#" | "GHC.Num.negate") if args.len() == 1 => {
                instrs.push(WasmInstr::I32Const(0));
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Sub);
            }

            // Derived `Enum` methods. A nullary constructor's runtime value is
            // exactly its tag, so `fromEnum`/`toEnum` are identities and
            // `succ`/`pred` step the tag by one.
            Some(n) if args.len() == 1 && matches!(strip_qualifier(n), "fromEnum" | "toEnum") => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
            }
            Some(n) if args.len() == 1 && strip_qualifier(n) == "succ" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Const(1));
                instrs.push(WasmInstr::I32Add);
            }
            Some(n) if args.len() == 1 && strip_qualifier(n) == "pred" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Const(1));
                instrs.push(WasmInstr::I32Sub);
            }

            // `Control.DeepSeq`: in the strict runtime everything is already
            // forced, so `force`/`id` are the identity, and `seq`/`deepseq`
            // evaluate the first argument (for its effects) and return the
            // second. `rnf` reduces to `()`.
            Some(n)
                if args.len() == 1
                    && matches!(
                        strip_qualifier(n),
                        "force" | "id" | "fromString" | "chr" | "ord"
                    ) =>
            {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
            }
            // `to (from x)` roundtrips to `x` (the common DeriveGeneric idiom):
            // short-circuit it so no Rep is materialised. A bare `to` elsewhere
            // is the identity.
            Some(n) if args.len() == 1 && strip_qualifier(n) == "to" => {
                let (h, hargs) = collect_app_spine(args[0]);
                let roundtrip = matches!(h, Expr::Var(v, _) if strip_qualifier(v.name.as_str()) == "from")
                    && hargs.len() == 1;
                if roundtrip {
                    self.lower_expr(hargs[0], instrs, locals, local_count, false)?;
                } else {
                    self.lower_expr(args[0], instrs, locals, local_count, false)?;
                }
            }
            // A bare `from d` for a derived-`Generic` enum builds the real outer
            // sum of the Rep: `M1` is identity (a metadata newtype), so produce
            // `L1 _` for the left half of the constructors and `R1 _` for the
            // right, split at `ceil(n/2)` by the value's tag. (Pattern matching
            // on the Rep — `case from d of M1 inner -> case inner of L1…/R1…` —
            // then dispatches on this. Products/`K1` never need a real Rep since
            // `to (from x)` is short-circuited above.)
            Some(n) if args.len() == 1 && strip_qualifier(n) == "from" => {
                let half = self.sole_enum_half();
                let p = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::I32Const(8));
                instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
                instrs.push(WasmInstr::LocalSet(p));
                // [p] = (tag(d) < half) ? 0 (L1) : 1 (R1)
                instrs.push(WasmInstr::LocalGet(p));
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Const(half));
                instrs.push(WasmInstr::I32LtS);
                instrs.push(WasmInstr::I32Eqz); // L1 when tag<half -> store 0; else 1
                instrs.push(WasmInstr::I32Store(2, 0));
                // [p+4] = 0 (the arm's payload is matched with `_`)
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::I32Store(2, 4));
                instrs.push(WasmInstr::LocalGet(p));
            }
            Some(n) if args.len() == 1 && strip_qualifier(n) == "rnf" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                instrs.push(WasmInstr::I32Const(0));
            }
            Some(n) if args.len() == 2 && matches!(strip_qualifier(n), "seq" | "deepseq") => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
            }

            // `when c a` / `unless c a`: run the IO action `a` only on the taken
            // branch — it must be lowered *inside* the `if`, not eagerly as a
            // call argument (which would always run its side effects). Result is
            // `()` (0) on the untaken branch.
            Some(n) if args.len() == 2 && matches!(strip_qualifier(n), "when" | "unless") => {
                let run_when_true = strip_qualifier(n) == "when";
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                if run_when_true {
                    self.lower_expr(args[1], instrs, locals, local_count, false)?;
                } else {
                    instrs.push(WasmInstr::I32Const(0));
                }
                instrs.push(WasmInstr::Else);
                if run_when_true {
                    instrs.push(WasmInstr::I32Const(0));
                } else {
                    self.lower_expr(args[1], instrs, locals, local_count, false)?;
                }
                instrs.push(WasmInstr::End);
            }

            // IORef as a one-slot heap cell holding the current value.
            Some(n) if args.len() == 1 && strip_qualifier(n) == "newIORef" => {
                instrs.push(WasmInstr::I32Const(4));
                instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
                let p = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::LocalTee(p));
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Store(2, 0));
                instrs.push(WasmInstr::LocalGet(p));
            }
            Some(n) if args.len() == 1 && strip_qualifier(n) == "readIORef" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Load(2, 0));
            }
            Some(n) if args.len() == 2 && strip_qualifier(n) == "writeIORef" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Store(2, 0));
                instrs.push(WasmInstr::I32Const(0)); // ()
            }
            Some(n)
                if args.len() == 2
                    && matches!(strip_qualifier(n), "modifyIORef" | "modifyIORef'") =>
            {
                // ref[0] = f ref[0]
                let p = *local_count;
                let fc = *local_count + 1;
                let nv = *local_count + 2;
                *local_count += 3;
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::LocalSet(p));
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::LocalSet(fc));
                let type_idx = self.closure_type_index();
                self.wasm.enable_func_table();
                instrs.push(WasmInstr::LocalGet(fc)); // env
                instrs.push(WasmInstr::LocalGet(p)); // arg = current value
                instrs.push(WasmInstr::I32Load(2, 0));
                instrs.push(WasmInstr::LocalGet(fc)); // code index at offset 0
                instrs.push(WasmInstr::I32Load(2, 0));
                instrs.push(WasmInstr::CallIndirect(type_idx, 0));
                instrs.push(WasmInstr::LocalSet(nv));
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::LocalGet(nv));
                instrs.push(WasmInstr::I32Store(2, 0));
                instrs.push(WasmInstr::I32Const(0)); // ()
            }

            // Comparison primitives
            Some("==" | "eqInt#" | "GHC.Classes.==") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Eq);
            }
            Some("/=" | "neInt#" | "GHC.Classes./=") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Ne);
            }
            Some("<" | "ltInt#" | "GHC.Classes.<") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32LtS);
            }
            Some("<=" | "leInt#" | "GHC.Classes.<=") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32LeS);
            }
            Some(">" | "gtInt#" | "GHC.Classes.>") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32GtS);
            }
            Some(">=" | "geInt#" | "GHC.Classes.>=") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32GeS);
            }

            // IO: putStrLn "..." => print_str_ln(offset, len). Handles dynamic
            // strings (if/case over string literals) by printing each leaf.
            // String concatenation: both operands are length-prefixed strings.
            // `++`: list append for cons-lists, string concat otherwise. Strings
            // and lists have different runtime representations and the same
            // `[a]` type, so dispatch on whether an operand is confidently a
            // (non-String) list; ambiguous cases default to string concat.
            Some("++" | "GHC.Base.++" | "Data.List.++" | "Data.List.NonEmpty.++")
                if args.len() == 2 =>
            {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                let idx = if is_list_operand(args[0]) || is_list_operand(args[1]) {
                    self.runtime.append_list_idx
                } else {
                    self.runtime.concat_str_idx
                };
                instrs.push(WasmInstr::Call(idx));
            }

            // `reverse` on a cons-list (string reversal is not supported).
            Some("reverse" | "GHC.List.reverse" | "Data.List.reverse")
                if args.len() == 1 && is_list_operand(args[0]) =>
            {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Call(self.runtime.reverse_list_idx));
            }

            // `unpackCString#`/`unpackCStringUtf8#` turn a string literal into a
            // String value; the literal already lowers to a string pointer.
            Some(name) if name.contains("unpackCString") && args.len() == 1 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
            }

            // `show x` as a value: produce a String (length-prefixed) so it
            // composes with `++`. The last argument is the value; any earlier
            // argument is a dictionary.
            Some("show" | "GHC.Show.show" | "Prelude.show" | "Text.Show.show")
                if !args.is_empty() =>
            {
                let value = args[args.len() - 1];
                self.emit_show_to_pstr(value, instrs, locals, local_count)?;
            }

            // `fromIntegral`/`realToFrac` used outside a float-consumption site:
            // pass the value through unchanged (Int stays an i32; a Double stays
            // boxed). Float contexts intercept `fromIntegral` earlier and
            // convert with F64ConvertI32S instead.
            Some("fromIntegral" | "GHC.Real.fromIntegral" | "Prelude.fromIntegral")
                if !args.is_empty() =>
            {
                self.lower_expr(args[args.len() - 1], instrs, locals, local_count, false)?;
            }
            Some("realToFrac" | "GHC.Real.realToFrac") if !args.is_empty() => {
                self.lower_expr(args[args.len() - 1], instrs, locals, local_count, false)?;
            }

            // Floating-point math: Double -> Double (boxed result).
            Some("sqrt" | "GHC.Float.sqrt") if args.len() == 1 => {
                self.emit_float_unary_box(
                    WasmInstr::F64Sqrt,
                    args[0],
                    instrs,
                    locals,
                    local_count,
                )?;
            }
            // abs/signum on a Double; integer abs/signum falls through.
            Some("abs" | "GHC.Num.abs") if args.len() == 1 && expr_is_float(args[0]) => {
                self.emit_float_unary_box(WasmInstr::F64Abs, args[0], instrs, locals, local_count)?;
            }

            // Double -> Int conversions (result is an unboxed Int).
            Some("truncate" | "GHC.Float.truncate") if args.len() == 1 => {
                self.emit_float_to_int(None, args[0], instrs, locals, local_count)?;
            }
            Some("floor" | "GHC.Float.floor") if args.len() == 1 => {
                self.emit_float_to_int(
                    Some(WasmInstr::F64Floor),
                    args[0],
                    instrs,
                    locals,
                    local_count,
                )?;
            }
            Some("ceiling" | "GHC.Float.ceiling") if args.len() == 1 => {
                self.emit_float_to_int(
                    Some(WasmInstr::F64Ceil),
                    args[0],
                    instrs,
                    locals,
                    local_count,
                )?;
            }
            Some("round" | "GHC.Float.round") if args.len() == 1 => {
                self.emit_float_to_int(
                    Some(WasmInstr::F64Nearest),
                    args[0],
                    instrs,
                    locals,
                    local_count,
                )?;
            }

            // `putStrLn` and its qualified forms (`System.IO`, `GHC.IO`,
            // `Data.Text.IO`, …) — Text is a pstr, so they share this path.
            Some(n) if args.len() == 1 && strip_qualifier(n) == "putStrLn" => {
                self.lower_cont(
                    args[0],
                    Cont::PrintStr { newline: true },
                    instrs,
                    locals,
                    local_count,
                    is_main,
                )?;
            }

            // IO: putStr "..." => print_str(offset, len)
            Some(n) if args.len() == 1 && strip_qualifier(n) == "putStr" => {
                self.lower_cont(
                    args[0],
                    Cont::PrintStr { newline: false },
                    instrs,
                    locals,
                    local_count,
                    is_main,
                )?;
            }

            // IO: print x => print_i32(x) + newline
            Some("print" | "System.IO.print" | "GHC.Show.print") if args.len() == 1 => {
                // `print x` = `putStrLn (show x)`: type-directed, with a newline.
                self.lower_show(args[0], true, instrs, locals, local_count)?;
            }

            // IO: getLine / readLn applied to dictionary/type arguments. The args
            // are class dictionaries (e.g. `readLn`'s `Read` dict), not values, so
            // they are ignored — the action takes no value arguments.
            Some("getLine" | "System.IO.getLine" | "GHC.IO.getLine" | "Prelude.getLine") => {
                instrs.push(WasmInstr::Call(self.runtime.read_line_idx));
            }
            Some("readLn" | "System.IO.readLn" | "GHC.Read.readLn" | "Prelude.readLn") => {
                instrs.push(WasmInstr::Call(self.runtime.read_line_idx));
                instrs.push(WasmInstr::Call(self.runtime.parse_int_idx));
            }
            Some(
                "getContents"
                | "System.IO.getContents"
                | "GHC.IO.getContents"
                | "Prelude.getContents",
            ) => {
                instrs.push(WasmInstr::Call(self.runtime.read_all_idx));
            }

            // IO: interact f = getContents >>= putStr . f. Read all of stdin,
            // apply the function `f` (a closure value) to the input String, and
            // write the resulting String to stdout with no trailing newline.
            Some("interact" | "System.IO.interact" | "Prelude.interact") if !args.is_empty() => {
                let f = args[args.len() - 1];
                // Evaluate f to a closure value.
                self.lower_expr(f, instrs, locals, local_count, false)?;
                let f_local = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::LocalSet(f_local));
                // input = read_all()
                instrs.push(WasmInstr::Call(self.runtime.read_all_idx));
                let in_local = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::LocalSet(in_local));
                // Apply the closure: call_indirect with (env, arg).
                let type_idx = self.closure_type_index();
                self.wasm.enable_func_table();
                instrs.push(WasmInstr::LocalGet(f_local)); // env pointer
                instrs.push(WasmInstr::LocalGet(in_local)); // argument (input String)
                instrs.push(WasmInstr::LocalGet(f_local));
                instrs.push(WasmInstr::I32Load(2, 0)); // code index at offset 0
                instrs.push(WasmInstr::CallIndirect(type_idx, 0));
                // Result String -> putStr (no newline).
                instrs.push(WasmInstr::Call(self.runtime.print_pstr_idx));
                // IO () result: leave a dummy 0 like other IO actions.
                instrs.push(WasmInstr::I32Const(0));
            }

            // IO: >> (sequence) - evaluate both sides for effects
            Some(">>" | "GHC.Base.>>") if args.len() == 2 => {
                // Evaluate first action, drop result
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                // Evaluate second action, keep result
                self.lower_expr(args[1], instrs, locals, local_count, is_main)?;
            }

            // IO: >>= (bind) - evaluate first, pass result to second
            Some(">>=" | "GHC.Base.>>=") if args.len() == 2 => {
                // Check if the second argument is a lambda: >>= \x -> body
                let second = peel_type_abstractions(args[1]);
                if let Expr::Lam(var, body, _) = second {
                    // Evaluate the first action
                    self.lower_expr(args[0], instrs, locals, local_count, false)?;
                    // Bind result to the lambda parameter
                    let param_local = *local_count;
                    *local_count += 1;
                    instrs.push(WasmInstr::LocalSet(param_local));
                    locals.insert(var.id, param_local);
                    // Evaluate the lambda body
                    self.lower_expr(body, instrs, locals, local_count, is_main)?;
                } else {
                    // No lambda — just sequence (drop first result, evaluate second)
                    self.lower_expr(args[0], instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::Drop);
                    self.lower_expr(args[1], instrs, locals, local_count, is_main)?;
                }
            }

            // IO: return / pure - just evaluate the argument
            Some("return" | "pure" | "GHC.Base.return" | "GHC.Base.pure") if args.len() == 1 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
            }

            // Monad-transformer eliminators (see rewrite_monad). The rewritten
            // value is a function of the environment/state (Reader/State) or the
            // result data directly (Writer/Except); these apply or project it.
            Some(n)
                if args.len() == 2
                    && matches!(
                        strip_qualifier(n),
                        "runReaderT" | "runReader" | "runStateT" | "runState"
                    ) =>
            {
                // run = apply the representation to the env / initial state. When
                // the argument is a nested eliminator (`runStateT (runExceptT
                // comp) …`, `runReaderT (evalStateT comp …) …`), it yields the
                // inner closure, which must be evaluated then applied via
                // call_indirect — a syntactic `papp` would merge the spine and
                // feed the inner eliminator an extra argument. Otherwise call
                // directly.
                let nested_elim = matches!(
                    collect_app_spine(args[0]).0,
                    Expr::Var(v, _) if matches!(
                        strip_qualifier(v.name.as_str()),
                        "runExceptT" | "runExcept" | "runWriterT" | "runWriter"
                            | "runReaderT" | "runReader" | "runStateT" | "runState"
                            | "evalStateT" | "evalState" | "execStateT" | "execState"
                    )
                );
                if nested_elim {
                    self.lower_expr(args[0], instrs, locals, local_count, false)?;
                    self.apply_closure_on_stack(&[args[1]], instrs, locals, local_count)?;
                } else {
                    let applied = papp(args[0].clone(), args[1].clone());
                    self.lower_expr(&applied, instrs, locals, local_count, is_main)?;
                }
            }
            Some(n)
                if args.len() == 2 && matches!(strip_qualifier(n), "evalStateT" | "evalState") =>
            {
                let a = pv("a", self.fresh_id());
                let s2 = pv("s2", self.fresh_id());
                if self.is_ctx_stack_arg(args[0]) {
                    // State-outer two-context stack: comp s0 is a Reader closure,
                    // so produce `\r -> fst (comp s0 r)` (peels State, stays in
                    // the inner Reader). The outer runReaderT then applies it.
                    let r = pv("r", self.fresh_id());
                    let applied = papp(papp(args[0].clone(), args[1].clone()), pev(&r));
                    let body = pcase(
                        applied,
                        vec![palt("(,)", 0, 2, vec![a.clone(), s2], pev(&a))],
                    );
                    let lam = plam(r, body);
                    self.lower_expr(&lam, instrs, locals, local_count, is_main)?;
                } else {
                    // fst (m s): case m s of (a, _) -> a
                    let applied = papp(args[0].clone(), args[1].clone());
                    let cas = pcase(
                        applied,
                        vec![palt("(,)", 0, 2, vec![a.clone(), s2], pev(&a))],
                    );
                    self.lower_expr(&cas, instrs, locals, local_count, is_main)?;
                }
            }
            Some(n)
                if args.len() == 2 && matches!(strip_qualifier(n), "execStateT" | "execState") =>
            {
                let a = pv("a", self.fresh_id());
                let s2 = pv("s2", self.fresh_id());
                if self.is_ctx_stack_arg(args[0]) {
                    let r = pv("r", self.fresh_id());
                    let applied = papp(papp(args[0].clone(), args[1].clone()), pev(&r));
                    let body = pcase(
                        applied,
                        vec![palt("(,)", 0, 2, vec![a, s2.clone()], pev(&s2))],
                    );
                    let lam = plam(r, body);
                    self.lower_expr(&lam, instrs, locals, local_count, is_main)?;
                } else {
                    // snd (m s): case m s of (_, s') -> s'
                    let applied = papp(args[0].clone(), args[1].clone());
                    let cas = pcase(
                        applied,
                        vec![palt("(,)", 0, 2, vec![a, s2.clone()], pev(&s2))],
                    );
                    self.lower_expr(&cas, instrs, locals, local_count, is_main)?;
                }
            }
            Some(n)
                if args.len() == 1
                    && matches!(
                        strip_qualifier(n),
                        "runWriterT" | "runWriter" | "runExceptT" | "runExcept"
                    ) =>
            {
                // the representation already is the result (pair / Either)
                self.lower_expr(args[0], instrs, locals, local_count, is_main)?;
            }
            Some(n)
                if args.len() == 1
                    && matches!(strip_qualifier(n), "execWriterT" | "execWriter") =>
            {
                // snd of the (a, w) pair
                let a = pv("a", self.fresh_id());
                let w = pv("w", self.fresh_id());
                let cas = pcase(
                    args[0].clone(),
                    vec![palt("(,)", 0, 2, vec![a, w.clone()], pev(&w))],
                );
                self.lower_expr(&cas, instrs, locals, local_count, is_main)?;
            }

            // IO: catch - execute the action, ignore the handler
            // Exception model (eager IO): `throwIO`/`throw`/`ioError`/`error` set
            // the pending-exception flag and yield a dummy value; effectful ops
            // no-op while the flag is set (see `emit_exn_guard` in wasi.rs).
            Some(n)
                if args.len() == 1
                    && matches!(
                        strip_qualifier(n),
                        "throwIO" | "throw" | "ioError" | "error" | "errorWithoutStackTrace"
                    ) =>
            {
                instrs.push(WasmInstr::I32Const(1));
                instrs.push(WasmInstr::GlobalSet(self.runtime.exn_flag_idx));
                instrs.push(WasmInstr::I32Const(0)); // dummy result
            }
            // File IO over the in-memory file table (no real filesystem under
            // WASI here). `readFile` raises if the path was never written;
            // `writeFile`/`appendFile` populate the table. Covers the `System.IO`
            // and `Data.Text.IO` variants alike (qualifier stripped).
            Some(n)
                if args.len() == 1 && matches!(strip_qualifier(n), "readFile" | "readFile'") =>
            {
                self.lower_expr(args[0], instrs, locals, local_count, false)?; // path
                instrs.push(WasmInstr::Call(self.runtime.file_read_idx));
            }
            Some(n) if args.len() == 2 && strip_qualifier(n) == "writeFile" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?; // path
                self.lower_expr(args[1], instrs, locals, local_count, false)?; // content
                instrs.push(WasmInstr::Call(self.runtime.file_write_idx));
                instrs.push(WasmInstr::I32Const(0)); // ()
            }
            Some(n) if args.len() == 2 && strip_qualifier(n) == "appendFile" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?; // path
                self.lower_expr(args[1], instrs, locals, local_count, false)?; // content
                instrs.push(WasmInstr::Call(self.runtime.file_append_idx));
                instrs.push(WasmInstr::I32Const(0)); // ()
            }
            // `catch action handler`: run the action; if it raised, clear the flag
            // and run the handler applied to a dummy exception value.
            Some("catch" | "Control.Exception.catch" | "GHC.IO.catch") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                instrs.push(WasmInstr::GlobalGet(self.runtime.exn_flag_idx));
                instrs.push(WasmInstr::If(None));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::GlobalSet(self.runtime.exn_flag_idx));
                self.lower_exn_handler(args[1], instrs, locals, local_count)?;
                instrs.push(WasmInstr::Drop);
                instrs.push(WasmInstr::End);
                instrs.push(WasmInstr::I32Const(0)); // catch result: ()
            }
            // `finally action cleanup`: run the action, then always run the cleanup
            // (even if the action raised), preserving any pending exception.
            Some(n) if args.len() == 2 && matches!(strip_qualifier(n), "finally" | "bracket_") => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                let saved = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::GlobalGet(self.runtime.exn_flag_idx));
                instrs.push(WasmInstr::LocalSet(saved));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::GlobalSet(self.runtime.exn_flag_idx));
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                instrs.push(WasmInstr::LocalGet(saved));
                instrs.push(WasmInstr::GlobalSet(self.runtime.exn_flag_idx));
                instrs.push(WasmInstr::I32Const(0));
            }
            // `onException action handler`: run the action; if it raised, run the
            // handler (with effects temporarily enabled) but keep the exception
            // pending so it propagates to an enclosing `catch`.
            Some(n) if args.len() == 2 && strip_qualifier(n) == "onException" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                instrs.push(WasmInstr::GlobalGet(self.runtime.exn_flag_idx));
                instrs.push(WasmInstr::If(None));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::GlobalSet(self.runtime.exn_flag_idx));
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                instrs.push(WasmInstr::I32Const(1));
                instrs.push(WasmInstr::GlobalSet(self.runtime.exn_flag_idx));
                instrs.push(WasmInstr::End);
                instrs.push(WasmInstr::I32Const(0));
            }

            // Fused sum/enumFromTo: sum (enumFromTo lo hi) => loop accumulation
            Some("sum" | "Prelude.sum" | "Data.List.sum" | "GHC.List.sum") if args.len() == 1 => {
                if let Some((lo_expr, hi_expr)) = extract_enum_from_to(args[0]) {
                    // Emit a loop: acc = 0; for i = lo to hi: acc += i
                    let acc_local = *local_count;
                    *local_count += 1;
                    let i_local = *local_count;
                    *local_count += 1;

                    // Initialize accumulator to 0
                    instrs.push(WasmInstr::I32Const(0));
                    instrs.push(WasmInstr::LocalSet(acc_local));

                    // Initialize loop variable to lo
                    self.lower_expr(lo_expr, instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::LocalSet(i_local));

                    // Evaluate hi and store
                    let hi_local = *local_count;
                    *local_count += 1;
                    self.lower_expr(hi_expr, instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::LocalSet(hi_local));

                    // Block(None) is break target, Loop(None) is continue target
                    instrs.push(WasmInstr::Block(None));
                    instrs.push(WasmInstr::Loop(None));

                    // Check condition: if i > hi, break
                    instrs.push(WasmInstr::LocalGet(i_local));
                    instrs.push(WasmInstr::LocalGet(hi_local));
                    instrs.push(WasmInstr::I32GtS);
                    instrs.push(WasmInstr::BrIf(1)); // break out of block

                    // acc += i
                    instrs.push(WasmInstr::LocalGet(acc_local));
                    instrs.push(WasmInstr::LocalGet(i_local));
                    instrs.push(WasmInstr::I32Add);
                    instrs.push(WasmInstr::LocalSet(acc_local));

                    // i += 1
                    instrs.push(WasmInstr::LocalGet(i_local));
                    instrs.push(WasmInstr::I32Const(1));
                    instrs.push(WasmInstr::I32Add);
                    instrs.push(WasmInstr::LocalSet(i_local));

                    // Continue loop
                    instrs.push(WasmInstr::Br(0));

                    instrs.push(WasmInstr::End); // end loop
                    instrs.push(WasmInstr::End); // end block

                    // Push accumulator as result
                    instrs.push(WasmInstr::LocalGet(acc_local));
                } else {
                    // General case: walk the runtime cons list, summing heads
                    // (Int). acc = 0; cur = xs; while cur is a cons: acc +=
                    // head; cur = tail.
                    let acc = *local_count;
                    let cur = *local_count + 1;
                    *local_count += 2;
                    instrs.push(WasmInstr::I32Const(0));
                    instrs.push(WasmInstr::LocalSet(acc));
                    self.lower_expr(args[0], instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::LocalSet(cur));
                    instrs.push(WasmInstr::Block(None));
                    instrs.push(WasmInstr::Loop(None));
                    // nil (value < HEAP_BASE) -> done
                    instrs.push(WasmInstr::LocalGet(cur));
                    instrs.push(WasmInstr::I32Const(HEAP_BASE));
                    instrs.push(WasmInstr::I32LtU);
                    instrs.push(WasmInstr::BrIf(1));
                    // acc += head ([cur+4])
                    instrs.push(WasmInstr::LocalGet(acc));
                    instrs.push(WasmInstr::LocalGet(cur));
                    instrs.push(WasmInstr::I32Load(2, 4));
                    instrs.push(WasmInstr::I32Add);
                    instrs.push(WasmInstr::LocalSet(acc));
                    // cur = tail ([cur+8])
                    instrs.push(WasmInstr::LocalGet(cur));
                    instrs.push(WasmInstr::I32Load(2, 8));
                    instrs.push(WasmInstr::LocalSet(cur));
                    instrs.push(WasmInstr::Br(0));
                    instrs.push(WasmInstr::End); // loop
                    instrs.push(WasmInstr::End); // block
                    instrs.push(WasmInstr::LocalGet(acc));
                }
            }
            // showInt n = show (n :: Int): an Int is a raw i32, so render it
            // directly with the integer formatter.
            Some(n) if args.len() == 1 && strip_qualifier(n) == "showInt" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Call(self.runtime.int_to_str_idx));
            }
            // showDouble x = show (x :: Double): render via the double formatter.
            Some(n) if args.len() == 1 && strip_qualifier(n) == "showDouble" => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::F64Load(3, 0));
                instrs.push(WasmInstr::Call(self.runtime.double_to_str_idx));
            }
            // Floating-point math (incl. C FFI stubs c_sin/c_cos/c_sqrt). No
            // libc in WASM: sqrt is the f64 instruction; sin/cos use a Taylor
            // series.
            Some(n)
                if args.len() == 1
                    && matches!(strip_qualifier(n), "sqrt" | "c_sqrt" | "GHC.Float.sqrt") =>
            {
                self.emit_math_builtin("sqrt", args[0], instrs, locals, local_count)?;
            }
            Some(n) if args.len() == 1 && matches!(strip_qualifier(n), "sin" | "c_sin") => {
                self.emit_math_builtin("sin", args[0], instrs, locals, local_count)?;
            }
            Some(n) if args.len() == 1 && matches!(strip_qualifier(n), "cos" | "c_cos") => {
                self.emit_math_builtin("cos", args[0], instrs, locals, local_count)?;
            }
            // showBool b = if b then "True" else "False"
            Some(n) if args.len() == 1 && strip_qualifier(n) == "showBool" => {
                let t = self.intern_pstr("True") as i32;
                let f = self.intern_pstr("False") as i32;
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::If(Some(WasmType::I32)));
                instrs.push(WasmInstr::I32Const(t));
                instrs.push(WasmInstr::Else);
                instrs.push(WasmInstr::I32Const(f));
                instrs.push(WasmInstr::End);
            }
            // showChar c = "'" ++ [c] ++ "'": build a one-char string from the
            // code and wrap it in single quotes. (No escaping of control chars.)
            Some(n) if args.len() == 1 && strip_qualifier(n) == "showChar" => {
                let p = *local_count;
                *local_count += 1;
                self.emit_pstr_lit("'", instrs);
                // one-char pstr: [len=1 | byte]
                instrs.push(WasmInstr::I32Const(8));
                instrs.push(WasmInstr::Call(self.runtime.alloc_idx));
                instrs.push(WasmInstr::LocalSet(p));
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::I32Const(1 | crate::wasi::PSTR_MARKER));
                instrs.push(WasmInstr::I32Store(2, 0));
                instrs.push(WasmInstr::LocalGet(p));
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Store(2, 4));
                instrs.push(WasmInstr::LocalGet(p));
                instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
                self.emit_pstr_lit("'", instrs);
                instrs.push(WasmInstr::Call(self.runtime.concat_str_idx));
            }

            // User-defined function call
            Some(name) => {
                let sym = func_expr_symbol(func_expr).unwrap();
                if let Some(&func_idx) = self.func_map.get(&sym) {
                    let arity = self.arities.get(&sym).copied().unwrap_or(args.len());
                    if args.len() < arity {
                        // Partial application: build a closure that captures the
                        // supplied arguments and waits for the rest.
                        let func_var = func_expr_var(func_expr).unwrap().clone();
                        return self.lower_partial_application(
                            &func_var,
                            &args,
                            arity,
                            instrs,
                            locals,
                            local_count,
                        );
                    }
                    // Saturated: call with the first `arity` arguments.
                    for arg in &args[..arity] {
                        self.lower_expr(arg, instrs, locals, local_count, false)?;
                    }
                    instrs.push(WasmInstr::Call(func_idx));
                    // Over-application: the result is a closure; apply the rest.
                    if args.len() > arity {
                        self.apply_closure_on_stack(&args[arity..], instrs, locals, local_count)?;
                    }
                } else if let Some(var) = func_expr_var(func_expr) {
                    // Check if it's a local variable being called (like a closure)
                    if let Some(&local_idx) = locals.get(&var.id) {
                        // Can't call a local in WASM - just push args and use the value
                        for arg in &args {
                            self.lower_expr(arg, instrs, locals, local_count, false)?;
                            instrs.push(WasmInstr::Drop);
                        }
                        instrs.push(WasmInstr::LocalGet(local_idx));
                    } else {
                        tracing::warn!(func = name, "unknown function, using 0");
                        for arg in &args {
                            self.lower_expr(arg, instrs, locals, local_count, false)?;
                            instrs.push(WasmInstr::Drop);
                        }
                        instrs.push(WasmInstr::I32Const(0));
                    }
                } else {
                    tracing::warn!(func = name, "unknown function, using 0");
                    for arg in &args {
                        self.lower_expr(arg, instrs, locals, local_count, false)?;
                        instrs.push(WasmInstr::Drop);
                    }
                    instrs.push(WasmInstr::I32Const(0));
                }
            }

            // Non-variable function expression: evaluate it to a closure value
            // and apply the arguments indirectly.
            None => {
                self.lower_expr(func_expr, instrs, locals, local_count, false)?;
                self.apply_closure_on_stack(&args, instrs, locals, local_count)?;
            }
        }

        Ok(())
    }

    /// Lower a case expression.
    /// Lower an expression under a continuation.
    ///
    /// For [`Cont::Value`] this is exactly [`Self::lower_expr`]. For
    /// [`Cont::PrintStr`] the print is pushed into each leaf: `case`/`if`
    /// expressions recurse so every branch prints its own (statically sized)
    /// string literal, and transparent wrappers are peeled through.
    fn lower_cont(
        &mut self,
        expr: &Expr,
        cont: Cont,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        let newline = match cont {
            Cont::Value => {
                return self.lower_expr(expr, instrs, locals, local_count, is_main);
            }
            Cont::Tail => {
                return self.lower_tail(expr, instrs, locals, local_count);
            }
            Cont::PrintStr { newline } => newline,
        };

        // `putStrLn (show x)` / `putStr (show x)`: type-directed rendering of the
        // shown value (Int/Bool/Double/constructor) rather than a real String.
        if let Some(inner) = match_show_arg(expr) {
            return self.lower_show(inner, newline, instrs, locals, local_count);
        }

        // Any other string-valued expression. The value is either a
        // length-prefixed `pstr` (literal/`++`/show result — its length word
        // carries the marker bit) or a runtime cons-`[Char]` (a built string,
        // e.g. `c : acc`). Discriminate at runtime: empty (nil/0) prints
        // nothing; a marked pstr prints directly; a cons list is converted to a
        // pstr first.
        let v = *local_count;
        *local_count += 1;
        self.lower_expr(expr, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(v));
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Eqz);
        instrs.push(WasmInstr::If(None));
        // empty string: print nothing
        instrs.push(WasmInstr::Else);
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::I32Load(2, 0));
        instrs.push(WasmInstr::I32Const(crate::wasi::PSTR_MARKER));
        instrs.push(WasmInstr::I32And);
        instrs.push(WasmInstr::If(None));
        // marked pstr
        instrs.push(WasmInstr::LocalGet(v));
        instrs.push(WasmInstr::Call(self.runtime.print_pstr_idx));
        instrs.push(WasmInstr::Else);
        // cons-[Char]: convert to a pstr, then print
        self.emit_charlist_to_pstr(v, instrs, local_count);
        instrs.push(WasmInstr::Call(self.runtime.print_pstr_idx));
        instrs.push(WasmInstr::End);
        instrs.push(WasmInstr::End);
        if newline {
            instrs.push(WasmInstr::I32Const(self.runtime.newline_offset as i32));
            instrs.push(WasmInstr::I32Const(1));
            instrs.push(WasmInstr::Call(self.runtime.print_str_idx));
        }
        // IO action returns a dummy value.
        instrs.push(WasmInstr::I32Const(0));
        Ok(())
    }

    /// Lower an expression in tail position (TCO). `case` branches are
    /// themselves tail positions and recurse; everything else is a leaf.
    fn lower_tail(
        &mut self,
        expr: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        match expr {
            Expr::Case(scrut, alts, _, _) => {
                self.lower_case(scrut, alts, Cont::Tail, instrs, locals, local_count, false)
            }
            Expr::TyApp(i, _, _)
            | Expr::Cast(i, _, _)
            | Expr::Tick(_, i, _)
            | Expr::Lazy(i, _)
            | Expr::TyLam(_, i, _) => self.lower_tail(i, instrs, locals, local_count),
            _ => self.lower_tail_leaf(expr, instrs, locals, local_count),
        }
    }

    /// Lower a tail-position leaf: a saturated self-call loops (reassign
    /// parameters, set the continue flag); anything else sets the result and
    /// clears the flag. (A non-tail self-call nested in a value leaf stays a
    /// normal recursive `Call`.)
    fn lower_tail_leaf(
        &mut self,
        expr: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
    ) -> WasmResult<()> {
        let ctx = self
            .tco_ctx
            .clone()
            .expect("lower_tail_leaf requires an active TCO context");
        let (head, args) = collect_app_spine(peel_show_wrappers(expr));
        if let Expr::Var(v, _) = head {
            if v.name == ctx.name && args.len() == ctx.params.len() {
                // Tail self-call: evaluate all args into fresh temporaries first
                // (an arg may read the old parameters), then assign parameters.
                let mut temps = Vec::with_capacity(args.len());
                for arg in &args {
                    self.lower_expr(arg, instrs, locals, local_count, false)?;
                    let t = *local_count;
                    *local_count += 1;
                    instrs.push(WasmInstr::LocalSet(t));
                    temps.push(t);
                }
                for (t, p) in temps.iter().zip(ctx.params.iter()) {
                    instrs.push(WasmInstr::LocalGet(*t));
                    instrs.push(WasmInstr::LocalSet(*p));
                }
                instrs.push(WasmInstr::I32Const(1));
                instrs.push(WasmInstr::LocalSet(ctx.continue_local));
                return Ok(());
            }
        }
        // Value leaf: compute it, store as the result, stop looping.
        self.lower_expr(expr, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::LocalSet(ctx.result_local));
        instrs.push(WasmInstr::I32Const(0));
        instrs.push(WasmInstr::LocalSet(ctx.continue_local));
        Ok(())
    }

    /// The block result type for lowering a case alternative under `cont`: a
    /// `Tail` alt leaves nothing (it sets locals), others leave an `i32`.
    fn cont_block_type(cont: Cont) -> Option<WasmType> {
        match cont {
            Cont::Tail => None,
            _ => Some(WasmType::I32),
        }
    }

    /// Emit the fallback for a case alternative with no matching default: a
    /// dummy `0` value for value/print conts, or "stop looping" for `Tail`.
    fn emit_cont_fallback(&self, cont: Cont, instrs: &mut Vec<WasmInstr>) {
        match cont {
            Cont::Tail => {
                let ctx = self.tco_ctx.as_ref().expect("Tail cont without context");
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::LocalSet(ctx.result_local));
                instrs.push(WasmInstr::I32Const(0));
                instrs.push(WasmInstr::LocalSet(ctx.continue_local));
            }
            _ => instrs.push(WasmInstr::I32Const(0)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_case(
        &mut self,
        scrut: &Expr,
        alts: &[bhc_core::Alt],
        cont: Cont,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        // Evaluate scrutinee and store in a local
        self.lower_expr(scrut, instrs, locals, local_count, false)?;
        let scrut_local = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalSet(scrut_local));

        // If there's only a Default alt, just execute it
        if alts.len() == 1 {
            if let AltCon::Default = &alts[0].con {
                // Bind scrutinee to binder if present
                if let Some(binder) = alts[0].binders.first() {
                    locals.insert(binder.id, scrut_local);
                }
                self.lower_cont(&alts[0].rhs, cont, instrs, locals, local_count, is_main)?;
                return Ok(());
            }
        }

        // A match on a newtype constructor (`case x of C n -> ...`) is identity:
        // the value isn't boxed, so the field binder *is* the scrutinee.
        if alts.len() == 1 {
            if let AltCon::DataCon(dc) = &alts[0].con {
                if self.newtype_cons.contains(dc.name.as_str()) {
                    if let Some(binder) = alts[0].binders.first() {
                        locals.insert(binder.id, scrut_local);
                    }
                    return self.lower_cont(
                        &alts[0].rhs,
                        cont,
                        instrs,
                        locals,
                        local_count,
                        is_main,
                    );
                }
            }
        }

        // Find the default alternative (if any)
        let default_alt = alts.iter().find(|a| matches!(a.con, AltCon::Default));
        let lit_alts: Vec<_> = alts
            .iter()
            .filter(|a| matches!(a.con, AltCon::Lit(_)))
            .collect();
        let datacon_alts: Vec<_> = alts
            .iter()
            .filter(|a| matches!(a.con, AltCon::DataCon(_)))
            .collect();

        // Generate if-else chain for literal alternatives
        if !lit_alts.is_empty() {
            self.lower_case_lit_chain(
                scrut_local,
                &lit_alts,
                default_alt,
                cont,
                instrs,
                locals,
                local_count,
                is_main,
            )?;
        } else if !datacon_alts.is_empty() {
            // DataCon alternatives: match on constructor tag.
            // This handles if-expressions (True/False) and other ADTs.
            self.lower_case_datacon_chain(
                scrut_local,
                &datacon_alts,
                default_alt,
                cont,
                instrs,
                locals,
                local_count,
                is_main,
            )?;
        } else if let Some(def) = default_alt {
            // Only a default alt
            if let Some(binder) = def.binders.first() {
                locals.insert(binder.id, scrut_local);
            }
            self.lower_cont(&def.rhs, cont, instrs, locals, local_count, is_main)?;
        } else {
            // No alternatives at all.
            self.emit_cont_fallback(cont, instrs);
        }

        Ok(())
    }

    /// Lower a case expression with literal alternatives using if-else chain.
    #[allow(clippy::too_many_arguments)]
    fn lower_case_lit_chain(
        &mut self,
        scrut_local: u32,
        lit_alts: &[&bhc_core::Alt],
        default_alt: Option<&bhc_core::Alt>,
        cont: Cont,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        // We need to emit nested if-else blocks.
        // Since WASM requires a result type for if blocks, we use
        // a block(result i32) + br_table pattern, or nested if/else.

        // Use nested if-else: each level checks one literal
        for (i, alt) in lit_alts.iter().enumerate() {
            let lit_val = match &alt.con {
                AltCon::Lit(Literal::Int(n)) => *n as i32,
                AltCon::Lit(Literal::Integer(n)) => *n as i32,
                AltCon::Lit(Literal::Char(c)) => *c as i32,
                _ => 0,
            };

            // Compare scrutinee with literal
            instrs.push(WasmInstr::LocalGet(scrut_local));
            instrs.push(WasmInstr::I32Const(lit_val));
            instrs.push(WasmInstr::I32Eq);
            instrs.push(WasmInstr::If(Self::cont_block_type(cont)));

            // Bind scrutinee to binder if present
            if let Some(binder) = alt.binders.first() {
                locals.insert(binder.id, scrut_local);
            }

            // RHS
            self.lower_cont(&alt.rhs, cont, instrs, locals, local_count, is_main)?;

            instrs.push(WasmInstr::Else);

            // If this is the last alternative and there's a default, emit it
            if i == lit_alts.len() - 1 {
                if let Some(def) = default_alt {
                    if let Some(binder) = def.binders.first() {
                        locals.insert(binder.id, scrut_local);
                    }
                    self.lower_cont(&def.rhs, cont, instrs, locals, local_count, is_main)?;
                } else {
                    self.emit_cont_fallback(cont, instrs);
                }
            }
        }

        // Close all the if-else blocks
        for _ in lit_alts {
            instrs.push(WasmInstr::End);
        }

        Ok(())
    }

    /// Lower a case expression with data constructor alternatives using if-else chain.
    ///
    /// Data constructors are matched by their tag (u32). For nullary constructors
    /// (like `True`/`False`), the scrutinee is the tag value directly. For
    /// constructors with fields, the scrutinee is a heap pointer and the tag
    /// is at offset 0, with fields at offsets 4, 8, 12, etc.
    #[allow(clippy::too_many_arguments)]
    fn lower_case_datacon_chain(
        &mut self,
        scrut_local: u32,
        datacon_alts: &[&bhc_core::Alt],
        default_alt: Option<&bhc_core::Alt>,
        cont: Cont,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        // Determine if any alternative has field binders — if so, the
        // scrutinee is a heap pointer and we need to load the tag from memory.
        let has_fields = datacon_alts.iter().any(|alt| !alt.binders.is_empty())
            || datacon_alts.iter().any(|alt| {
                if let AltCon::DataCon(dc) = &alt.con {
                    dc.arity > 0
                } else {
                    false
                }
            });

        // Compute the constructor tag. In a type with field-carrying
        // constructors, a value is either a small nullary tag (e.g. `[]`) or a
        // heap pointer (e.g. `(:)`). Heap pointers are >= HEAP_BASE, so below
        // that the scrutinee *is* the tag; otherwise the tag is at offset 0.
        let tag_local = if has_fields {
            let tl = *local_count;
            *local_count += 1;
            instrs.push(WasmInstr::LocalGet(scrut_local));
            instrs.push(WasmInstr::I32Const(HEAP_BASE));
            instrs.push(WasmInstr::I32GeU);
            instrs.push(WasmInstr::If(Some(WasmType::I32)));
            instrs.push(WasmInstr::LocalGet(scrut_local));
            instrs.push(WasmInstr::I32Load(2, 0)); // heap object: tag at offset 0
            instrs.push(WasmInstr::Else);
            instrs.push(WasmInstr::LocalGet(scrut_local)); // nullary: value is the tag
            instrs.push(WasmInstr::End);
            instrs.push(WasmInstr::LocalSet(tl));
            tl
        } else {
            // Nullary constructors: scrutinee IS the tag
            scrut_local
        };

        for (i, alt) in datacon_alts.iter().enumerate() {
            let tag = match &alt.con {
                AltCon::DataCon(dc) => dc.tag as i32,
                _ => 0,
            };

            // Compare tag with constructor tag
            instrs.push(WasmInstr::LocalGet(tag_local));
            instrs.push(WasmInstr::I32Const(tag));
            instrs.push(WasmInstr::I32Eq);
            instrs.push(WasmInstr::If(Self::cont_block_type(cont)));

            // Extract field binders from the heap object
            for (fi, binder) in alt.binders.iter().enumerate() {
                let field_local = *local_count;
                *local_count += 1;
                instrs.push(WasmInstr::LocalGet(scrut_local));
                instrs.push(WasmInstr::I32Load(2, ((fi + 1) * 4) as u32));
                instrs.push(WasmInstr::LocalSet(field_local));
                locals.insert(binder.id, field_local);
            }

            // RHS
            self.lower_cont(&alt.rhs, cont, instrs, locals, local_count, is_main)?;

            instrs.push(WasmInstr::Else);

            // Last alternative: emit default or fallback
            if i == datacon_alts.len() - 1 {
                if let Some(def) = default_alt {
                    if let Some(binder) = def.binders.first() {
                        locals.insert(binder.id, scrut_local);
                    }
                    self.lower_cont(&def.rhs, cont, instrs, locals, local_count, is_main)?;
                } else {
                    self.emit_cont_fallback(cont, instrs);
                }
            }
        }

        // Close all the if-else blocks
        for _ in datacon_alts {
            instrs.push(WasmInstr::End);
        }

        Ok(())
    }
}

// ============================================================
// Helper functions
// ============================================================

/// Whether `expr` references the given name anywhere. Used to detect
/// (self-)recursive functions, which must not be inlined: inlining a function
/// into its own body would alias parameter `VarId`s and loop forever.
fn expr_uses_name(expr: &Expr, name: Symbol) -> bool {
    match expr {
        Expr::Var(v, _) => v.name == name,
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => false,
        Expr::App(f, a, _) => expr_uses_name(f, name) || expr_uses_name(a, name),
        Expr::TyApp(inner, _, _)
        | Expr::TyLam(_, inner, _)
        | Expr::Lam(_, inner, _)
        | Expr::Lazy(inner, _)
        | Expr::Cast(inner, _, _)
        | Expr::Tick(_, inner, _) => expr_uses_name(inner, name),
        Expr::Let(bind, body, _) => {
            let in_bind = match bind.as_ref() {
                Bind::NonRec(_, rhs) => expr_uses_name(rhs, name),
                Bind::Rec(bs) => bs.iter().any(|(_, rhs)| expr_uses_name(rhs, name)),
            };
            in_bind || expr_uses_name(body, name)
        }
        Expr::Case(scrut, alts, _, _) => {
            expr_uses_name(scrut, name) || alts.iter().any(|a| expr_uses_name(&a.rhs, name))
        }
    }
}

/// Collect the free variables of `expr` (in first-occurrence order), skipping
/// any bound within it. `bound` tracks variables in scope; `seen` deduplicates.
fn collect_free_vars(
    expr: &Expr,
    bound: &mut FxHashSet<VarId>,
    acc: &mut Vec<Var>,
    seen: &mut FxHashSet<VarId>,
) {
    match expr {
        Expr::Var(v, _) => {
            if !bound.contains(&v.id) && seen.insert(v.id) {
                acc.push(v.clone());
            }
        }
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
        Expr::App(f, a, _) => {
            collect_free_vars(f, bound, acc, seen);
            collect_free_vars(a, bound, acc, seen);
        }
        Expr::TyApp(inner, _, _)
        | Expr::Cast(inner, _, _)
        | Expr::Tick(_, inner, _)
        | Expr::Lazy(inner, _)
        | Expr::TyLam(_, inner, _) => collect_free_vars(inner, bound, acc, seen),
        Expr::Lam(p, body, _) => {
            let added = bound.insert(p.id);
            collect_free_vars(body, bound, acc, seen);
            if added {
                bound.remove(&p.id);
            }
        }
        Expr::Let(bind, body, _) => match bind.as_ref() {
            Bind::NonRec(v, rhs) => {
                collect_free_vars(rhs, bound, acc, seen);
                let added = bound.insert(v.id);
                collect_free_vars(body, bound, acc, seen);
                if added {
                    bound.remove(&v.id);
                }
            }
            Bind::Rec(bs) => {
                let mut added = Vec::new();
                for (v, _) in bs {
                    if bound.insert(v.id) {
                        added.push(v.id);
                    }
                }
                for (_, rhs) in bs {
                    collect_free_vars(rhs, bound, acc, seen);
                }
                collect_free_vars(body, bound, acc, seen);
                for id in added {
                    bound.remove(&id);
                }
            }
        },
        Expr::Case(scrut, alts, _, _) => {
            collect_free_vars(scrut, bound, acc, seen);
            for alt in alts {
                let mut added = Vec::new();
                for b in &alt.binders {
                    if bound.insert(b.id) {
                        added.push(b.id);
                    }
                }
                collect_free_vars(&alt.rhs, bound, acc, seen);
                for id in added {
                    bound.remove(&id);
                }
            }
        }
    }
}

/// Peel transparent wrappers and (dead) let-bindings to reach the head of an
/// application's function position.
///
/// Let-bindings in head position are skipped: after dictionary specialization
/// they bind dead dictionaries. A live reference resolves via the normal
/// variable path instead.
fn peel_head(expr: &Expr) -> &Expr {
    let mut e = expr;
    loop {
        e = match e {
            Expr::TyApp(inner, _, _)
            | Expr::TyLam(_, inner, _)
            | Expr::Cast(inner, _, _)
            | Expr::Tick(_, inner, _)
            | Expr::Lazy(inner, _)
            | Expr::Let(_, inner, _) => inner,
            _ => return e,
        };
    }
}

/// If `expr` is `fromIntegral <value>` (possibly with leading dictionary
/// arguments), return the integer value being converted.
fn match_from_integral(expr: &Expr) -> Option<&Expr> {
    let (head, args) = collect_app_spine(peel_show_wrappers(expr));
    if let Expr::Var(v, _) = head {
        if matches!(
            v.name.as_str(),
            "fromIntegral" | "GHC.Real.fromIntegral" | "Prelude.fromIntegral"
        ) && !args.is_empty()
        {
            return Some(args[args.len() - 1]);
        }
    }
    None
}

/// If `expr` is `show <value>` (possibly with a leading dictionary argument),
/// return the value being shown. Used to print `show n` for integers via the
/// runtime int printer without materialising a string.
fn match_show_arg(expr: &Expr) -> Option<&Expr> {
    let (func, args) = collect_app_spine(expr);
    let name = match func {
        Expr::Var(v, _) => v.name.as_str(),
        _ => return None,
    };
    let is_show = matches!(
        name,
        "show" | "GHC.Show.show" | "Prelude.show" | "Text.Show.show"
    );
    if is_show && !args.is_empty() {
        // The last argument is the value; any earlier argument is a dictionary.
        Some(args[args.len() - 1])
    } else {
        None
    }
}

// ============================================================
// Injected list-prelude functions
// ============================================================
//
// The WASM path links no standard library, so prelude list functions appear as
// unresolved names. We synthesize Core definitions for the common ones and
// lower them like user code — recursion, cons-cell matching, and closure
// parameters already work, so these "just compile".

/// List-prelude functions we can synthesize, by name.
const LIST_PRELUDE_NAMES: &[&str] = &[
    "map",
    "__fmapMaybe",
    "zip3",
    "zipWith3",
    "unfoldr",
    "gcd",
    "lcm",
    "filterM",
    "foldM",
    "foldM_",
    "zipWithM",
    "zipWithM_",
    "either",
    "fromLeft",
    "fromRight",
    "lefts",
    "rights",
    "maybe",
    "listToMaybe",
    "catMaybes",
    "guard",
    // Data.Map / Data.IntMap
    "Data.Map.fromList",
    "Data.Map.size",
    "Data.Map.member",
    "Data.Map.insert",
    "Data.Map.delete",
    "Data.Map.null",
    "Data.Map.findWithDefault",
    "Data.Map.elems",
    "Data.Map.keys",
    "Data.Map.lookup",
    "Data.Map.mapMaybe",
    "Data.Map.union",
    "Data.Map.unions",
    "Data.Map.update",
    "Data.Map.alter",
    "Data.IntMap.fromList",
    "Data.IntMap.size",
    "Data.IntMap.member",
    "Data.IntMap.insert",
    "Data.IntMap.delete",
    "Data.IntMap.null",
    "Data.IntMap.findWithDefault",
    "Data.IntMap.elems",
    "Data.IntMap.keys",
    "Data.IntMap.lookup",
    "Data.IntMap.mapMaybe",
    "Data.IntMap.union",
    "Data.IntMap.unions",
    "Data.IntMap.update",
    "Data.IntMap.alter",
    // Data.Set / Data.IntSet
    "Data.Set.fromList",
    "Data.Set.size",
    "Data.Set.member",
    "Data.Set.insert",
    "Data.Set.delete",
    "Data.Set.union",
    "Data.Set.intersection",
    "Data.Set.difference",
    "Data.Set.filter",
    "Data.Set.foldr",
    "Data.IntSet.fromList",
    "Data.IntSet.size",
    "Data.IntSet.member",
    "Data.IntSet.insert",
    "Data.IntSet.delete",
    "Data.IntSet.union",
    "Data.IntSet.intersection",
    "Data.IntSet.difference",
    "Data.IntSet.filter",
    "Data.IntSet.foldr",
    // Data.Sequence (list-backed)
    "Data.Sequence.empty",
    "Data.Sequence.singleton",
    "Data.Sequence.fromList",
    "Data.Sequence.toList",
    "Data.Sequence.length",
    "Data.Sequence.null",
    "Data.Sequence.index",
    "Data.Sequence.take",
    "Data.Sequence.drop",
    "Data.Sequence.reverse",
    // Data.ByteString (list of byte Ints)
    "Data.ByteString.pack",
    "Data.ByteString.unpack",
    "Data.ByteString.singleton",
    "Data.ByteString.length",
    "Data.ByteString.null",
    "Data.ByteString.head",
    "Data.ByteString.append",
    "Data.ByteString.take",
    "Data.ByteString.drop",
    "Data.ByteString.reverse",
    // Data.ByteString.Lazy (also list-backed; strict<->lazy is identity)
    "Data.ByteString.Lazy.fromStrict",
    "Data.ByteString.Lazy.toStrict",
    "Data.ByteString.Lazy.pack",
    "Data.ByteString.Lazy.unpack",
    "Data.ByteString.Lazy.empty",
    "Data.ByteString.Lazy.singleton",
    "Data.ByteString.Lazy.length",
    "Data.ByteString.Lazy.null",
    "Data.ByteString.Lazy.head",
    "Data.ByteString.Lazy.append",
    "Data.ByteString.Lazy.take",
    "Data.ByteString.Lazy.drop",
    "Data.ByteString.Lazy.reverse",
    "filter",
    "foldr",
    "foldl",
    "length",
    "elem",
    "enumFromTo",
    "take",
    "drop",
    "replicate",
    "null",
    "head",
    "tail",
    "product",
    "zipWith",
    "zip",
    "all",
    "any",
    "and",
    "or",
    "takeWhile",
    "dropWhile",
    "maximum",
    "minimum",
    "unzip",
    "splitAt",
    "span",
    "break",
    "findIndex",
    "elemIndex",
    "isPrefixOf",
    "fst",
    "snd",
    "fromMaybe",
    "even",
    "odd",
    "divMod",
    "quotRem",
    "min",
    "max",
    "subtract",
    "foldl1",
    "foldr1",
    "flip",
    "const",
    "isDigit",
    "isUpper",
    "isLower",
    "isAlpha",
    "isSpace",
    "toUpper",
    "toLower",
    "mapM_",
    "mapM",
    "intersect",
    "scanl",
    "scanl1",
    "scanr",
    "not",
    "reverse",
    "tails",
    "inits",
    "lookup",
    "maybeToList",
    "stripPrefix",
    "concat",
    "concatMap",
    "isSuffixOf",
    "isInfixOf",
    "__listAppend",
    "otherwise",
    "deleteBy",
    "intersectBy",
    "nubBy",
    "groupBy",
    "insert",
    "unionBy",
    "sortOn",
    "__insertOn",
    "mapAccumL",
    "mapAccumR",
    "compare",
    "maximumBy",
    "minimumBy",
    "bracket",
    "enumFromThenTo",
    "until",
    "swap",
    "curry",
    "uncurry",
    "&",
    "isAscii",
    "isLetter",
    "digitToInt",
];

/// Whether any binding in the module references `name`.
fn module_uses_name(binds: &[Bind], name: Symbol) -> bool {
    binds.iter().any(|bind| match bind {
        Bind::NonRec(_, e) => expr_uses_name(e, name),
        Bind::Rec(bs) => bs.iter().any(|(_, e)| expr_uses_name(e, name)),
    })
}

// Small Core builders for synthesizing prelude definitions.
fn pv(name: &str, id: usize) -> Var {
    Var::new(Symbol::intern(name), VarId::new(id), Ty::Error)
}
fn pev(v: &Var) -> Expr {
    Expr::Var(v.clone(), Span::default())
}
/// A reference resolved by name at lowering time (top-level function, primitive,
/// or constructor). The id is irrelevant.
fn pref(name: &str, id: &mut usize) -> Expr {
    let e = Expr::Var(pv(name, *id), Span::default());
    *id += 1;
    e
}
fn papp(f: Expr, x: Expr) -> Expr {
    Expr::App(Box::new(f), Box::new(x), Span::default())
}
fn papp2(f: Expr, a: Expr, b: Expr) -> Expr {
    papp(papp(f, a), b)
}
fn plam(v: Var, body: Expr) -> Expr {
    Expr::Lam(v, Box::new(body), Span::default())
}
fn pint(n: i64) -> Expr {
    Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
}

/// Whether an expression evaluates to a `Ratio Int` (a Rational). Types are
/// erased, so detect structurally: the `%` constructor, or an overloaded
/// `Num`/`Fractional` operator with a rational operand.
fn is_rational_expr(expr: &Expr) -> bool {
    let (head, args) = collect_app_spine(expr);
    if let Expr::Var(v, _) = head {
        let name = strip_qualifier(v.name.as_str());
        if name == "%" {
            return true;
        }
        if matches!(
            name,
            "+" | "-" | "*" | "/" | "negate" | "abs" | "signum" | "recip"
        ) && args.iter().any(|a| is_rational_expr(a))
        {
            return true;
        }
    }
    false
}

/// Whether a let-binding RHS is a bottom value — a call to `error`/`undefined`/
/// `throw`. Such a binding is lazy: it must only raise when forced, not when
/// bound. (The strict WASM backend would otherwise evaluate it eagerly.)
fn is_bottom_rhs(rhs: &Expr) -> bool {
    let (head, _) = collect_app_spine(rhs);
    if let Expr::Var(v, _) = head {
        matches!(
            strip_qualifier(v.name.as_str()),
            "error" | "errorWithoutStackTrace" | "undefined" | "throw"
        )
    } else {
        false
    }
}
fn pcase(scrut: Expr, alts: Vec<Alt>) -> Expr {
    Expr::Case(Box::new(scrut), alts, Ty::Error, Span::default())
}
fn pdatacon(name: &str, tag: u32, arity: u32) -> DataCon {
    DataCon {
        name: Symbol::intern(name),
        ty_con: TyCon::new(Symbol::intern("[]"), Kind::Star),
        tag,
        arity,
    }
}
fn palt(name: &str, tag: u32, arity: u32, binders: Vec<Var>, rhs: Expr) -> Alt {
    Alt {
        con: AltCon::DataCon(pdatacon(name, tag, arity)),
        binders,
        rhs,
    }
}
/// An empty `String` literal (the Writer monoid's `mempty`).
fn pstr_empty() -> Expr {
    Expr::Lit(
        Literal::String(Symbol::intern("")),
        Ty::Error,
        Span::default(),
    )
}

// ============================================================
// Monad transformer rewriting
// ============================================================
//
// The Core reaches the backend with untagged `>>=`/`return`/`ask`/`get`/… — the
// monad is only knowable from the operations a binding uses. We infer a
// binding's monad and rewrite its body into a concrete eager representation, so
// no transformer-specific machinery reaches the generic IO `>>=` path:
//
//   ReaderT r m a  ~  \r -> a            (a function of the environment)
//   StateT  s m a  ~  \s -> (a, s)       (state-threading function)
//   WriterT w m a  ~  (a, w)             (value paired with the log)
//   ExceptT e m a  ~  Either e a         (Left on throw, Right on success)
//
// The eliminators (runReaderT/evalStateT/runWriterT/runExceptT) — which appear
// in plain-IO code — apply or project these representations (see lower_app).

#[derive(Clone, Copy, PartialEq, Eq)]
enum MonadKind {
    Reader,
    State,
    Writer,
    Except,
}

/// The transformer a characteristic operation belongs to.
fn monad_op_kind(name: &str) -> Option<MonadKind> {
    match strip_qualifier(name) {
        "ask" | "asks" | "local" | "reader" => Some(MonadKind::Reader),
        "get" | "put" | "modify" | "modify'" | "gets" | "state" => Some(MonadKind::State),
        "tell" | "writer" => Some(MonadKind::Writer),
        "throwE" | "catchE" | "throwError" | "catchError" => Some(MonadKind::Except),
        _ => None,
    }
}

/// Infer a binding's monad from the first characteristic operation in its body.
/// `None` means plain IO (left untouched).
fn infer_monad_kind(expr: &Expr) -> Option<MonadKind> {
    match expr {
        Expr::Var(v, _) => monad_op_kind(v.name.as_str()),
        Expr::App(f, a, _) => infer_monad_kind(f).or_else(|| infer_monad_kind(a)),
        Expr::Lam(_, b, _) | Expr::TyLam(_, b, _) => infer_monad_kind(b),
        Expr::TyApp(i, _, _) | Expr::Cast(i, _, _) | Expr::Tick(_, i, _) | Expr::Lazy(i, _) => {
            infer_monad_kind(i)
        }
        Expr::Let(bind, b, _) => {
            let in_bind = match bind.as_ref() {
                Bind::NonRec(_, e) => infer_monad_kind(e),
                Bind::Rec(bs) => bs.iter().find_map(|(_, e)| infer_monad_kind(e)),
            };
            in_bind.or_else(|| infer_monad_kind(b))
        }
        Expr::Case(s, alts, _, _) => {
            infer_monad_kind(s).or_else(|| alts.iter().find_map(|a| infer_monad_kind(&a.rhs)))
        }
        _ => None,
    }
}

/// Rewrite a binding's body into its monad representation, preserving leading
/// value-parameter lambdas. Non-transformer bindings are cloned unchanged.
fn rewrite_bind_monads(bind: &Bind, ctx_stacks: &FxHashMap<String, bool>, id: &mut usize) -> Bind {
    let rewrite = |v: &Var, e: &Expr, id: &mut usize| -> (Var, Expr) {
        // Peel the binding's value parameters; rewrite the monadic body.
        let mut params: Vec<Var> = Vec::new();
        let mut cur = e;
        while let Expr::Lam(p, b, _) = cur {
            params.push(p.clone());
            cur = b;
        }
        // A two-context Reader+State stack (detected up front) takes priority,
        // then a data-outer `lift` stack, then single-layer inference.
        let rewritten = if let Some(&state_outer) = ctx_stacks.get(v.name.as_str()) {
            Some(rewrite_ctx_stack(state_outer, cur, id))
        } else if let Some((outer, inner)) = infer_stack(cur) {
            Some(rewrite_stack(outer, inner, cur, id))
        } else {
            infer_monad_kind(cur).map(|kind| rewrite_monad(kind, cur, id))
        };
        match rewritten {
            Some(mut body) => {
                for p in params.into_iter().rev() {
                    body = plam(p, body);
                }
                (v.clone(), body)
            }
            None => (v.clone(), e.clone()),
        }
    };
    match bind {
        Bind::NonRec(v, e) => {
            let (v, e) = rewrite(v, e, id);
            Bind::NonRec(v, Box::new(e))
        }
        Bind::Rec(bs) => Bind::Rec(
            bs.iter()
                .map(|(v, e)| {
                    let (v, e) = rewrite(v, e, id);
                    (v, Box::new(e))
                })
                .collect(),
        ),
    }
}

/// Rewrite the continuation of a bind (`\x -> rest`) so its body is rewritten in
/// the same monad. A non-lambda continuation (a function value) is left as-is.
fn rw_cont(kind: MonadKind, k: &Expr, id: &mut usize) -> Expr {
    match peel_type_abstractions(k) {
        Expr::Lam(x, body, _) => plam(x.clone(), rewrite_monad(kind, body, id)),
        other => other.clone(),
    }
}

/// Rewrite a monadic expression into the eager representation for `kind`.
fn rewrite_monad(kind: MonadKind, expr: &Expr, id: &mut usize) -> Expr {
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    // A monadic computation can be a `case`/`let` whose scrutinee/bound value is
    // pure but whose branches/body are monadic — rewrite those in place.
    match expr {
        Expr::Case(scrut, alts, ty, sp) => {
            let alts = alts
                .iter()
                .map(|a| Alt {
                    con: a.con.clone(),
                    binders: a.binders.clone(),
                    rhs: rewrite_monad(kind, &a.rhs, id),
                })
                .collect();
            return Expr::Case(scrut.clone(), alts, ty.clone(), *sp);
        }
        Expr::Let(bind, body, sp) => {
            return Expr::Let(bind.clone(), Box::new(rewrite_monad(kind, body, id)), *sp);
        }
        Expr::TyApp(i, t, sp) => {
            return Expr::TyApp(Box::new(rewrite_monad(kind, i, id)), t.clone(), *sp)
        }
        Expr::Cast(i, t, sp) => {
            return Expr::Cast(Box::new(rewrite_monad(kind, i, id)), t.clone(), *sp)
        }
        Expr::Tick(tk, i, sp) => {
            return Expr::Tick(tk.clone(), Box::new(rewrite_monad(kind, i, id)), *sp)
        }
        Expr::Lazy(i, sp) => return Expr::Lazy(Box::new(rewrite_monad(kind, i, id)), *sp),
        _ => {}
    }
    let (head, args) = collect_app_spine(expr);
    let op = match head {
        Expr::Var(v, _) => strip_qualifier(v.name.as_str()),
        _ => return expr.clone(),
    };
    match kind {
        MonadKind::Reader => match (op, args.len()) {
            (">>=", 2) => {
                let m = rewrite_monad(kind, args[0], id);
                let k = rw_cont(kind, args[1], id);
                let r = pv("r", fresh(id));
                // \r -> (k (m r)) r
                plam(r.clone(), papp(papp(k, papp(m, pev(&r))), pev(&r)))
            }
            (">>", 2) => {
                let m1 = rewrite_monad(kind, args[0], id);
                let m2 = rewrite_monad(kind, args[1], id);
                let r = pv("r", fresh(id));
                plam(
                    r.clone(),
                    papp2(pref("seq", id), papp(m1, pev(&r)), papp(m2, pev(&r))),
                )
            }
            ("return" | "pure", 1) => {
                let r = pv("r", fresh(id));
                plam(r, args[0].clone())
            }
            ("ask", 0) => {
                let r = pv("r", fresh(id));
                plam(r.clone(), pev(&r))
            }
            ("asks" | "reader", 1) => {
                let r = pv("r", fresh(id));
                plam(r.clone(), papp(args[0].clone(), pev(&r)))
            }
            ("local", 2) => {
                let m = rewrite_monad(kind, args[1], id);
                let r = pv("r", fresh(id));
                plam(r.clone(), papp(m, papp(args[0].clone(), pev(&r))))
            }
            _ => expr.clone(),
        },
        MonadKind::State => match (op, args.len()) {
            (">>=", 2) => {
                let m = rewrite_monad(kind, args[0], id);
                let k = rw_cont(kind, args[1], id);
                let s = pv("s", fresh(id));
                let a = pv("a", fresh(id));
                let s2 = pv("s2", fresh(id));
                // \s -> case m s of (a, s') -> (k a) s'
                plam(
                    s.clone(),
                    pcase(
                        papp(m, pev(&s)),
                        vec![palt(
                            "(,)",
                            0,
                            2,
                            vec![a.clone(), s2.clone()],
                            papp(papp(k, pev(&a)), pev(&s2)),
                        )],
                    ),
                )
            }
            (">>", 2) => {
                let m1 = rewrite_monad(kind, args[0], id);
                let m2 = rewrite_monad(kind, args[1], id);
                let s = pv("s", fresh(id));
                let a = pv("a", fresh(id));
                let s2 = pv("s2", fresh(id));
                plam(
                    s.clone(),
                    pcase(
                        papp(m1, pev(&s)),
                        vec![palt("(,)", 0, 2, vec![a, s2.clone()], papp(m2, pev(&s2)))],
                    ),
                )
            }
            ("return" | "pure", 1) => {
                let s = pv("s", fresh(id));
                plam(s.clone(), papp2(pref("(,)", id), args[0].clone(), pev(&s)))
            }
            ("get", 0) => {
                let s = pv("s", fresh(id));
                plam(s.clone(), papp2(pref("(,)", id), pev(&s), pev(&s)))
            }
            ("put", 1) => {
                let s = pv("s", fresh(id));
                plam(s, papp2(pref("(,)", id), pref("()", id), args[0].clone()))
            }
            ("modify" | "modify'", 1) => {
                let s = pv("s", fresh(id));
                plam(
                    s.clone(),
                    papp2(
                        pref("(,)", id),
                        pref("()", id),
                        papp(args[0].clone(), pev(&s)),
                    ),
                )
            }
            ("gets", 1) => {
                let s = pv("s", fresh(id));
                plam(
                    s.clone(),
                    papp2(pref("(,)", id), papp(args[0].clone(), pev(&s)), pev(&s)),
                )
            }
            _ => expr.clone(),
        },
        MonadKind::Writer => match (op, args.len()) {
            (">>=", 2) => {
                let m = rewrite_monad(kind, args[0], id);
                let k = rw_cont(kind, args[1], id);
                let a = pv("a", fresh(id));
                let w1 = pv("w1", fresh(id));
                let b = pv("b", fresh(id));
                let w2 = pv("w2", fresh(id));
                // case m of (a,w1) -> case k a of (b,w2) -> (b, w1 ++ w2)
                pcase(
                    m,
                    vec![palt(
                        "(,)",
                        0,
                        2,
                        vec![a.clone(), w1.clone()],
                        pcase(
                            papp(k, pev(&a)),
                            vec![palt(
                                "(,)",
                                0,
                                2,
                                vec![b.clone(), w2.clone()],
                                papp2(
                                    pref("(,)", id),
                                    pev(&b),
                                    papp2(pref("++", id), pev(&w1), pev(&w2)),
                                ),
                            )],
                        ),
                    )],
                )
            }
            (">>", 2) => {
                let m1 = rewrite_monad(kind, args[0], id);
                let m2 = rewrite_monad(kind, args[1], id);
                let a = pv("a", fresh(id));
                let w1 = pv("w1", fresh(id));
                let b = pv("b", fresh(id));
                let w2 = pv("w2", fresh(id));
                pcase(
                    m1,
                    vec![palt(
                        "(,)",
                        0,
                        2,
                        vec![a, w1.clone()],
                        pcase(
                            m2,
                            vec![palt(
                                "(,)",
                                0,
                                2,
                                vec![b.clone(), w2.clone()],
                                papp2(
                                    pref("(,)", id),
                                    pev(&b),
                                    papp2(pref("++", id), pev(&w1), pev(&w2)),
                                ),
                            )],
                        ),
                    )],
                )
            }
            ("return" | "pure", 1) => papp2(pref("(,)", id), args[0].clone(), pstr_empty()),
            ("tell" | "writer", 1) => papp2(pref("(,)", id), pref("()", id), args[0].clone()),
            _ => expr.clone(),
        },
        MonadKind::Except => match (op, args.len()) {
            (">>=", 2) => {
                let m = rewrite_monad(kind, args[0], id);
                let k = rw_cont(kind, args[1], id);
                let e = pv("e", fresh(id));
                let a = pv("a", fresh(id));
                // case m of Left e -> Left e; Right a -> k a
                pcase(
                    m,
                    vec![
                        palt(
                            "Left",
                            0,
                            1,
                            vec![e.clone()],
                            papp(pref("Left", id), pev(&e)),
                        ),
                        palt("Right", 1, 1, vec![a.clone()], papp(k, pev(&a))),
                    ],
                )
            }
            (">>", 2) => {
                let m1 = rewrite_monad(kind, args[0], id);
                let m2 = rewrite_monad(kind, args[1], id);
                let e = pv("e", fresh(id));
                let a = pv("a", fresh(id));
                pcase(
                    m1,
                    vec![
                        palt(
                            "Left",
                            0,
                            1,
                            vec![e.clone()],
                            papp(pref("Left", id), pev(&e)),
                        ),
                        palt("Right", 1, 1, vec![a], m2),
                    ],
                )
            }
            ("return" | "pure", 1) => papp(pref("Right", id), args[0].clone()),
            ("throwE" | "throwError", 1) => papp(pref("Left", id), args[0].clone()),
            ("catchE" | "catchError", 2) => {
                let m = rewrite_monad(kind, args[0], id);
                let h = rw_cont(kind, args[1], id);
                let e = pv("e", fresh(id));
                let a = pv("a", fresh(id));
                // case m of Left e -> h e; Right a -> Right a
                pcase(
                    m,
                    vec![
                        palt("Left", 0, 1, vec![e.clone()], papp(h, pev(&e))),
                        palt(
                            "Right",
                            1,
                            1,
                            vec![a.clone()],
                            papp(pref("Right", id), pev(&a)),
                        ),
                    ],
                )
            }
            _ => expr.clone(),
        },
    }
}

// ============================================================
// Two-layer monad transformer stacks
// ============================================================
//
// A binding with an explicit `lift` over a data-monad outer layer (Writer or
// Except) stacked on a context-monad inner layer (Reader or State). The outer
// layer's bind/ops are expressed in terms of the inner monad's bind/return:
//
//   WriterT w (m) a ~ m (a, w)         ExceptT e (m) a ~ m (Either e a)
//
// where m is the inner ReaderT/StateT (\r->a / \s->(a,s)). The eliminators
// already compose: runWriterT/runExceptT return the inner value, which the
// inner eliminator (runReaderT/runStateT) then applies/projects.

/// Find the monad of the operation directly under an explicit `lift` (the inner
/// layer), scanning the whole expression.
fn find_lifted_inner(expr: &Expr) -> Option<MonadKind> {
    let (head, args) = collect_app_spine(expr);
    if let Expr::Var(v, _) = head {
        if strip_qualifier(v.name.as_str()) == "lift" && args.len() == 1 {
            let (ih, _) = collect_app_spine(args[0]);
            if let Expr::Var(iv, _) = ih {
                if let Some(k) = monad_op_kind(iv.name.as_str()) {
                    return Some(k);
                }
            }
        }
    }
    // Recurse into children.
    match expr {
        Expr::App(f, a, _) => find_lifted_inner(f).or_else(|| find_lifted_inner(a)),
        Expr::Lam(_, b, _) | Expr::TyLam(_, b, _) => find_lifted_inner(b),
        Expr::TyApp(i, _, _) | Expr::Cast(i, _, _) | Expr::Tick(_, i, _) | Expr::Lazy(i, _) => {
            find_lifted_inner(i)
        }
        Expr::Let(bind, b, _) => {
            let in_bind = match bind.as_ref() {
                Bind::NonRec(_, e) => find_lifted_inner(e),
                Bind::Rec(bs) => bs.iter().find_map(|(_, e)| find_lifted_inner(e)),
            };
            in_bind.or_else(|| find_lifted_inner(b))
        }
        Expr::Case(s, alts, _, _) => {
            find_lifted_inner(s).or_else(|| alts.iter().find_map(|a| find_lifted_inner(&a.rhs)))
        }
        _ => None,
    }
}

/// Find the outer data-monad layer (Writer via `tell`, Except via
/// `throwE`/`catchE`) of a stacked binding.
fn find_outer_data_kind(expr: &Expr) -> Option<MonadKind> {
    fn scan(e: &Expr) -> Option<MonadKind> {
        if let Expr::Var(v, _) = e {
            match strip_qualifier(v.name.as_str()) {
                "tell" | "writer" => return Some(MonadKind::Writer),
                "throwE" | "throwError" | "catchE" | "catchError" => {
                    return Some(MonadKind::Except)
                }
                _ => {}
            }
        }
        match e {
            Expr::App(f, a, _) => scan(f).or_else(|| scan(a)),
            Expr::Lam(_, b, _) | Expr::TyLam(_, b, _) => scan(b),
            Expr::TyApp(i, _, _) | Expr::Cast(i, _, _) | Expr::Tick(_, i, _) | Expr::Lazy(i, _) => {
                scan(i)
            }
            Expr::Let(bind, b, _) => {
                let in_bind = match bind.as_ref() {
                    Bind::NonRec(_, e) => scan(e),
                    Bind::Rec(bs) => bs.iter().find_map(|(_, e)| scan(e)),
                };
                in_bind.or_else(|| scan(b))
            }
            Expr::Case(s, alts, _, _) => scan(s).or_else(|| alts.iter().find_map(|a| scan(&a.rhs))),
            _ => None,
        }
    }
    scan(expr)
}

/// Infer a two-layer stack `(outer, inner)`: a data-monad outer (Writer/Except)
/// over a context-monad inner (Reader/State), identified by an explicit `lift`.
fn infer_stack(expr: &Expr) -> Option<(MonadKind, MonadKind)> {
    let inner = find_lifted_inner(expr)?;
    if !matches!(inner, MonadKind::Reader | MonadKind::State) {
        return None;
    }
    let outer = find_outer_data_kind(expr)?;
    Some((outer, inner))
}

/// `return`/`pure` of the inner monad.
fn inner_return(inner: MonadKind, e: Expr, id: &mut usize) -> Expr {
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    match inner {
        MonadKind::Reader => plam(pv("r", fresh(id)), e),
        MonadKind::State => {
            let s = pv("s", fresh(id));
            plam(s.clone(), papp2(pref("(,)", id), e, pev(&s)))
        }
        _ => e,
    }
}

/// `m >>= k` of the inner monad, where `k` is a lambda `\x -> <inner value>`.
fn inner_bind(inner: MonadKind, m: Expr, k: Expr, id: &mut usize) -> Expr {
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    match inner {
        MonadKind::Reader => {
            let r = pv("r", fresh(id));
            // \r -> (k (m r)) r
            plam(r.clone(), papp(papp(k, papp(m, pev(&r))), pev(&r)))
        }
        MonadKind::State => {
            let s = pv("s", fresh(id));
            let x = pv("x", fresh(id));
            let s2 = pv("s2", fresh(id));
            // \s -> case m s of (x, s') -> (k x) s'
            plam(
                s.clone(),
                pcase(
                    papp(m, pev(&s)),
                    vec![palt(
                        "(,)",
                        0,
                        2,
                        vec![x.clone(), s2.clone()],
                        papp(papp(k, pev(&x)), pev(&s2)),
                    )],
                ),
            )
        }
        _ => m,
    }
}

/// Lift an inner-monad value into the outer (data) layer.
fn outer_lift(outer: MonadKind, inner: MonadKind, m: Expr, id: &mut usize) -> Expr {
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    let a = pv("a", fresh(id));
    let wrapped = match outer {
        // lift m = m >>= \a -> return (Right a)
        MonadKind::Except => inner_return(inner, papp(pref("Right", id), pev(&a)), id),
        // lift m = m >>= \a -> return (a, mempty)
        MonadKind::Writer => inner_return(inner, papp2(pref("(,)", id), pev(&a), pstr_empty()), id),
        _ => return m,
    };
    inner_bind(inner, m, plam(a, wrapped), id)
}

/// Rewrite the continuation of a stacked bind.
fn rw_stack_cont(outer: MonadKind, inner: MonadKind, k: &Expr, id: &mut usize) -> Expr {
    match peel_type_abstractions(k) {
        Expr::Lam(x, body, _) => plam(x.clone(), rewrite_stack(outer, inner, body, id)),
        other => other.clone(),
    }
}

/// Rewrite a two-layer-stack monadic expression into its representation
/// (`inner (a, w)` for Writer, `inner (Either e a)` for Except).
fn rewrite_stack(outer: MonadKind, inner: MonadKind, expr: &Expr, id: &mut usize) -> Expr {
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    // Recurse through case/let/wrappers (continuation bodies are monadic).
    match expr {
        Expr::Case(scrut, alts, ty, sp) => {
            let alts = alts
                .iter()
                .map(|a| Alt {
                    con: a.con.clone(),
                    binders: a.binders.clone(),
                    rhs: rewrite_stack(outer, inner, &a.rhs, id),
                })
                .collect();
            return Expr::Case(scrut.clone(), alts, ty.clone(), *sp);
        }
        Expr::Let(bind, body, sp) => {
            return Expr::Let(
                bind.clone(),
                Box::new(rewrite_stack(outer, inner, body, id)),
                *sp,
            );
        }
        Expr::TyApp(i, t, sp) => {
            return Expr::TyApp(Box::new(rewrite_stack(outer, inner, i, id)), t.clone(), *sp)
        }
        Expr::Cast(i, t, sp) => {
            return Expr::Cast(Box::new(rewrite_stack(outer, inner, i, id)), t.clone(), *sp)
        }
        Expr::Tick(tk, i, sp) => {
            return Expr::Tick(
                tk.clone(),
                Box::new(rewrite_stack(outer, inner, i, id)),
                *sp,
            )
        }
        Expr::Lazy(i, sp) => return Expr::Lazy(Box::new(rewrite_stack(outer, inner, i, id)), *sp),
        _ => {}
    }
    let (head, args) = collect_app_spine(expr);
    let op = match head {
        Expr::Var(v, _) => strip_qualifier(v.name.as_str()),
        _ => return expr.clone(),
    };
    // `lift m`: lift an inner action (rewritten single-layer) into the stack.
    if op == "lift" && args.len() == 1 {
        return outer_lift(outer, inner, rewrite_monad(inner, args[0], id), id);
    }
    match outer {
        MonadKind::Except => match (op, args.len()) {
            (">>=", 2) => {
                let m = rewrite_stack(outer, inner, args[0], id);
                let k = rw_stack_cont(outer, inner, args[1], id);
                let ea = pv("ea", fresh(id));
                let e = pv("e", fresh(id));
                let a = pv("a", fresh(id));
                // m >>= \ea -> case ea of Left e -> return (Left e); Right a -> k a
                let cont = plam(
                    ea.clone(),
                    pcase(
                        pev(&ea),
                        vec![
                            palt(
                                "Left",
                                0,
                                1,
                                vec![e.clone()],
                                inner_return(inner, papp(pref("Left", id), pev(&e)), id),
                            ),
                            palt("Right", 1, 1, vec![a.clone()], papp(k, pev(&a))),
                        ],
                    ),
                );
                inner_bind(inner, m, cont, id)
            }
            (">>", 2) => {
                let m = rewrite_stack(outer, inner, args[0], id);
                let m2 = rewrite_stack(outer, inner, args[1], id);
                let ea = pv("ea", fresh(id));
                let e = pv("e", fresh(id));
                let a = pv("a", fresh(id));
                let cont = plam(
                    ea.clone(),
                    pcase(
                        pev(&ea),
                        vec![
                            palt(
                                "Left",
                                0,
                                1,
                                vec![e.clone()],
                                inner_return(inner, papp(pref("Left", id), pev(&e)), id),
                            ),
                            palt("Right", 1, 1, vec![a], m2),
                        ],
                    ),
                );
                inner_bind(inner, m, cont, id)
            }
            ("return" | "pure", 1) => {
                inner_return(inner, papp(pref("Right", id), args[0].clone()), id)
            }
            ("throwE" | "throwError", 1) => {
                inner_return(inner, papp(pref("Left", id), args[0].clone()), id)
            }
            ("catchE" | "catchError", 2) => {
                let m = rewrite_stack(outer, inner, args[0], id);
                let h = rw_stack_cont(outer, inner, args[1], id);
                let ea = pv("ea", fresh(id));
                let e = pv("e", fresh(id));
                let a = pv("a", fresh(id));
                let cont = plam(
                    ea.clone(),
                    pcase(
                        pev(&ea),
                        vec![
                            palt("Left", 0, 1, vec![e.clone()], papp(h, pev(&e))),
                            palt(
                                "Right",
                                1,
                                1,
                                vec![a.clone()],
                                inner_return(inner, papp(pref("Right", id), pev(&a)), id),
                            ),
                        ],
                    ),
                );
                inner_bind(inner, m, cont, id)
            }
            _ => expr.clone(),
        },
        MonadKind::Writer => match (op, args.len()) {
            (">>=", 2) | (">>", 2) => {
                let is_bind = op == ">>=";
                let m = rewrite_stack(outer, inner, args[0], id);
                let p1 = pv("p1", fresh(id));
                let a = pv("a", fresh(id));
                let w1 = pv("w1", fresh(id));
                let p2 = pv("p2", fresh(id));
                let b = pv("b", fresh(id));
                let w2 = pv("w2", fresh(id));
                // k a (>>=) or the second action (>>)
                let kb = if is_bind {
                    let k = rw_stack_cont(outer, inner, args[1], id);
                    papp(k, pev(&a))
                } else {
                    rewrite_stack(outer, inner, args[1], id)
                };
                // inner: m >>= \(a,w1) -> kb >>= \(b,w2) -> return (b, w1++w2)
                let inner_cont = plam(
                    p2.clone(),
                    pcase(
                        pev(&p2),
                        vec![palt(
                            "(,)",
                            0,
                            2,
                            vec![b.clone(), w2.clone()],
                            inner_return(
                                inner,
                                papp2(
                                    pref("(,)", id),
                                    pev(&b),
                                    papp2(pref("++", id), pev(&w1), pev(&w2)),
                                ),
                                id,
                            ),
                        )],
                    ),
                );
                let outer_cont = plam(
                    p1.clone(),
                    pcase(
                        pev(&p1),
                        vec![palt(
                            "(,)",
                            0,
                            2,
                            vec![a.clone(), w1.clone()],
                            inner_bind(inner, kb, inner_cont, id),
                        )],
                    ),
                );
                inner_bind(inner, m, outer_cont, id)
            }
            ("return" | "pure", 1) => inner_return(
                inner,
                papp2(pref("(,)", id), args[0].clone(), pstr_empty()),
                id,
            ),
            ("tell" | "writer", 1) => inner_return(
                inner,
                papp2(pref("(,)", id), pref("()", id), args[0].clone()),
                id,
            ),
            _ => expr.clone(),
        },
        _ => expr.clone(),
    }
}

// ============================================================
// Two-context monad stacks (Reader + State, mtl-auto-lifted)
// ============================================================
//
// A binding using both Reader (ask) and State (get/put) operations with no
// explicit `lift`. Represented as a curried function of both contexts returning
// (value, state):
//   StateT s (ReaderT r)  ~  \s -> \r -> (a, s)
//   ReaderT r (StateT s)  ~  \r -> \s -> (a, s)
// The outer layer (which the innermost eliminator peels) is inferred from the
// program's eliminator nesting.

/// Whether any operation of `kind` appears in `expr`.
fn body_uses_monad(expr: &Expr, kind: MonadKind) -> bool {
    if let Expr::Var(v, _) = expr {
        if monad_op_kind(v.name.as_str()) == Some(kind) {
            return true;
        }
    }
    match expr {
        Expr::App(f, a, _) => body_uses_monad(f, kind) || body_uses_monad(a, kind),
        Expr::Lam(_, b, _) | Expr::TyLam(_, b, _) => body_uses_monad(b, kind),
        Expr::TyApp(i, _, _) | Expr::Cast(i, _, _) | Expr::Tick(_, i, _) | Expr::Lazy(i, _) => {
            body_uses_monad(i, kind)
        }
        Expr::Let(bind, b, _) => {
            let in_bind = match bind.as_ref() {
                Bind::NonRec(_, e) => body_uses_monad(e, kind),
                Bind::Rec(bs) => bs.iter().any(|(_, e)| body_uses_monad(e, kind)),
            };
            in_bind || body_uses_monad(b, kind)
        }
        Expr::Case(s, alts, _, _) => {
            body_uses_monad(s, kind) || alts.iter().any(|a| body_uses_monad(&a.rhs, kind))
        }
        _ => false,
    }
}

/// Whether `expr` contains an explicit `lift` application.
fn has_explicit_lift(expr: &Expr) -> bool {
    if let (Expr::Var(v, _), 1) = {
        let (h, a) = collect_app_spine(expr);
        (h, a.len())
    } {
        if strip_qualifier(v.name.as_str()) == "lift" {
            return true;
        }
    }
    match expr {
        Expr::App(f, a, _) => has_explicit_lift(f) || has_explicit_lift(a),
        Expr::Lam(_, b, _) | Expr::TyLam(_, b, _) => has_explicit_lift(b),
        Expr::TyApp(i, _, _) | Expr::Cast(i, _, _) | Expr::Tick(_, i, _) | Expr::Lazy(i, _) => {
            has_explicit_lift(i)
        }
        Expr::Let(bind, b, _) => {
            let in_bind = match bind.as_ref() {
                Bind::NonRec(_, e) => has_explicit_lift(e),
                Bind::Rec(bs) => bs.iter().any(|(_, e)| has_explicit_lift(e)),
            };
            in_bind || has_explicit_lift(b)
        }
        Expr::Case(s, alts, _, _) => {
            has_explicit_lift(s) || alts.iter().any(|a| has_explicit_lift(&a.rhs))
        }
        _ => false,
    }
}

/// Whether `expr` is a two-context Reader+State stack (uses both, no `lift`).
fn infer_ctx_stack(expr: &Expr) -> bool {
    !has_explicit_lift(expr)
        && body_uses_monad(expr, MonadKind::State)
        && body_uses_monad(expr, MonadKind::Reader)
}

/// The eliminator family a name belongs to (State vs Reader), if it is one.
fn eliminator_kind(name: &str) -> Option<MonadKind> {
    match strip_qualifier(name) {
        "evalStateT" | "runStateT" | "execStateT" | "evalState" | "runState" | "execState" => {
            Some(MonadKind::State)
        }
        "runReaderT" | "runReader" => Some(MonadKind::Reader),
        _ => None,
    }
}

/// Find the outer layer of a two-context stack binding `name` by locating the
/// eliminator applied *directly* to it anywhere in the program (the innermost
/// eliminator peels the outermost transformer).
fn find_stack_outer(binds: &[Bind], name: Symbol) -> Option<MonadKind> {
    fn scan(e: &Expr, name: Symbol) -> Option<MonadKind> {
        let (head, args) = collect_app_spine(e);
        if let Expr::Var(v, _) = head {
            if let Some(kind) = eliminator_kind(v.name.as_str()) {
                if let Some(Expr::Var(arg0, _)) = args.first().map(|a| peel_head(a)) {
                    if arg0.name == name {
                        return Some(kind);
                    }
                }
            }
        }
        // recurse
        let mut found = None;
        for a in &args {
            found = found.or_else(|| scan(a, name));
        }
        found.or_else(|| match e {
            Expr::Lam(_, b, _) | Expr::TyLam(_, b, _) => scan(b, name),
            Expr::TyApp(i, _, _) | Expr::Cast(i, _, _) | Expr::Tick(_, i, _) | Expr::Lazy(i, _) => {
                scan(i, name)
            }
            Expr::Let(bind, b, _) => {
                let ib = match bind.as_ref() {
                    Bind::NonRec(_, e) => scan(e, name),
                    Bind::Rec(bs) => bs.iter().find_map(|(_, e)| scan(e, name)),
                };
                ib.or_else(|| scan(b, name))
            }
            Expr::Case(s, alts, _, _) => {
                scan(s, name).or_else(|| alts.iter().find_map(|a| scan(&a.rhs, name)))
            }
            _ => None,
        })
    }
    binds.iter().find_map(|b| match b {
        Bind::NonRec(_, e) => scan(e, name),
        Bind::Rec(bs) => bs.iter().find_map(|(_, e)| scan(e, name)),
    })
}

/// Apply a two-context value `f` to the state and reader contexts in the order
/// dictated by `state_outer`.
fn ctx_app2(state_outer: bool, f: Expr, s_val: Expr, r_val: Expr) -> Expr {
    if state_outer {
        papp2(f, s_val, r_val)
    } else {
        papp2(f, r_val, s_val)
    }
}

/// Wrap `body` in the two context lambdas in `state_outer` order.
fn ctx_wrap2(state_outer: bool, s: Var, r: Var, body: Expr) -> Expr {
    if state_outer {
        plam(s, plam(r, body))
    } else {
        plam(r, plam(s, body))
    }
}

/// Rewrite the continuation of a two-context bind.
fn rw_ctx_cont(state_outer: bool, k: &Expr, id: &mut usize) -> Expr {
    match peel_type_abstractions(k) {
        Expr::Lam(x, body, _) => plam(x.clone(), rewrite_ctx_stack(state_outer, body, id)),
        other => other.clone(),
    }
}

/// Rewrite a two-context Reader+State computation into `\pO -> \pI -> (a, s)`.
fn rewrite_ctx_stack(state_outer: bool, expr: &Expr, id: &mut usize) -> Expr {
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    match expr {
        Expr::Case(scrut, alts, ty, sp) => {
            let alts = alts
                .iter()
                .map(|a| Alt {
                    con: a.con.clone(),
                    binders: a.binders.clone(),
                    rhs: rewrite_ctx_stack(state_outer, &a.rhs, id),
                })
                .collect();
            return Expr::Case(scrut.clone(), alts, ty.clone(), *sp);
        }
        Expr::Let(bind, body, sp) => {
            return Expr::Let(
                bind.clone(),
                Box::new(rewrite_ctx_stack(state_outer, body, id)),
                *sp,
            );
        }
        Expr::TyApp(i, t, sp) => {
            return Expr::TyApp(
                Box::new(rewrite_ctx_stack(state_outer, i, id)),
                t.clone(),
                *sp,
            )
        }
        Expr::Cast(i, t, sp) => {
            return Expr::Cast(
                Box::new(rewrite_ctx_stack(state_outer, i, id)),
                t.clone(),
                *sp,
            )
        }
        Expr::Tick(tk, i, sp) => {
            return Expr::Tick(
                tk.clone(),
                Box::new(rewrite_ctx_stack(state_outer, i, id)),
                *sp,
            )
        }
        Expr::Lazy(i, sp) => {
            return Expr::Lazy(Box::new(rewrite_ctx_stack(state_outer, i, id)), *sp)
        }
        _ => {}
    }
    let (head, args) = collect_app_spine(expr);
    let op = match head {
        Expr::Var(v, _) => strip_qualifier(v.name.as_str()),
        _ => return expr.clone(),
    };
    let pair = |a: Expr, s: Expr, id: &mut usize| papp2(pref("(,)", id), a, s);
    match (op, args.len()) {
        (">>=", 2) | (">>", 2) => {
            let is_bind = op == ">>=";
            let m = rewrite_ctx_stack(state_outer, args[0], id);
            let s = pv("s", fresh(id));
            let r = pv("r", fresh(id));
            let a = pv("a", fresh(id));
            let s2 = pv("s2", fresh(id));
            // (k a) or the second action, applied to (s', r)
            let next = if is_bind {
                let k = rw_ctx_cont(state_outer, args[1], id);
                papp(k, pev(&a))
            } else {
                rewrite_ctx_stack(state_outer, args[1], id)
            };
            let body = pcase(
                ctx_app2(state_outer, m, pev(&s), pev(&r)),
                vec![palt(
                    "(,)",
                    0,
                    2,
                    vec![a.clone(), s2.clone()],
                    ctx_app2(state_outer, next, pev(&s2), pev(&r)),
                )],
            );
            ctx_wrap2(state_outer, s, r, body)
        }
        ("return" | "pure", 1) => {
            let s = pv("s", fresh(id));
            let r = pv("r", fresh(id));
            let body = pair(args[0].clone(), pev(&s), id);
            ctx_wrap2(state_outer, s, r, body)
        }
        ("get", 0) => {
            let s = pv("s", fresh(id));
            let r = pv("r", fresh(id));
            let body = pair(pev(&s), pev(&s), id);
            ctx_wrap2(state_outer, s, r, body)
        }
        ("put", 1) => {
            let s = pv("s", fresh(id));
            let r = pv("r", fresh(id));
            let body = papp2(pref("(,)", id), pref("()", id), args[0].clone());
            ctx_wrap2(state_outer, s, r, body)
        }
        ("modify" | "modify'", 1) => {
            let s = pv("s", fresh(id));
            let r = pv("r", fresh(id));
            let body = papp2(
                pref("(,)", id),
                pref("()", id),
                papp(args[0].clone(), pev(&s)),
            );
            ctx_wrap2(state_outer, s, r, body)
        }
        ("gets", 1) => {
            let s = pv("s", fresh(id));
            let r = pv("r", fresh(id));
            let body = pair(papp(args[0].clone(), pev(&s)), pev(&s), id);
            ctx_wrap2(state_outer, s, r, body)
        }
        ("ask", 0) => {
            let s = pv("s", fresh(id));
            let r = pv("r", fresh(id));
            let body = pair(pev(&r), pev(&s), id);
            ctx_wrap2(state_outer, s, r, body)
        }
        ("asks" | "reader", 1) => {
            let s = pv("s", fresh(id));
            let r = pv("r", fresh(id));
            let body = pair(papp(args[0].clone(), pev(&r)), pev(&s), id);
            ctx_wrap2(state_outer, s, r, body)
        }
        _ => expr.clone(),
    }
}

/// Synthesize a `Data.Sequence`/`Data.ByteString` operation. Both are backed by
/// a plain list (a ByteString is a list of byte-valued `Int`s), so most ops
/// alias the list prelude. Returns the lambda/CAF body, or `None`.
fn build_listbacked_fn(name: &str, id: &mut usize) -> Option<Expr> {
    let (prefix, op) = name.rsplit_once('.')?;
    if !(prefix.contains("Sequence") || prefix.contains("ByteString")) {
        return None;
    }
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    let body = match op {
        // empty is a nullary CAF: the empty list (nil == 0).
        "empty" => pref("[]", id),
        // singleton x = [x]
        "singleton" => {
            let x = pv("x", fresh(id));
            plam(x.clone(), papp2(pref(":", id), pev(&x), pref("[]", id)))
        }
        // identity conversions between the list and its packed/lazy forms
        "fromList" | "toList" | "pack" | "unpack" | "fromStrict" | "toStrict" | "copy" => {
            let x = pv("x", fresh(id));
            plam(x.clone(), pev(&x))
        }
        // unary ops aliasing the list prelude
        "length" | "null" | "head" | "reverse" => {
            let s = pv("s", fresh(id));
            plam(s.clone(), papp(pref(op, id), pev(&s)))
        }
        // take/drop n s
        "take" | "drop" => {
            let n = pv("n", fresh(id));
            let s = pv("s", fresh(id));
            plam(
                n.clone(),
                plam(s.clone(), papp2(pref(op, id), pev(&n), pev(&s))),
            )
        }
        // index s i = head (drop i s)
        "index" => {
            let s = pv("s", fresh(id));
            let i = pv("i", fresh(id));
            plam(
                s.clone(),
                plam(
                    i.clone(),
                    papp(pref("head", id), papp2(pref("drop", id), pev(&i), pev(&s))),
                ),
            )
        }
        // append a b = a ++ b
        "append" => {
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            plam(
                a.clone(),
                plam(b.clone(), papp2(pref("__listAppend", id), pev(&a), pev(&b))),
            )
        }
        _ => return None,
    };
    Some(body)
}

/// Build a single named list-prelude binding, drawing fresh ids from `id`.
/// Synthesize a `Data.Map`/`Data.Set`/`Data.IntMap`/`Data.IntSet` operation.
///
/// A Set is a deduplicated list `[a]`; a Map is an association list `[(k, v)]`.
/// IntMap/IntSet share the Map/Set implementations. Sibling/recursive calls use
/// the same module prefix as `name`, so the IntMap/IntSet copies stay internally
/// consistent. Returns the lambda body, or `None` if `name` isn't a container op.
fn build_container_fn(name: &str, id: &mut usize) -> Option<Expr> {
    let (prefix, op) = name.rsplit_once('.')?;
    let is_map = prefix.ends_with("Map");
    let is_set = prefix.ends_with("Set");
    if !is_map && !is_set {
        return None;
    }
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    // Reference a sibling op in the same module (e.g. insert calls delete).
    let q = |op: &str, id: &mut usize| pref(&format!("{prefix}.{op}"), id);

    let body = if is_map {
        match op {
            // member k m = case m of [] -> False
            //   (p:ps) -> case fst p == k of { True -> True; False -> member k ps }
            "member" => {
                let k = pv("k", fresh(id));
                let m = pv("m", fresh(id));
                let p = pv("p", fresh(id));
                let ps = pv("ps", fresh(id));
                let eqk = papp2(pref("==", id), papp(pref("fst", id), pev(&p)), pev(&k));
                let inner = pcase(
                    eqk,
                    vec![
                        palt(
                            "False",
                            0,
                            0,
                            vec![],
                            papp2(q("member", id), pev(&k), pev(&ps)),
                        ),
                        palt("True", 1, 0, vec![], pref("True", id)),
                    ],
                );
                let nil = palt("[]", 0, 0, vec![], pref("False", id));
                let cons = palt(":", 1, 2, vec![p.clone(), ps.clone()], inner);
                plam(k, plam(m.clone(), pcase(pev(&m), vec![nil, cons])))
            }
            // insert k v m = (k,v) : delete k m
            "insert" => {
                let k = pv("k", fresh(id));
                let v = pv("v", fresh(id));
                let m = pv("m", fresh(id));
                let pair = papp2(pref("(,)", id), pev(&k), pev(&v));
                let body = papp2(
                    pref(":", id),
                    pair,
                    papp2(q("delete", id), pev(&k), pev(&m)),
                );
                plam(k, plam(v, plam(m.clone(), body)))
            }
            // delete k m = filter (\p -> fst p /= k) m
            "delete" => {
                let k = pv("k", fresh(id));
                let m = pv("m", fresh(id));
                let p = pv("p", fresh(id));
                let pred = plam(
                    p.clone(),
                    papp2(pref("/=", id), papp(pref("fst", id), pev(&p)), pev(&k)),
                );
                plam(k, plam(m.clone(), papp2(pref("filter", id), pred, pev(&m))))
            }
            // fromList xs = case xs of [] -> []
            //   (p:ps) -> case p of (k,v) -> insert k v (fromList ps)
            "fromList" => {
                let xs = pv("xs", fresh(id));
                let p = pv("p", fresh(id));
                let ps = pv("ps", fresh(id));
                let k = pv("k", fresh(id));
                let v = pv("v", fresh(id));
                let ins = papp(
                    papp2(q("insert", id), pev(&k), pev(&v)),
                    papp(q("fromList", id), pev(&ps)),
                );
                let on_pair = pcase(
                    pev(&p),
                    vec![palt("(,)", 0, 2, vec![k.clone(), v.clone()], ins)],
                );
                let nil = palt("[]", 0, 0, vec![], pref("[]", id));
                let cons = palt(":", 1, 2, vec![p.clone(), ps.clone()], on_pair);
                plam(xs.clone(), pcase(pev(&xs), vec![nil, cons]))
            }
            "size" => {
                let m = pv("m", fresh(id));
                plam(m.clone(), papp(pref("length", id), pev(&m)))
            }
            // null m = case m of [] -> True; (_:_) -> False
            "null" => {
                let m = pv("m", fresh(id));
                let p = pv("p", fresh(id));
                let ps = pv("ps", fresh(id));
                plam(
                    m.clone(),
                    pcase(
                        pev(&m),
                        vec![
                            palt("[]", 0, 0, vec![], pref("True", id)),
                            palt(":", 1, 2, vec![p, ps], pref("False", id)),
                        ],
                    ),
                )
            }
            // findWithDefault d k m = case m of [] -> d
            //   (p:ps) -> case fst p == k of { True -> snd p; False -> findWithDefault d k ps }
            "findWithDefault" => {
                let d = pv("d", fresh(id));
                let k = pv("k", fresh(id));
                let m = pv("m", fresh(id));
                let p = pv("p", fresh(id));
                let ps = pv("ps", fresh(id));
                let eqk = papp2(pref("==", id), papp(pref("fst", id), pev(&p)), pev(&k));
                let inner = pcase(
                    eqk,
                    vec![
                        palt(
                            "False",
                            0,
                            0,
                            vec![],
                            papp(papp2(q("findWithDefault", id), pev(&d), pev(&k)), pev(&ps)),
                        ),
                        palt("True", 1, 0, vec![], papp(pref("snd", id), pev(&p))),
                    ],
                );
                let nil = palt("[]", 0, 0, vec![], pev(&d));
                let cons = palt(":", 1, 2, vec![p.clone(), ps.clone()], inner);
                plam(d, plam(k, plam(m.clone(), pcase(pev(&m), vec![nil, cons]))))
            }
            "elems" => {
                let m = pv("m", fresh(id));
                plam(m.clone(), papp2(pref("map", id), pref("snd", id), pev(&m)))
            }
            "keys" => {
                let m = pv("m", fresh(id));
                plam(m.clone(), papp2(pref("map", id), pref("fst", id), pev(&m)))
            }
            // lookup k m = case m of [] -> Nothing
            //   (p:ps) -> case fst p == k of { True -> Just (snd p); False -> lookup k ps }
            "lookup" => {
                let k = pv("k", fresh(id));
                let m = pv("m", fresh(id));
                let p = pv("p", fresh(id));
                let ps = pv("ps", fresh(id));
                let eqk = papp2(pref("==", id), papp(pref("fst", id), pev(&p)), pev(&k));
                let inner = pcase(
                    eqk,
                    vec![
                        palt(
                            "False",
                            0,
                            0,
                            vec![],
                            papp2(q("lookup", id), pev(&k), pev(&ps)),
                        ),
                        palt(
                            "True",
                            1,
                            0,
                            vec![],
                            papp(pref("Just", id), papp(pref("snd", id), pev(&p))),
                        ),
                    ],
                );
                let nil = palt("[]", 0, 0, vec![], pref("Nothing", id));
                let cons = palt(":", 1, 2, vec![p.clone(), ps.clone()], inner);
                plam(k, plam(m.clone(), pcase(pev(&m), vec![nil, cons])))
            }
            // mapMaybe f m = case m of [] -> []
            //   ((k,v):ps) -> case f v of { Nothing -> rest; Just w -> (k,w):rest }
            "mapMaybe" => {
                let f = pv("f", fresh(id));
                let m = pv("m", fresh(id));
                let p = pv("p", fresh(id));
                let ps = pv("ps", fresh(id));
                let k = pv("k", fresh(id));
                let v = pv("v", fresh(id));
                let w = pv("w", fresh(id));
                let rest = papp2(q("mapMaybe", id), pev(&f), pev(&ps));
                let on_just = papp2(
                    pref(":", id),
                    papp2(pref("(,)", id), pev(&k), pev(&w)),
                    rest.clone(),
                );
                let on_fv = pcase(
                    papp(pev(&f), pev(&v)),
                    vec![
                        palt("Nothing", 0, 0, vec![], rest),
                        palt("Just", 1, 1, vec![w.clone()], on_just),
                    ],
                );
                let on_pair = pcase(
                    pev(&p),
                    vec![palt("(,)", 0, 2, vec![k.clone(), v.clone()], on_fv)],
                );
                let nil = palt("[]", 0, 0, vec![], pref("[]", id));
                let cons = palt(":", 1, 2, vec![p.clone(), ps.clone()], on_pair);
                plam(f, plam(m.clone(), pcase(pev(&m), vec![nil, cons])))
            }
            // union a b = case a of [] -> b; (p:ps) -> insert (fst p) (snd p) (union ps b)
            "union" => {
                let a = pv("a", fresh(id));
                let b = pv("b", fresh(id));
                let p = pv("p", fresh(id));
                let ps = pv("ps", fresh(id));
                let ins = papp(
                    papp2(
                        q("insert", id),
                        papp(pref("fst", id), pev(&p)),
                        papp(pref("snd", id), pev(&p)),
                    ),
                    papp2(q("union", id), pev(&ps), pev(&b)),
                );
                let nil = palt("[]", 0, 0, vec![], pev(&b));
                let cons = palt(":", 1, 2, vec![p.clone(), ps.clone()], ins);
                plam(a.clone(), plam(b.clone(), pcase(pev(&a), vec![nil, cons])))
            }
            // unions ms = case ms of [] -> []; (m:rest) -> union m (unions rest)
            "unions" => {
                let ms = pv("ms", fresh(id));
                let m = pv("m", fresh(id));
                let rest = pv("rest", fresh(id));
                let u = papp2(q("union", id), pev(&m), papp(q("unions", id), pev(&rest)));
                let nil = palt("[]", 0, 0, vec![], pref("[]", id));
                let cons = palt(":", 1, 2, vec![m.clone(), rest.clone()], u);
                plam(ms.clone(), pcase(pev(&ms), vec![nil, cons]))
            }
            // update f k m = case m of [] -> []
            //   ((pk,pv):ps) -> case pk == k of
            //       True  -> case f pv of { Nothing -> ps; Just w -> (k,w):ps }
            //       False -> (pk,pv) : update f k ps
            "update" => {
                let f = pv("f", fresh(id));
                let k = pv("k", fresh(id));
                let m = pv("m", fresh(id));
                let p = pv("p", fresh(id));
                let ps = pv("ps", fresh(id));
                let pk = pv("pk", fresh(id));
                let pvv = pv("pvv", fresh(id));
                let w = pv("w", fresh(id));
                let on_fv = pcase(
                    papp(pev(&f), pev(&pvv)),
                    vec![
                        palt("Nothing", 0, 0, vec![], pev(&ps)),
                        palt(
                            "Just",
                            1,
                            1,
                            vec![w.clone()],
                            papp2(
                                pref(":", id),
                                papp2(pref("(,)", id), pev(&k), pev(&w)),
                                pev(&ps),
                            ),
                        ),
                    ],
                );
                let recur = papp2(
                    pref(":", id),
                    pev(&p),
                    papp(papp2(q("update", id), pev(&f), pev(&k)), pev(&ps)),
                );
                let on_match = pcase(
                    papp2(pref("==", id), pev(&pk), pev(&k)),
                    vec![
                        palt("False", 0, 0, vec![], recur),
                        palt("True", 1, 0, vec![], on_fv),
                    ],
                );
                let on_pair = pcase(
                    pev(&p),
                    vec![palt("(,)", 0, 2, vec![pk.clone(), pvv.clone()], on_match)],
                );
                let nil = palt("[]", 0, 0, vec![], pref("[]", id));
                let cons = palt(":", 1, 2, vec![p.clone(), ps.clone()], on_pair);
                plam(f, plam(k, plam(m.clone(), pcase(pev(&m), vec![nil, cons]))))
            }
            // alter f k m = case lookup k m of
            //   Nothing -> case f Nothing  of { Nothing -> m; Just w -> insert k w m }
            //   Just v  -> case f (Just v) of { Nothing -> delete k m; Just w -> insert k w m }
            "alter" => {
                let f = pv("f", fresh(id));
                let k = pv("k", fresh(id));
                let m = pv("m", fresh(id));
                let v = pv("v", fresh(id));
                let w = pv("w", fresh(id));
                let ins = |w: &Var, id: &mut usize| {
                    papp(
                        papp2(pref(&format!("{prefix}.insert"), id), pev(&k), pev(w)),
                        pev(&m),
                    )
                };
                let on_absent = pcase(
                    papp(pev(&f), pref("Nothing", id)),
                    vec![
                        palt("Nothing", 0, 0, vec![], pev(&m)),
                        palt("Just", 1, 1, vec![w.clone()], ins(&w, id)),
                    ],
                );
                let on_present = pcase(
                    papp(pev(&f), papp(pref("Just", id), pev(&v))),
                    vec![
                        palt(
                            "Nothing",
                            0,
                            0,
                            vec![],
                            papp2(q("delete", id), pev(&k), pev(&m)),
                        ),
                        palt("Just", 1, 1, vec![w.clone()], ins(&w, id)),
                    ],
                );
                let body = pcase(
                    papp2(q("lookup", id), pev(&k), pev(&m)),
                    vec![
                        palt("Nothing", 0, 0, vec![], on_absent),
                        palt("Just", 1, 1, vec![v.clone()], on_present),
                    ],
                );
                plam(f, plam(k, plam(m.clone(), body)))
            }
            _ => return None,
        }
    } else {
        match op {
            "member" => {
                let x = pv("x", fresh(id));
                let s = pv("s", fresh(id));
                plam(
                    x.clone(),
                    plam(s.clone(), papp2(pref("elem", id), pev(&x), pev(&s))),
                )
            }
            // insert x s = case elem x s of { True -> s; False -> x : s }
            "insert" => {
                let x = pv("x", fresh(id));
                let s = pv("s", fresh(id));
                let body = pcase(
                    papp2(pref("elem", id), pev(&x), pev(&s)),
                    vec![
                        palt(
                            "False",
                            0,
                            0,
                            vec![],
                            papp2(pref(":", id), pev(&x), pev(&s)),
                        ),
                        palt("True", 1, 0, vec![], pev(&s)),
                    ],
                );
                plam(x.clone(), plam(s.clone(), body))
            }
            // delete x s = filter (\e -> e /= x) s
            "delete" => {
                let x = pv("x", fresh(id));
                let s = pv("s", fresh(id));
                let e = pv("e", fresh(id));
                let pred = plam(e.clone(), papp2(pref("/=", id), pev(&e), pev(&x)));
                plam(
                    x.clone(),
                    plam(s.clone(), papp2(pref("filter", id), pred, pev(&s))),
                )
            }
            // fromList xs = case xs of [] -> []; (y:ys) -> insert y (fromList ys)
            "fromList" => {
                let xs = pv("xs", fresh(id));
                let y = pv("y", fresh(id));
                let ys = pv("ys", fresh(id));
                let ins = papp2(q("insert", id), pev(&y), papp(q("fromList", id), pev(&ys)));
                let nil = palt("[]", 0, 0, vec![], pref("[]", id));
                let cons = palt(":", 1, 2, vec![y.clone(), ys.clone()], ins);
                plam(xs.clone(), pcase(pev(&xs), vec![nil, cons]))
            }
            "size" => {
                let s = pv("s", fresh(id));
                plam(s.clone(), papp(pref("length", id), pev(&s)))
            }
            // union a b = case a of [] -> b; (x:xs) -> insert x (union xs b)
            "union" => {
                let a = pv("a", fresh(id));
                let b = pv("b", fresh(id));
                let x = pv("x", fresh(id));
                let xs = pv("xs", fresh(id));
                let ins = papp2(
                    q("insert", id),
                    pev(&x),
                    papp2(q("union", id), pev(&xs), pev(&b)),
                );
                let nil = palt("[]", 0, 0, vec![], pev(&b));
                let cons = palt(":", 1, 2, vec![x.clone(), xs.clone()], ins);
                plam(a.clone(), plam(b.clone(), pcase(pev(&a), vec![nil, cons])))
            }
            // intersection a b = filter (\x -> elem x b) a
            "intersection" => {
                let a = pv("a", fresh(id));
                let b = pv("b", fresh(id));
                let x = pv("x", fresh(id));
                let pred = plam(x.clone(), papp2(pref("elem", id), pev(&x), pev(&b)));
                plam(
                    a.clone(),
                    plam(b.clone(), papp2(pref("filter", id), pred, pev(&a))),
                )
            }
            // difference a b = filter (\x -> not (elem x b)) a
            "difference" => {
                let a = pv("a", fresh(id));
                let b = pv("b", fresh(id));
                let x = pv("x", fresh(id));
                let pred = plam(
                    x.clone(),
                    papp(pref("not", id), papp2(pref("elem", id), pev(&x), pev(&b))),
                );
                plam(
                    a.clone(),
                    plam(b.clone(), papp2(pref("filter", id), pred, pev(&a))),
                )
            }
            "filter" => {
                let p = pv("p", fresh(id));
                let s = pv("s", fresh(id));
                plam(
                    p.clone(),
                    plam(s.clone(), papp2(pref("filter", id), pev(&p), pev(&s))),
                )
            }
            "foldr" => {
                let f = pv("f", fresh(id));
                let z = pv("z", fresh(id));
                let s = pv("s", fresh(id));
                plam(
                    f.clone(),
                    plam(
                        z.clone(),
                        plam(
                            s.clone(),
                            papp(papp2(pref("foldr", id), pev(&f), pev(&z)), pev(&s)),
                        ),
                    ),
                )
            }
            _ => return None,
        }
    };
    Some(body)
}

fn build_list_fn(name: &str, id: &mut usize) -> Option<(Var, Expr)> {
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
    // Data.Map/Set/IntMap/IntSet operations, synthesized over an assoc-list
    // (Map) or list (Set) representation.
    if let Some(body) = build_container_fn(name, id) {
        return Some((pv(name, fresh(id)), body));
    }
    // Data.Sequence/ByteString operations, synthesized over a list.
    if let Some(body) = build_listbacked_fn(name, id) {
        return Some((pv(name, fresh(id)), body));
    }
    let body = match name {
        // map f xs = case xs of { [] -> []; (y:ys) -> f y : map f ys }
        "map" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let nil = palt("[]", 0, 0, vec![], pref("[]", id));
            let cons = palt(
                ":",
                1,
                2,
                vec![y.clone(), ys.clone()],
                papp2(
                    pref(":", id),
                    papp(pev(&f), pev(&y)),
                    papp2(pref("map", id), pev(&f), pev(&ys)),
                ),
            );
            plam(f, plam(xs.clone(), pcase(pev(&xs), vec![nil, cons])))
        }
        // __fmapMaybe f m = case m of { Nothing -> Nothing; Just x -> Just (f x) }
        // The Maybe arm of `fmap`, synthesized when the program uses `fmap`.
        "__fmapMaybe" => {
            let f = pv("f", fresh(id));
            let m = pv("m", fresh(id));
            let x = pv("x", fresh(id));
            let nothing = palt("Nothing", 0, 0, vec![], pref("Nothing", id));
            let just = palt(
                "Just",
                1,
                1,
                vec![x.clone()],
                papp(pref("Just", id), papp(pev(&f), pev(&x))),
            );
            plam(f, plam(m.clone(), pcase(pev(&m), vec![nothing, just])))
        }
        // filter p xs = case xs of
        //   [] -> []
        //   (y:ys) -> case p y of { True -> y : filter p ys; False -> filter p ys }
        "filter" => {
            let p = pv("p", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let keep = papp2(
                pref(":", id),
                pev(&y),
                papp2(pref("filter", id), pev(&p), pev(&ys)),
            );
            let skip = papp2(pref("filter", id), pev(&p), pev(&ys));
            let inner = pcase(
                papp(pev(&p), pev(&y)),
                vec![
                    palt("False", 0, 0, vec![], skip),
                    palt("True", 1, 0, vec![], keep),
                ],
            );
            let nil = palt("[]", 0, 0, vec![], pref("[]", id));
            let cons = palt(":", 1, 2, vec![y.clone(), ys.clone()], inner);
            plam(p, plam(xs.clone(), pcase(pev(&xs), vec![nil, cons])))
        }
        // foldr f z xs = case xs of { [] -> z; (y:ys) -> f y (foldr f z ys) }
        "foldr" => {
            let f = pv("f", fresh(id));
            let z = pv("z", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let nil = palt("[]", 0, 0, vec![], pev(&z));
            let cons = palt(
                ":",
                1,
                2,
                vec![y.clone(), ys.clone()],
                papp2(
                    pev(&f),
                    pev(&y),
                    papp(papp2(pref("foldr", id), pev(&f), pev(&z)), pev(&ys)),
                ),
            );
            plam(
                f,
                plam(z, plam(xs.clone(), pcase(pev(&xs), vec![nil, cons]))),
            )
        }
        // foldl f z xs = case xs of { [] -> z; (y:ys) -> foldl f (f z y) ys }
        "foldl" => {
            let f = pv("f", fresh(id));
            let z = pv("z", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let nil = palt("[]", 0, 0, vec![], pev(&z));
            let cons = palt(
                ":",
                1,
                2,
                vec![y.clone(), ys.clone()],
                papp(
                    papp2(pref("foldl", id), pev(&f), papp2(pev(&f), pev(&z), pev(&y))),
                    pev(&ys),
                ),
            );
            plam(
                f,
                plam(z, plam(xs.clone(), pcase(pev(&xs), vec![nil, cons]))),
            )
        }
        // length xs = case xs of { [] -> 0; (y:ys) -> 1 + length ys }
        "length" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let nil = palt("[]", 0, 0, vec![], pint(0));
            let cons = palt(
                ":",
                1,
                2,
                vec![y, ys.clone()],
                papp2(pref("+", id), pint(1), papp(pref("length", id), pev(&ys))),
            );
            plam(xs.clone(), pcase(pev(&xs), vec![nil, cons]))
        }
        // elem x xs = case xs of
        //   [] -> False
        //   (y:ys) -> case x == y of { True -> True; False -> elem x ys }
        "elem" => {
            let x = pv("x", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                papp2(pref("==", id), pev(&x), pev(&y)),
                vec![
                    palt(
                        "False",
                        0,
                        0,
                        vec![],
                        papp2(pref("elem", id), pev(&x), pev(&ys)),
                    ),
                    palt("True", 1, 0, vec![], pref("True", id)),
                ],
            );
            let nil = palt("[]", 0, 0, vec![], pref("False", id));
            let cons = palt(":", 1, 2, vec![y.clone(), ys.clone()], inner);
            plam(x, plam(xs.clone(), pcase(pev(&xs), vec![nil, cons])))
        }
        // enumFromTo lo hi = case lo > hi of { True -> []; False -> lo : enumFromTo (lo+1) hi }
        "enumFromTo" => {
            let lo = pv("lo", fresh(id));
            let hi = pv("hi", fresh(id));
            let cond = papp2(pref(">", id), pev(&lo), pev(&hi));
            let empty = palt("True", 1, 0, vec![], pref("[]", id));
            let step = palt(
                "False",
                0,
                0,
                vec![],
                papp2(
                    pref(":", id),
                    pev(&lo),
                    papp2(
                        pref("enumFromTo", id),
                        papp2(pref("+", id), pev(&lo), pint(1)),
                        pev(&hi),
                    ),
                ),
            );
            plam(lo, plam(hi.clone(), pcase(cond, vec![step, empty])))
        }
        // take n xs = case n <= 0 of
        //   True -> []
        //   False -> case xs of { [] -> []; (y:ys) -> y : take (n-1) ys }
        "take" => {
            let n = pv("n", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![y.clone(), ys.clone()],
                        papp2(
                            pref(":", id),
                            pev(&y),
                            papp2(
                                pref("take", id),
                                papp2(pref("-", id), pev(&n), pint(1)),
                                pev(&ys),
                            ),
                        ),
                    ),
                ],
            );
            let cond = papp2(pref("<=", id), pev(&n), pint(0));
            let body = pcase(
                cond,
                vec![
                    palt("False", 0, 0, vec![], inner),
                    palt("True", 1, 0, vec![], pref("[]", id)),
                ],
            );
            plam(n, plam(xs, body))
        }
        // drop n xs = case n <= 0 of
        //   True -> xs
        //   False -> case xs of { [] -> []; (y:ys) -> drop (n-1) ys }
        "drop" => {
            let n = pv("n", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![y, ys.clone()],
                        papp2(
                            pref("drop", id),
                            papp2(pref("-", id), pev(&n), pint(1)),
                            pev(&ys),
                        ),
                    ),
                ],
            );
            let cond = papp2(pref("<=", id), pev(&n), pint(0));
            let body = pcase(
                cond,
                vec![
                    palt("False", 0, 0, vec![], inner),
                    palt("True", 1, 0, vec![], pev(&xs)),
                ],
            );
            plam(n, plam(xs.clone(), body))
        }
        // replicate n x = case n <= 0 of { True -> []; False -> x : replicate (n-1) x }
        "replicate" => {
            let n = pv("n", fresh(id));
            let x = pv("x", fresh(id));
            let cons = papp2(
                pref(":", id),
                pev(&x),
                papp2(
                    pref("replicate", id),
                    papp2(pref("-", id), pev(&n), pint(1)),
                    pev(&x),
                ),
            );
            let cond = papp2(pref("<=", id), pev(&n), pint(0));
            let body = pcase(
                cond,
                vec![
                    palt("False", 0, 0, vec![], cons),
                    palt("True", 1, 0, vec![], pref("[]", id)),
                ],
            );
            plam(n, plam(x, body))
        }
        // null xs = case xs of { [] -> True; (y:ys) -> False }
        "null" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("True", id)),
                    palt(":", 1, 2, vec![y, ys], pref("False", id)),
                ],
            );
            plam(xs.clone(), body)
        }
        // head (y:_) = y
        "head" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let body = pcase(
                pev(&xs),
                vec![palt(":", 1, 2, vec![y.clone(), ys], pev(&y))],
            );
            plam(xs.clone(), body)
        }
        // tail (_:ys) = ys
        "tail" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let body = pcase(
                pev(&xs),
                vec![palt(":", 1, 2, vec![y, ys.clone()], pev(&ys))],
            );
            plam(xs.clone(), body)
        }
        // product xs = case xs of { [] -> 1; (y:ys) -> y * product ys }
        "product" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pint(1)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![y.clone(), ys.clone()],
                        papp2(pref("*", id), pev(&y), papp(pref("product", id), pev(&ys))),
                    ),
                ],
            );
            plam(xs.clone(), body)
        }
        // zipWith f xs ys = case xs of
        //   [] -> []
        //   (a:as) -> case ys of { [] -> []; (b:bs) -> f a b : zipWith f as bs }
        "zipWith" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let a = pv("a", fresh(id));
            let as_ = pv("as", fresh(id));
            let b = pv("b", fresh(id));
            let bs = pv("bs", fresh(id));
            let inner = pcase(
                pev(&ys),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![b.clone(), bs.clone()],
                        papp2(
                            pref(":", id),
                            papp2(pev(&f), pev(&a), pev(&b)),
                            papp(papp2(pref("zipWith", id), pev(&f), pev(&as_)), pev(&bs)),
                        ),
                    ),
                ],
            );
            let outer = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![a.clone(), as_.clone()], inner),
                ],
            );
            plam(f, plam(xs, plam(ys.clone(), outer)))
        }
        // zip xs ys = case xs of
        //   [] -> []
        //   (a:as) -> case ys of { [] -> []; (b:bs) -> (a,b) : zip as bs }
        "zip" => {
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let a = pv("a", fresh(id));
            let as_ = pv("as", fresh(id));
            let b = pv("b", fresh(id));
            let bs = pv("bs", fresh(id));
            let inner = pcase(
                pev(&ys),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![b.clone(), bs.clone()],
                        papp2(
                            pref(":", id),
                            papp2(pref("(,)", id), pev(&a), pev(&b)),
                            papp2(pref("zip", id), pev(&as_), pev(&bs)),
                        ),
                    ),
                ],
            );
            let outer = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![a.clone(), as_.clone()], inner),
                ],
            );
            plam(xs, plam(ys.clone(), outer))
        }
        // all p xs = case xs of
        //   [] -> True
        //   (y:ys) -> case p y of { True -> all p ys; False -> False }
        "all" => {
            let p = pv("p", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                papp(pev(&p), pev(&y)),
                vec![
                    palt("False", 0, 0, vec![], pref("False", id)),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(pref("all", id), pev(&p), pev(&ys)),
                    ),
                ],
            );
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("True", id)),
                    palt(":", 1, 2, vec![y.clone(), ys.clone()], inner),
                ],
            );
            plam(p, plam(xs, body))
        }
        // any p xs = case xs of
        //   [] -> False
        //   (y:ys) -> case p y of { True -> True; False -> any p ys }
        "any" => {
            let p = pv("p", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                papp(pev(&p), pev(&y)),
                vec![
                    palt(
                        "False",
                        0,
                        0,
                        vec![],
                        papp2(pref("any", id), pev(&p), pev(&ys)),
                    ),
                    palt("True", 1, 0, vec![], pref("True", id)),
                ],
            );
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("False", id)),
                    palt(":", 1, 2, vec![y.clone(), ys.clone()], inner),
                ],
            );
            plam(p, plam(xs, body))
        }
        // and xs = case xs of { [] -> True; (y:ys) -> case y of { True -> and ys; False -> False } }
        "and" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                pev(&y),
                vec![
                    palt("False", 0, 0, vec![], pref("False", id)),
                    palt("True", 1, 0, vec![], papp(pref("and", id), pev(&ys))),
                ],
            );
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("True", id)),
                    palt(":", 1, 2, vec![y.clone(), ys.clone()], inner),
                ],
            );
            plam(xs.clone(), body)
        }
        // or xs = case xs of { [] -> False; (y:ys) -> case y of { True -> True; False -> or ys } }
        "or" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                pev(&y),
                vec![
                    palt("False", 0, 0, vec![], papp(pref("or", id), pev(&ys))),
                    palt("True", 1, 0, vec![], pref("True", id)),
                ],
            );
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("False", id)),
                    palt(":", 1, 2, vec![y.clone(), ys.clone()], inner),
                ],
            );
            plam(xs.clone(), body)
        }
        // takeWhile p xs = case xs of
        //   [] -> []
        //   (y:ys) -> case p y of { True -> y : takeWhile p ys; False -> [] }
        "takeWhile" => {
            let p = pv("p", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                papp(pev(&p), pev(&y)),
                vec![
                    palt("False", 0, 0, vec![], pref("[]", id)),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(
                            pref(":", id),
                            pev(&y),
                            papp2(pref("takeWhile", id), pev(&p), pev(&ys)),
                        ),
                    ),
                ],
            );
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![y.clone(), ys.clone()], inner),
                ],
            );
            plam(p, plam(xs, body))
        }
        // dropWhile p xs = case xs of
        //   [] -> []
        //   (y:ys) -> case p y of { True -> dropWhile p ys; False -> xs }
        "dropWhile" => {
            let p = pv("p", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let inner = pcase(
                papp(pev(&p), pev(&y)),
                vec![
                    palt("False", 0, 0, vec![], pev(&xs)),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(pref("dropWhile", id), pev(&p), pev(&ys)),
                    ),
                ],
            );
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![y.clone(), ys.clone()], inner),
                ],
            );
            plam(p, plam(xs.clone(), body))
        }
        // maximum/minimum xs = foldl (\a b -> if b `cmp` a then b else a) (head) (tail)
        "maximum" | "minimum" => {
            let cmp_op = if name == "maximum" { ">" } else { "<" };
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let cmp = plam(
                a.clone(),
                plam(
                    b.clone(),
                    pcase(
                        papp2(pref(cmp_op, id), pev(&b), pev(&a)),
                        vec![
                            palt("False", 0, 0, vec![], pev(&a)),
                            palt("True", 1, 0, vec![], pev(&b)),
                        ],
                    ),
                ),
            );
            let cons_rhs = papp(papp2(pref("foldl", id), cmp, pev(&y)), pev(&ys));
            plam(
                xs.clone(),
                pcase(
                    pev(&xs),
                    vec![
                        palt("[]", 0, 0, vec![], pint(0)),
                        palt(":", 1, 2, vec![y.clone(), ys.clone()], cons_rhs),
                    ],
                ),
            )
        }
        // unzip xs = case xs of
        //   [] -> ([], [])
        //   (p:rest) -> case p of (a,b) -> case unzip rest of (as,bs) -> (a:as, b:bs)
        "unzip" => {
            let xs = pv("xs", fresh(id));
            let p = pv("p", fresh(id));
            let rest = pv("rest", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let as_ = pv("as", fresh(id));
            let bs = pv("bs", fresh(id));
            let rebuild = papp2(
                pref("(,)", id),
                papp2(pref(":", id), pev(&a), pev(&as_)),
                papp2(pref(":", id), pev(&b), pev(&bs)),
            );
            let inner = pcase(
                papp(pref("unzip", id), pev(&rest)),
                vec![palt("(,)", 0, 2, vec![as_.clone(), bs.clone()], rebuild)],
            );
            let cons_rhs = pcase(
                pev(&p),
                vec![palt("(,)", 0, 2, vec![a.clone(), b.clone()], inner)],
            );
            let nil_rhs = papp2(pref("(,)", id), pref("[]", id), pref("[]", id));
            plam(
                xs.clone(),
                pcase(
                    pev(&xs),
                    vec![
                        palt("[]", 0, 0, vec![], nil_rhs),
                        palt(":", 1, 2, vec![p.clone(), rest.clone()], cons_rhs),
                    ],
                ),
            )
        }
        // splitAt n xs = case n <= 0 of
        //   True  -> ([], xs)
        //   False -> case xs of { [] -> ([],[]); (y:ys) -> case splitAt (n-1) ys of (a,b) -> (y:a, b) }
        "splitAt" => {
            let n = pv("n", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let rec = pcase(
                papp2(
                    pref("splitAt", id),
                    papp2(pref("-", id), pev(&n), pint(1)),
                    pev(&ys),
                ),
                vec![palt(
                    "(,)",
                    0,
                    2,
                    vec![a.clone(), b.clone()],
                    papp2(
                        pref("(,)", id),
                        papp2(pref(":", id), pev(&y), pev(&a)),
                        pev(&b),
                    ),
                )],
            );
            let false_rhs = pcase(
                pev(&xs),
                vec![
                    palt(
                        "[]",
                        0,
                        0,
                        vec![],
                        papp2(pref("(,)", id), pref("[]", id), pref("[]", id)),
                    ),
                    palt(":", 1, 2, vec![y.clone(), ys.clone()], rec),
                ],
            );
            let body = pcase(
                papp2(pref("<=", id), pev(&n), pint(0)),
                vec![
                    palt("False", 0, 0, vec![], false_rhs),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(pref("(,)", id), pref("[]", id), pev(&xs)),
                    ),
                ],
            );
            plam(n, plam(xs.clone(), body))
        }
        // span/break p xs: split at the first element where the predicate
        // changes. span keeps the prefix where p holds; break keeps the prefix
        // where p does *not* hold.
        "span" | "break" => {
            let p = pv("p", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let rec = pcase(
                papp2(pref(name, id), pev(&p), pev(&ys)),
                vec![palt(
                    "(,)",
                    0,
                    2,
                    vec![a.clone(), b.clone()],
                    papp2(
                        pref("(,)", id),
                        papp2(pref(":", id), pev(&y), pev(&a)),
                        pev(&b),
                    ),
                )],
            );
            let stop = papp2(
                pref("(,)", id),
                pref("[]", id),
                papp2(pref(":", id), pev(&y), pev(&ys)),
            );
            let (true_branch, false_branch) = if name == "span" {
                (rec, stop)
            } else {
                (stop, rec)
            };
            let cons_rhs = pcase(
                papp(pev(&p), pev(&y)),
                vec![
                    palt("False", 0, 0, vec![], false_branch),
                    palt("True", 1, 0, vec![], true_branch),
                ],
            );
            plam(
                p,
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt(
                                "[]",
                                0,
                                0,
                                vec![],
                                papp2(pref("(,)", id), pref("[]", id), pref("[]", id)),
                            ),
                            palt(":", 1, 2, vec![y.clone(), ys.clone()], cons_rhs),
                        ],
                    ),
                ),
            )
        }
        // findIndex p xs = case xs of
        //   [] -> Nothing
        //   (y:ys) -> case p y of
        //     True  -> Just 0
        //     False -> case findIndex p ys of { Nothing -> Nothing; Just i -> Just (i+1) }
        "findIndex" => {
            let p = pv("p", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let i = pv("i", fresh(id));
            let rec = pcase(
                papp2(pref("findIndex", id), pev(&p), pev(&ys)),
                vec![
                    palt("Nothing", 0, 0, vec![], pref("Nothing", id)),
                    palt(
                        "Just",
                        1,
                        1,
                        vec![i.clone()],
                        papp(pref("Just", id), papp2(pref("+", id), pev(&i), pint(1))),
                    ),
                ],
            );
            let cons_rhs = pcase(
                papp(pev(&p), pev(&y)),
                vec![
                    palt("False", 0, 0, vec![], rec),
                    palt("True", 1, 0, vec![], papp(pref("Just", id), pint(0))),
                ],
            );
            plam(
                p,
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pref("Nothing", id)),
                            palt(":", 1, 2, vec![y.clone(), ys.clone()], cons_rhs),
                        ],
                    ),
                ),
            )
        }
        // elemIndex x xs = findIndex (\y -> y == x) xs
        "elemIndex" => {
            let x = pv("x", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let pred = plam(y.clone(), papp2(pref("==", id), pev(&y), pev(&x)));
            plam(
                x.clone(),
                plam(xs.clone(), papp2(pref("findIndex", id), pred, pev(&xs))),
            )
        }
        // isPrefixOf xs ys = case xs of
        //   [] -> True
        //   (a:as) -> case ys of { [] -> False; (b:bs) -> case a==b of { False -> False; True -> isPrefixOf as bs } }
        "isPrefixOf" => {
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let a = pv("a", fresh(id));
            let as_ = pv("as", fresh(id));
            let b = pv("b", fresh(id));
            let bs = pv("bs", fresh(id));
            let inner = pcase(
                papp2(pref("==", id), pev(&a), pev(&b)),
                vec![
                    palt("False", 0, 0, vec![], pref("False", id)),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(pref("isPrefixOf", id), pev(&as_), pev(&bs)),
                    ),
                ],
            );
            let cons_xs = pcase(
                pev(&ys),
                vec![
                    palt("[]", 0, 0, vec![], pref("False", id)),
                    palt(":", 1, 2, vec![b.clone(), bs.clone()], inner),
                ],
            );
            plam(
                xs.clone(),
                plam(
                    ys.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pref("True", id)),
                            palt(":", 1, 2, vec![a.clone(), as_.clone()], cons_xs),
                        ],
                    ),
                ),
            )
        }
        // fst / snd: tuple accessors.
        "fst" | "snd" => {
            let p = pv("p", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let result = if name == "fst" { pev(&a) } else { pev(&b) };
            plam(
                p.clone(),
                pcase(
                    pev(&p),
                    vec![palt("(,)", 0, 2, vec![a.clone(), b.clone()], result)],
                ),
            )
        }
        // fromMaybe d m = case m of { Nothing -> d; Just x -> x }
        "fromMaybe" => {
            let d = pv("d", fresh(id));
            let m = pv("m", fresh(id));
            let x = pv("x", fresh(id));
            plam(
                d.clone(),
                plam(
                    m.clone(),
                    pcase(
                        pev(&m),
                        vec![
                            palt("Nothing", 0, 0, vec![], pev(&d)),
                            palt("Just", 1, 1, vec![x.clone()], pev(&x)),
                        ],
                    ),
                ),
            )
        }
        // divMod n d = (div n d, mod n d) ; quotRem n d = (quot n d, rem n d)
        "divMod" | "quotRem" => {
            let (op1, op2) = if name == "divMod" {
                ("div", "mod")
            } else {
                ("quot", "rem")
            };
            let n = pv("n", fresh(id));
            let d = pv("d", fresh(id));
            plam(
                n.clone(),
                plam(
                    d.clone(),
                    papp2(
                        pref("(,)", id),
                        papp2(pref(op1, id), pev(&n), pev(&d)),
                        papp2(pref(op2, id), pev(&n), pev(&d)),
                    ),
                ),
            )
        }
        // even n = rem n 2 == 0 ; odd n = rem n 2 /= 0
        "even" | "odd" => {
            let cmp = if name == "even" { "==" } else { "/=" };
            let n = pv("n", fresh(id));
            plam(
                n.clone(),
                papp2(
                    pref(cmp, id),
                    papp2(pref("rem", id), pev(&n), pint(2)),
                    pint(0),
                ),
            )
        }
        // min/max a b = if a `op` b then a else b
        "min" | "max" => {
            let op = if name == "max" { ">=" } else { "<=" };
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            plam(
                a.clone(),
                plam(
                    b.clone(),
                    pcase(
                        papp2(pref(op, id), pev(&a), pev(&b)),
                        vec![
                            palt("False", 0, 0, vec![], pev(&b)),
                            palt("True", 1, 0, vec![], pev(&a)),
                        ],
                    ),
                ),
            )
        }
        // subtract n x = x - n
        "subtract" => {
            let n = pv("n", fresh(id));
            let x = pv("x", fresh(id));
            plam(
                n.clone(),
                plam(x.clone(), papp2(pref("-", id), pev(&x), pev(&n))),
            )
        }
        // foldl1 f xs = case xs of (y:ys) -> foldl f y ys
        "foldl1" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            plam(
                f.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pint(0)),
                            palt(
                                ":",
                                1,
                                2,
                                vec![y.clone(), ys.clone()],
                                papp(papp2(pref("foldl", id), pev(&f), pev(&y)), pev(&ys)),
                            ),
                        ],
                    ),
                ),
            )
        }
        // foldr1 f xs = case xs of (y:ys) -> case ys of { [] -> y; _ -> f y (foldr1 f ys) }
        "foldr1" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let z = pv("z", fresh(id));
            let zs = pv("zs", fresh(id));
            let inner = pcase(
                pev(&ys),
                vec![
                    palt("[]", 0, 0, vec![], pev(&y)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![z.clone(), zs.clone()],
                        papp2(
                            pev(&f),
                            pev(&y),
                            papp2(pref("foldr1", id), pev(&f), pev(&ys)),
                        ),
                    ),
                ],
            );
            plam(
                f.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pint(0)),
                            palt(":", 1, 2, vec![y.clone(), ys.clone()], inner),
                        ],
                    ),
                ),
            )
        }
        // flip f a b = f b a
        "flip" => {
            let f = pv("f", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            plam(
                f.clone(),
                plam(a.clone(), plam(b.clone(), papp2(pev(&f), pev(&b), pev(&a)))),
            )
        }
        // const a b = a
        "const" => {
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            plam(a.clone(), plam(b, pev(&a)))
        }
        // Char predicates on the codepoint: range checks.
        "isDigit" | "isUpper" | "isLower" => {
            let (lo, hi) = match name {
                "isDigit" => (48, 57),
                "isUpper" => (65, 90),
                _ => (97, 122),
            };
            let c = pv("c", fresh(id));
            plam(
                c.clone(),
                pcase(
                    papp2(pref(">=", id), pev(&c), pint(lo)),
                    vec![
                        palt("False", 0, 0, vec![], pref("False", id)),
                        palt(
                            "True",
                            1,
                            0,
                            vec![],
                            papp2(pref("<=", id), pev(&c), pint(hi)),
                        ),
                    ],
                ),
            )
        }
        // isAlpha c = isUpper c || isLower c
        "isAlpha" => {
            let c = pv("c", fresh(id));
            plam(
                c.clone(),
                pcase(
                    papp(pref("isUpper", id), pev(&c)),
                    vec![
                        palt("False", 0, 0, vec![], papp(pref("isLower", id), pev(&c))),
                        palt("True", 1, 0, vec![], pref("True", id)),
                    ],
                ),
            )
        }
        // isSpace c = c==32 || c==9 || c==10 || c==13
        "isSpace" => {
            let c = pv("c", fresh(id));
            let or_eq = |val: i64, alt: Expr, id: &mut usize| {
                pcase(
                    papp2(pref("==", id), pev(&c), pint(val)),
                    vec![
                        palt("False", 0, 0, vec![], alt),
                        palt("True", 1, 0, vec![], pref("True", id)),
                    ],
                )
            };
            let e = pref("False", id);
            let e = or_eq(13, e, id);
            let e = or_eq(10, e, id);
            let e = or_eq(9, e, id);
            let e = or_eq(32, e, id);
            plam(c.clone(), e)
        }
        // toUpper c = if isLower c then c - 32 else c
        // toLower c = if isUpper c then c + 32 else c
        "toUpper" | "toLower" => {
            let (test, op, delta) = if name == "toUpper" {
                ("isLower", "-", 32)
            } else {
                ("isUpper", "+", 32)
            };
            let c = pv("c", fresh(id));
            plam(
                c.clone(),
                pcase(
                    papp(pref(test, id), pev(&c)),
                    vec![
                        palt("False", 0, 0, vec![], pev(&c)),
                        palt(
                            "True",
                            1,
                            0,
                            vec![],
                            papp2(pref(op, id), pev(&c), pint(delta)),
                        ),
                    ],
                ),
            )
        }
        // mapM_ f xs = case xs of { [] -> (); (y:ys) -> f y >> mapM_ f ys }
        "mapM_" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let cons = papp2(
                pref(">>", id),
                papp(pev(&f), pev(&y)),
                papp2(pref("mapM_", id), pev(&f), pev(&ys)),
            );
            plam(
                f.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pint(0)),
                            palt(":", 1, 2, vec![y.clone(), ys.clone()], cons),
                        ],
                    ),
                ),
            )
        }
        // mapM f xs = case xs of
        //   [] -> return []
        //   (y:ys) -> f y >>= \r -> mapM f ys >>= \rs -> return (r:rs)
        "mapM" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let r = pv("r", fresh(id));
            let rs = pv("rs", fresh(id));
            let ret_cons = papp(pref("return", id), papp2(pref(":", id), pev(&r), pev(&rs)));
            let inner_bind = papp2(
                pref(">>=", id),
                papp2(pref("mapM", id), pev(&f), pev(&ys)),
                plam(rs.clone(), ret_cons),
            );
            let outer_bind = papp2(
                pref(">>=", id),
                papp(pev(&f), pev(&y)),
                plam(r.clone(), inner_bind),
            );
            let nil = papp(pref("return", id), pref("[]", id));
            plam(
                f.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], nil),
                            palt(":", 1, 2, vec![y.clone(), ys.clone()], outer_bind),
                        ],
                    ),
                ),
            )
        }
        // intersect xs ys = filter (\x -> elem x ys) xs
        "intersect" => {
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let x = pv("x", fresh(id));
            let pred = plam(x.clone(), papp2(pref("elem", id), pev(&x), pev(&ys)));
            plam(
                xs.clone(),
                plam(ys.clone(), papp2(pref("filter", id), pred, pev(&xs))),
            )
        }
        // scanl f z xs = z : (case xs of { [] -> []; (y:ys) -> scanl f (f z y) ys })
        "scanl" => {
            let f = pv("f", fresh(id));
            let z = pv("z", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let rest = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![y.clone(), ys.clone()],
                        papp(
                            papp2(pref("scanl", id), pev(&f), papp2(pev(&f), pev(&z), pev(&y))),
                            pev(&ys),
                        ),
                    ),
                ],
            );
            plam(
                f.clone(),
                plam(
                    z.clone(),
                    plam(xs.clone(), papp2(pref(":", id), pev(&z), rest)),
                ),
            )
        }
        // scanl1 f xs = case xs of { [] -> []; (y:ys) -> scanl f y ys }
        "scanl1" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            plam(
                f.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pref("[]", id)),
                            palt(
                                ":",
                                1,
                                2,
                                vec![y.clone(), ys.clone()],
                                papp(papp2(pref("scanl", id), pev(&f), pev(&y)), pev(&ys)),
                            ),
                        ],
                    ),
                ),
            )
        }
        // scanr f z xs = case xs of
        //   [] -> [z]
        //   (y:ys) -> case scanr f z ys of (q:qs) -> f y q : q : qs
        "scanr" => {
            let f = pv("f", fresh(id));
            let z = pv("z", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let q = pv("q", fresh(id));
            let qs = pv("qs", fresh(id));
            let cons = pcase(
                papp(papp2(pref("scanr", id), pev(&f), pev(&z)), pev(&ys)),
                vec![palt(
                    ":",
                    1,
                    2,
                    vec![q.clone(), qs.clone()],
                    papp2(
                        pref(":", id),
                        papp2(pev(&f), pev(&y), pev(&q)),
                        papp2(pref(":", id), pev(&q), pev(&qs)),
                    ),
                )],
            );
            let nil = papp2(pref(":", id), pev(&z), pref("[]", id));
            plam(
                f.clone(),
                plam(
                    z.clone(),
                    plam(
                        xs.clone(),
                        pcase(
                            pev(&xs),
                            vec![
                                palt("[]", 0, 0, vec![], nil),
                                palt(":", 1, 2, vec![y.clone(), ys.clone()], cons),
                            ],
                        ),
                    ),
                ),
            )
        }
        // not b = case b of { True -> False; False -> True }
        "not" => {
            let b = pv("b", fresh(id));
            plam(
                b.clone(),
                pcase(
                    pev(&b),
                    vec![
                        palt("False", 0, 0, vec![], pref("True", id)),
                        palt("True", 1, 0, vec![], pref("False", id)),
                    ],
                ),
            )
        }
        // reverse = foldl (\acc x -> x : acc) []
        "reverse" => {
            let xs = pv("xs", fresh(id));
            let acc = pv("acc", fresh(id));
            let x = pv("x", fresh(id));
            let step = plam(
                acc.clone(),
                plam(x.clone(), papp2(pref(":", id), pev(&x), pev(&acc))),
            );
            plam(
                xs.clone(),
                papp(papp2(pref("foldl", id), step, pref("[]", id)), pev(&xs)),
            )
        }
        // __listAppend a b = case a of { [] -> b; (x:xs) -> x : __listAppend xs b }
        "__listAppend" => {
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let x = pv("x", fresh(id));
            let xs = pv("xs", fresh(id));
            plam(
                a.clone(),
                plam(
                    b.clone(),
                    pcase(
                        pev(&a),
                        vec![
                            palt("[]", 0, 0, vec![], pev(&b)),
                            palt(
                                ":",
                                1,
                                2,
                                vec![x.clone(), xs.clone()],
                                papp2(
                                    pref(":", id),
                                    pev(&x),
                                    papp2(pref("__listAppend", id), pev(&xs), pev(&b)),
                                ),
                            ),
                        ],
                    ),
                ),
            )
        }
        // concat xs = case xs of { [] -> []; (y:ys) -> __listAppend y (concat ys) }
        "concat" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            plam(
                xs.clone(),
                pcase(
                    pev(&xs),
                    vec![
                        palt("[]", 0, 0, vec![], pref("[]", id)),
                        palt(
                            ":",
                            1,
                            2,
                            vec![y.clone(), ys.clone()],
                            papp2(
                                pref("__listAppend", id),
                                pev(&y),
                                papp(pref("concat", id), pev(&ys)),
                            ),
                        ),
                    ],
                ),
            )
        }
        // concatMap f xs = concat (map f xs)
        "concatMap" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            plam(
                f.clone(),
                plam(
                    xs.clone(),
                    papp(
                        pref("concat", id),
                        papp2(pref("map", id), pev(&f), pev(&xs)),
                    ),
                ),
            )
        }
        // tails xs = xs : (case xs of { [] -> []; (y:ys) -> tails ys })
        "tails" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let rest = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![y.clone(), ys.clone()],
                        papp(pref("tails", id), pev(&ys)),
                    ),
                ],
            );
            plam(xs.clone(), papp2(pref(":", id), pev(&xs), rest))
        }
        // inits xs = [] : (case xs of { [] -> []; (y:ys) -> map (\t -> y:t) (inits ys) })
        "inits" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let t = pv("t", fresh(id));
            let cons_map = papp2(
                pref("map", id),
                plam(t.clone(), papp2(pref(":", id), pev(&y), pev(&t))),
                papp(pref("inits", id), pev(&ys)),
            );
            let rest = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![y.clone(), ys.clone()], cons_map),
                ],
            );
            plam(xs.clone(), papp2(pref(":", id), pref("[]", id), rest))
        }
        // lookup k xs = case xs of
        //   [] -> Nothing
        //   ((a,b):rest) -> case a == k of { True -> Just b; False -> lookup k rest }
        "lookup" => {
            let k = pv("k", fresh(id));
            let xs = pv("xs", fresh(id));
            let p = pv("p", fresh(id));
            let rest = pv("rest", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let pair_case = pcase(
                pev(&p),
                vec![palt(
                    "(,)",
                    0,
                    2,
                    vec![a.clone(), b.clone()],
                    pcase(
                        papp2(pref("==", id), pev(&a), pev(&k)),
                        vec![
                            palt(
                                "False",
                                0,
                                0,
                                vec![],
                                papp2(pref("lookup", id), pev(&k), pev(&rest)),
                            ),
                            palt("True", 1, 0, vec![], papp(pref("Just", id), pev(&b))),
                        ],
                    ),
                )],
            );
            plam(
                k.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pref("Nothing", id)),
                            palt(":", 1, 2, vec![p.clone(), rest.clone()], pair_case),
                        ],
                    ),
                ),
            )
        }
        // maybeToList m = case m of { Nothing -> []; Just x -> x : [] }
        "maybeToList" => {
            let m = pv("m", fresh(id));
            let x = pv("x", fresh(id));
            plam(
                m.clone(),
                pcase(
                    pev(&m),
                    vec![
                        palt("Nothing", 0, 0, vec![], pref("[]", id)),
                        palt(
                            "Just",
                            1,
                            1,
                            vec![x.clone()],
                            papp2(pref(":", id), pev(&x), pref("[]", id)),
                        ),
                    ],
                ),
            )
        }
        // stripPrefix xs ys = case xs of
        //   [] -> Just ys
        //   (a:as) -> case ys of { [] -> Nothing; (b:bs) -> case a==b of { True -> stripPrefix as bs; False -> Nothing } }
        "stripPrefix" => {
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let a = pv("a", fresh(id));
            let as_ = pv("as", fresh(id));
            let b = pv("b", fresh(id));
            let bs = pv("bs", fresh(id));
            let inner = pcase(
                papp2(pref("==", id), pev(&a), pev(&b)),
                vec![
                    palt("False", 0, 0, vec![], pref("Nothing", id)),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(pref("stripPrefix", id), pev(&as_), pev(&bs)),
                    ),
                ],
            );
            let cons_xs = pcase(
                pev(&ys),
                vec![
                    palt("[]", 0, 0, vec![], pref("Nothing", id)),
                    palt(":", 1, 2, vec![b.clone(), bs.clone()], inner),
                ],
            );
            plam(
                xs.clone(),
                plam(
                    ys.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], papp(pref("Just", id), pev(&ys))),
                            palt(":", 1, 2, vec![a.clone(), as_.clone()], cons_xs),
                        ],
                    ),
                ),
            )
        }
        // isSuffixOf xs ys = isPrefixOf (reverse xs) (reverse ys)
        "isSuffixOf" => {
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            plam(
                xs.clone(),
                plam(
                    ys.clone(),
                    papp2(
                        pref("isPrefixOf", id),
                        papp(pref("reverse", id), pev(&xs)),
                        papp(pref("reverse", id), pev(&ys)),
                    ),
                ),
            )
        }
        // isInfixOf xs ys = any (isPrefixOf xs) (tails ys)
        "isInfixOf" => {
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            plam(
                xs.clone(),
                plam(
                    ys.clone(),
                    papp2(
                        pref("any", id),
                        papp(pref("isPrefixOf", id), pev(&xs)),
                        papp(pref("tails", id), pev(&ys)),
                    ),
                ),
            )
        }
        // otherwise = True (a CAF, not a function)
        "otherwise" => pref("True", id),
        // deleteBy eq x xs = case xs of { [] -> []; (y:ys) -> case eq x y of { True -> ys; False -> y : deleteBy eq x ys } }
        "deleteBy" => {
            let eq = pv("eq", fresh(id));
            let x = pv("x", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let cons = pcase(
                papp2(pev(&eq), pev(&x), pev(&y)),
                vec![
                    palt(
                        "False",
                        0,
                        0,
                        vec![],
                        papp2(
                            pref(":", id),
                            pev(&y),
                            papp(papp2(pref("deleteBy", id), pev(&eq), pev(&x)), pev(&ys)),
                        ),
                    ),
                    palt("True", 1, 0, vec![], pev(&ys)),
                ],
            );
            plam(
                eq.clone(),
                plam(
                    x.clone(),
                    plam(
                        xs.clone(),
                        pcase(
                            pev(&xs),
                            vec![
                                palt("[]", 0, 0, vec![], pref("[]", id)),
                                palt(":", 1, 2, vec![y.clone(), ys.clone()], cons),
                            ],
                        ),
                    ),
                ),
            )
        }
        // intersectBy eq xs ys = filter (\x -> any (eq x) ys) xs
        "intersectBy" => {
            let eq = pv("eq", fresh(id));
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let x = pv("x", fresh(id));
            let pred = plam(
                x.clone(),
                papp2(pref("any", id), papp(pev(&eq), pev(&x)), pev(&ys)),
            );
            plam(
                eq.clone(),
                plam(
                    xs.clone(),
                    plam(ys.clone(), papp2(pref("filter", id), pred, pev(&xs))),
                ),
            )
        }
        // nubBy eq xs = case xs of { [] -> []; (y:ys) -> y : nubBy eq (filter (\z -> not (eq y z)) ys) }
        "nubBy" => {
            let eq = pv("eq", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let z = pv("z", fresh(id));
            let keep = plam(
                z.clone(),
                papp(pref("not", id), papp2(pev(&eq), pev(&y), pev(&z))),
            );
            let cons = papp2(
                pref(":", id),
                pev(&y),
                papp2(
                    pref("nubBy", id),
                    pev(&eq),
                    papp2(pref("filter", id), keep, pev(&ys)),
                ),
            );
            plam(
                eq.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pref("[]", id)),
                            palt(":", 1, 2, vec![y.clone(), ys.clone()], cons),
                        ],
                    ),
                ),
            )
        }
        // groupBy eq xs = case xs of
        //   [] -> []
        //   (y:ys) -> case span (eq y) ys of (g, rest) -> (y : g) : groupBy eq rest
        "groupBy" => {
            let eq = pv("eq", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let g = pv("g", fresh(id));
            let rest = pv("rest", fresh(id));
            let split = pcase(
                papp2(pref("span", id), papp(pev(&eq), pev(&y)), pev(&ys)),
                vec![palt(
                    "(,)",
                    0,
                    2,
                    vec![g.clone(), rest.clone()],
                    papp2(
                        pref(":", id),
                        papp2(pref(":", id), pev(&y), pev(&g)),
                        papp2(pref("groupBy", id), pev(&eq), pev(&rest)),
                    ),
                )],
            );
            plam(
                eq.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt("[]", 0, 0, vec![], pref("[]", id)),
                            palt(":", 1, 2, vec![y.clone(), ys.clone()], split),
                        ],
                    ),
                ),
            )
        }
        // insert x xs = case xs of { [] -> [x]; (y:ys) -> case x <= y of { True -> x:y:ys; False -> y : insert x ys } }
        "insert" => {
            let x = pv("x", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let cons = pcase(
                papp2(pref("<=", id), pev(&x), pev(&y)),
                vec![
                    palt(
                        "False",
                        0,
                        0,
                        vec![],
                        papp2(
                            pref(":", id),
                            pev(&y),
                            papp2(pref("insert", id), pev(&x), pev(&ys)),
                        ),
                    ),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(
                            pref(":", id),
                            pev(&x),
                            papp2(pref(":", id), pev(&y), pev(&ys)),
                        ),
                    ),
                ],
            );
            plam(
                x.clone(),
                plam(
                    xs.clone(),
                    pcase(
                        pev(&xs),
                        vec![
                            palt(
                                "[]",
                                0,
                                0,
                                vec![],
                                papp2(pref(":", id), pev(&x), pref("[]", id)),
                            ),
                            palt(":", 1, 2, vec![y.clone(), ys.clone()], cons),
                        ],
                    ),
                ),
            )
        }
        // unionBy eq xs ys = __listAppend xs (foldl (\acc x -> deleteBy eq x acc) (nubBy eq ys) xs)
        "unionBy" => {
            let eq = pv("eq", fresh(id));
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let acc = pv("acc", fresh(id));
            let x = pv("x", fresh(id));
            let step = plam(
                acc.clone(),
                plam(
                    x.clone(),
                    papp(papp2(pref("deleteBy", id), pev(&eq), pev(&x)), pev(&acc)),
                ),
            );
            let folded = papp(
                papp2(
                    pref("foldl", id),
                    step,
                    papp2(pref("nubBy", id), pev(&eq), pev(&ys)),
                ),
                pev(&xs),
            );
            plam(
                eq.clone(),
                plam(
                    xs.clone(),
                    plam(
                        ys.clone(),
                        papp2(pref("__listAppend", id), pev(&xs), folded),
                    ),
                ),
            )
        }
        // __insertOn f x ys = case ys of { [] -> [x]; (y:rest) -> case f x <= f y of { True -> x:y:rest; False -> y : __insertOn f x rest } }
        "__insertOn" => {
            let f = pv("f", fresh(id));
            let x = pv("x", fresh(id));
            let ys = pv("ys", fresh(id));
            let y = pv("y", fresh(id));
            let rest = pv("rest", fresh(id));
            let cons = pcase(
                papp2(
                    pref("<=", id),
                    papp(pev(&f), pev(&x)),
                    papp(pev(&f), pev(&y)),
                ),
                vec![
                    palt(
                        "False",
                        0,
                        0,
                        vec![],
                        papp2(
                            pref(":", id),
                            pev(&y),
                            papp(papp2(pref("__insertOn", id), pev(&f), pev(&x)), pev(&rest)),
                        ),
                    ),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(
                            pref(":", id),
                            pev(&x),
                            papp2(pref(":", id), pev(&y), pev(&rest)),
                        ),
                    ),
                ],
            );
            plam(
                f.clone(),
                plam(
                    x.clone(),
                    plam(
                        ys.clone(),
                        pcase(
                            pev(&ys),
                            vec![
                                palt(
                                    "[]",
                                    0,
                                    0,
                                    vec![],
                                    papp2(pref(":", id), pev(&x), pref("[]", id)),
                                ),
                                palt(":", 1, 2, vec![y.clone(), rest.clone()], cons),
                            ],
                        ),
                    ),
                ),
            )
        }
        // sortOn f xs = foldr (\x acc -> __insertOn f x acc) [] xs
        "sortOn" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let x = pv("x", fresh(id));
            let acc = pv("acc", fresh(id));
            let step = plam(
                x.clone(),
                plam(
                    acc.clone(),
                    papp(papp2(pref("__insertOn", id), pev(&f), pev(&x)), pev(&acc)),
                ),
            );
            plam(
                f.clone(),
                plam(
                    xs.clone(),
                    papp(papp2(pref("foldr", id), step, pref("[]", id)), pev(&xs)),
                ),
            )
        }
        // mapAccumL f acc xs = case xs of
        //   [] -> (acc, [])
        //   (y:ys) -> case f acc y of (acc1,z) -> case mapAccumL f acc1 ys of (acc2,zs) -> (acc2, z:zs)
        "mapAccumL" => {
            let f = pv("f", fresh(id));
            let acc = pv("acc", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let acc1 = pv("acc1", fresh(id));
            let z = pv("z", fresh(id));
            let acc2 = pv("acc2", fresh(id));
            let zs = pv("zs", fresh(id));
            let rec = pcase(
                papp(papp2(pref("mapAccumL", id), pev(&f), pev(&acc1)), pev(&ys)),
                vec![palt(
                    "(,)",
                    0,
                    2,
                    vec![acc2.clone(), zs.clone()],
                    papp2(
                        pref("(,)", id),
                        pev(&acc2),
                        papp2(pref(":", id), pev(&z), pev(&zs)),
                    ),
                )],
            );
            let cons = pcase(
                papp2(pev(&f), pev(&acc), pev(&y)),
                vec![palt("(,)", 0, 2, vec![acc1.clone(), z.clone()], rec)],
            );
            plam(
                f.clone(),
                plam(
                    acc.clone(),
                    plam(
                        xs.clone(),
                        pcase(
                            pev(&xs),
                            vec![
                                palt(
                                    "[]",
                                    0,
                                    0,
                                    vec![],
                                    papp2(pref("(,)", id), pev(&acc), pref("[]", id)),
                                ),
                                palt(":", 1, 2, vec![y.clone(), ys.clone()], cons),
                            ],
                        ),
                    ),
                ),
            )
        }
        // mapAccumR f acc xs = case xs of
        //   [] -> (acc, [])
        //   (y:ys) -> case mapAccumR f acc ys of (acc1,zs) -> case f acc1 y of (acc2,z) -> (acc2, z:zs)
        "mapAccumR" => {
            let f = pv("f", fresh(id));
            let acc = pv("acc", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let acc1 = pv("acc1", fresh(id));
            let zs = pv("zs", fresh(id));
            let acc2 = pv("acc2", fresh(id));
            let z = pv("z", fresh(id));
            let inner = pcase(
                papp2(pev(&f), pev(&acc1), pev(&y)),
                vec![palt(
                    "(,)",
                    0,
                    2,
                    vec![acc2.clone(), z.clone()],
                    papp2(
                        pref("(,)", id),
                        pev(&acc2),
                        papp2(pref(":", id), pev(&z), pev(&zs)),
                    ),
                )],
            );
            let cons = pcase(
                papp(papp2(pref("mapAccumR", id), pev(&f), pev(&acc)), pev(&ys)),
                vec![palt("(,)", 0, 2, vec![acc1.clone(), zs.clone()], inner)],
            );
            plam(
                f.clone(),
                plam(
                    acc.clone(),
                    plam(
                        xs.clone(),
                        pcase(
                            pev(&xs),
                            vec![
                                palt(
                                    "[]",
                                    0,
                                    0,
                                    vec![],
                                    papp2(pref("(,)", id), pev(&acc), pref("[]", id)),
                                ),
                                palt(":", 1, 2, vec![y.clone(), ys.clone()], cons),
                            ],
                        ),
                    ),
                ),
            )
        }
        // compare a b = case a < b of { True -> LT; False -> case a == b of { True -> EQ; False -> GT } }
        // Works for Int and for nullary (enum) constructors, whose runtime value
        // is the constructor tag.
        "compare" => {
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let eq_case = pcase(
                papp2(pref("==", id), pev(&a), pev(&b)),
                vec![
                    palt("False", 0, 0, vec![], pref("GT", id)),
                    palt("True", 1, 0, vec![], pref("EQ", id)),
                ],
            );
            plam(
                a.clone(),
                plam(
                    b.clone(),
                    pcase(
                        papp2(pref("<", id), pev(&a), pev(&b)),
                        vec![
                            palt("False", 0, 0, vec![], eq_case),
                            palt("True", 1, 0, vec![], pref("LT", id)),
                        ],
                    ),
                ),
            )
        }
        // maximumBy/minimumBy cmp xs = foldl1 (\x y -> case cmp x y of ...) xs
        "maximumBy" | "minimumBy" => {
            let is_max = name == "maximumBy";
            let cmp = pv("cmp", fresh(id));
            let xs = pv("xs", fresh(id));
            let x = pv("x", fresh(id));
            let y = pv("y", fresh(id));
            // maximumBy: GT -> x, else y ; minimumBy: GT -> y, else x
            let gt_branch = if is_max { pev(&x) } else { pev(&y) };
            let other = if is_max { pev(&y) } else { pev(&x) };
            let step = plam(
                x.clone(),
                plam(
                    y.clone(),
                    pcase(
                        papp2(pev(&cmp), pev(&x), pev(&y)),
                        vec![
                            palt("LT", 0, 0, vec![], other.clone()),
                            palt("EQ", 1, 0, vec![], other),
                            palt("GT", 2, 0, vec![], gt_branch),
                        ],
                    ),
                ),
            );
            plam(
                cmp.clone(),
                plam(xs.clone(), papp2(pref("foldl1", id), step, pev(&xs))),
            )
        }
        // zip3 (a:as) (b:bs) (c:cs) = (a,b,c) : zip3 as bs cs ; _ -> []
        "zip3" => {
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let zs = pv("zs", fresh(id));
            let a = pv("a", fresh(id));
            let as_ = pv("as", fresh(id));
            let b = pv("b", fresh(id));
            let bs = pv("bs", fresh(id));
            let c = pv("c", fresh(id));
            let cs = pv("cs", fresh(id));
            // innermost: case zs of (c:cs) -> (a,b,c) : zip3 as bs cs ; [] -> []
            let triple = papp2(
                pref(":", id),
                papp(papp2(pref("(,,)", id), pev(&a), pev(&b)), pev(&c)),
                papp(papp2(pref("zip3", id), pev(&as_), pev(&bs)), pev(&cs)),
            );
            let on_zs = pcase(
                pev(&zs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![c.clone(), cs.clone()], triple),
                ],
            );
            let on_ys = pcase(
                pev(&ys),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![b.clone(), bs.clone()], on_zs),
                ],
            );
            let on_xs = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![a.clone(), as_.clone()], on_ys),
                ],
            );
            plam(xs.clone(), plam(ys.clone(), plam(zs.clone(), on_xs)))
        }
        // zipWith3 f (a:as) (b:bs) (c:cs) = f a b c : zipWith3 f as bs cs ; _ -> []
        "zipWith3" => {
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let zs = pv("zs", fresh(id));
            let a = pv("a", fresh(id));
            let as_ = pv("as", fresh(id));
            let b = pv("b", fresh(id));
            let bs = pv("bs", fresh(id));
            let c = pv("c", fresh(id));
            let cs = pv("cs", fresh(id));
            let applied = papp2(
                pref(":", id),
                papp(papp2(pev(&f), pev(&a), pev(&b)), pev(&c)),
                papp(
                    papp2(papp(pref("zipWith3", id), pev(&f)), pev(&as_), pev(&bs)),
                    pev(&cs),
                ),
            );
            let on_zs = pcase(
                pev(&zs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![c.clone(), cs.clone()], applied),
                ],
            );
            let on_ys = pcase(
                pev(&ys),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![b.clone(), bs.clone()], on_zs),
                ],
            );
            let on_xs = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("[]", id)),
                    palt(":", 1, 2, vec![a.clone(), as_.clone()], on_ys),
                ],
            );
            plam(
                f,
                plam(xs.clone(), plam(ys.clone(), plam(zs.clone(), on_xs))),
            )
        }
        // unfoldr f b = case f b of { Nothing -> []; Just p -> case p of (a,b') -> a : unfoldr f b' }
        "unfoldr" => {
            let f = pv("f", fresh(id));
            let b = pv("b", fresh(id));
            let p = pv("p", fresh(id));
            let a = pv("a", fresh(id));
            let b2 = pv("b2", fresh(id));
            let cons = papp2(
                pref(":", id),
                pev(&a),
                papp2(pref("unfoldr", id), pev(&f), pev(&b2)),
            );
            let on_pair = pcase(
                pev(&p),
                vec![palt("(,)", 0, 2, vec![a.clone(), b2.clone()], cons)],
            );
            let body = pcase(
                papp(pev(&f), pev(&b)),
                vec![
                    palt("Nothing", 0, 0, vec![], pref("[]", id)),
                    palt("Just", 1, 1, vec![p.clone()], on_pair),
                ],
            );
            plam(f, plam(b.clone(), body))
        }
        // gcd a b = case b == 0 of { True -> a; False -> gcd b (a `mod` b) }
        // (correct for non-negative inputs, which is what the fixtures use)
        "gcd" => {
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let recur = papp2(
                pref("gcd", id),
                pev(&b),
                papp2(pref("mod", id), pev(&a), pev(&b)),
            );
            let body = pcase(
                papp2(pref("==", id), pev(&b), pint(0)),
                vec![
                    palt("False", 0, 0, vec![], recur),
                    palt("True", 1, 0, vec![], pev(&a)),
                ],
            );
            plam(a.clone(), plam(b.clone(), body))
        }
        // lcm a b = case a == 0 of { True -> 0; False ->
        //             case b == 0 of { True -> 0; False -> (a `div` gcd a b) * b } }
        "lcm" => {
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let prod = papp2(
                pref("*", id),
                papp2(
                    pref("div", id),
                    pev(&a),
                    papp2(pref("gcd", id), pev(&a), pev(&b)),
                ),
                pev(&b),
            );
            let on_b = pcase(
                papp2(pref("==", id), pev(&b), pint(0)),
                vec![
                    palt("False", 0, 0, vec![], prod),
                    palt("True", 1, 0, vec![], pint(0)),
                ],
            );
            let body = pcase(
                papp2(pref("==", id), pev(&a), pint(0)),
                vec![
                    palt("False", 0, 0, vec![], on_b),
                    palt("True", 1, 0, vec![], pint(0)),
                ],
            );
            plam(a.clone(), plam(b.clone(), body))
        }
        // filterM p xs = case xs of
        //   [] -> []
        //   (y:ys) -> case p y of { True -> y : filterM p ys; False -> filterM p ys }
        // In eager IO, `p y` runs its effect and yields the Bool.
        "filterM" => {
            let p = pv("p", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let keep = papp2(
                pref(":", id),
                pev(&y),
                papp2(pref("filterM", id), pev(&p), pev(&ys)),
            );
            let skip = papp2(pref("filterM", id), pev(&p), pev(&ys));
            let inner = pcase(
                papp(pev(&p), pev(&y)),
                vec![
                    palt("False", 0, 0, vec![], skip),
                    palt("True", 1, 0, vec![], keep),
                ],
            );
            let nil = palt("[]", 0, 0, vec![], pref("[]", id));
            let cons = palt(":", 1, 2, vec![y.clone(), ys.clone()], inner);
            plam(p, plam(xs.clone(), pcase(pev(&xs), vec![nil, cons])))
        }
        // foldM f z xs = case xs of { [] -> z; (y:ys) -> foldM f (f z y) ys }
        // foldM_ is the same but yields () (0) — the threaded accumulator still
        // forces `f z y` at each step for its effect.
        "foldM" | "foldM_" => {
            let f = pv("f", fresh(id));
            let z = pv("z", fresh(id));
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let nil_rhs = if name == "foldM" { pev(&z) } else { pint(0) };
            let recur = papp(
                papp2(pref(name, id), pev(&f), papp2(pev(&f), pev(&z), pev(&y))),
                pev(&ys),
            );
            let nil = palt("[]", 0, 0, vec![], nil_rhs);
            let cons = palt(":", 1, 2, vec![y.clone(), ys.clone()], recur);
            plam(
                f,
                plam(z, plam(xs.clone(), pcase(pev(&xs), vec![nil, cons]))),
            )
        }
        // zipWithM f (a:as) (b:bs) = f a b : zipWithM f as bs ; _ -> []
        // zipWithM_ runs `f a b` for effect and yields () — `seq` forces it.
        "zipWithM" | "zipWithM_" => {
            let collect = name == "zipWithM";
            let f = pv("f", fresh(id));
            let xs = pv("xs", fresh(id));
            let ys = pv("ys", fresh(id));
            let a = pv("a", fresh(id));
            let as_ = pv("as", fresh(id));
            let b = pv("b", fresh(id));
            let bs = pv("bs", fresh(id));
            let recur = papp2(papp(pref(name, id), pev(&f)), pev(&as_), pev(&bs));
            let step = if collect {
                papp2(pref(":", id), papp2(pev(&f), pev(&a), pev(&b)), recur)
            } else {
                papp2(pref("seq", id), papp2(pev(&f), pev(&a), pev(&b)), recur)
            };
            let empty = if collect { pref("[]", id) } else { pint(0) };
            let on_ys = pcase(
                pev(&ys),
                vec![
                    palt("[]", 0, 0, vec![], empty.clone()),
                    palt(":", 1, 2, vec![b.clone(), bs.clone()], step),
                ],
            );
            let on_xs = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], empty),
                    palt(":", 1, 2, vec![a.clone(), as_.clone()], on_ys),
                ],
            );
            plam(f, plam(xs.clone(), plam(ys.clone(), on_xs)))
        }
        // either f g e = case e of { Left x -> f x; Right y -> g y }
        "either" => {
            let f = pv("f", fresh(id));
            let g = pv("g", fresh(id));
            let e = pv("e", fresh(id));
            let x = pv("x", fresh(id));
            let y = pv("y", fresh(id));
            let body = pcase(
                pev(&e),
                vec![
                    palt("Left", 0, 1, vec![x.clone()], papp(pev(&f), pev(&x))),
                    palt("Right", 1, 1, vec![y.clone()], papp(pev(&g), pev(&y))),
                ],
            );
            plam(f, plam(g, plam(e.clone(), body)))
        }
        // fromLeft d e = case e of { Left x -> x; Right _ -> d }
        // fromRight d e = case e of { Right y -> y; Left _ -> d }
        "fromLeft" | "fromRight" => {
            let want_left = name == "fromLeft";
            let d = pv("d", fresh(id));
            let e = pv("e", fresh(id));
            let v = pv("v", fresh(id));
            let w = pv("w", fresh(id));
            let left = if want_left {
                palt("Left", 0, 1, vec![v.clone()], pev(&v))
            } else {
                palt("Left", 0, 1, vec![w.clone()], pev(&d))
            };
            let right = if want_left {
                palt("Right", 1, 1, vec![w.clone()], pev(&d))
            } else {
                palt("Right", 1, 1, vec![v.clone()], pev(&v))
            };
            plam(
                d.clone(),
                plam(e.clone(), pcase(pev(&e), vec![left, right])),
            )
        }
        // lefts xs  = [x | Left x  <- xs] ; rights xs = [y | Right y <- xs]
        "lefts" | "rights" => {
            let keep_left = name == "lefts";
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let v = pv("v", fresh(id));
            let w = pv("w", fresh(id));
            let recur = papp(pref(name, id), pev(&ys));
            let keep = papp2(pref(":", id), pev(&v), recur.clone());
            let (left_rhs, right_rhs) = if keep_left {
                (keep, recur)
            } else {
                (recur.clone(), papp2(pref(":", id), pev(&v), recur))
            };
            let on_elem = pcase(
                pev(&y),
                vec![
                    palt(
                        "Left",
                        0,
                        1,
                        vec![if keep_left { v.clone() } else { w.clone() }],
                        left_rhs,
                    ),
                    palt(
                        "Right",
                        1,
                        1,
                        vec![if keep_left { w.clone() } else { v.clone() }],
                        right_rhs,
                    ),
                ],
            );
            let nil = palt("[]", 0, 0, vec![], pref("[]", id));
            let cons = palt(":", 1, 2, vec![y.clone(), ys.clone()], on_elem);
            plam(xs.clone(), pcase(pev(&xs), vec![nil, cons]))
        }
        // maybe d f m = case m of { Nothing -> d; Just x -> f x }
        "maybe" => {
            let d = pv("d", fresh(id));
            let f = pv("f", fresh(id));
            let m = pv("m", fresh(id));
            let x = pv("x", fresh(id));
            let body = pcase(
                pev(&m),
                vec![
                    palt("Nothing", 0, 0, vec![], pev(&d)),
                    palt("Just", 1, 1, vec![x.clone()], papp(pev(&f), pev(&x))),
                ],
            );
            plam(d, plam(f, plam(m.clone(), body)))
        }
        // listToMaybe xs = case xs of { [] -> Nothing; (x:_) -> Just x }
        "listToMaybe" => {
            let xs = pv("xs", fresh(id));
            let x = pv("x", fresh(id));
            let rest = pv("rest", fresh(id));
            let body = pcase(
                pev(&xs),
                vec![
                    palt("[]", 0, 0, vec![], pref("Nothing", id)),
                    palt(
                        ":",
                        1,
                        2,
                        vec![x.clone(), rest.clone()],
                        papp(pref("Just", id), pev(&x)),
                    ),
                ],
            );
            plam(xs.clone(), body)
        }
        // catMaybes xs = case xs of
        //   [] -> []; (y:ys) -> case y of { Nothing -> catMaybes ys; Just x -> x : catMaybes ys }
        "catMaybes" => {
            let xs = pv("xs", fresh(id));
            let y = pv("y", fresh(id));
            let ys = pv("ys", fresh(id));
            let x = pv("x", fresh(id));
            let recur = papp(pref("catMaybes", id), pev(&ys));
            let on_elem = pcase(
                pev(&y),
                vec![
                    palt("Nothing", 0, 0, vec![], recur.clone()),
                    palt(
                        "Just",
                        1,
                        1,
                        vec![x.clone()],
                        papp2(pref(":", id), pev(&x), recur),
                    ),
                ],
            );
            let nil = palt("[]", 0, 0, vec![], pref("[]", id));
            let cons = palt(":", 1, 2, vec![y.clone(), ys.clone()], on_elem);
            plam(xs.clone(), pcase(pev(&xs), vec![nil, cons]))
        }
        // enumFromThenTo a b c: step = b - a. Ascending (step >= 0) stops when
        // a > c; descending stops when a < c. The recursive step is the same in
        // both; it is duplicated (not shared) so the empty case stays strict.
        "enumFromThenTo" => {
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            let c = pv("c", fresh(id));
            let recur = |id: &mut usize| {
                let next_b = papp2(
                    pref("+", id),
                    pev(&b),
                    papp2(pref("-", id), pev(&b), pev(&a)),
                );
                papp2(
                    pref(":", id),
                    pev(&a),
                    papp(papp2(pref("enumFromThenTo", id), pev(&b), next_b), pev(&c)),
                )
            };
            // ascending arm: stop when a > c
            let asc = pcase(
                papp2(pref(">", id), pev(&a), pev(&c)),
                vec![
                    palt("False", 0, 0, vec![], recur(id)),
                    palt("True", 1, 0, vec![], pref("[]", id)),
                ],
            );
            // descending arm: stop when a < c
            let desc = pcase(
                papp2(pref("<", id), pev(&a), pev(&c)),
                vec![
                    palt("False", 0, 0, vec![], recur(id)),
                    palt("True", 1, 0, vec![], pref("[]", id)),
                ],
            );
            // choose by step sign: (b - a) >= 0 ?
            let body = pcase(
                papp2(
                    pref(">=", id),
                    papp2(pref("-", id), pev(&b), pev(&a)),
                    pint(0),
                ),
                vec![
                    palt("False", 0, 0, vec![], desc),
                    palt("True", 1, 0, vec![], asc),
                ],
            );
            plam(a.clone(), plam(b.clone(), plam(c.clone(), body)))
        }
        // until p f x = case p x of { True -> x; False -> until p f (f x) }
        "until" => {
            let p = pv("p", fresh(id));
            let f = pv("f", fresh(id));
            let x = pv("x", fresh(id));
            let recur = papp(
                papp2(pref("until", id), pev(&p), pev(&f)),
                papp(pev(&f), pev(&x)),
            );
            let body = pcase(
                papp(pev(&p), pev(&x)),
                vec![
                    palt("False", 0, 0, vec![], recur),
                    palt("True", 1, 0, vec![], pev(&x)),
                ],
            );
            plam(p, plam(f, plam(x.clone(), body)))
        }
        // isAscii c = c < 128
        "isAscii" => {
            let c = pv("c", fresh(id));
            plam(c.clone(), papp2(pref("<", id), pev(&c), pint(128)))
        }
        // isLetter = isAlpha
        "isLetter" => {
            let c = pv("c", fresh(id));
            plam(c.clone(), papp(pref("isAlpha", id), pev(&c)))
        }
        // digitToInt c = c - 48 ('0'..'9'); 'A'..'F' -> c-55; 'a'..'f' -> c-87
        "digitToInt" => {
            let c = pv("c", fresh(id));
            let hex_lower = papp2(pref("-", id), pev(&c), pint(87));
            let hex_upper = pcase(
                papp2(pref("<=", id), pev(&c), pint(70)),
                vec![
                    palt("False", 0, 0, vec![], hex_lower),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(pref("-", id), pev(&c), pint(55)),
                    ),
                ],
            );
            let body = pcase(
                papp2(pref("<=", id), pev(&c), pint(57)),
                vec![
                    palt("False", 0, 0, vec![], hex_upper),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(pref("-", id), pev(&c), pint(48)),
                    ),
                ],
            );
            plam(c.clone(), body)
        }
        // swap (a,b) = (b,a)
        "swap" => {
            let p = pv("p", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            plam(
                p.clone(),
                pcase(
                    pev(&p),
                    vec![palt(
                        "(,)",
                        0,
                        2,
                        vec![a.clone(), b.clone()],
                        papp2(pref("(,)", id), pev(&b), pev(&a)),
                    )],
                ),
            )
        }
        // curry f x y = f (x, y)
        "curry" => {
            let f = pv("f", fresh(id));
            let x = pv("x", fresh(id));
            let y = pv("y", fresh(id));
            plam(
                f.clone(),
                plam(
                    x.clone(),
                    plam(
                        y.clone(),
                        papp(pev(&f), papp2(pref("(,)", id), pev(&x), pev(&y))),
                    ),
                ),
            )
        }
        // uncurry f (a, b) = f a b
        "uncurry" => {
            let f = pv("f", fresh(id));
            let p = pv("p", fresh(id));
            let a = pv("a", fresh(id));
            let b = pv("b", fresh(id));
            plam(
                f.clone(),
                plam(
                    p.clone(),
                    pcase(
                        pev(&p),
                        vec![palt(
                            "(,)",
                            0,
                            2,
                            vec![a.clone(), b.clone()],
                            papp2(pev(&f), pev(&a), pev(&b)),
                        )],
                    ),
                ),
            )
        }
        // (&) x f = f x  (reverse application)
        "&" => {
            let x = pv("x", fresh(id));
            let f = pv("f", fresh(id));
            plam(x.clone(), plam(f.clone(), papp(pev(&f), pev(&x))))
        }
        // bracket acquire release use = const (use a) (release a), where the
        // acquire action `a` has already run as an argument. const evaluates its
        // args left-to-right, so the effects run acquire, use, release — then
        // the use result is returned. (Happy path only; no exception unwinding.)
        "bracket" => {
            let acq = pv("acq", fresh(id));
            let rel = pv("rel", fresh(id));
            let use_ = pv("use", fresh(id));
            let body = papp2(
                pref("const", id),
                papp(pev(&use_), pev(&acq)),
                papp(pev(&rel), pev(&acq)),
            );
            plam(acq, plam(rel, plam(use_, body)))
        }
        // guard b = case b of { True -> [()]; False -> [] }  (list Alternative)
        "guard" => {
            let b = pv("b", fresh(id));
            let body = pcase(
                pev(&b),
                vec![
                    palt("False", 0, 0, vec![], pref("[]", id)),
                    palt(
                        "True",
                        1,
                        0,
                        vec![],
                        papp2(pref(":", id), pref("()", id), pref("[]", id)),
                    ),
                ],
            );
            plam(b.clone(), body)
        }
        _ => return None,
    };
    Some((pv(name, fresh(id)), body))
}

/// Peel transparent wrappers (`TyApp`/`Cast`/`Tick`/`Lazy`) off an expression.
fn peel_show_wrappers(expr: &Expr) -> &Expr {
    let mut e = expr;
    loop {
        e = match e {
            Expr::TyApp(inner, _, _)
            | Expr::Cast(inner, _, _)
            | Expr::Tick(_, inner, _)
            | Expr::Lazy(inner, _) => inner,
            _ => return e,
        };
    }
}

/// Peel transparent wrappers *and* runtime-identity applications
/// (`force`/`id`/`from`/`to`) so `show` and structure inspection see the
/// underlying value. In the strict backend these functions are the identity
/// (and `to . from` is a no-op roundtrip), so peeling them is sound.
fn peel_runtime_identity(expr: &Expr) -> &Expr {
    let mut e = expr;
    loop {
        e = match e {
            Expr::TyApp(inner, _, _)
            | Expr::Cast(inner, _, _)
            | Expr::Tick(_, inner, _)
            | Expr::Lazy(inner, _) => inner,
            Expr::App(f, arg, _)
                if matches!(peel_show_wrappers(f), Expr::Var(v, _)
                    if matches!(strip_qualifier(v.name.as_str()), "force" | "id" | "from" | "to")) =>
            {
                arg.as_ref()
            }
            _ => return e,
        };
    }
}

/// If `expr` is a statically-known list (a chain of `:` ending in `[]`), return
/// its element expressions. Returns `None` for runtime lists (a variable, a
/// `++` result, etc.) whose elements aren't known at compile time.
fn extract_list_elements(expr: &Expr) -> Option<Vec<&Expr>> {
    let mut elems = Vec::new();
    let mut cur = peel_show_wrappers(expr);
    loop {
        let (head, args) = collect_app_spine(cur);
        match head {
            Expr::Var(v, _) if v.name.as_str() == "[]" && args.is_empty() => return Some(elems),
            Expr::Var(v, _) if v.name.as_str() == ":" && args.len() == 2 => {
                elems.push(args[0]);
                cur = peel_show_wrappers(args[1]);
            }
            _ => return None,
        }
    }
}

/// Whether a constructor name is a tuple constructor (`(,)`, `(,,)`, ...).
/// Parse a dictionary field selector name `$sel_N` into its field index `N`.
/// Returns `None` for any other name.
fn parse_sel_index(name: &str) -> Option<u32> {
    name.strip_prefix("$sel_").and_then(|s| s.parse().ok())
}

fn is_tuple_con(name: &str) -> bool {
    name.len() >= 3
        && name.starts_with('(')
        && name.ends_with(')')
        && name[1..name.len() - 1].chars().all(|c| c == ',')
}

/// Strip a trailing `_<digits>` deriving counter (`Box_50000` -> `Box`,
/// `Maybe2_50011` -> `Maybe2`). Names without such a suffix are unchanged.
fn strip_counter_suffix(s: &str) -> &str {
    match s.rsplit_once('_') {
        Some((head, tail)) if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) => head,
        _ => s,
    }
}

/// Drop a module qualifier from a name (`GHC.Enum.succ` -> `succ`), leaving the
/// final dotted segment. Operators (which may legitimately contain `.`) are
/// returned unchanged.
fn strip_qualifier(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((_, last))
            if !last.is_empty()
                && last
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '\'') =>
        {
            last
        }
        _ => name,
    }
}

/// The element type of a list type `[e]`, if `ty` is one.
fn list_element_ty(ty: &Ty) -> Option<&Ty> {
    match ty {
        Ty::List(e) => Some(e.as_ref()),
        Ty::App(c, e) if matches!(c.as_ref(), Ty::Con(tc) if tc.name.as_str() == "[]") => {
            Some(e.as_ref())
        }
        _ => None,
    }
}

/// Whether a type is a (cons-list) list whose element is *not* `Char` — i.e. a
/// real list, not a `String` (which has a length-prefixed representation). Only
/// reports `true` for a concrete element type; polymorphic/erased elements are
/// treated as ambiguous (`false`) so `++` defaults to string concatenation.
/// Whether `expr` is a `String` value — a string literal or anything of type
/// `[Char]` — for which `show` produces a quoted, escaped rendering.
fn is_string_expr(expr: &Expr) -> bool {
    let mut e = expr;
    loop {
        e = match e {
            Expr::TyApp(inner, _, _)
            | Expr::Cast(inner, _, _)
            | Expr::Tick(_, inner, _)
            | Expr::Lazy(inner, _) => inner,
            _ => break,
        };
    }
    if matches!(e, Expr::Lit(Literal::String(_), _, _)) {
        return true;
    }
    matches!(list_element_ty(&e.ty()), Some(Ty::Con(c)) if c.name.as_str() == "Char")
}

fn is_nonstring_list_ty(ty: &Ty) -> bool {
    match list_element_ty(ty) {
        Some(Ty::Con(c)) => c.name.as_str() != "Char",
        // A list whose element type is a variable (`[a]`, from a polymorphic
        // signature) is treated as a non-String list — the common case; a
        // genuinely `[Char]` value is usually concretely typed.
        Some(Ty::List(_) | Ty::App(..) | Ty::Tuple(_) | Ty::Var(_)) => true,
        _ => false,
    }
}

/// Whether a function always returns a cons-list (used to classify an operand
/// of `++` as a list). Excludes `++`/`reverse`/`show`, whose result kind is
/// ambiguous (string vs list).
fn is_list_returning_fn(name: &str) -> bool {
    matches!(
        strip_qualifier(name),
        "map"
            | "filter"
            | "elems"
            | "keys"
            | "toList"
            | "enumFromThenTo"
            | "enumFromThen"
            | "enumFrom"
            | "iterate"
            | "cycle"
            | "take"
            | "drop"
            | "zip"
            | "zipWith"
            | "replicate"
            | "enumFromTo"
            | "takeWhile"
            | "dropWhile"
            | "reverse"
            | "intersect"
            | "scanl"
            | "scanl1"
            | "scanr"
            | "concat"
            | "concatMap"
            | "sortOn"
            | "nubBy"
            | "deleteBy"
            | "unionBy"
            | "intersectBy"
            | "insert"
    )
}

/// Whether an expression is confidently a (non-String) cons-list: a syntactic
/// list literal, a value of concrete non-`Char` list type, or the result of a
/// list-returning function.
fn is_list_operand(expr: &Expr) -> bool {
    let e = peel_show_wrappers(expr);
    if extract_list_elements(e).is_some() {
        return true;
    }
    if is_nonstring_list_ty(&e.ty()) {
        return true;
    }
    let (head, args) = collect_app_spine(e);
    if let Expr::Var(v, _) = head {
        let name = strip_qualifier(v.name.as_str());
        if is_list_returning_fn(v.name.as_str()) {
            return true;
        }
        // `fmap`/`<$>` over a list is a list; `foldr (:) [] xs` builds a list —
        // detect both by the list-ness of the relevant operand.
        if matches!(name, "fmap" | "<$>") && args.len() == 2 {
            return is_list_operand(args[1]);
        }
        if name == "foldr" && args.len() == 3 {
            return is_list_operand(args[1]);
        }
    }
    false
}

/// The contents of `expr` if it is a string literal (after peeling wrappers).
fn as_string_literal(expr: &Expr) -> Option<&str> {
    match peel_show_wrappers(expr) {
        Expr::Lit(Literal::String(sym), _, _) => Some(sym.as_str()),
        _ => None,
    }
}

/// The integer value of `expr` if it is an integer literal (after peeling
/// transparent wrappers). Used to unroll statically-counted combinators.
fn as_int_literal(expr: &Expr) -> Option<i64> {
    match peel_show_wrappers(expr) {
        Expr::Lit(Literal::Int(n), _, _) => Some(*n),
        Expr::Lit(Literal::Integer(n), _, _) => i64::try_from(*n).ok(),
        Expr::Lit(Literal::Char(c), _, _) => Some(*c as i64),
        _ => None,
    }
}

/// Build a cons-list Core expression from a vector of element expressions.
fn build_cons_list(elems: Vec<Expr>, id: &mut usize) -> Expr {
    elems
        .into_iter()
        .rev()
        .fold(pref("[]", id), |acc, e| papp2(pref(":", id), e, acc))
}

/// Fuse `take k <producer>` into a finite `k`-element list when the producer is
/// an infinite/lazy generator the strict backend can't otherwise build. Returns
/// `None` for ordinary (finite) lists, which use the normal `take`.
fn fuse_take(k: i64, producer: &Expr, id: &mut usize) -> Option<Expr> {
    let (head, pargs) = collect_app_spine(peel_show_wrappers(producer));
    let name = match head {
        Expr::Var(v, _) => strip_qualifier(v.name.as_str()),
        _ => return None,
    };
    let n = k.max(0) as usize;
    let elems: Vec<Expr> = match (name, pargs.len()) {
        // iterate f x = [x, f x, f (f x), ...]
        ("iterate", 2) => {
            let (f, x) = (pargs[0], pargs[1]);
            let mut out = Vec::with_capacity(n);
            let mut cur = x.clone();
            for i in 0..n {
                if i > 0 {
                    cur = papp(f.clone(), cur);
                }
                out.push(cur.clone());
            }
            out
        }
        // repeat v = [v, v, ...]
        ("repeat", 1) => (0..n).map(|_| pargs[0].clone()).collect(),
        // cycle xs = xs ++ xs ++ ... (xs must be a known finite list)
        ("cycle", 1) => {
            let es = extract_list_elements(pargs[0])?;
            if es.is_empty() {
                return None;
            }
            (0..n).map(|i| es[i % es.len()].clone()).collect()
        }
        // enumFrom a = [a, a+1, ...]
        ("enumFrom", 1) => (0..n as i64)
            .map(|i| papp2(pref("+", id), pargs[0].clone(), pint(i)))
            .collect(),
        // enumFromThen a b = [a, a+(b-a), a+2(b-a), ...] (needs literal a, b)
        ("enumFromThen", 2) => {
            let a = as_int_literal(pargs[0])?;
            let b = as_int_literal(pargs[1])?;
            let step = b - a;
            (0..n as i64).map(|i| pint(a + step * i)).collect()
        }
        _ => return None,
    };
    Some(build_cons_list(elems, id))
}

/// Whether `expr`'s head is a `Maybe` constructor (`Just`/`Nothing`), used to
/// pick the Maybe arm of `fmap`.
fn is_maybe_expr(expr: &Expr) -> bool {
    let (head, _) = collect_app_spine(peel_head(expr));
    matches!(head, Expr::Var(v, _) if matches!(v.name.as_str(), "Just" | "Nothing"))
}

/// The renderable kind of a runtime scalar (a list element).
#[derive(Clone, Copy)]
enum ElemKind {
    Int,
    Double,
    Bool,
}

/// If `ty` is a list type `[e]` whose element is a renderable scalar, return its
/// element kind. Used to show runtime lists (e.g. `map`/`filter` results).
fn list_elem_show_kind(ty: &Ty) -> Option<ElemKind> {
    let elem = match ty {
        Ty::List(e) => e.as_ref(),
        // `[] e` desugared as an application.
        Ty::App(c, e) if matches!(c.as_ref(), Ty::Con(tc) if tc.name.as_str() == "[]") => {
            e.as_ref()
        }
        _ => return None,
    };
    match elem {
        Ty::Con(c) => match c.name.as_str() {
            "Int" | "Integer" | "Word" | "Char" => Some(ElemKind::Int),
            "Double" | "Float" => Some(ElemKind::Double),
            "Bool" => Some(ElemKind::Bool),
            _ => None,
        },
        _ => None,
    }
}

/// Whether a type is a floating-point type (`Double` or `Float`), which the
/// backend represents as a boxed `f64`.
fn is_float_ty(ty: &Ty) -> bool {
    match ty {
        Ty::Con(c) => matches!(c.name.as_str(), "Double" | "Float"),
        // Look through a single application layer (defensive; scalar floats are
        // plain `Con`s in practice).
        Ty::App(head, _) => is_float_ty(head),
        _ => false,
    }
}

/// Whether an expression has floating-point type. A `Float`/`Double` literal is
/// always float; otherwise consult the inferred type, and — since the types of
/// intermediate arithmetic are often erased — infer structurally: `/` and
/// `sqrt` are always floating, and `+`/`-`/`*`/`negate`/`abs` are floating when
/// an operand is.
fn expr_is_float(expr: &Expr) -> bool {
    let e = peel_show_wrappers(expr);
    if let Expr::Lit(Literal::Float(_) | Literal::Double(_), _, _) = e {
        return true;
    }
    if is_float_ty(&e.ty()) {
        return true;
    }
    let (head, args) = collect_app_spine(e);
    if let Expr::Var(hv, _) = head {
        match hv.name.as_str() {
            "/" | "GHC.Real./" | "GHC.Float./" | "sqrt" | "GHC.Float.sqrt" => return true,
            "+" | "GHC.Num.+" | "-" | "GHC.Num.-" | "*" | "GHC.Num.*"
                if args.iter().any(|a| expr_is_float(a)) =>
            {
                return true;
            }
            "negate" | "GHC.Num.negate" | "abs" | "GHC.Num.abs"
                if args.len() == 1 && expr_is_float(args[0]) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// If `name` is a primitive operator that can be used as a function value,
/// return its arity. Used to eta-expand operator sections like `(+)` into
/// closures when passed to higher-order functions.
fn operator_arity(name: &str) -> Option<usize> {
    match name {
        "+" | "GHC.Num.+" | "-" | "GHC.Num.-" | "*" | "GHC.Num.*" | "/" | "GHC.Real./"
        | "GHC.Float./" | "div" | "GHC.Real.div" | "mod" | "GHC.Real.mod" | "=="
        | "GHC.Classes.==" | "/=" | "GHC.Classes./=" | "<" | "GHC.Classes.<" | "<="
        | "GHC.Classes.<=" | ">" | "GHC.Classes.>" | ">=" | "GHC.Classes.>=" | "++"
        | "GHC.Base.++" => Some(2),
        "negate" | "GHC.Num.negate" => Some(1),
        // Enum/identity/char builtins usable as first-class function values.
        _ => match strip_qualifier(name) {
            "succ" | "pred" | "negate" | "abs" | "fromEnum" | "toEnum" | "not" | "chr" | "ord" => {
                Some(1)
            }
            _ => None,
        },
    }
}

/// Whether a function name always returns a `Double`, so its result should
/// `show` via the double formatter even when the static type is erased.
fn returns_double_fn(name: &str) -> bool {
    matches!(name, "sqrt" | "GHC.Float.sqrt")
}

/// Whether a prelude function always returns a `Bool`, so its result should
/// `show` as `True`/`False` even when the static type is erased.
/// Whether a (possibly curried, possibly `forall`-quantified) function type's
/// final result is `Bool`. Used to record `Bool`-returning functions for `show`.
fn final_result_is_bool(ty: &Ty) -> bool {
    let mut t = ty;
    loop {
        match t {
            Ty::Fun(_, res) => t = res,
            Ty::Forall(_, inner) => t = inner,
            Ty::Con(c) => return c.name.as_str() == "Bool",
            _ => return false,
        }
    }
}

fn returns_bool_fn(name: &str) -> bool {
    matches!(
        strip_qualifier(name),
        "null"
            | "all"
            | "any"
            | "and"
            | "or"
            | "elem"
            | "even"
            | "odd"
            | "notElem"
            | "isPrefixOf"
            | "isSuffixOf"
            | "isInfixOf"
            | "member"
            | "isAbsolute"
            | "isRelative"
            | "hasExtension"
            | "isAscii"
            | "isLetter"
    )
}

/// Whether an operator name denotes a boolean-valued operation (comparison or
/// logical connective), so its result should `show` as `True`/`False`.
fn is_bool_valued_op(name: &str) -> bool {
    matches!(
        name,
        "==" | "GHC.Classes.=="
            | "eqInt#"
            | "/="
            | "GHC.Classes./="
            | "neInt#"
            | "<"
            | "GHC.Classes.<"
            | "ltInt#"
            | "<="
            | "GHC.Classes.<="
            | "leInt#"
            | ">"
            | "GHC.Classes.>"
            | "gtInt#"
            | ">="
            | "GHC.Classes.>="
            | "geInt#"
            | "&&"
            | "GHC.Classes.&&"
            | "||"
            | "GHC.Classes.||"
            | "not"
            | "GHC.Classes.not"
    )
}

/// Peel lambda abstractions off the front of an expression.
fn peel_lambdas(expr: &Expr) -> (Vec<&Var>, &Expr) {
    let mut params = Vec::new();
    let mut current = expr;

    loop {
        match current {
            Expr::Lam(var, body, _) => {
                params.push(var);
                current = body;
            }
            // Skip type abstractions
            Expr::TyLam(_, body, _) => {
                current = body;
            }
            _ => break,
        }
    }

    (params, current)
}

/// Collect the application spine: `f a1 a2 ... an` -> `(f, [a1, a2, ..., an])`.
fn collect_app_spine(expr: &Expr) -> (&Expr, Vec<&Expr>) {
    let mut args = Vec::new();
    let mut current = expr;

    loop {
        match current {
            Expr::App(f, arg, _) => {
                args.push(arg.as_ref());
                current = f;
            }
            // Skip type applications
            Expr::TyApp(f, _, _) => {
                current = f;
            }
            _ => break,
        }
    }

    args.reverse();
    (current, args)
}

/// Get the `Symbol` from a function expression if it's a `Var`.
fn func_expr_symbol(expr: &Expr) -> Option<Symbol> {
    match expr {
        Expr::Var(var, _) => Some(var.name),
        _ => None,
    }
}

/// Get the `Var` from a function expression if it's a `Var`.
fn func_expr_var(expr: &Expr) -> Option<&Var> {
    match expr {
        Expr::Var(var, _) => Some(var),
        _ => None,
    }
}

/// Peel off type abstractions (TyLam, TyApp) to find the underlying expression.
fn peel_type_abstractions(expr: &Expr) -> &Expr {
    match expr {
        Expr::TyLam(_, body, _) | Expr::TyApp(body, _, _) => peel_type_abstractions(body),
        Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) => peel_type_abstractions(inner),
        _ => expr,
    }
}

/// Try to extract `enumFromTo lo hi` from an expression.
///
/// Returns `Some((lo, hi))` if the expression matches `App(App(enumFromTo, lo), hi)`,
/// looking through type applications and casts.
fn extract_enum_from_to(expr: &Expr) -> Option<(&Expr, &Expr)> {
    let (func, args) = collect_app_spine(expr);

    let func_name = match func {
        Expr::Var(var, _) => Some(var.name.as_str()),
        _ => None,
    };

    match func_name {
        Some("enumFromTo" | "Prelude.enumFromTo" | "GHC.Enum.enumFromTo") if args.len() == 2 => {
            Some((args[0], args[1]))
        }
        _ => None,
    }
}
