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

    // Record non-recursive top-level functions so saturated calls can be
    // inlined; record every top-level function's arity so partial and
    // over-applications can be handled via closures.
    lowering.register_inline_bodies(core);
    lowering.register_arities(core);

    // First pass: register all top-level function names so we can resolve calls
    for bind in &core.bindings {
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
    loop {
        let mut added = false;
        for &name in LIST_PRELUDE_NAMES {
            let sym = Symbol::intern(name);
            if lowering.func_map.contains_key(&sym) || synthesized.contains(&sym) {
                continue;
            }
            let used =
                module_uses_name(core, sym) || prelude.iter().any(|(_, e)| expr_uses_name(e, sym));
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
    for bind in &core.bindings {
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
    /// Substitution environment: var id -> expression to lower in its place.
    /// Used to inline function/lambda arguments without alpha-renaming —
    /// when a parameter is referenced, its argument expression is lowered.
    subst: FxHashMap<VarId, Expr>,
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
            subst: FxHashMap::default(),
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
        for con in &core.constructors {
            if con.is_newtype {
                continue;
            }
            self.con_map.insert(
                con.name.clone(),
                ConInfo {
                    tag: con.tag,
                    arity: con.arity,
                },
            );
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
                "toEnum" | "minBound" | "maxBound" => return self.sole_enum().cloned(),
                _ => {}
            }
        }
        None
    }

    /// Record non-recursive top-level functions (those with at least one
    /// value parameter) so saturated applications can be inlined. `main` is
    /// excluded — it is the entry point, never a callee.
    fn register_inline_bodies(&mut self, core: &CoreModule) {
        for bind in &core.bindings {
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

    /// Record the arity (leading lambda parameter count) of every top-level
    /// binding, so partial/over-application can be detected at call sites.
    fn register_arities(&mut self, core: &CoreModule) {
        let mut record = |var: &Var, expr: &Expr| {
            let (params, _) = peel_lambdas(expr);
            self.arities.insert(var.name, params.len());
        };
        for bind in &core.bindings {
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
        let mut bytes = (len as i32).to_le_bytes().to_vec();
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
                    _ => {}
                }
                // Check if it's a nullary constructor
                if let Some((tag, 0)) = self.lookup_constructor(name) {
                    instrs.push(WasmInstr::I32Const(tag as i32));
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
        // Peel transparent wrappers.
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
                    if is_bool_valued_op(name) || returns_bool_fn(name) {
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
            if args.len() == 2 {
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

            Some("putStrLn" | "System.IO.putStrLn" | "GHC.IO.putStrLn") if args.len() == 1 => {
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
            Some("putStr" | "System.IO.putStr" | "GHC.IO.putStr") if args.len() == 1 => {
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

            // IO: catch - execute the action, ignore the handler
            Some("catch" | "Control.Exception.catch" | "GHC.IO.catch") if args.len() == 2 => {
                // Simple implementation: just run the first argument (the IO action).
                // The exception handler (args[1]) is never invoked since we don't throw.
                self.lower_expr(args[0], instrs, locals, local_count, is_main)?;
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
                    // Can't handle non-enumFromTo sum arguments
                    tracing::warn!("sum with non-enumFromTo argument, using 0");
                    self.lower_expr(args[0], instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::Drop);
                    instrs.push(WasmInstr::I32Const(0));
                }
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

        // Any other string-valued expression — a literal, an `if`/`case` over
        // strings, or a `++` result — evaluates to a length-prefixed string
        // pointer that carries its own length, so it prints uniformly.
        self.lower_expr(expr, instrs, locals, local_count, false)?;
        instrs.push(WasmInstr::Call(self.runtime.print_pstr_idx));
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
];

/// Whether any binding in the module references `name`.
fn module_uses_name(core: &CoreModule, name: Symbol) -> bool {
    core.bindings.iter().any(|bind| match bind {
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

/// Build a single named list-prelude binding, drawing fresh ids from `id`.
fn build_list_fn(name: &str, id: &mut usize) -> Option<(Var, Expr)> {
    let fresh = |id: &mut usize| {
        let v = *id;
        *id += 1;
        v
    };
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
        Some(Ty::List(_) | Ty::App(..) | Ty::Tuple(_)) => true,
        _ => false,
    }
}

/// Whether a function always returns a cons-list (used to classify an operand
/// of `++` as a list). Excludes `++`/`reverse`/`show`, whose result kind is
/// ambiguous (string vs list).
fn is_list_returning_fn(name: &str) -> bool {
    matches!(
        name,
        "map"
            | "filter"
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
    let (head, _) = collect_app_spine(e);
    matches!(head, Expr::Var(v, _) if is_list_returning_fn(v.name.as_str()))
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
        _ => None,
    }
}

/// Whether a function name always returns a `Double`, so its result should
/// `show` via the double formatter even when the static type is erased.
fn returns_double_fn(name: &str) -> bool {
    matches!(name, "sqrt" | "GHC.Float.sqrt")
}

/// Whether a prelude function always returns a `Bool`, so its result should
/// `show` as `True`/`False` even when the static type is erased.
fn returns_bool_fn(name: &str) -> bool {
    matches!(name, "null" | "all" | "any" | "and" | "or" | "elem")
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
