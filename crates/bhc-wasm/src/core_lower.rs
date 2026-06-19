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

use bhc_core::{AltCon, Bind, CoreModule, Expr, Literal, Var, VarId};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
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

    // Second pass: lower each binding to a WASM function
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
    /// String pool: maps string content to (data_offset, length).
    string_pool: FxHashMap<String, (u32, u32)>,
    /// Next available offset in the data segment for string storage.
    next_data_offset: u32,
    /// Counter for pre-registering function indices.
    next_func_idx: u32,
    /// Constructor map: name -> (tag, arity). Seeded with well-known
    /// constructors and extended with user-defined ones from the module.
    con_map: FxHashMap<String, ConInfo>,
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
}

impl<'a> WasmLowering<'a> {
    fn new(wasm: &'a mut WasmModule, runtime: &'a RuntimeIndices) -> Self {
        // Count existing functions (imports + defined) to know where new functions start
        let next_func_idx = wasm.next_function_index();

        Self {
            wasm,
            runtime,
            func_map: FxHashMap::default(),
            string_pool: FxHashMap::default(),
            next_data_offset: STRING_DATA_BASE,
            next_func_idx,
            con_map: well_known_constructors(),
            subst: FxHashMap::default(),
            inline_bodies: FxHashMap::default(),
            inlining: FxHashSet::default(),
            pending_closures: Vec::new(),
            arities: FxHashMap::default(),
            next_synthetic_id: 5_000_000,
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

    /// Intern a string into the data segment, returning (offset, length).
    fn intern_string(&mut self, s: &str) -> (u32, u32) {
        if let Some(&entry) = self.string_pool.get(s) {
            return entry;
        }

        let offset = self.next_data_offset;
        let len = s.len() as u32;
        let bytes = s.as_bytes().to_vec();
        self.wasm.add_data_segment(offset, bytes);
        self.next_data_offset += len;
        // Align to 4 bytes
        self.next_data_offset = (self.next_data_offset + 3) & !3;

        self.string_pool.insert(s.to_string(), (offset, len));
        (offset, len)
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

        // Lower the body expression
        let mut instrs = Vec::new();
        let mut local_count = params.len() as u32;
        self.lower_expr(body, &mut instrs, &mut locals, &mut local_count, is_main)?;

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
                self.lower_literal(lit, instrs)?;
            }

            Expr::Var(var, _) => {
                let name = var.name.as_str();
                // Inlined parameter: lower its bound argument expression.
                if let Some(arg) = self.subst.get(&var.id).cloned() {
                    return self.lower_expr(&arg, instrs, locals, local_count, is_main);
                }
                // Check if it's a nullary constructor
                if let Some((tag, 0)) = self.lookup_constructor(name) {
                    instrs.push(WasmInstr::I32Const(tag as i32));
                } else if let Some(&local_idx) = locals.get(&var.id) {
                    // Check if it's a local variable
                    instrs.push(WasmInstr::LocalGet(local_idx));
                } else if let Some(&func_idx) = self.func_map.get(&var.name) {
                    // It's a reference to a top-level function (as a value).
                    // For now, just call it with no args (for nullary functions).
                    instrs.push(WasmInstr::Call(func_idx));
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
    fn lower_literal(&mut self, lit: &Literal, instrs: &mut Vec<WasmInstr>) -> WasmResult<()> {
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
                // Intern the string and push its offset as the "value"
                let s = sym.as_str();
                let (offset, _len) = self.intern_string(s);
                instrs.push(WasmInstr::I32Const(offset as i32));
            }
            Literal::Float(f) => {
                instrs.push(WasmInstr::I32Const((*f as i32).max(0)));
            }
            Literal::Double(d) => {
                instrs.push(WasmInstr::I32Const((*d as i32).max(0)));
            }
        }
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
            Some("div" | "divInt#" | "GHC.Real.div") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32DivS);
            }
            Some("mod" | "modInt#" | "GHC.Real.mod") if args.len() == 2 => {
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                self.lower_expr(args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32RemS);
            }
            Some("negate" | "negateInt#" | "GHC.Num.negate") if args.len() == 1 => {
                instrs.push(WasmInstr::I32Const(0));
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
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
                self.lower_expr(args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Call(self.runtime.print_i32_idx));
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
            Cont::PrintStr { newline } => newline,
        };

        match expr {
            // Push the print into each branch so the leaf literal's length is
            // known statically.
            Expr::Case(scrut, alts, _, _) => {
                self.lower_case(scrut, alts, cont, instrs, locals, local_count, is_main)?;
            }

            // Transparent wrappers: see through to the inner expression.
            Expr::TyApp(inner, _, _)
            | Expr::TyLam(_, inner, _)
            | Expr::Cast(inner, _, _)
            | Expr::Tick(_, inner, _)
            | Expr::Lazy(inner, _) => {
                self.lower_cont(inner, cont, instrs, locals, local_count, is_main)?;
            }

            // Leaf: `putStrLn (show n)` / `putStr (show n)` prints the integer
            // directly via the runtime's int printer (which appends a newline).
            _ if match_show_arg(expr).is_some() => {
                let inner = match_show_arg(expr).unwrap();
                self.lower_expr(inner, instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Call(self.runtime.print_i32_idx));
                instrs.push(WasmInstr::I32Const(0));
            }

            // Leaf: a string literal whose length we know.
            _ => {
                if let Some(s) = extract_string_literal(expr) {
                    let (offset, len) = self.intern_string(s);
                    let print_idx = if newline {
                        self.runtime.print_str_ln_idx
                    } else {
                        self.runtime.print_str_idx
                    };
                    instrs.push(WasmInstr::I32Const(offset as i32));
                    instrs.push(WasmInstr::I32Const(len as i32));
                    instrs.push(WasmInstr::Call(print_idx));
                } else {
                    // Dynamic string with no statically known length — we can't
                    // recover it from the offset alone, so evaluate for effect
                    // and drop.
                    tracing::warn!("print of dynamic string with unknown length");
                    self.lower_expr(expr, instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::Drop);
                }
                // IO action returns a dummy value.
                instrs.push(WasmInstr::I32Const(0));
            }
        }

        Ok(())
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
            // No alternatives at all - push 0
            instrs.push(WasmInstr::I32Const(0));
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
            instrs.push(WasmInstr::If(Some(WasmType::I32)));

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
                    // No default - unreachable or just push 0
                    instrs.push(WasmInstr::I32Const(0));
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

        // If the scrutinee is a heap pointer, load the tag into a separate local.
        let tag_local = if has_fields {
            let tl = *local_count;
            *local_count += 1;
            instrs.push(WasmInstr::LocalGet(scrut_local));
            instrs.push(WasmInstr::I32Load(2, 0)); // load tag from offset 0
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
            instrs.push(WasmInstr::If(Some(WasmType::I32)));

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
                    instrs.push(WasmInstr::I32Const(0));
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

/// Extract a string literal from an expression, looking through casts/ticks.
fn extract_string_literal(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Lit(Literal::String(sym), _, _) => Some(sym.as_str()),
        Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) => extract_string_literal(inner),
        Expr::TyApp(inner, _, _) => extract_string_literal(inner),
        Expr::App(f, arg, _) => {
            // Sometimes strings appear as `unpackCString# "literal"`
            if let Expr::Var(var, _) = f.as_ref() {
                if var.name.as_str().contains("unpackCString") {
                    return extract_string_literal(arg);
                }
            }
            None
        }
        _ => None,
    }
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
