//! Kind checking: a type must be applied to the arguments its constructor
//! takes, and a value's type must be a type of values (#97).
//!
//! `f :: Maybe -> Int`, `g :: Int Int -> Int`, `data W = W (Maybe Maybe)`
//! and `instance Functor Int` all used to compile.
//!
//! # How kinds are found
//!
//! Kinds are inferred with kind variables and unification:
//!
//! - The data types, newtypes and type synonyms the module declares are
//!   inferred together per strongly connected group, from their
//!   constructors' fields and synonym right-hand sides, and then
//!   generalised: a parameter nothing constrains stays polymorphic (as
//!   under `PolyKinds`) instead of defaulting to `*`, so `data Proxy a =
//!   Proxy` can be used at `Proxy Maybe`.
//! - A class's parameter kinds come from its method signatures.
//! - The Prelude's types and classes have their standard kinds.
//! - Every other type constructor (an imported type, a type family, a
//!   promoted constructor, a type operator) gets a fresh, unconstrained
//!   kind at each occurrence. What the checker does not know never makes a
//!   program fail; it only reports a type it can see is ill-kinded.
//!
//! # What is checked
//!
//! Each constructor field and each top-level and class method signature
//! has kind `*`, every application matches its constructor's kind, and a
//! class constraint or instance head gives a class arguments of the
//! class's parameter kinds.

use bhc_hir::{ConDef, ConFields, InstanceDef, Item, Module};
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Kind, Ty, TyList};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::context::TyCtxt;
use crate::diagnostics::pretty_ty;

/// A kind with its own kind variables quantified: each use instantiates
/// them afresh.
#[derive(Clone, Debug)]
struct KindScheme {
    vars: Vec<u32>,
    kind: Kind,
}

/// Why a type is ill-kinded, reported once per declaration.
struct KindError {
    message: String,
    note: String,
}

/// Kind inference state.
#[derive(Default)]
struct KindChecker {
    /// Solutions of kind variables, indexed by `Kind::Var` number.
    solution: Vec<Option<Kind>>,
    /// The generalised kinds of the type constructors this module declares.
    local: FxHashMap<Symbol, KindScheme>,
    /// The kinds of the declarations being inferred together, before they
    /// are generalised.
    group: FxHashMap<Symbol, Kind>,
    /// Type names the module declares in a way the checker does not infer
    /// (type and data families): these must not fall back to a standard
    /// type of the same name.
    opaque: FxHashSet<Symbol>,
    /// Parameter kinds of the classes this module declares.
    classes: FxHashMap<Symbol, Vec<KindScheme>>,
    /// The first error in the declaration being checked.
    error: Option<KindError>,
}

/// Check the kinds of the types in `module`'s declarations and signatures,
/// reporting each ill-kinded declaration once.
pub(crate) fn check_module_kinds(ctx: &mut TyCtxt, module: &Module) {
    let mut checker = KindChecker::default();
    let mut errors = Vec::new();
    checker.infer_type_declarations(module, &mut errors);
    checker.infer_classes(module, &mut errors);
    for item in &module.items {
        match item {
            Item::Value(value) => {
                if let Some(sig) = &value.sig {
                    let mut vars = FxHashMap::default();
                    checker.constraints(&sig.constraints, &mut vars);
                    checker.expect_star(&sig.ty, &mut vars);
                }
                checker.take_error(value.span, &mut errors);
            }
            Item::Instance(instance) => {
                checker.instance_head(instance);
                checker.take_error(instance.span, &mut errors);
            }
            _ => {}
        }
    }
    for (span, error) in errors {
        let diag = bhc_diagnostics::Diagnostic::error(error.message)
            .with_code("E0007")
            .with_label(ctx.full_span(span), "in this declaration")
            .with_note(error.note);
        ctx.emit_error(diag);
    }
}

impl KindChecker {
    // ------------------------------------------------------------------
    // Kind variables
    // ------------------------------------------------------------------

    fn fresh(&mut self) -> Kind {
        let index = u32::try_from(self.solution.len()).unwrap_or(u32::MAX);
        self.solution.push(None);
        Kind::Var(index)
    }

    /// `kind` with its outermost solved variables replaced.
    fn shallow(&self, kind: &Kind) -> Kind {
        let mut kind = kind.clone();
        while let Kind::Var(v) = kind {
            match self.solution.get(v as usize).and_then(Option::as_ref) {
                Some(solved) => kind = solved.clone(),
                None => break,
            }
        }
        kind
    }

    /// `kind` with every solved variable replaced.
    fn zonk(&self, kind: &Kind) -> Kind {
        match self.shallow(kind) {
            Kind::Arrow(a, b) => Kind::Arrow(Box::new(self.zonk(&a)), Box::new(self.zonk(&b))),
            Kind::List(k) => Kind::List(Box::new(self.zonk(&k))),
            other => other,
        }
    }

    fn occurs(&self, var: u32, kind: &Kind) -> bool {
        match self.shallow(kind) {
            Kind::Var(v) => v == var,
            Kind::Arrow(a, b) => self.occurs(var, &a) || self.occurs(var, &b),
            Kind::List(k) => self.occurs(var, &k),
            _ => false,
        }
    }

    /// Make two kinds equal, or report that they cannot be.
    fn unify(&mut self, a: &Kind, b: &Kind) -> bool {
        let (a, b) = (self.shallow(a), self.shallow(b));
        match (&a, &b) {
            (Kind::Var(x), Kind::Var(y)) if x == y => true,
            (Kind::Var(v), other) | (other, Kind::Var(v)) => {
                if self.occurs(*v, other) {
                    return false;
                }
                self.solution[*v as usize] = Some(other.clone());
                true
            }
            (Kind::Arrow(a1, r1), Kind::Arrow(a2, r2)) => self.unify(a1, a2) && self.unify(r1, r2),
            (Kind::List(k1), Kind::List(k2)) => self.unify(k1, k2),
            _ => a == b,
        }
    }

    fn free_vars(&self, kind: &Kind, out: &mut Vec<u32>) {
        match self.shallow(kind) {
            Kind::Var(v) => {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
            Kind::Arrow(a, b) => {
                self.free_vars(&a, out);
                self.free_vars(&b, out);
            }
            Kind::List(k) => self.free_vars(&k, out),
            _ => {}
        }
    }

    fn generalize(&self, kind: &Kind) -> KindScheme {
        let kind = self.zonk(kind);
        let mut vars = Vec::new();
        self.free_vars(&kind, &mut vars);
        KindScheme { vars, kind }
    }

    fn instantiate(&mut self, scheme: &KindScheme) -> Kind {
        let fresh: FxHashMap<u32, Kind> = scheme.vars.iter().map(|v| (*v, self.fresh())).collect();
        substitute(&scheme.kind, &fresh)
    }

    // ------------------------------------------------------------------
    // Types
    // ------------------------------------------------------------------

    /// The kind of the type constructor `name`, or a fresh variable when
    /// the checker does not know it.
    fn constructor_kind(&mut self, name: Symbol) -> Kind {
        if let Some(kind) = self.group.get(&name) {
            return kind.clone();
        }
        if let Some(scheme) = self.local.get(&name).cloned() {
            return self.instantiate(&scheme);
        }
        if self.opaque.contains(&name) {
            return self.fresh();
        }
        standard_type_kind(name.as_str()).unwrap_or_else(|| self.fresh())
    }

    /// Infer the kind of `ty`. Type variables are looked up in, or added
    /// to, `vars`.
    fn infer(&mut self, ty: &Ty, vars: &mut FxHashMap<u32, Kind>) -> Kind {
        match ty {
            Ty::Var(v) => {
                if let Some(kind) = vars.get(&v.id) {
                    return kind.clone();
                }
                let kind = self.fresh();
                vars.insert(v.id, kind.clone());
                kind
            }
            Ty::Con(con) => self.constructor_kind(con.name),
            Ty::Prim(_) => Kind::Star,
            Ty::App(fun, arg) => {
                let fun_kind = self.infer(fun, vars);
                let arg_kind = self.infer(arg, vars);
                let result = self.fresh();
                let expected = Kind::Arrow(Box::new(arg_kind.clone()), Box::new(result.clone()));
                if !self.unify(&fun_kind, &expected) {
                    self.report_application(fun, &fun_kind, arg, &arg_kind);
                }
                result
            }
            Ty::Fun(from, to) => {
                self.expect_star(from, vars);
                self.expect_star(to, vars);
                Kind::Star
            }
            Ty::Tuple(elems) => {
                for elem in elems {
                    self.expect_star(elem, vars);
                }
                Kind::Star
            }
            Ty::List(elem) => {
                self.expect_star(elem, vars);
                Kind::Star
            }
            Ty::Forall(bound, body) => {
                let saved: Vec<(u32, Option<Kind>)> = bound
                    .iter()
                    .map(|v| (v.id, vars.get(&v.id).cloned()))
                    .collect();
                for v in bound {
                    let kind = self.fresh();
                    vars.insert(v.id, kind);
                }
                let kind = self.infer(body, vars);
                for (id, previous) in saved {
                    match previous {
                        Some(kind) => vars.insert(id, kind),
                        None => vars.remove(&id),
                    };
                }
                kind
            }
            Ty::Nat(_) => Kind::Nat,
            Ty::TyList(list) => {
                let elem = self.fresh();
                self.infer_ty_list(list, &elem, vars);
                Kind::List(Box::new(elem))
            }
            Ty::Error => self.fresh(),
        }
    }

    fn infer_ty_list(&mut self, list: &TyList, elem: &Kind, vars: &mut FxHashMap<u32, Kind>) {
        match list {
            TyList::Nil | TyList::Var(_) => {}
            TyList::Cons(head, tail) => {
                let head_kind = self.infer(head, vars);
                // A mismatch inside a type-level list is left to the type
                // checker: these lists are only used for tensor shapes.
                let _ = self.unify(&head_kind, elem);
                self.infer_ty_list(tail, elem, vars);
            }
            TyList::Append(a, b) => {
                self.infer_ty_list(a, elem, vars);
                self.infer_ty_list(b, elem, vars);
            }
        }
    }

    /// Require `ty` to be a type of values.
    fn expect_star(&mut self, ty: &Ty, vars: &mut FxHashMap<u32, Kind>) {
        let kind = self.infer(ty, vars);
        if !self.unify(&kind, &Kind::Star) {
            let kind = self.zonk(&kind);
            self.report(KindError {
                message: format!(
                    "kind mismatch: `{}` has kind `{}`, but a type of kind `*` is expected",
                    pretty_ty(ty),
                    pretty_kind(&kind)
                ),
                note: "a type of values has kind `*`; maybe a type argument is missing".to_string(),
            });
        }
    }

    /// Apply the classes' parameter kinds to the arguments of `constraints`.
    fn constraints(&mut self, constraints: &[Constraint], vars: &mut FxHashMap<u32, Kind>) {
        for constraint in constraints {
            self.class_arguments(constraint.class, &constraint.args, vars);
        }
    }

    /// Require `args` to have the parameter kinds of `class`, when the
    /// checker knows them.
    fn class_arguments(&mut self, class: Symbol, args: &[Ty], vars: &mut FxHashMap<u32, Kind>) {
        let Some(params) = self.class_kinds(class) else {
            return;
        };
        if params.len() != args.len() {
            return;
        }
        for (arg, expected) in args.iter().zip(params) {
            let kind = self.infer(arg, vars);
            if !self.unify(&kind, &expected) {
                let (kind, expected) = (self.zonk(&kind), self.zonk(&expected));
                self.report(KindError {
                    message: format!(
                        "kind mismatch: `{}` has kind `{}`, but `{class}` expects a type of kind `{}`",
                        pretty_ty(arg),
                        pretty_kind(&kind),
                        pretty_kind(&expected)
                    ),
                    note: format!("the parameter of `{class}` has kind `{}`", pretty_kind(&expected)),
                });
            }
        }
    }

    fn class_kinds(&mut self, class: Symbol) -> Option<Vec<Kind>> {
        if let Some(schemes) = self.classes.get(&class).cloned() {
            return Some(schemes.iter().map(|s| self.instantiate(s)).collect());
        }
        standard_class_kind(class.as_str()).map(|kind| vec![kind])
    }

    fn instance_head(&mut self, instance: &InstanceDef) {
        let mut vars = FxHashMap::default();
        self.constraints(&instance.constraints, &mut vars);
        self.class_arguments(instance.class, &instance.types, &mut vars);
    }

    fn report_application(&mut self, fun: &Ty, fun_kind: &Kind, arg: &Ty, arg_kind: &Kind) {
        let fun_kind = self.zonk(fun_kind);
        let arg_kind = self.zonk(arg_kind);
        let error = match &fun_kind {
            Kind::Arrow(expected, _) => KindError {
                message: format!(
                    "kind mismatch: `{}` expects an argument of kind `{}`, but `{}` has kind `{}`",
                    pretty_ty(fun),
                    pretty_kind(expected),
                    pretty_ty(arg),
                    pretty_kind(&arg_kind)
                ),
                note: format!("`{}` has kind `{}`", pretty_ty(fun), pretty_kind(&fun_kind)),
            },
            _ => KindError {
                message: format!(
                    "kind mismatch: `{}` has kind `{}` and cannot be applied to `{}`",
                    pretty_ty(fun),
                    pretty_kind(&fun_kind),
                    pretty_ty(arg)
                ),
                note: "it is applied to more type arguments than it takes".to_string(),
            },
        };
        self.report(error);
    }

    fn report(&mut self, error: KindError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    fn take_error(&mut self, span: Span, errors: &mut Vec<(Span, KindError)>) {
        if let Some(error) = self.error.take() {
            errors.push((span, error));
        }
    }

    // ------------------------------------------------------------------
    // Declarations
    // ------------------------------------------------------------------

    /// Infer the kinds of the module's data types, newtypes and synonyms,
    /// one strongly connected group at a time.
    fn infer_type_declarations(&mut self, module: &Module, errors: &mut Vec<(Span, KindError)>) {
        let decls: Vec<TypeDecl<'_>> = module
            .items
            .iter()
            .filter_map(TypeDecl::from_item)
            .collect();
        for item in &module.items {
            match item {
                Item::TypeFamily(family) => {
                    self.opaque.insert(family.name);
                }
                Item::DataFamily(family) => {
                    self.opaque.insert(family.name);
                }
                _ => {}
            }
        }
        let names: FxHashMap<Symbol, usize> = decls
            .iter()
            .enumerate()
            .map(|(i, d)| (d.name(), i))
            .collect();
        let edges: Vec<Vec<usize>> = decls
            .iter()
            .map(|d| {
                let mut used = Vec::new();
                d.for_each_type(&mut |ty| collect_constructors(ty, &mut used));
                let mut targets: Vec<usize> =
                    used.iter().filter_map(|n| names.get(n).copied()).collect();
                targets.sort_unstable();
                targets.dedup();
                targets
            })
            .collect();
        for group in strongly_connected_components(&edges) {
            self.infer_group(
                &group.iter().map(|&i| &decls[i]).collect::<Vec<_>>(),
                errors,
            );
        }
    }

    fn infer_group(&mut self, group: &[&TypeDecl<'_>], errors: &mut Vec<(Span, KindError)>) {
        let mut params: Vec<FxHashMap<u32, Kind>> = Vec::new();
        let mut results: Vec<Kind> = Vec::new();
        for decl in group {
            let mut vars = FxHashMap::default();
            let mut kinds = Vec::new();
            for param in decl.params() {
                let kind = self.fresh();
                vars.insert(param.id, kind.clone());
                kinds.push(kind);
            }
            // A GADT may give its parameters in a kind signature instead
            // (`data T :: * -> * where`), and a synonym's result is its
            // right-hand side's kind.
            let result = match decl {
                TypeDecl::Data(data) if !data.is_gadt => Kind::Star,
                TypeDecl::Newtype(_) => Kind::Star,
                _ => self.fresh(),
            };
            let kind = kinds.into_iter().rev().fold(result.clone(), |acc, k| {
                Kind::Arrow(Box::new(k), Box::new(acc))
            });
            self.group.insert(decl.name(), kind);
            params.push(vars);
            results.push(result);
        }
        for ((decl, vars), result) in group.iter().zip(&params).zip(&results) {
            match decl {
                TypeDecl::Data(data) => {
                    for con in &data.cons {
                        self.constructor(con, vars);
                    }
                }
                TypeDecl::Newtype(newtype) => self.constructor(&newtype.con, vars),
                TypeDecl::Alias(alias) => {
                    let mut vars = vars.clone();
                    let kind = self.infer(&alias.ty, &mut vars);
                    // A synonym's result kind is whatever its right-hand
                    // side has; this cannot fail.
                    let _ = self.unify(result, &kind);
                }
            }
            self.take_error(decl.span(), errors);
        }
        for decl in group {
            if let Some(kind) = self.group.remove(&decl.name()) {
                let scheme = self.generalize(&kind);
                self.local.insert(decl.name(), scheme);
            }
        }
    }

    fn constructor(&mut self, con: &ConDef, params: &FxHashMap<u32, Kind>) {
        let mut vars = params.clone();
        match &con.fields {
            ConFields::Positional(fields) => {
                for field in fields {
                    self.expect_star(field, &mut vars);
                }
            }
            ConFields::Named(fields) => {
                for field in fields {
                    self.expect_star(&field.ty, &mut vars);
                }
            }
        }
        if let Some(result) = &con.gadt_return_ty {
            self.expect_star(result, &mut vars);
        }
    }

    /// Infer the parameter kinds of the module's classes from their method
    /// signatures.
    fn infer_classes(&mut self, module: &Module, errors: &mut Vec<(Span, KindError)>) {
        for item in &module.items {
            let Item::Class(class) = item else {
                continue;
            };
            let mut vars = FxHashMap::default();
            let mut kinds = Vec::new();
            for param in &class.params {
                let kind = self.fresh();
                vars.insert(param.id, kind.clone());
                kinds.push(kind);
            }
            // A single-parameter superclass constrains the same parameter.
            if let [kind] = kinds.as_slice() {
                for sup in &class.supers {
                    if let Some([sup_kind]) = self.class_kinds(*sup).as_deref() {
                        let _ = self.unify(kind, sup_kind);
                    }
                }
            }
            for method in &class.methods {
                let mut method_vars = vars.clone();
                self.constraints(&method.ty.constraints, &mut method_vars);
                self.expect_star(&method.ty.ty, &mut method_vars);
                self.take_error(method.span, errors);
            }
            let schemes = kinds.iter().map(|k| self.generalize(k)).collect();
            self.classes.insert(class.name, schemes);
        }
    }
}

/// A declaration that introduces a type constructor whose kind is inferred.
enum TypeDecl<'a> {
    Data(&'a bhc_hir::DataDef),
    Newtype(&'a bhc_hir::NewtypeDef),
    Alias(&'a bhc_hir::TypeAlias),
}

impl<'a> TypeDecl<'a> {
    fn from_item(item: &'a Item) -> Option<Self> {
        match item {
            Item::Data(data) => Some(Self::Data(data)),
            Item::Newtype(newtype) => Some(Self::Newtype(newtype)),
            Item::TypeAlias(alias) => Some(Self::Alias(alias)),
            _ => None,
        }
    }

    fn name(&self) -> Symbol {
        match self {
            Self::Data(d) => d.name,
            Self::Newtype(n) => n.name,
            Self::Alias(a) => a.name,
        }
    }

    fn params(&self) -> &'a [bhc_types::TyVar] {
        match self {
            Self::Data(d) => &d.params,
            Self::Newtype(n) => &n.params,
            Self::Alias(a) => &a.params,
        }
    }

    fn span(&self) -> Span {
        match self {
            Self::Data(d) => d.span,
            Self::Newtype(n) => n.span,
            Self::Alias(a) => a.span,
        }
    }

    fn for_each_type(&self, f: &mut dyn FnMut(&Ty)) {
        let mut con = |con: &ConDef| {
            match &con.fields {
                ConFields::Positional(fields) => fields.iter().for_each(&mut *f),
                ConFields::Named(fields) => fields.iter().for_each(|field| f(&field.ty)),
            }
            if let Some(result) = &con.gadt_return_ty {
                f(result);
            }
        };
        match self {
            Self::Data(d) => d.cons.iter().for_each(&mut con),
            Self::Newtype(n) => con(&n.con),
            Self::Alias(a) => f(&a.ty),
        }
    }
}

/// The type constructor names `ty` mentions.
fn collect_constructors(ty: &Ty, out: &mut Vec<Symbol>) {
    match ty {
        Ty::Con(con) => out.push(con.name),
        Ty::App(a, b) | Ty::Fun(a, b) => {
            collect_constructors(a, out);
            collect_constructors(b, out);
        }
        Ty::Tuple(elems) => elems.iter().for_each(|t| collect_constructors(t, out)),
        Ty::List(elem) => collect_constructors(elem, out),
        Ty::Forall(_, body) => collect_constructors(body, out),
        Ty::Var(_) | Ty::Prim(_) | Ty::Nat(_) | Ty::TyList(_) | Ty::Error => {}
    }
}

/// The strongly connected components of a graph, dependencies first
/// (Tarjan's algorithm).
fn strongly_connected_components(edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct State<'e> {
        edges: &'e [Vec<usize>],
        index: Vec<Option<usize>>,
        low: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        next: usize,
        components: Vec<Vec<usize>>,
    }
    fn visit(s: &mut State<'_>, v: usize) {
        s.index[v] = Some(s.next);
        s.low[v] = s.next;
        s.next += 1;
        s.stack.push(v);
        s.on_stack[v] = true;
        for &w in &s.edges[v] {
            match s.index[w] {
                None => {
                    visit(s, w);
                    s.low[v] = s.low[v].min(s.low[w]);
                }
                Some(w_index) if s.on_stack[w] => s.low[v] = s.low[v].min(w_index),
                Some(_) => {}
            }
        }
        if Some(s.low[v]) == s.index[v] {
            let mut component = Vec::new();
            while let Some(w) = s.stack.pop() {
                s.on_stack[w] = false;
                component.push(w);
                if w == v {
                    break;
                }
            }
            s.components.push(component);
        }
    }
    let n = edges.len();
    let mut state = State {
        edges,
        index: vec![None; n],
        low: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        next: 0,
        components: Vec::new(),
    };
    for v in 0..n {
        if state.index[v].is_none() {
            visit(&mut state, v);
        }
    }
    state.components
}

fn substitute(kind: &Kind, fresh: &FxHashMap<u32, Kind>) -> Kind {
    match kind {
        Kind::Var(v) => fresh.get(v).cloned().unwrap_or(Kind::Var(*v)),
        Kind::Arrow(a, b) => Kind::Arrow(
            Box::new(substitute(a, fresh)),
            Box::new(substitute(b, fresh)),
        ),
        Kind::List(k) => Kind::List(Box::new(substitute(k, fresh))),
        other => other.clone(),
    }
}

fn arrows(n: usize) -> Kind {
    (0..n).fold(Kind::Star, |acc, _| {
        Kind::Arrow(Box::new(Kind::Star), Box::new(acc))
    })
}

/// The kind of a standard Prelude type, by its unqualified name.
fn standard_type_kind(name: &str) -> Option<Kind> {
    let arity = match name {
        "Int" | "Integer" | "Double" | "Float" | "Char" | "Bool" | "Word" | "Ordering" | "()"
        | "String" | "Rational" => 0,
        "Maybe" | "IO" | "[]" => 1,
        "Either" | "(,)" | "->" | "(->)" => 2,
        _ => return None,
    };
    Some(arrows(arity))
}

/// The parameter kind of a standard single-parameter class.
fn standard_class_kind(name: &str) -> Option<Kind> {
    let arity = match name {
        "Show" | "Eq" | "Ord" | "Num" | "Enum" | "Bounded" | "Integral" | "Real" | "Fractional"
        | "Floating" | "RealFrac" | "RealFloat" | "Read" | "Semigroup" | "Monoid" => 0,
        "Functor" | "Applicative" | "Monad" | "Foldable" | "Traversable" | "Alternative"
        | "MonadPlus" | "MonadFail" => 1,
        _ => return None,
    };
    Some(arrows(arity))
}

/// A kind as Haskell writes it: `*`, `* -> *`, `(* -> *) -> *`.
fn pretty_kind(kind: &Kind) -> String {
    match kind {
        Kind::Star => "*".to_string(),
        Kind::Constraint => "Constraint".to_string(),
        Kind::Nat => "Nat".to_string(),
        Kind::Var(_) => "k".to_string(),
        Kind::List(k) => format!("[{}]", pretty_kind(k)),
        Kind::Arrow(a, b) => {
            let a = match a.as_ref() {
                Kind::Arrow(..) => format!("({})", pretty_kind(a)),
                other => pretty_kind(other),
            };
            format!("{a} -> {}", pretty_kind(b))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unify_solves_variables_and_rejects_mismatches() {
        let mut checker = KindChecker::default();
        let k = checker.fresh();
        assert!(checker.unify(&k, &arrows(1)));
        assert_eq!(checker.zonk(&k), arrows(1));
        assert!(!checker.unify(&k, &Kind::Star));
        let j = checker.fresh();
        let loops = Kind::Arrow(Box::new(j.clone()), Box::new(Kind::Star));
        assert!(!checker.unify(&j, &loops), "occurs check");
    }

    #[test]
    fn generalised_kinds_are_instantiated_afresh() {
        let mut checker = KindChecker::default();
        let param = checker.fresh();
        let proxy = Kind::Arrow(Box::new(param), Box::new(Kind::Star));
        let scheme = checker.generalize(&proxy);
        let at_star = checker.instantiate(&scheme);
        let at_arrow = checker.instantiate(&scheme);
        assert!(checker.unify(&at_star, &arrows(1)));
        let maybe_arg = Kind::Arrow(Box::new(arrows(1)), Box::new(Kind::Star));
        assert!(checker.unify(&at_arrow, &maybe_arg));
    }

    #[test]
    fn components_put_dependencies_first() {
        // 0 -> 1 <-> 2, 3 alone
        let edges = vec![vec![1], vec![2], vec![1], vec![]];
        let components = strongly_connected_components(&edges);
        let position = |v: usize| components.iter().position(|c| c.contains(&v)).unwrap();
        assert_eq!(position(1), position(2));
        assert!(position(1) < position(0));
        assert_eq!(components.len(), 3);
    }

    #[test]
    fn kinds_print_as_haskell_writes_them() {
        let higher = Kind::Arrow(Box::new(arrows(1)), Box::new(Kind::Star));
        assert_eq!(pretty_kind(&higher), "(* -> *) -> *");
        assert_eq!(pretty_kind(&arrows(2)), "* -> * -> *");
    }
}
