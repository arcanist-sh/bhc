//! # BHC Type Checker
//!
//! This crate implements Hindley-Milner type inference for the Basel Haskell Compiler.
//! It operates on HIR (High-level Intermediate Representation) and produces typed HIR
//! suitable for lowering to Core IR.
//!
//! ## Overview
//!
//! The type checker implements Algorithm W with the following features:
//!
//! - **Let-polymorphism**: Types are generalized at let-bindings
//! - **Mutual recursion**: Binding groups are analyzed via SCC decomposition
//! - **Type signatures**: User-provided signatures are checked against inferred types
//! - **Error recovery**: Inference continues after errors using error types
//!
//! ## Algorithm
//!
//! Type inference proceeds in several phases:
//!
//! 1. **Binding group analysis**: Identify mutually recursive groups via SCC
//! 2. **Constraint generation**: Walk HIR and generate type constraints
//! 3. **Unification**: Solve constraints via substitution
//! 4. **Generalization**: Generalize types at let-bindings
//!
//! ## Usage
//!
//! ```ignore
//! use bhc_typeck::type_check_module;
//! use bhc_hir::Module;
//! use bhc_span::FileId;
//!
//! let result = type_check_module(&hir_module, file_id);
//! match result {
//!     Ok(typed_module) => {
//!         // Use typed_module.expr_types to get inferred types
//!     }
//!     Err(diagnostics) => {
//!         // Report type errors
//!     }
//! }
//! ```
//!
//! ## See Also
//!
//! - `bhc-hir`: Input HIR types
//! - `bhc-types`: Type representation
//! - `bhc-core`: Output Core IR (after lowering)

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

mod binding_groups;
pub mod builtins;
mod context;
mod diagnostics;
mod env;
mod generalize;
mod infer;
mod instantiate;
pub mod kind_check;
pub mod nat_solver;
mod pattern;
pub mod shape_bridge;
pub mod shape_diagrams;
pub mod suggest;
pub mod type_families;
mod unify;

pub use context::TyCtxt;
pub use env::{DataConInfo, TypeEnv};
pub use kind_check::KindEnv;

use bhc_diagnostics::Diagnostic;
use bhc_hir::{DefId, Module};
use bhc_intern::Symbol;
use bhc_span::{FileId, Span};
use bhc_types::{Scheme, Ty, TyVar};
use rustc_hash::FxHashMap;

/// The result of type checking a module.
///
/// Contains the original HIR along with type annotations for all
/// expressions and definitions.
#[derive(Debug)]
pub struct TypedModule {
    /// The original HIR module.
    pub hir: Module,
    /// Inferred types for each expression, keyed by source `Span` (a stable id
    /// both typeck and HIR→Core lowering can compute). Populated during
    /// inference (spec/BHC-BRIEF-0002); values have the final substitution
    /// applied.
    pub expr_types: FxHashMap<Span, Ty>,
    /// Like [`expr_types`](Self::expr_types), but with the final substitution
    /// applied to a *fixpoint* (see [`bhc_types::Subst::apply_resolved`]). The plain
    /// `expr_types` uses a single-pass apply, which leaves nested variables
    /// unresolved under a triangular substitution — e.g. `return 42 :: M a`
    /// records `App(Var m, Int)` with `m` still a variable even though `m` is
    /// bound. This map resolves those chains.
    ///
    /// It exists ONLY as a side channel for lowering's dispatch decisions —
    /// choosing which `pure`/`return`/method implementation to call at a
    /// user-defined monad. It MUST NOT be baked into Core node types: codegen
    /// infers value widths assuming many `expr_types` leaves stay unresolved,
    /// and the fuller resolution here regresses it.
    pub resolved_expr_types: FxHashMap<Span, Ty>,
    /// Type schemes for each definition (indexed by `DefId`).
    pub def_schemes: FxHashMap<DefId, Scheme>,
    /// Type synonyms in scope at this module's compilation (local + imported),
    /// mapping the synonym name to its `(parameters, right-hand side)`. Exposed
    /// so interface generation can expand synonyms in exported signatures,
    /// making the `.bhi` self-contained: a consumer that does NOT import the
    /// synonym's defining module can still unify against the expanded type.
    pub type_aliases: FxHashMap<Symbol, (Vec<TyVar>, Ty)>,
}

// Re-export definition types from bhc-lower
pub use bhc_lower::{DefKind, DefMap};

/// Type check a HIR module.
///
/// This is the main entry point for type checking. It takes a resolved
/// HIR module and produces either a typed module or a list of diagnostics.
///
/// # Arguments
///
/// * `hir` - The HIR module to type check
/// * `file_id` - The file ID for error reporting
///
/// # Returns
///
/// * `Ok(TypedModule)` - Successfully typed module
/// * `Err(Vec<Diagnostic>)` - Type errors encountered
///
/// # Errors
///
/// Returns a `Vec<Diagnostic>` containing all type errors found during
/// type checking, such as type mismatches, unbound variables, and
/// occurs check failures.
///
/// # Example
///
/// ```ignore
/// let result = type_check_module(&module, file_id);
/// ```
pub fn type_check_module(hir: &Module, file_id: FileId) -> Result<TypedModule, Vec<Diagnostic>> {
    type_check_module_with_defs(hir, file_id, None)
}

pub(crate) use bhc_types::io_closure_mode;

/// The instance a newtype's deriving clause produces by newtype deriving,
/// under the closure IO model (spec/BHC-BRIEF-0005), for a class that is not
/// stock-derivable: `newtype App a = App (ExceptT String (StateT Int IO) a)
/// deriving (MonadError String)` gives `instance MonadError String
/// (ExceptT String (StateT Int IO)) => MonadError String App`. The class's
/// last parameter is the newtype minus its last parameter; the context is the
/// same class at the wrapped type, so solving goes on through its instances.
fn newtype_derived_instance(
    newtype: &bhc_hir::NewtypeDef,
    clause: &bhc_hir::DerivingClause,
) -> Option<env::InstanceInfo> {
    // Functor/Applicative/Monad are newtype-derived too: hir-to-core builds
    // their dictionaries from the wrapped type's.
    const STOCK: &[&str] = &[
        "Eq",
        "Ord",
        "Show",
        "Read",
        "Enum",
        "Bounded",
        "Ix",
        "Generic",
        "NFData",
        "Foldable",
        "Traversable",
        "Typeable",
        "Data",
    ];
    if !io_closure_mode() || STOCK.contains(&clause.class.as_str()) {
        return None;
    }
    let (last, init) = newtype.params.split_last()?;
    let field = match &newtype.con.fields {
        bhc_hir::ConFields::Positional(tys) => tys.first()?.clone(),
        bhc_hir::ConFields::Named(fields) => fields.first()?.ty.clone(),
    };
    // The field must be `U last` for the class to see through the newtype.
    let Ty::App(wrapped, arg) = field else {
        return None;
    };
    if !matches!(arg.as_ref(), Ty::Var(v) if v.id == last.id) {
        return None;
    }
    let head = context::TyCtxt::build_applied_type(newtype.name, init);
    let mut types = clause.args.clone();
    types.push(head);
    let mut wrapped_types = clause.args.clone();
    wrapped_types.push(*wrapped);
    Some(env::InstanceInfo {
        class: clause.class,
        types,
        context: vec![bhc_types::Constraint::new_multi(
            clause.class,
            wrapped_types,
            newtype.span,
        )],
        methods: rustc_hash::FxHashMap::default(),
        assoc_type_impls: Vec::new(),
    })
}

/// Register the imported classes that have functional dependencies, with all
/// their imported instances, so `apply_fundep_improvement` can use them, and
/// the imported instances of the builtin monad-family classes.
fn register_imported_fundep_classes(ctx: &mut TyCtxt, imported: &ImportedClasses<'_>) {
    let Some(fundeps) = imported.fundeps else {
        return;
    };
    for entry in imported.classes {
        let (name, supers, param_count) = (entry.0, &entry.2, entry.5);
        let Some(class_fundeps) = fundeps.get(&name) else {
            continue;
        };
        if ctx.env.lookup_class(name).is_some() {
            continue;
        }
        // Its methods' constraints are emitted at use sites like a local
        // class's (`instantiate` only emits those of user classes).
        ctx.user_defined_classes.insert(name);
        let params = (0..param_count).map(|_| ctx.fresh_ty_var()).collect();
        ctx.env.register_class(env::ClassInfo {
            name,
            params,
            fundeps: class_fundeps
                .iter()
                .map(|(from, to)| env::FunDep {
                    from: from.clone(),
                    to: to.clone(),
                })
                .collect(),
            supers: supers.clone(),
            methods: rustc_hash::FxHashMap::default(),
            assoc_types: Vec::new(),
        });
    }
    // The monad-family classes have no `ClassInfo` (they are builtins), but
    // their instances for library types must be known: `MonadError e
    // (ExceptT e m)` needs `Monad m`, and at `m = StateT Int IO` only the
    // library's `Monad (StateT s m)` answers.
    const MONAD_FAMILY: &[&str] = &[
        "Functor",
        "Applicative",
        "Monad",
        "MonadFail",
        "Alternative",
        "MonadPlus",
    ];
    for (class, types, _methods, context) in imported.instances {
        if !fundeps.contains_key(class) && !MONAD_FAMILY.contains(&class.as_str()) {
            continue;
        }
        let info = expand_instance_aliases(
            ctx,
            env::InstanceInfo {
                class: *class,
                types: types.clone(),
                context: context.clone(),
                methods: rustc_hash::FxHashMap::default(),
                assoc_type_impls: Vec::new(),
            },
        );
        ctx.env.register_instance(info);
    }
}

/// Expand type synonyms in an instance's head and context: instance heads
/// are matched structurally against constraints whose synonyms the solver
/// has expanded (`MonadError String App` must meet `MonadError [Char] App`).
fn expand_instance_aliases(ctx: &TyCtxt, mut info: env::InstanceInfo) -> env::InstanceInfo {
    for ty in &mut info.types {
        *ty = unify::expand_aliases_deep(ctx, ty);
    }
    for c in &mut info.context {
        for ty in &mut c.args {
            *ty = unify::expand_aliases_deep(ctx, ty);
        }
    }
    info
}

/// The builtin values whose type ends in `IO _` — the IO primitives.
///
/// Codegen's closure IO model (`BHC_IO_MODEL=closure`, spec/BHC-BRIEF-0005
/// Phase 0) needs to know which builtin applications are IO actions, and Core
/// types are too sparse to tell; the builtin schemes say it exactly.
#[must_use]
pub fn io_primitive_names() -> Vec<Symbol> {
    fn result_is_io(ty: &Ty) -> bool {
        match ty {
            Ty::Fun(_, r) => result_is_io(r),
            Ty::Forall(_, body) => result_is_io(body),
            Ty::App(f, _) => matches!(f.as_ref(), Ty::Con(c) if c.name.as_str() == "IO"),
            _ => false,
        }
    }
    let mut ctx = TyCtxt::new(FileId::new(0));
    ctx.register_builtins();
    let mut names = ctx
        .env
        .top_level_names_where(|scheme| result_is_io(&scheme.ty));
    // The builtins typed by name when lowering binds them
    // (`register_lowered_builtins`: `doesFileExist`, `createDirectory`, …),
    // over every name the lowering context predefines.
    let lowered = bhc_lower::LowerContext::with_builtins();
    ctx.register_lowered_builtins(&lowered.defs, &rustc_hash::FxHashSet::default());
    for (id, info) in &lowered.defs {
        if ctx
            .env
            .lookup_global(*id)
            .is_some_and(|scheme| result_is_io(&scheme.ty))
            && !names.contains(&info.name)
        {
            names.push(info.name);
        }
    }
    names
}

/// Type check a HIR module with definition mappings from the lowering pass.
///
/// This function accepts the DefMap from the lowering context, which allows
/// the type checker to register builtins with the correct DefIds assigned
/// during lowering.
///
/// # Arguments
///
/// * `hir` - The HIR module to type check
/// * `file_id` - The file ID for error reporting
/// * `defs` - Optional definition map from the lowering context
pub fn type_check_module_with_defs(
    hir: &Module,
    file_id: FileId,
    defs: Option<&DefMap>,
) -> Result<TypedModule, Vec<Diagnostic>> {
    type_check_module_full(hir, file_id, defs, &[])
}

/// Type check a HIR module with definition mappings and imported type aliases.
///
/// This is the most complete entry point for type checking. In addition to
/// the DefMap, it accepts type aliases from imported modules so the unifier
/// can expand cross-module type aliases transparently.
pub fn type_check_module_full(
    hir: &Module,
    file_id: FileId,
    defs: Option<&DefMap>,
    imported_aliases: &[(Symbol, Vec<bhc_types::TyVar>, Ty)],
) -> Result<TypedModule, Vec<Diagnostic>> {
    type_check_module_with_imports(
        hir,
        file_id,
        defs,
        imported_aliases,
        &ImportedClasses::default(),
    )
}

/// Classes and instances of imported modules, as the lowering pass collected
/// them from the module registry and `.bhi` interfaces.
#[derive(Default)]
pub struct ImportedClasses<'a> {
    /// Imported classes (see `bhc_lower::InterfaceClassEntry`).
    pub classes: &'a [bhc_lower::InterfaceClassEntry],
    /// Functional dependencies of those classes, keyed by class name.
    pub fundeps: Option<&'a rustc_hash::FxHashMap<Symbol, Vec<bhc_lower::ClassFunDep>>>,
    /// Imported instances (see `bhc_lower::InterfaceInstanceEntry`).
    pub instances: &'a [bhc_lower::InterfaceInstanceEntry],
}

/// Type check a HIR module, additionally knowing imported classes.
///
/// Under the closure IO model (spec/BHC-BRIEF-0005) the imported classes that
/// have functional dependencies are registered together with their instances,
/// so a use of an imported method improves its constraint: `get` from the
/// transformers library at `StateT Int IO` fixes its state type to `Int`.
pub fn type_check_module_with_imports(
    hir: &Module,
    file_id: FileId,
    defs: Option<&DefMap>,
    imported_aliases: &[(Symbol, Vec<bhc_types::TyVar>, Ty)],
    imported_classes: &ImportedClasses<'_>,
) -> Result<TypedModule, Vec<Diagnostic>> {
    let mut ctx = TyCtxt::new(file_id);
    ctx.overloaded_strings = hir.overloaded_strings;
    ctx.overloaded_lists = hir.overloaded_lists;
    ctx.scoped_type_variables = hir.scoped_type_variables;

    // Register built-in types
    ctx.register_builtins();

    if io_closure_mode() {
        register_imported_fundep_classes(&mut ctx, imported_classes);
    }

    // If we have definition mappings from the lowering pass, use them
    // to register builtins with the correct DefIds.
    //
    // Names defined by this module's own source (top-level values, class
    // methods, instance methods) shadow Prelude builtins, so they must not
    // receive hardcoded builtin schemes — e.g. a user class method named
    // `combine` is unrelated to System.FilePath's `combine`.
    if let Some(def_map) = defs {
        let local_value_ids: rustc_hash::FxHashSet<DefId> = hir
            .items
            .iter()
            .flat_map(|item| -> Vec<DefId> {
                match item {
                    bhc_hir::Item::Value(vd) => vec![vd.id],
                    bhc_hir::Item::Class(c) => c
                        .methods
                        .iter()
                        .map(|m| m.id)
                        .chain(c.defaults.iter().map(|d| d.id))
                        .collect(),
                    bhc_hir::Item::Instance(i) => i.methods.iter().map(|m| m.id).collect(),
                    _ => vec![],
                }
            })
            .collect();
        ctx.register_lowered_builtins(def_map, &local_value_ids);
    }

    // Register data types from the module
    for item in &hir.items {
        if let bhc_hir::Item::Data(data) = item {
            ctx.register_data_type(data);
        }
        if let bhc_hir::Item::Newtype(newtype) = item {
            ctx.register_newtype(newtype);
        }
    }

    // Register standard Haskell type aliases so the unifier can expand them.
    {
        use bhc_types::{Kind, TyVar};
        let a = TyVar::new_star(0xFFFE_0000);

        // type String = [Char]
        ctx.type_aliases.insert(
            Symbol::intern("String"),
            (
                vec![],
                Ty::List(Box::new(Ty::Con(bhc_types::TyCon::new(
                    Symbol::intern("Char"),
                    Kind::Star,
                )))),
            ),
        );

        // type ShowS = String -> String
        let string_ty = Ty::List(Box::new(Ty::Con(bhc_types::TyCon::new(
            Symbol::intern("Char"),
            Kind::Star,
        ))));
        ctx.type_aliases.insert(
            Symbol::intern("ShowS"),
            (
                vec![],
                Ty::Fun(Box::new(string_ty.clone()), Box::new(string_ty.clone())),
            ),
        );

        // type ReadS a = String -> [(a, String)]
        let pair = Ty::Tuple(vec![Ty::Var(a.clone()), string_ty.clone()]);
        let list_pair = Ty::List(Box::new(pair));
        ctx.type_aliases.insert(
            Symbol::intern("ReadS"),
            (
                vec![a.clone()],
                Ty::Fun(Box::new(string_ty.clone()), Box::new(list_pair)),
            ),
        );

        // type FilePath = String
        ctx.type_aliases
            .insert(Symbol::intern("FilePath"), (vec![], string_ty));

        // Pandoc type aliases (from pandoc-types)
        let text_ty = Ty::Con(bhc_types::TyCon::new(Symbol::intern("Text"), Kind::Star));

        // type Attr = (Text, [Text], [(Text, Text)])
        // Pandoc's type alias (XML.Light has a data type Attr, but that uses
        // a different mechanism — data types vs type aliases don't conflict).
        ctx.type_aliases.insert(
            Symbol::intern("Attr"),
            (
                vec![],
                Ty::Tuple(vec![
                    text_ty.clone(),
                    Ty::List(Box::new(text_ty.clone())),
                    Ty::List(Box::new(Ty::Tuple(vec![text_ty.clone(), text_ty.clone()]))),
                ]),
            ),
        );

        // type Target = (Text, Text)
        ctx.type_aliases.insert(
            Symbol::intern("Target"),
            (vec![], Ty::Tuple(vec![text_ty.clone(), text_ty.clone()])),
        );

        // TagSoup: type Attribute str = (str, str). Left opaque it clashes
        // with the tuples `lookup` produces over attribute lists (Readers.HTML
        // `pSelfClosing (== "img") (isJust . lookup "src")`).
        ctx.type_aliases.insert(
            Symbol::intern("Attribute"),
            (
                vec![a.clone()],
                Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(a.clone())]),
            ),
        );

        // blaze-html: type Markup = MarkupM (); type Html = Markup. Left
        // opaque, `Html` can't unify with the `m a` that do-notation over
        // markup infers (all 77 unmasked Writers.HTML errors were
        // `expected Html, found (t t)` pairs).
        {
            let markupm_con = Ty::Con(bhc_types::TyCon::new(
                Symbol::intern("MarkupM"),
                Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)),
            ));
            let markup_unit = Ty::App(Box::new(markupm_con), Box::new(Ty::unit()));
            ctx.type_aliases
                .insert(Symbol::intern("Markup"), (vec![], markup_unit.clone()));
            ctx.type_aliases
                .insert(Symbol::intern("Html"), (vec![], markup_unit));
        }

        // Approximate `NonEmpty a` as `[a]`. BHC already models the `:|`
        // constructor and the `Data.List.NonEmpty` operations as list-valued
        // (see context.rs `":|"`), so the *type* must agree — otherwise a
        // signature like `multicolumnDescriptor :: … -> NonEmpty ColWidth -> …`
        // stays a distinct `NonEmpty` con and clashes with the list values it
        // receives (`expected NonEmpty, found []`). A real NonEmpty type would
        // need Foldable-generic reductions; this keeps the approximation coherent.
        ctx.type_aliases.insert(
            Symbol::intern("NonEmpty"),
            (vec![a.clone()], Ty::List(Box::new(Ty::Var(a.clone())))),
        );

        // Approximate `Seq a` as `[a]`, for the same reason as NonEmpty: the
        // Data.Sequence operations are permissive stubs, and OverloadedLists
        // code matches Seq values with LIST patterns (Readers.Typst.Math's
        // `(base :: Seq Content)` against `[Elt "math.op" _ fs]`).
        ctx.type_aliases.insert(
            Symbol::intern("Seq"),
            (vec![a.clone()], Ty::List(Box::new(Ty::Var(a.clone())))),
        );

        // type Parsec s u = ParsecT s u Identity
        // The parsec monad specialized to the Identity base functor. Without
        // this alias `Parsec Sources st a` (e.g. the argument of
        // `Text.Pandoc.Parsing.General`'s `readWith`) stays a distinct head and
        // won't unify with `ParsecT Sources st Identity a` (what `readWithM`
        // produces at `m = Identity`). Two params; the trailing result arg is
        // applied to the expansion (`Parsec s u a` → `ParsecT s u Identity a`).
        {
            let s = TyVar::new_star(0xFFFE_0001);
            let u = TyVar::new_star(0xFFFE_0002);
            let star_to_star = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
            // ParsecT :: * -> * -> (* -> *) -> * -> *
            let parsect_kind = Kind::Arrow(
                Box::new(Kind::Star),
                Box::new(Kind::Arrow(
                    Box::new(Kind::Star),
                    Box::new(Kind::Arrow(
                        Box::new(star_to_star.clone()),
                        Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
                    )),
                )),
            );
            let parsect_con = Ty::Con(bhc_types::TyCon::new(
                Symbol::intern("ParsecT"),
                parsect_kind,
            ));
            let identity_con = Ty::Con(bhc_types::TyCon::new(
                Symbol::intern("Identity"),
                star_to_star,
            ));
            // ParsecT s u Identity
            let expansion = Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::App(Box::new(parsect_con), Box::new(Ty::Var(s.clone())))),
                    Box::new(Ty::Var(u.clone())),
                )),
                Box::new(identity_con),
            );
            ctx.type_aliases
                .insert(Symbol::intern("Parsec"), (vec![s, u], expansion));
        }

        // skylighting type aliases (external, stubbed opaquely otherwise):
        //   type SyntaxMap  = Map Text Syntax
        //   type Token      = (TokenType, Text)
        //   type SourceLine = [Token]
        let map_con = Ty::Con(bhc_types::TyCon::new(
            Symbol::intern("Map"),
            Kind::Arrow(
                Box::new(Kind::Star),
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
            ),
        ));
        let syntax_ty = Ty::Con(bhc_types::TyCon::new(Symbol::intern("Syntax"), Kind::Star));
        ctx.type_aliases.insert(
            Symbol::intern("SyntaxMap"),
            (
                vec![],
                Ty::App(
                    Box::new(Ty::App(Box::new(map_con), Box::new(text_ty.clone()))),
                    Box::new(syntax_ty),
                ),
            ),
        );
        let token_tuple = Ty::Tuple(vec![
            Ty::Con(bhc_types::TyCon::new(
                Symbol::intern("TokenType"),
                Kind::Star,
            )),
            text_ty.clone(),
        ]);
        ctx.type_aliases
            .insert(Symbol::intern("Token"), (vec![], token_tuple.clone()));
        ctx.type_aliases.insert(
            Symbol::intern("SourceLine"),
            (vec![], Ty::List(Box::new(token_tuple))),
        );

        // type ColSpec = (Alignment, ColWidth)
        ctx.type_aliases.insert(
            Symbol::intern("ColSpec"),
            (
                vec![],
                Ty::Tuple(vec![
                    Ty::Con(bhc_types::TyCon::new(
                        Symbol::intern("Alignment"),
                        Kind::Star,
                    )),
                    Ty::Con(bhc_types::TyCon::new(
                        Symbol::intern("ColWidth"),
                        Kind::Star,
                    )),
                ]),
            ),
        );

        // type ListAttributes = (Int, ListNumberStyle, ListNumberDelim)
        ctx.type_aliases.insert(
            Symbol::intern("ListAttributes"),
            (
                vec![],
                Ty::Tuple(vec![
                    Ty::Con(bhc_types::TyCon::new(Symbol::intern("Int"), Kind::Star)),
                    Ty::Con(bhc_types::TyCon::new(
                        Symbol::intern("ListNumberStyle"),
                        Kind::Star,
                    )),
                    Ty::Con(bhc_types::TyCon::new(
                        Symbol::intern("ListNumberDelim"),
                        Kind::Star,
                    )),
                ]),
            ),
        );

        // type ShortCaption = [Inline]
        ctx.type_aliases.insert(
            Symbol::intern("ShortCaption"),
            (
                vec![],
                Ty::List(Box::new(Ty::Con(bhc_types::TyCon::new(
                    Symbol::intern("Inline"),
                    Kind::Star,
                )))),
            ),
        );

        // Many is a newtype: newtype Many a = Many (Seq a)
        // But for type-checking purposes, we treat Blocks/Inlines as [Block]/[Inline]
        // since BHC doesn't have a real Seq type. The Many newtype is isomorphic to lists
        // in our simplified model.
        let _many_con = bhc_types::TyCon::new(
            Symbol::intern("Many"),
            Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)),
        );
        let a_alias = TyVar::new_star(0xFFFE_0001);

        // type Blocks = Many Block  (treated as [Block])
        let block_con = bhc_types::TyCon::new(Symbol::intern("Block"), Kind::Star);
        ctx.type_aliases.insert(
            Symbol::intern("Blocks"),
            (vec![], Ty::List(Box::new(Ty::Con(block_con)))),
        );

        // type Inlines = Many Inline  (treated as [Inline])
        let inline_con = bhc_types::TyCon::new(Symbol::intern("Inline"), Kind::Star);
        ctx.type_aliases.insert(
            Symbol::intern("Inlines"),
            (vec![], Ty::List(Box::new(Ty::Con(inline_con)))),
        );

        // type Many a = [a] (simplification: treat Many as a list)
        ctx.type_aliases.insert(
            Symbol::intern("Many"),
            (vec![a_alias.clone()], Ty::List(Box::new(Ty::Var(a_alias)))),
        );
    }

    // Register type aliases from imported modules (cross-module propagation).
    // The caller scopes `imported_aliases` to this module's actual imports, so a
    // module-local alias from an unrelated module can't leak in here.
    for (name, params, ty) in imported_aliases {
        ctx.type_aliases.insert(*name, (params.clone(), ty.clone()));
    }

    // Register user-defined type aliases so the unifier can expand them.
    for item in &hir.items {
        if let bhc_hir::Item::TypeAlias(alias) = item {
            ctx.type_aliases
                .insert(alias.name, (alias.params.clone(), alias.ty.clone()));
        }
    }

    // The built-in `Many a = [a]` (with `Inlines`/`Blocks` as lists) only
    // stands in for a STUBBED pandoc-types Builder. With the real `Many` in
    // scope — declared here, or `Inlines` imported from the compiled Builder —
    // it is a newtype over `Seq`, and the list reading resolved its `<>` and
    // `mempty` to `Data.List.append`/`[]` applied to a Seq.
    let declares_many = hir
        .items
        .iter()
        .any(|item| matches!(item, bhc_hir::Item::Newtype(nt) if nt.name.as_str() == "Many"));
    let imports_many = imported_aliases
        .iter()
        .any(|(name, _, ty)| name.as_str() == "Inlines" && !matches!(ty, Ty::List(_)));
    if declares_many || imports_many {
        ctx.type_aliases.remove(&Symbol::intern("Many"));
        for name in ["Inlines", "Blocks"] {
            let sym = Symbol::intern(name);
            if matches!(ctx.type_aliases.get(&sym), Some((_, Ty::List(_)))) {
                ctx.type_aliases.remove(&sym);
            }
        }
    }

    // Register derived instances so the type checker knows about them.
    // Derived instances come from `deriving (Eq, Ord, Show, ...)` clauses
    // on data type and newtype declarations.
    for item in &hir.items {
        match item {
            bhc_hir::Item::Data(data) => {
                for clause in &data.deriving {
                    let instance_type =
                        context::TyCtxt::build_applied_type(data.name, &data.params);
                    let info = env::InstanceInfo {
                        class: clause.class,
                        types: vec![instance_type],
                        context: vec![],
                        methods: rustc_hash::FxHashMap::default(),
                        assoc_type_impls: vec![],
                    };
                    ctx.env.register_instance(info);
                }
            }
            bhc_hir::Item::Newtype(newtype) => {
                for clause in &newtype.deriving {
                    if let Some(info) = newtype_derived_instance(newtype, clause) {
                        let info = expand_instance_aliases(&ctx, info);
                        ctx.env.register_instance(info);
                        continue;
                    }
                    let instance_type =
                        context::TyCtxt::build_applied_type(newtype.name, &newtype.params);
                    let info = env::InstanceInfo {
                        class: clause.class,
                        types: vec![instance_type],
                        context: vec![],
                        methods: rustc_hash::FxHashMap::default(),
                        assoc_type_impls: vec![],
                    };
                    ctx.env.register_instance(info);
                }
            }
            _ => {}
        }
    }

    // Register type classes and instances
    for item in &hir.items {
        if let bhc_hir::Item::Class(class) = item {
            ctx.register_class(class);
        }
    }
    for item in &hir.items {
        if let bhc_hir::Item::Instance(instance) = item {
            ctx.register_instance(instance);
        }
    }

    // Register standalone type families
    for item in &hir.items {
        if let bhc_hir::Item::TypeFamily(tf) = item {
            ctx.register_type_family(tf);
        }
    }
    for item in &hir.items {
        if let bhc_hir::Item::TypeFamilyInst(inst) = item {
            ctx.register_type_family_instance(inst);
        }
    }

    // Register data families and their instances
    for item in &hir.items {
        if let bhc_hir::Item::DataFamily(df) = item {
            ctx.register_data_family(df);
        }
    }
    for item in &hir.items {
        if let bhc_hir::Item::DataFamilyInst(inst) = item {
            ctx.register_data_family_instance(inst);
        }
    }

    // Compute binding groups (SCCs) for mutual recursion
    let groups = binding_groups::compute_binding_groups(&hir.items);

    // Type check each binding group
    for group in groups {
        ctx.check_binding_group(&group);
    }

    // Solve type class constraints (defaults ambiguous type variables)
    ctx.solve_constraints();

    if ctx.has_errors() {
        Err(ctx.take_diagnostics())
    } else {
        Ok(ctx.into_typed_module(hir.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_module() {
        use bhc_intern::Symbol;

        let module = Module {
            name: Symbol::intern("Test"),
            exports: None,
            imports: Vec::new(),
            items: Vec::new(),
            span: bhc_span::Span::DUMMY,
            overloaded_strings: false,
            scoped_type_variables: false,
            generalized_newtype_deriving: false,
            flexible_instances: false,
            flexible_contexts: false,
            gadts: false,
            strict_data: false,
            overloaded_lists: false,
        };

        let result = type_check_module(&module, FileId::new(0));
        assert!(result.is_ok());
    }
}
