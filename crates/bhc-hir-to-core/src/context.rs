//! Lowering context for HIR to Core transformation.
//!
//! The `LowerContext` tracks state during the lowering process, including:
//! - Fresh variable generation
//! - Error collection
//! - Type environment
//! - Constructor metadata for ADTs

use bhc_core::{self as core, Bind, CoreConstructor, CoreModule, Var, VarId};
use bhc_hir::{DefId, Item, Module as HirModule, ValueDef};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Kind, Scheme, Ty, TyCon, TyVar};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::deriving::{DerivedInstance, DerivingContext};
use crate::dictionary::{ClassInfo, ClassRegistry, DictContext, InstanceInfo};

/// Metadata about a data constructor.
///
/// This stores information needed to generate correct pattern matching code,
/// particularly the constructor's tag (position within its data type).
#[derive(Clone, Debug)]
pub struct ConstructorInfo {
    /// The name of the constructor.
    pub name: Symbol,
    /// The name of the data type this constructor belongs to.
    pub type_name: Symbol,
    /// The constructor's tag (0-based index within the data type's constructor list).
    pub tag: u32,
    /// The number of fields this constructor has.
    pub arity: u32,
    /// Field names for record constructors (in canonical order).
    /// Empty for positional constructors.
    pub field_names: Vec<Symbol>,
    /// Whether this constructor is a newtype constructor (identity at runtime).
    pub is_newtype: bool,
    /// Number of existential dictionary fields prepended to the constructor.
    /// These are implicit fields that carry typeclass dictionaries for
    /// existential type variables (e.g., `forall a. C a => MkT a` has 1 dict field).
    pub existential_dict_count: u32,
    /// Class names for existential dictionary fields (in order).
    pub existential_classes: Vec<Symbol>,
}

/// Metadata about a record field selector function.
///
/// This stores information needed to generate field access code.
#[derive(Clone, Debug)]
pub struct FieldSelectorInfo {
    /// The field name.
    pub field_name: Symbol,
    /// The constructor's `DefId`.
    pub con_id: DefId,
    /// The constructor's name.
    pub con_name: Symbol,
    /// The data type name.
    pub type_name: Symbol,
    /// The field's index within the constructor (0-based).
    pub field_index: usize,
    /// The total number of fields in the constructor.
    pub total_fields: usize,
}

use crate::expr::lower_expr;
use crate::{LowerError, LowerResult, TypeSchemeMap};

/// Context for the HIR to Core lowering pass.
pub struct LowerContext {
    /// Counter for generating fresh variable names.
    fresh_counter: u32,

    /// Mapping from HIR `DefIds` to Core variables.
    var_map: FxHashMap<DefId, Var>,

    /// Type schemes from the type checker (`DefId` -> Scheme).
    type_schemes: TypeSchemeMap,

    /// Per-HIR-node inferred types from the type checker, keyed by source `Span`
    /// (spec/BHC-BRIEF-0002). Threaded in by the driver and populated by typeck's
    /// `infer_expr` (Path A). Read via `expr_ty_opt(span)` so lowering can give
    /// Core nodes real types; ~100% coverage measured on real-expression nodes.
    expr_types: crate::ExprTypeMap,

    /// Fixpoint-resolved per-node types, a dispatch-only side channel keyed by
    /// `Span`. Read via `resolved_expr_ty_opt(span)` to choose which method to
    /// call at a user-defined monad; never used to type a Core node. Empty
    /// unless threaded from `TypedModule::resolved_expr_types`.
    resolved_expr_types: crate::ExprTypeMap,

    /// The type a CALLER expects a sub-expression to have, keyed by `Span`.
    ///
    /// Typeck records an occurrence's type as it inferred it, and for a parser
    /// nested inside another argument that is a row of bare variables:
    /// `readWithM (try (id pB))` leaves `pB` at `ParsecT ?s ?u ?m Int`, so its
    /// own `Stream ?s ?m Char` constraint resolves to nothing and the slot
    /// becomes a null placeholder the parser is then run through. The concrete
    /// stream type is only known one level out, at `readWithM`'s parameter —
    /// `lower_value_arg` pushes it inward here, so the leaf can refine what
    /// typeck recorded. Read via `expected_ty_opt`; a hint never replaces a
    /// recorded type, it only substitutes that type's variables.
    expected_arg_tys: crate::ExprTypeMap,

    /// Constructor metadata (`DefId` -> `ConstructorInfo`).
    /// This maps constructor `DefIds` to their metadata including tag and type.
    constructor_map: FxHashMap<DefId, ConstructorInfo>,
    /// Declared field types per constructor name, from user data declarations.
    /// Threaded into `CoreConstructor::field_types` for codegen `show` dispatch.
    constructor_field_types: FxHashMap<Symbol, Vec<Ty>>,

    /// Field selector metadata (field name -> `FieldSelectorInfo`).
    /// This maps field names to their selector information for generating field access code.
    field_selector_map: FxHashMap<Symbol, FieldSelectorInfo>,

    /// Stack of in-scope dictionary variables.
    ///
    /// When lowering a constrained function like `f :: Num a => a -> a`,
    /// we push the dictionary variable `$dNum` onto this stack before lowering
    /// the body. When we encounter a reference to another constrained function
    /// that requires the same constraint, we can look up the dictionary here.
    ///
    /// Each entry maps constraint class names to their dictionary variables.
    dict_scope: Vec<FxHashMap<Symbol, Var>>,
    /// The constrained TYPE each in-scope dictionary is for, where known.
    ///
    /// `dict_scope` is keyed by CLASS ALONE, so a lookup happily returns a
    /// dictionary for a DIFFERENT type. Harmless while at most one dictionary
    /// per class is ever in scope, but a binding constrained over its own monad
    /// puts a `Monad m` dictionary in scope for someone else's INNER monad —
    /// parsec's `getPosition :: Monad m => ParsecT s u m SourcePos` then ran its
    /// do-block's `>>=` out of that dictionary instead of ParsecT's, and every
    /// parser started from a garbage state. Kept in lockstep with `dict_scope`.
    dict_scope_ty: Vec<FxHashMap<Symbol, Vec<Ty>>>,

    /// Registry of type classes and instances for dictionary construction.
    class_registry: ClassRegistry,

    /// Accumulated errors.
    errors: Vec<LowerError>,

    /// Accumulated warnings (non-fatal diagnostics).
    warnings: Vec<String>,

    /// Whether `GeneralizedNewtypeDeriving` is enabled for the current module.
    pub generalized_newtype_deriving: bool,

    /// Collected foreign import declarations for the Core module.
    foreign_imports: Vec<core::ForeignImport>,

    /// Stack of monad types for resolving return/pure in do-notation lambdas.
    /// Pushed when lowering >>=/>>'s lambda argument for non-builtin monads.
    monad_type_stack: Vec<Ty>,

    /// The type of the instance whose method bodies are currently being lowered
    /// (e.g. `ParsecT s u m` for `instance Alternative (ParsecT s u m)`). Set
    /// while lowering instance methods so a bare monad-family method used as a
    /// value — a point-free `(<|>) = mplus` — resolves to that type's instance
    /// method instead of stubbing as a builtin.
    current_instance_type: Option<Ty>,
    /// The class of the instance whose method bodies are being lowered
    /// (paired with `current_instance_type`).
    current_instance_class: Option<Symbol>,
    /// The declared scheme TYPE of the top-level binding currently being
    /// lowered. An occurrence's typeck-recorded type inside a signed binding
    /// can keep the signature's instantiation VARIABLES (`optional (char 'z')`
    /// inside `poly :: Monad m => ParsecT String () m SourcePos` records
    /// `ParsecT s u m (Maybe Char)`); matching the occurrence's result against
    /// this signature's result pins them (`s := [Char]`, `u := ()`), so the
    /// Stream dictionary resolves instead of being silently omitted.
    current_binding_sig: Option<Ty>,

    /// Head type constructors of instances that arrived through a module
    /// interface, keyed by class — `(Monad, ParsecT)` when compiling a module
    /// that *imports* parsec.
    ///
    /// This distinguishes a use site from the instance's own implementation.
    /// Dispatching a method inside the module that defines the instance would
    /// rewrite the generic implementation into a call to itself; only a
    /// consumer may dispatch. See the signature fallback in
    /// `lower_app`'s monad-family case.
    imported_instance_heads: FxHashSet<(Symbol, Symbol)>,
    /// Type-synonym definitions (name -> (params, rhs)), local + imported,
    /// threaded from typeck. Declared signatures keep synonyms unexpanded
    /// (`MarkdownParser m a`); dictionary resolution expands them before
    /// matching against constructor-shaped occurrence types.
    type_aliases: FxHashMap<Symbol, (Vec<bhc_types::TyVar>, Ty)>,
    /// Depth of superclass-dispatched bind operand lowering (see
    /// `select_method_via_superclass` and expr.rs Case 3a').
    superclass_bind_depth: usize,

    /// Pre-created existential dict binder variables.
    /// Set before lowering a case alternative RHS so that pattern lowering
    /// can reuse the same vars (instead of creating different fresh ones).
    pub(crate) existential_dict_binders: Vec<Var>,
}

impl LowerContext {
    /// Create a new lowering context.
    #[must_use]
    pub fn new() -> Self {
        let mut ctx = Self {
            // Fresh VarIds share one namespace with DefId-derived VarIds
            // (`VarId::new(def_id.index())`), the deriving context (50000+),
            // and RTS builtins (1_000_000+). Starting at 100 collided with
            // DefIds in any module with >100 defs: two distinct variables got
            // the same VarId, and codegen's env lookups crossed function
            // boundaries ("Referring to an argument in another function" —
            // ImageSize's pWebpSize captured webpSize's argument). 200_000 is
            // above any realistic DefId count (~11k today) and deriving usage,
            // and below the builtin range.
            fresh_counter: 200_000,
            var_map: FxHashMap::default(),
            type_schemes: FxHashMap::default(),
            expr_types: FxHashMap::default(),
            resolved_expr_types: FxHashMap::default(),
            expected_arg_tys: FxHashMap::default(),
            constructor_map: FxHashMap::default(),
            constructor_field_types: FxHashMap::default(),
            field_selector_map: FxHashMap::default(),
            dict_scope: vec![FxHashMap::default()], // Start with empty root scope
            dict_scope_ty: vec![FxHashMap::default()],
            class_registry: ClassRegistry::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            generalized_newtype_deriving: false,
            foreign_imports: Vec::new(),
            monad_type_stack: Vec::new(),
            current_instance_type: None,
            current_binding_sig: None,
            imported_instance_heads: FxHashSet::default(),
            type_aliases: FxHashMap::default(),
            current_instance_class: None,
            superclass_bind_depth: 0,
            existential_dict_binders: Vec::new(),
        };
        ctx.register_builtins();
        ctx.register_builtin_constructors();
        ctx.register_builtin_classes();
        ctx
    }

    /// Set the type schemes from the type checker.
    pub fn set_type_schemes(&mut self, schemes: TypeSchemeMap) {
        self.type_schemes = schemes;
    }

    /// Set the per-HIR-node inferred types from the type checker
    /// (spec/BHC-BRIEF-0002).
    pub fn set_expr_types(&mut self, expr_types: crate::ExprTypeMap) {
        self.expr_types = expr_types;
    }

    /// Set the fixpoint-resolved per-node types (dispatch-only side channel).
    /// See `resolved_expr_ty_opt` (the private reader this map feeds).
    pub fn set_resolved_expr_types(&mut self, resolved: crate::ExprTypeMap) {
        self.resolved_expr_types = resolved;
    }

    /// Look up a HIR expression's inferred type by its source `Span`, returning
    /// `None` if absent (spec/BHC-BRIEF-0002). Populated by typeck as of Path A;
    /// used to give Core nodes real types instead of `Ty::Error`.
    #[must_use]
    pub(crate) fn expr_ty_opt(&self, span: bhc_span::Span) -> Option<Ty> {
        self.expr_types.get(&span).cloned()
    }

    /// Like [`expr_ty_opt`](Self::expr_ty_opt), but from the fixpoint-resolved
    /// map (`TypedModule::resolved_expr_types`). Use this ONLY to decide which
    /// method/instance to dispatch to (e.g. `return`/`pure` at a user-defined
    /// monad whose type a single-pass apply leaves partially unresolved) — never
    /// to type a Core node, since the fuller resolution regresses codegen width
    /// inference. Falls back to `expr_ty_opt` when the resolved map is empty
    /// (e.g. lowering paths that don't thread it).
    #[must_use]
    pub(crate) fn resolved_expr_ty_opt(&self, span: bhc_span::Span) -> Option<Ty> {
        self.resolved_expr_types
            .get(&span)
            .cloned()
            .or_else(|| self.expr_ty_opt(span))
    }

    /// Record the type a caller expects the expression at `span` to have. See
    /// [`expected_arg_tys`](Self::expected_arg_tys).
    pub(crate) fn record_expected_ty(&mut self, span: bhc_span::Span, ty: Ty) {
        self.expected_arg_tys.insert(span, ty);
    }

    /// The caller's expected type for the expression at `span`, if one was
    /// pushed inward.
    #[must_use]
    pub(crate) fn expected_ty_opt(&self, span: bhc_span::Span) -> Option<Ty> {
        self.expected_arg_tys.get(&span).cloned()
    }

    /// Look up the type for a definition from the type checker.
    ///
    /// Returns the monomorphic type from the scheme, or `Ty::Error` if not found.
    #[must_use]
    pub fn lookup_type(&self, def_id: DefId) -> Ty {
        self.type_schemes
            .get(&def_id)
            .map_or(Ty::Error, |scheme| scheme.ty.clone())
    }

    /// Look up the full type scheme for a definition, including constraints.
    ///
    /// Returns the complete scheme if found, or None if not found.
    #[must_use]
    pub fn lookup_scheme(&self, def_id: DefId) -> Option<&Scheme> {
        self.type_schemes.get(&def_id)
    }

    /// Register builtin constructor metadata.
    ///
    /// This sets up the constructor tags for builtin types (Bool, Maybe, Either, etc.)
    /// so pattern matching generates correct code.
    fn register_builtin_constructors(&mut self) {
        // Bool: False = 0, True = 1
        let bool_sym = Symbol::intern("Bool");
        self.constructor_map.insert(
            DefId::new(9),
            ConstructorInfo {
                name: Symbol::intern("True"),
                type_name: bool_sym,
                tag: 1,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(10),
            ConstructorInfo {
                name: Symbol::intern("False"),
                type_name: bool_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        // Maybe: Nothing = 0, Just = 1
        let maybe_sym = Symbol::intern("Maybe");
        self.constructor_map.insert(
            DefId::new(11),
            ConstructorInfo {
                name: Symbol::intern("Nothing"),
                type_name: maybe_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(12),
            ConstructorInfo {
                name: Symbol::intern("Just"),
                type_name: maybe_sym,
                tag: 1,
                arity: 1,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        // Either: Left = 0, Right = 1
        let either_sym = Symbol::intern("Either");
        self.constructor_map.insert(
            DefId::new(13),
            ConstructorInfo {
                name: Symbol::intern("Left"),
                type_name: either_sym,
                tag: 0,
                arity: 1,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(14),
            ConstructorInfo {
                name: Symbol::intern("Right"),
                type_name: either_sym,
                tag: 1,
                arity: 1,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        // List: [] = 0, : = 1
        let list_sym = Symbol::intern("List");
        self.constructor_map.insert(
            DefId::new(15),
            ConstructorInfo {
                name: Symbol::intern("[]"),
                type_name: list_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(16),
            ConstructorInfo {
                name: Symbol::intern(":"),
                type_name: list_sym,
                tag: 1,
                arity: 2,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        // Unit: () = 0
        let unit_sym = Symbol::intern("Unit");
        self.constructor_map.insert(
            DefId::new(17),
            ConstructorInfo {
                name: Symbol::intern("()"),
                type_name: unit_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        // GHC.Generics representation constructors (DefIds 12410-12415)
        let sum_sym = Symbol::intern(":+:");
        self.constructor_map.insert(
            DefId::new(12410),
            ConstructorInfo {
                name: Symbol::intern("U1"),
                type_name: Symbol::intern("U1"),
                tag: 0,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(12411),
            ConstructorInfo {
                name: Symbol::intern("K1"),
                type_name: Symbol::intern("K1"),
                tag: 0,
                arity: 1,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(12412),
            ConstructorInfo {
                name: Symbol::intern("M1"),
                type_name: Symbol::intern("M1"),
                tag: 0,
                arity: 1,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(12413),
            ConstructorInfo {
                name: Symbol::intern("L1"),
                type_name: sum_sym,
                tag: 0,
                arity: 1,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(12414),
            ConstructorInfo {
                name: Symbol::intern("R1"),
                type_name: sum_sym,
                tag: 1,
                arity: 1,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(12415),
            ConstructorInfo {
                name: Symbol::intern(":*:"),
                type_name: Symbol::intern(":*:"),
                tag: 0,
                arity: 2,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
    }

    /// Register builtin operators and constructors.
    ///
    /// `DefIds` must match the allocation order in bhc-lower and bhc-typeck.
    fn register_builtins(&mut self) {
        // DefIds 0-8: Types (not values, skip)
        // DefIds 9-14: Data constructors (True, False, Nothing, Just, Left, Right)
        let constructors = [
            (9, "True"),
            (10, "False"),
            (11, "Nothing"),
            (12, "Just"),
            (13, "Left"),
            (14, "Right"),
            (15, "[]"),
            (16, ":"),
            (17, "()"),
        ];

        for (id, name) in constructors {
            let def_id = DefId::new(id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(id),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_id, var);
        }

        // GHC.Generics representation constructors (fixed DefIds 12410-12415)
        let generics_cons = [
            (12410, "U1"),
            (12411, "K1"),
            (12412, "M1"),
            (12413, "L1"),
            (12414, "R1"),
            (12415, ":*:"),
        ];
        for (id, name) in generics_cons {
            let def_id = DefId::new(id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // DefIds 18+: Operators and functions
        // Order must match bhc-lower/src/context.rs define_builtins
        let operators = [
            // Arithmetic operators (18-26)
            "+",
            "-",
            "*",
            "/",
            "div",
            "mod",
            "^",
            "^^",
            "**",
            // Comparison operators (27-32)
            "==",
            "/=",
            "<",
            "<=",
            ">",
            ">=",
            // Boolean operators (33-34)
            "&&",
            "||",
            // List operators (35-37)
            ":",
            "append",
            "!!",
            // Function composition (38-39)
            ".",
            "$",
            // Monadic operators (40-41)
            ">>=",
            ">>",
            // Applicative operators (42-45)
            "<*>",
            "<$>",
            "*>",
            "<*",
            // Alternative operator (46)
            "<|>",
            // Monadic operations (47-48)
            "return",
            "pure",
            // List operations (49-62)
            "map",
            "filter",
            "foldr",
            "foldl",
            "foldl'",
            "concatMap",
            "head",
            "tail",
            "length",
            "null",
            "reverse",
            "take",
            "drop",
            "elem",
            // More list operations (63-70)
            "sum",
            "product",
            "and",
            "or",
            "any",
            "all",
            "maximum",
            "minimum",
            // Zip operations (71-72)
            "zip",
            "zipWith",
            // Prelude functions (73-79)
            "id",
            "const",
            "flip",
            "error",
            "undefined",
            "seq",
            // Numeric operations (80-88)
            "fromInteger",
            "fromRational",
            "negate",
            "abs",
            "signum",
            "sqrt",
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            // Comparison (89-90)
            "compare",
            "min",
            "max",
            // Show (91)
            "show",
            // Boolean (92-93)
            "not",
            "otherwise",
        ];

        // Start after constructors (id 18).
        for (offset, name) in operators.into_iter().enumerate() {
            let id = 18 + offset;
            let def_id = DefId::new(id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(id),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_id, var);
        }

        // IO monad method implementations (DefIds 150-154)
        // These are referenced by dictionary construction for Functor/Applicative/Monad IO
        let io_methods: [(usize, &str); 5] = [
            (150, "fmap"),
            (151, "pure"),
            (152, "<*>"),
            (153, ">>="),
            (154, ">>"),
        ];
        for (method_id, name) in io_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // Identity type + methods (DefIds 10000-10006)
        let identity_methods: [(usize, &str); 7] = [
            (10000, "Identity"),
            (10001, "runIdentity"),
            (10002, "Identity.fmap"),
            (10003, "Identity.pure"),
            (10004, "Identity.<*>"),
            (10005, "Identity.>>="),
            (10006, "Identity.>>"),
        ];
        for (method_id, name) in identity_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // MonadTrans/MonadIO class methods + IO MonadIO instance (DefIds 10010-10012)
        let class_methods: [(usize, &str); 3] =
            [(10010, "lift"), (10011, "liftIO"), (10012, "IO.liftIO")];
        for (method_id, name) in class_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // ReaderT type + instances + operations (DefIds 10020-10031)
        let reader_t_methods: [(usize, &str); 12] = [
            (10020, "ReaderT"),
            (10021, "runReaderT"),
            (10022, "ReaderT.fmap"),
            (10023, "ReaderT.pure"),
            (10024, "ReaderT.<*>"),
            (10025, "ReaderT.>>="),
            (10026, "ReaderT.>>"),
            (10027, "ReaderT.lift"),
            (10028, "ReaderT.liftIO"),
            (10029, "ask"),
            (10030, "asks"),
            (10031, "local"),
        ];
        for (method_id, name) in reader_t_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // StateT type + instances + operations (DefIds 10040-10055)
        let state_t_methods: [(usize, &str); 15] = [
            (10040, "StateT"),
            (10041, "runStateT"),
            (10042, "StateT.fmap"),
            (10043, "StateT.pure"),
            (10044, "StateT.<*>"),
            (10045, "StateT.>>="),
            (10046, "StateT.>>"),
            (10047, "StateT.lift"),
            (10048, "StateT.liftIO"),
            (10049, "get"),
            (10050, "put"),
            (10051, "modify"),
            (10053, "gets"),
            (10054, "evalStateT"),
            (10055, "execStateT"),
        ];
        for (method_id, name) in state_t_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }
    }

    /// Register built-in type classes and their instances.
    ///
    /// This sets up the type class hierarchy (Eq, Ord, Num, etc.) and
    /// registers instances for built-in types (Int, Float, Bool, Char).
    fn register_builtin_classes(&mut self) {
        // Helper to create a type constructor
        let make_ty = |name: &str| -> Ty { Ty::Con(TyCon::new(Symbol::intern(name), Kind::Star)) };

        // === Register Eq class ===
        // Methods: == (/=)
        // DefIds: == is 27, /= is 28
        let eq_class = ClassInfo {
            name: Symbol::intern("Eq"),
            param_count: 1,
            methods: vec![Symbol::intern("=="), Symbol::intern("/=")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(eq_class);

        // === Register Ord class ===
        // Methods: compare, <, <=, >, >=, min, max
        // DefIds: < is 29, <= is 30, > is 31, >= is 32, compare is 89, min is 90, max is 91
        let ord_class = ClassInfo {
            name: Symbol::intern("Ord"),
            param_count: 1,
            methods: vec![
                Symbol::intern("compare"),
                Symbol::intern("<"),
                Symbol::intern("<="),
                Symbol::intern(">"),
                Symbol::intern(">="),
                Symbol::intern("min"),
                Symbol::intern("max"),
            ],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Eq")],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(ord_class);

        // === Register Num class ===
        // Methods: +, -, *, negate, abs, signum, fromInteger
        // DefIds: + is 18, - is 19, * is 20, negate is 82, abs is 83, signum is 84, fromInteger is 80
        let num_class = ClassInfo {
            name: Symbol::intern("Num"),
            param_count: 1,
            methods: vec![
                Symbol::intern("+"),
                Symbol::intern("-"),
                Symbol::intern("*"),
                Symbol::intern("negate"),
                Symbol::intern("abs"),
                Symbol::intern("signum"),
                Symbol::intern("fromInteger"),
            ],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(num_class);

        // === Register Fractional class ===
        // Methods: /, recip, fromRational
        // DefIds: / is 21, fromRational is 81
        let fractional_class = ClassInfo {
            name: Symbol::intern("Fractional"),
            param_count: 1,
            methods: vec![
                Symbol::intern("/"),
                Symbol::intern("recip"),
                Symbol::intern("fromRational"),
            ],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Num")],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(fractional_class);

        // === Register Show class ===
        // Methods: show
        // DefIds: show is 92
        let show_class = ClassInfo {
            name: Symbol::intern("Show"),
            param_count: 1,
            methods: vec![Symbol::intern("show")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(show_class);

        // === Register instances for Int ===
        let int_ty = make_ty("Int");
        self.register_builtin_instance("Eq", &int_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &int_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance(
            "Num",
            &int_ty,
            &[
                (18, "+"),
                (19, "-"),
                (20, "*"),
                (82, "negate"),
                (83, "abs"),
                (84, "signum"),
                (80, "fromInteger"),
            ],
        );
        self.register_builtin_instance("Show", &int_ty, &[(92, "show")]);

        // === Register instances for Float ===
        let float_ty = make_ty("Float");
        self.register_builtin_instance("Eq", &float_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &float_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance(
            "Num",
            &float_ty,
            &[
                (18, "+"),
                (19, "-"),
                (20, "*"),
                (82, "negate"),
                (83, "abs"),
                (84, "signum"),
                (80, "fromInteger"),
            ],
        );
        self.register_builtin_instance("Fractional", &float_ty, &[(21, "/"), (81, "fromRational")]);
        self.register_builtin_instance("Show", &float_ty, &[(92, "show")]);

        // === Register instances for Bool ===
        let bool_ty = make_ty("Bool");
        self.register_builtin_instance("Eq", &bool_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &bool_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance("Show", &bool_ty, &[(92, "show")]);

        // === Register instances for Char ===
        let char_ty = make_ty("Char");
        self.register_builtin_instance("Eq", &char_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &char_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance("Show", &char_ty, &[(92, "show")]);

        // === Register instances for Rational ===
        let rational_ty = make_ty("Rational");
        self.register_builtin_instance("Eq", &rational_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &rational_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance(
            "Num",
            &rational_ty,
            &[
                (18, "+"),
                (19, "-"),
                (20, "*"),
                (82, "negate"),
                (83, "abs"),
                (84, "signum"),
                (80, "fromInteger"),
            ],
        );
        self.register_builtin_instance(
            "Fractional",
            &rational_ty,
            &[(21, "/"), (81, "fromRational")],
        );
        self.register_builtin_instance("Show", &rational_ty, &[(92, "show")]);

        // === Register Functor class ===
        // Methods: fmap, <$ (fmap is also known as <$>, DefId 43). `<$` is
        // dispatched by rewriting to `fmap (const x)` when the instance has no
        // `$instance_<$` of its own (parsec's ParsecT relies on the default).
        let functor_class = ClassInfo {
            name: Symbol::intern("Functor"),
            param_count: 1,
            methods: vec![Symbol::intern("fmap"), Symbol::intern("<$")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(functor_class);

        // === Register Applicative class ===
        // Methods: pure, <*>
        // Superclass: Functor
        let applicative_class = ClassInfo {
            name: Symbol::intern("Applicative"),
            param_count: 1,
            // `*>`/`<*` appended so a point-free instance alias (parsec's
            // `(>>) = (Applicative.*>)`) resolves them as class methods and
            // dispatches to the named instance method
            // (`$instance_*>_ParsecT`). Appending keeps existing method
            // indices unchanged; builtin monads still take the codegen fast
            // path (dispatch falls through when no named instance exists).
            methods: vec![
                Symbol::intern("pure"),
                Symbol::intern("<*>"),
                Symbol::intern("*>"),
                Symbol::intern("<*"),
            ],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Functor")],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(applicative_class);

        // === Register Monad class ===
        // Methods: >>=, >>
        // Superclass: Applicative
        let monad_class = ClassInfo {
            name: Symbol::intern("Monad"),
            param_count: 1,
            methods: vec![Symbol::intern(">>="), Symbol::intern(">>")],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Applicative")],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(monad_class);

        // === Register Alternative class ===
        // Methods: <|>, empty. Dispatch resolves to a named instance method
        // (`$instance_<|>_ParsecT`), so the method-index layout does not
        // matter here. A bare `empty`/`mzero` occurrence dispatches via its
        // typeck-recorded result type (`monad_head_of_method_occurrence`);
        // when no named instance exists it falls through to the builtin as
        // before. Superclass: Applicative.
        let alternative_class = ClassInfo {
            name: Symbol::intern("Alternative"),
            param_count: 1,
            methods: vec![Symbol::intern("<|>"), Symbol::intern("empty")],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Applicative")],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(alternative_class);

        // === Register Semigroup / Monoid classes ===
        // Dispatch resolves to named instance methods; `mconcat` with no
        // dedicated instance method is rewritten to `foldr mappend mempty`
        // at the dispatch site (its class default).
        let semigroup_class = ClassInfo {
            name: Symbol::intern("Semigroup"),
            param_count: 1,
            methods: vec![Symbol::intern("<>")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(semigroup_class);
        let monoid_class = ClassInfo {
            name: Symbol::intern("Monoid"),
            param_count: 1,
            methods: vec![
                Symbol::intern("mempty"),
                Symbol::intern("mappend"),
                Symbol::intern("mconcat"),
            ],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Semigroup")],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(monoid_class);

        // Builtin `Semigroup Text` / `Monoid Text` instances: Text is a
        // builtin type with no Haskell instance declaration anywhere, so
        // dispatch at the concrete type would otherwise find nothing and
        // lower `t <> u` to an unresolved-method error closure (pandoc's
        // ensureFinalNewlines hit exactly that at runtime). Route the
        // methods at vars named for the codegen Text builtins.
        {
            let text_ty = Ty::Con(bhc_types::TyCon::new(
                Symbol::intern("Text"),
                bhc_types::Kind::Star,
            ));
            self.register_builtin_text_instance_vars();
            let append_id = DefId::new(790_000);
            let empty_id = DefId::new(790_001);
            let concat_id = DefId::new(790_002);
            let mut sg_methods = FxHashMap::default();
            sg_methods.insert(Symbol::intern("<>"), append_id);
            self.class_registry
                .register_instance(crate::dictionary::InstanceInfo {
                    class: Symbol::intern("Semigroup"),
                    instance_types: vec![text_ty.clone()],
                    methods: sg_methods,
                    superclass_instances: Vec::new(),
                    assoc_type_impls: FxHashMap::default(),
                    instance_constraints: Vec::new(),
                });
            let mut mo_methods = FxHashMap::default();
            mo_methods.insert(Symbol::intern("mempty"), empty_id);
            mo_methods.insert(Symbol::intern("mappend"), append_id);
            mo_methods.insert(Symbol::intern("mconcat"), concat_id);
            self.class_registry
                .register_instance(crate::dictionary::InstanceInfo {
                    class: Symbol::intern("Monoid"),
                    instance_types: vec![text_ty],
                    methods: mo_methods,
                    superclass_instances: vec![Ty::Con(bhc_types::TyCon::new(
                        Symbol::intern("Text"),
                        bhc_types::Kind::Star,
                    ))],
                    assoc_type_impls: FxHashMap::default(),
                    instance_constraints: Vec::new(),
                });
        }

        // Lists get builtin `Semigroup` and `Monoid` instances.
        // Without this, `"ab" <> "cd"` and `[1,2] <> [3]` lowered to an
        // unresolved-method stub. `mappend` and `mconcat` would work too, but
        // registering the Monoid instance also exposes `mempty`, and `mempty`
        // at a list type does not currently produce an empty list —
        // `length (mempty :: [Int])` answers 5. A loud stub for the Monoid
        // methods beats a silent wrong answer for `mempty`.
        {
            let list_head = Ty::List(Box::new(Ty::Var(bhc_types::TyVar::new_star(0xFFF7_0004))));
            let mut sg_methods = FxHashMap::default();
            sg_methods.insert(Symbol::intern("<>"), DefId::new(790_009));
            self.class_registry
                .register_instance(crate::dictionary::InstanceInfo {
                    class: Symbol::intern("Semigroup"),
                    instance_types: vec![list_head.clone()],
                    methods: sg_methods,
                    superclass_instances: Vec::new(),
                    assoc_type_impls: FxHashMap::default(),
                    instance_constraints: Vec::new(),
                });
            let mut mo_methods = FxHashMap::default();
            mo_methods.insert(Symbol::intern("mempty"), DefId::new(790_010));
            mo_methods.insert(Symbol::intern("mappend"), DefId::new(790_009));
            mo_methods.insert(Symbol::intern("mconcat"), DefId::new(790_011));
            self.class_registry
                .register_instance(crate::dictionary::InstanceInfo {
                    class: Symbol::intern("Monoid"),
                    instance_types: vec![list_head.clone()],
                    methods: mo_methods,
                    superclass_instances: vec![list_head],
                    assoc_type_impls: FxHashMap::default(),
                    instance_constraints: Vec::new(),
                });
        }

        // Builtin `Semigroup`/`Monoid` instances for the container builtins
        // Set and Map (same reasoning as Text above): `mempty :: Set a` has
        // no Haskell instance anywhere, so pandoc's
        // `emptyExtensions = Extensions mempty` lowered to a garbage-valued
        // stub and `Set.member` walked it (Rust BTreeSet panic). Route the
        // methods at the codegen container builtins; left-biased
        // `Data.Map.union` matches the Monoid (Map k v) contract.
        {
            let star_to_star =
                bhc_types::Kind::Arrow(Box::new(bhc_types::Kind::Star), Box::new(Kind::Star));
            let set_head = Ty::App(
                Box::new(Ty::Con(bhc_types::TyCon::new(
                    Symbol::intern("Set"),
                    star_to_star.clone(),
                ))),
                Box::new(Ty::Var(bhc_types::TyVar::new_star(0xFFF7_0001))),
            );
            let map_head = Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(bhc_types::TyCon::new(
                        Symbol::intern("Map"),
                        bhc_types::Kind::Arrow(Box::new(Kind::Star), Box::new(star_to_star)),
                    ))),
                    Box::new(Ty::Var(bhc_types::TyVar::new_star(0xFFF7_0002))),
                )),
                Box::new(Ty::Var(bhc_types::TyVar::new_star(0xFFF7_0003))),
            );
            for (head, union_id, empty_id, unions_id) in [
                (set_head, 790_003, 790_004, 790_005),
                (map_head, 790_006, 790_007, 790_008),
            ] {
                let mut sg_methods = FxHashMap::default();
                sg_methods.insert(Symbol::intern("<>"), DefId::new(union_id));
                self.class_registry
                    .register_instance(crate::dictionary::InstanceInfo {
                        class: Symbol::intern("Semigroup"),
                        instance_types: vec![head.clone()],
                        methods: sg_methods,
                        superclass_instances: Vec::new(),
                        assoc_type_impls: FxHashMap::default(),
                        instance_constraints: Vec::new(),
                    });
                let mut mo_methods = FxHashMap::default();
                mo_methods.insert(Symbol::intern("mempty"), DefId::new(empty_id));
                mo_methods.insert(Symbol::intern("mappend"), DefId::new(union_id));
                mo_methods.insert(Symbol::intern("mconcat"), DefId::new(unions_id));
                self.class_registry
                    .register_instance(crate::dictionary::InstanceInfo {
                        class: Symbol::intern("Monoid"),
                        instance_types: vec![head.clone()],
                        methods: mo_methods,
                        superclass_instances: vec![head],
                        assoc_type_impls: FxHashMap::default(),
                        instance_constraints: Vec::new(),
                    });
            }
        }

        // === Register MonadPlus class ===
        // Methods: mplus, mzero (dispatched like `empty` above; parsec's
        // `choice ps = foldr (<|>) mzero ps` needs the bare `mzero` resolved
        // to `$instance_mzero_ParsecT`). Superclass: Monad.
        let monad_plus_class = ClassInfo {
            name: Symbol::intern("MonadPlus"),
            param_count: 1,
            methods: vec![Symbol::intern("mplus"), Symbol::intern("mzero")],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Monad")],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(monad_plus_class);

        // === Register IO instances for Functor/Applicative/Monad ===
        // IO has kind * -> *, so we construct it as a type application
        let io_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
        let io_ty = Ty::Con(TyCon::new(Symbol::intern("IO"), io_kind));

        // IO's Functor/Applicative/Monad methods live in the same reserved
        // range as the other builtin monads (10000+), NOT at 150-154.
        // `DictContext::method_reference` consults the module's var_map before
        // its own name table, and DefIds that low are real module exports —
        // 150-154 land on `Control.Exception`'s `throwIO`/`bracket`/`bracket_`,
        // so an IO dictionary built from them had those in its method slots.
        self.register_builtin_instance("Functor", &io_ty, &[(10060, "fmap")]);
        self.register_builtin_instance("Applicative", &io_ty, &[(10061, "pure"), (10062, "<*>")]);
        self.register_builtin_instance("Monad", &io_ty, &[(10063, ">>="), (10064, ">>")]);

        // === Register MonadTrans class ===
        // Methods: lift
        let monad_trans_class = ClassInfo {
            name: Symbol::intern("MonadTrans"),
            param_count: 1,
            methods: vec![Symbol::intern("lift")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(monad_trans_class);

        // === Register MonadIO class ===
        // Methods: liftIO
        // Superclass: Monad
        let monad_io_class = ClassInfo {
            name: Symbol::intern("MonadIO"),
            param_count: 1,
            methods: vec![Symbol::intern("liftIO")],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Monad")],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(monad_io_class);

        // MonadIO IO: liftIO = id (DefId 10012)
        self.register_builtin_instance("MonadIO", &io_ty, &[(10012, "liftIO")]);

        // === Register Identity type and instances ===
        let identity_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
        let identity_ty = Ty::Con(TyCon::new(Symbol::intern("Identity"), identity_kind));

        self.register_builtin_instance("Functor", &identity_ty, &[(10002, "fmap")]);
        self.register_builtin_instance(
            "Applicative",
            &identity_ty,
            &[(10003, "pure"), (10004, "<*>")],
        );
        self.register_builtin_instance("Monad", &identity_ty, &[(10005, ">>="), (10006, ">>")]);

        // === Register ReaderT instances ===
        // ReaderT r m is represented as a partially applied type constructor
        // For codegen, we match on the name "ReaderT" rather than the full type
        let reader_t_kind = Kind::Arrow(
            Box::new(Kind::Star),
            Box::new(Kind::Arrow(
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
            )),
        );
        let reader_t_ty = Ty::Con(TyCon::new(Symbol::intern("ReaderT"), reader_t_kind));

        self.register_builtin_instance("Functor", &reader_t_ty, &[(10022, "fmap")]);
        self.register_builtin_instance(
            "Applicative",
            &reader_t_ty,
            &[(10023, "pure"), (10024, "<*>")],
        );
        self.register_builtin_instance("Monad", &reader_t_ty, &[(10025, ">>="), (10026, ">>")]);
        self.register_builtin_instance("MonadTrans", &reader_t_ty, &[(10027, "lift")]);
        self.register_builtin_instance("MonadIO", &reader_t_ty, &[(10028, "liftIO")]);

        // === Register StateT instances ===
        let state_t_kind = Kind::Arrow(
            Box::new(Kind::Star),
            Box::new(Kind::Arrow(
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
            )),
        );
        let state_t_ty = Ty::Con(TyCon::new(Symbol::intern("StateT"), state_t_kind));

        // === Register ExceptT instances ===
        // Only fmap/pure/>>= — the methods with a value form in codegen's
        // `lower_builtin_direct`. `<*>` and `>>` are deliberately left out so
        // `missing_method_field` fills them with a deferred-error lambda; a
        // registered method with no value form aborts with `unknown builtin`
        // the moment the dictionary is BUILT, which is worse.
        let except_t_kind = Kind::Arrow(
            Box::new(Kind::Star),
            Box::new(Kind::Arrow(
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
            )),
        );
        let except_t_ty = Ty::Con(TyCon::new(Symbol::intern("ExceptT"), except_t_kind));
        self.register_builtin_instance("Functor", &except_t_ty, &[(10080, "fmap")]);
        self.register_builtin_instance("Applicative", &except_t_ty, &[(10081, "pure")]);
        self.register_builtin_instance("Monad", &except_t_ty, &[(10083, ">>=")]);

        self.register_builtin_instance("Functor", &state_t_ty, &[(10042, "fmap")]);
        self.register_builtin_instance(
            "Applicative",
            &state_t_ty,
            &[(10043, "pure"), (10044, "<*>")],
        );
        self.register_builtin_instance("Monad", &state_t_ty, &[(10045, ">>="), (10046, ">>")]);
        self.register_builtin_instance("MonadTrans", &state_t_ty, &[(10047, "lift")]);
        self.register_builtin_instance("MonadIO", &state_t_ty, &[(10048, "liftIO")]);
    }

    /// Helper to register a builtin instance with method `DefIds`.
    fn register_builtin_instance(
        &mut self,
        class_name: &str,
        instance_type: &Ty,
        methods: &[(usize, &str)],
    ) {
        let mut method_map = FxHashMap::default();
        for (def_id, name) in methods {
            method_map.insert(Symbol::intern(name), DefId::new(*def_id));
        }

        // For superclass instances, use the same instance type
        let class_info = self.class_registry.lookup_class(Symbol::intern(class_name));
        let superclass_instances = class_info
            .map(|c| {
                c.superclasses
                    .iter()
                    .map(|_| instance_type.clone())
                    .collect()
            })
            .unwrap_or_default();

        let instance_info = InstanceInfo {
            class: Symbol::intern(class_name),
            instance_types: vec![instance_type.clone()],
            methods: method_map,
            superclass_instances,
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        self.class_registry.register_instance(instance_info);
    }

    /// Register builtins using `DefIds` from the lowering pass.
    ///
    /// This replaces the hardcoded `DefIds` with the actual `DefIds` assigned
    /// during AST-to-HIR lowering, ensuring consistency across passes.
    pub fn register_lowered_builtins(&mut self, defs: &crate::DefMap) {
        // Clear the existing hardcoded builtins
        self.var_map.clear();

        // Register all definitions from the lowering pass
        for (_def_id, def_info) in defs {
            let var = Var {
                name: def_info.name,
                id: VarId::new(def_info.id.index()),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_info.id, var);
        }

        // The clear above wiped the synthetic method vars behind the builtin
        // `Semigroup Text` / `Monoid Text` instances (DefIds 790_000–790_002,
        // registered at context construction); restore them so dictionary
        // construction still finds `Data.Text.append`/`empty`/`concat`
        // instead of falling back to a bare `<>` stub.
        self.register_builtin_text_instance_vars();
    }

    /// (Re-)register the vars backing the builtin Text/Set/Map
    /// Semigroup/Monoid instance methods under their fixed DefIds.
    fn register_builtin_text_instance_vars(&mut self) {
        for (offset, builtin) in [
            "Data.Text.append",
            "Data.Text.empty",
            "Data.Text.concat",
            "Data.Set.union",
            "Data.Set.empty",
            "Data.Set.unions",
            "Data.Map.union",
            "Data.Map.empty",
            "Data.Map.unions",
            "Data.List.append",
            "Data.List.empty",
            "Data.List.concat",
        ]
        .iter()
        .enumerate()
        {
            let def_id = DefId::new(790_000 + offset);
            let var = self.named_var(Symbol::intern(builtin), Ty::Error);
            self.register_var(def_id, var);
        }
    }

    /// Generate a fresh variable with the given base name.
    ///
    /// The name will be mangled with a counter to ensure uniqueness.
    /// For top-level bindings that need to preserve their original name,
    /// use `named_var` instead.
    pub fn fresh_var(&mut self, base: &str, ty: Ty, _span: Span) -> Var {
        let name = Symbol::intern(&format!("{}_{}", base, self.fresh_counter));
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
            ty,
        }
    }

    /// Create a variable with a specific name (preserving the original name).
    ///
    /// Use this for top-level bindings where the name must be preserved
    /// for external visibility (e.g., `main`).
    pub fn named_var(&mut self, name: Symbol, ty: Ty) -> Var {
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
            ty,
        }
    }

    /// Generate a fresh variable ID.
    pub fn fresh_id(&mut self) -> VarId {
        self.fresh_counter += 1;
        VarId::new(self.fresh_counter as usize)
    }

    /// Record an error.
    pub fn error(&mut self, err: LowerError) {
        self.errors.push(err);
    }

    /// Check if any errors have been recorded.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Take all recorded errors.
    pub fn take_errors(&mut self) -> Vec<LowerError> {
        std::mem::take(&mut self.errors)
    }

    /// Record a warning (non-fatal diagnostic).
    pub fn warn(&mut self, message: String) {
        self.warnings.push(message);
    }

    /// Take all recorded warnings.
    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    /// Get all constructors for a given type name.
    ///
    /// Returns a vector of `(tag, name, arity)` tuples sorted by tag.
    #[must_use]
    pub fn constructors_for_type_name(&self, type_name: Symbol) -> Vec<(u32, Symbol, u32)> {
        let mut cons: Vec<(u32, Symbol, u32)> = self
            .constructor_map
            .values()
            .filter(|info| info.type_name == type_name)
            .map(|info| (info.tag, info.name, info.arity))
            .collect();
        cons.sort_by_key(|(tag, _, _)| *tag);
        cons.dedup_by_key(|(tag, _, _)| *tag);
        cons
    }

    /// Register a HIR definition with a Core variable.
    pub fn register_var(&mut self, def_id: DefId, var: Var) {
        self.var_map.insert(def_id, var);
    }

    /// Look up the Core variable for a HIR definition.
    #[must_use]
    pub fn lookup_var(&self, def_id: DefId) -> Option<&Var> {
        self.var_map.get(&def_id)
    }

    /// Register a data constructor with its metadata.
    pub fn register_constructor(&mut self, def_id: DefId, info: ConstructorInfo) {
        self.constructor_map.insert(def_id, info);
    }

    /// Look up constructor metadata for a given `DefId`.
    #[must_use]
    pub fn lookup_constructor(&self, def_id: DefId) -> Option<&ConstructorInfo> {
        self.constructor_map.get(&def_id)
    }

    /// Look up constructor info by name (for decision tree lowering where `DefId` isn't available).
    #[must_use]
    pub fn lookup_constructor_by_name(&self, name: Symbol) -> Option<&ConstructorInfo> {
        self.constructor_map.values().find(|info| info.name == name)
    }

    /// Look up constructor info by `DefId`, falling back to the constructor's
    /// name. A cross-module reference to a constructor can carry a different
    /// `DefId` than the one it was registered under (in the defining module), so
    /// the direct `DefId` lookup misses; the by-name fallback then finds the
    /// canonical metadata — crucially its position-based `tag` — keeping
    /// construction and matching agreed on the same tag across modules.
    #[must_use]
    pub fn lookup_constructor_or_by_name(&self, def_id: DefId) -> Option<&ConstructorInfo> {
        self.lookup_constructor(def_id).or_else(|| {
            self.lookup_var(def_id)
                .and_then(|v| self.lookup_constructor_by_name(v.name))
        })
    }

    /// Register a field selector function.
    pub fn register_field_selector(&mut self, field_name: Symbol, info: FieldSelectorInfo) {
        self.field_selector_map.insert(field_name, info);
    }

    /// Look up field selector metadata for a given field name.
    #[must_use]
    pub fn lookup_field_selector(&self, field_name: Symbol) -> Option<&FieldSelectorInfo> {
        self.field_selector_map.get(&field_name)
    }

    /// Push a new dictionary scope.
    pub fn push_dict_scope(&mut self) {
        self.dict_scope.push(FxHashMap::default());
        self.dict_scope_ty.push(FxHashMap::default());
    }

    /// Pop the current dictionary scope.
    pub fn pop_dict_scope(&mut self) {
        if self.dict_scope.len() > 1 {
            self.dict_scope.pop();
            self.dict_scope_ty.pop();
        }
    }

    /// Register a dictionary variable for a constraint in the current scope.
    pub fn register_dict(&mut self, class_name: Symbol, dict_var: Var) {
        if let Some(scope) = self.dict_scope.last_mut() {
            scope.insert(class_name, dict_var);
        }
    }

    /// Register a dictionary along with the TYPE it is for, so a later lookup
    /// can tell whether it is the right dictionary and not merely the right
    /// class. See `dict_scope_ty`.
    ///
    /// Whether hopping to `needed`'s dictionary through an in-scope `holder`
    /// dictionary is the RIGHT hop for the binding being lowered.
    ///
    /// A multi-parameter class's superclass constrains one specific
    /// parameter — `class Monad m => Stream s m t` is about `m` — so the hop
    /// only makes sense in a binding that works in that parameter.
    /// parsec's `runPT` is such a place: its do-block genuinely runs in `m`,
    /// which is why selecting its `>>=` through `Stream`'s superclass is
    /// correct there. Any ParsecT-level binding under the same constraint is
    /// not, and taking the hop there ran the parser in the wrong monad —
    /// silently, as wrong answers rather than crashes.
    ///
    /// Refuses whenever anything is unknown, which leaves the historical
    /// "skip multi-parameter classes entirely" behaviour in place.
    fn superclass_hop_matches(&self, holder: Symbol, needed: Symbol) -> bool {
        let Some(info) = self.class_registry.lookup_class(holder) else {
            return false;
        };
        // `needed` may be reached TRANSITIVELY — `return` is Applicative's
        // `pure`, two hops from a `Stream` dictionary — so find the DIRECT
        // superclass that leads to it and use that one's parameter.
        let Some(idx) = info
            .superclasses
            .iter()
            .position(|s| *s == needed || self.superclass_field_path(*s, needed, 0).is_some())
        else {
            return false;
        };
        let Some(param) = info
            .superclass_params
            .get(idx)
            .and_then(|p| p.first())
            .copied()
        else {
            return false;
        };
        let Some(constrained) = self
            .lookup_dict_args(holder)
            .and_then(|a| a.get(param))
            .cloned()
        else {
            return false;
        };
        // Decide from the ENCLOSING BINDING's signature, not from the
        // occurrence's recorded type.
        //
        // The occurrence route cannot work: comparing type-variable HEADS
        // treats any two variables as equal, so every occurrence whose monad
        // typeck left unresolved matches — and a parser library is full of
        // those. Comparing variable IDENTITY fails the other way, because an
        // occurrence's inference variables are not the signature's.
        //
        // The signature route has neither problem. `runPT :: Stream s m t =>
        // … -> m (Either ParseError a)` returns in `m`, the very parameter
        // the superclass constrains, so its do-block wants the hop. A
        // ParsecT-level binding under the same constraint returns in
        // `ParsecT s u m`, so it does not. Both the signature and the
        // constraint come from one declaration, so their variables ARE the
        // same objects and identity is meaningful here.
        fn result_of(t: &Ty) -> &Ty {
            match t {
                Ty::Fun(_, r) => result_of(r),
                other => other,
            }
        }
        fn head(ty: &Ty) -> &Ty {
            match ty {
                Ty::App(f, _) => head(f),
                other => other,
            }
        }
        let Some(sig) = self.current_binding_sig() else {
            return false;
        };
        let sig_monad = head(result_of(sig));
        matches!(
            (head(&constrained), sig_monad),
            (Ty::Var(a), Ty::Var(b)) if a.id == b.id
        )
    }

    /// The type a `holder` dictionary's superclass path to `needed` is ABOUT.
    ///
    /// `class Monad m => Stream s m t`'s `Monad` superclass constrains `m`, so
    /// the answer is whatever `m` is bound to in the holder's own constraint
    /// arguments. A single-parameter holder has only one thing it can be
    /// about, which matters because `superclass_params` is recorded where a
    /// class is DECLARED — an imported class like `PandocMonad` arrives
    /// without it.
    fn superclass_constrained_ty(&self, holder: Symbol, needed: Symbol) -> Option<Ty> {
        let info = self.class_registry.lookup_class(holder)?;
        let idx = info
            .superclasses
            .iter()
            .position(|s| *s == needed || self.superclass_field_path(*s, needed, 0).is_some())?;
        let param = info
            .superclass_params
            .get(idx)
            .and_then(|p| p.first())
            .copied()
            .or(if info.param_count == 1 { Some(0) } else { None })?;
        self.lookup_dict_args(holder)
            .and_then(|a| a.get(param))
            .cloned()
    }

    /// Whether the OCCURRENCE being lowered rules out taking `holder`'s
    /// superclass hop to `needed`.
    ///
    /// The enclosing binding's signature is the wrong thing to ask (see
    /// `superclass_hop_matches`): a `ParsecT`-level binding can still contain
    /// sub-expressions that work in `m`, and a binding in `m` can contain an
    /// inline parser that works in `ParsecT`. What settles it is the monad
    /// THIS occurrence works in, which typeck records span-keyed.
    ///
    /// Used only to REFUSE, and only when the occurrence's monad is a concrete
    /// constructor while the hop's dictionary is about something else. A type
    /// variable never refuses anything — typeck leaves plenty of them
    /// unresolved in a parser library, and reading those as agreement is the
    /// mistake that kept every earlier scoping of this guard wrong in one
    /// direction or the other.
    fn occurrence_refutes_hop(&self, holder: Symbol, needed: Symbol, span: Span) -> bool {
        if !self.is_monad_family_class(needed) {
            return false;
        }
        fn result_of(t: &Ty) -> &Ty {
            match t {
                Ty::Fun(_, r) => result_of(r),
                other => other,
            }
        }
        fn head(ty: &Ty) -> &Ty {
            match ty {
                Ty::App(f, _) => head(f),
                other => other,
            }
        }
        let dbg = std::env::var("BHC_DBG_HOP").is_ok();
        let Some(occ) = self
            .resolved_expr_ty_opt(span)
            .or_else(|| self.expr_ty_opt(span))
        else {
            if dbg {
                eprintln!("[hop] {holder} -> {needed} @ {span:?}: no occurrence type");
            }
            return false;
        };
        let occ_head = head(result_of(&occ)).clone();
        let constrained = self.superclass_constrained_ty(holder, needed);
        if dbg {
            eprintln!(
                "[hop] {holder} -> {needed} @ {span:?}: occ_head={occ_head:?} constrained={constrained:?}"
            );
        }
        let Ty::Con(occ_con) = &occ_head else {
            return false;
        };
        match constrained.as_ref().map(head) {
            Some(Ty::Var(_)) => true,
            Some(Ty::Con(c)) => c.name != occ_con.name,
            _ => false,
        }
    }

    /// The dictionaries currently in scope, with the constraint arguments each
    /// is for, in the shape `DictContext` wants for filling superclass slots.
    fn scope_dicts_for_construction(&self) -> Vec<(Symbol, Vec<Ty>, Var)> {
        let mut out = Vec::new();
        for scope in self.dict_scope.iter().rev() {
            for (class, var) in scope {
                if out.iter().any(|(c, _, _)| c == class) {
                    continue; // innermost wins
                }
                let args = self.lookup_dict_args(*class).unwrap_or(&[]).to_vec();
                out.push((*class, args, var.clone()));
            }
        }
        out
    }

    /// The dictionaries reachable from one in scope by SUPERCLASS selection,
    /// as (superclass, arguments, base dictionary, field-index path).
    ///
    /// `scope_dicts_for_construction` above only reports the classes a binding
    /// is constrained by directly. That is not enough to fill an instance's own
    /// constraint slot when the constraint sits ABOVE the one in scope:
    /// `myRead :: PandocMonad m => …` builds `Stream Sources m Char`, whose
    /// `Monad m` slot has no instance to come from while `m` is a variable, and
    /// no `Monad` dictionary is in scope either — only `PandocMonad`, which has
    /// Monad among its superclasses.
    ///
    /// Only single-parameter hops are offered. Which class parameter a
    /// superclass of a multi-parameter class constrains is recorded separately
    /// (`superclass_params`), and guessing it here would hand the slot a
    /// dictionary for the wrong type — worse than leaving it null.
    fn scope_super_dicts_for_construction(&self) -> Vec<(Symbol, Vec<Ty>, Var, Vec<usize>)> {
        let mut out: Vec<(Symbol, Vec<Ty>, Var, Vec<usize>)> = Vec::new();
        for scope in self.dict_scope.iter().rev() {
            for (class, var) in scope {
                let Some(args) = self.lookup_dict_args(*class) else {
                    continue;
                };
                if args.len() != 1 {
                    continue;
                }
                let args = args.to_vec();
                for (sup, path) in self.superclass_reachable(*class, 0) {
                    if sup == *class {
                        continue;
                    }
                    let single_param = self
                        .class_registry
                        .lookup_class(sup)
                        .is_some_and(|c| c.param_count == 1);
                    if !single_param {
                        continue;
                    }
                    if out.iter().any(|(c, _, _, _)| *c == sup) {
                        continue; // innermost, shallowest wins
                    }
                    out.push((sup, args.clone(), var.clone(), path));
                }
            }
        }
        // Multi-parameter dictionaries, whose superclass constrains a
        // parameter that is not the first: `class Monad m => Stream s m t`
        // records `superclass_params = [[1]]`, so a `Stream s m a` in scope
        // does carry the `Monad m` its uses need. Skipping these left parsec's
        // `getState :: Monad m => …` with no dictionary inside any
        // `Stream s m a =>` binding, and its continuation argument landed in
        // the dictionary's slot.
        //
        // Offered AFTER the single-parameter hops so an exact single-parameter
        // match still wins, and only for a DIRECT superclass — a longer chain
        // would need each hop's parameter mapping, and guessing one is worse
        // than leaving the slot empty.
        for scope in self.dict_scope.iter().rev() {
            for (class, var) in scope {
                let Some(args) = self.lookup_dict_args(*class) else {
                    continue;
                };
                if args.len() < 2 {
                    continue;
                }
                let Some(info) = self.class_registry.lookup_class(*class) else {
                    continue;
                };
                let supers = info.superclasses.clone();
                let super_params = info.superclass_params.clone();
                for (i, sup) in supers.iter().enumerate() {
                    if *sup == *class || out.iter().any(|(c, _, _, _)| c == sup) {
                        continue;
                    }
                    let single_param = self
                        .class_registry
                        .lookup_class(*sup)
                        .is_some_and(|c| c.param_count == 1);
                    if !single_param {
                        continue;
                    }
                    let Some([idx]) = super_params.get(i).map(Vec::as_slice) else {
                        continue;
                    };
                    let Some(arg) = args.get(*idx) else {
                        continue;
                    };
                    out.push((*sup, vec![arg.clone()], var.clone(), vec![i]));
                }
            }
        }
        out
    }

    /// A dictionary for `class` from one in scope — the dictionary itself, or
    /// one of its superclasses selected out of it.
    ///
    /// A constrained VALUE gets its dictionaries at its use site, and only
    /// from `lookup_dict`, which sees a binding's OWN constraints. parsec's
    /// `getState :: Monad m => ParsecT s u m u` used inside a `PandocMonad m
    /// =>` function found no `Monad` there — `PandocMonad` has it as a
    /// superclass — so `getState` was emitted with its dictionary missing and
    /// its first value argument landed in the dictionary's slot.
    pub(crate) fn dict_in_scope_or_via_superclass(
        &self,
        class: Symbol,
        span: Span,
    ) -> Option<core::Expr> {
        if let Some(v) = self.lookup_dict(class) {
            return Some(core::Expr::Var(v.clone(), span));
        }
        let (_, _, base, path) = self
            .scope_super_dicts_for_construction()
            .into_iter()
            .find(|(c, _, _, _)| *c == class)?;
        let mut expr = core::Expr::Var(base, span);
        for idx in path {
            let sel = Var {
                name: Symbol::intern(&format!("$sel_{idx}")),
                id: VarId::new(idx),
                ty: Ty::Error,
            };
            expr = core::Expr::App(Box::new(core::Expr::Var(sel, span)), Box::new(expr), span);
        }
        Some(expr)
    }

    /// Every class reachable from `from` by superclass edges, with the
    /// field-index path taken to get there. Breadth-first, so the shallowest
    /// path to a class is the one reported.
    fn superclass_reachable(&self, from: Symbol, depth: usize) -> Vec<(Symbol, Vec<usize>)> {
        let mut out = Vec::new();
        if depth > 8 {
            return out;
        }
        let Some(info) = self.class_registry.lookup_class(from) else {
            return out;
        };
        let sups: Vec<Symbol> = info.superclasses.clone();
        for (i, sup) in sups.iter().enumerate() {
            if !out.iter().any(|(c, _): &(Symbol, Vec<usize>)| c == sup) {
                out.push((*sup, vec![i]));
            }
        }
        for (i, sup) in sups.iter().enumerate() {
            for (deeper, mut rest) in self.superclass_reachable(*sup, depth + 1) {
                if out.iter().any(|(c, _): &(Symbol, Vec<usize>)| *c == deeper) {
                    continue;
                }
                let mut path = vec![i];
                path.append(&mut rest);
                out.push((deeper, path));
            }
        }
        out
    }

    /// A dictionary expression for `class_name` from the dictionaries in scope:
    /// the dictionary itself when one is in scope for exactly these arguments,
    /// else a superclass selection from one that is.
    ///
    /// `anyChar`'s leading `Monad m` is wholly a type variable, so no instance
    /// can be selected for it at all — but inside a `Monad m =>` binding the
    /// binding's own dictionary is precisely it. `want` must match what the
    /// in-scope dictionary is recorded for: parsec's internals have several
    /// distinct type variables live at once, and matching on the class alone
    /// hands back a dictionary for a different one.
    #[must_use]
    pub fn dict_expr_for_class(
        &self,
        class_name: Symbol,
        want: &[Ty],
        span: Span,
    ) -> Option<core::Expr> {
        let args_match = |have: Option<&[Ty]>| {
            have.is_some_and(|h| {
                h.len() == want.len()
                    && h.iter().zip(want).all(|(a, b)| match (a, b) {
                        // Two variables line up: the dictionary the binding
                        // received IS the one for its own type variable, even
                        // though the occurrence carries a freshly instantiated
                        // id rather than the signature's.
                        (Ty::Var(_), Ty::Var(_)) => true,
                        _ => a == b,
                    })
            })
        };
        if let Some(var) = self.lookup_dict(class_name) {
            if args_match(self.lookup_dict_args(class_name)) {
                return Some(core::Expr::Var(var.clone(), span));
            }
            return None;
        }
        let mut best: Option<(Var, Vec<usize>)> = None;
        for scope in self.dict_scope.iter().rev() {
            for (have, var) in scope {
                if !args_match(self.lookup_dict_args(*have)) {
                    continue;
                }
                if let Some(path) = self.superclass_field_path(*have, class_name, 0) {
                    if path.is_empty() {
                        continue;
                    }
                    if best.as_ref().is_none_or(|(_, b)| path.len() < b.len()) {
                        best = Some((var.clone(), path));
                    }
                }
            }
        }
        let (var, path) = best?;
        let mut cur = core::Expr::Var(var, span);
        for idx in path {
            let sel = Var {
                name: Symbol::intern(&format!("$sel_{idx}")),
                id: VarId::new(0),
                ty: Ty::Error,
            };
            cur = core::Expr::App(Box::new(core::Expr::Var(sel, span)), Box::new(cur), span);
        }
        Some(cur)
    }

    /// Register a dictionary along with the constraint arguments it is for.
    ///
    /// Records the FULL argument list, not just the first type: a superclass
    /// of a multi-parameter class is about a specific parameter (`class Monad
    /// m => Stream s m t` is about `m`, index 1), so deciding whether that
    /// superclass hop applies needs the argument at that index.
    pub fn register_dict_at(&mut self, class_name: Symbol, args: Vec<Ty>, dict_var: Var) {
        self.register_dict(class_name, dict_var);
        if let Some(scope) = self.dict_scope_ty.last_mut() {
            scope.insert(class_name, args);
        }
    }

    /// The constraint arguments an in-scope dictionary for `class_name` is
    /// for, if recorded.
    #[must_use]
    pub fn lookup_dict_args(&self, class_name: Symbol) -> Option<&[Ty]> {
        for scope in self.dict_scope_ty.iter().rev() {
            if let Some(args) = scope.get(&class_name) {
                return Some(args.as_slice());
            }
        }
        None
    }

    /// The first constraint argument, i.e. the type a single-parameter class's
    /// dictionary is for.
    #[must_use]
    pub fn lookup_dict_ty(&self, class_name: Symbol) -> Option<&Ty> {
        self.lookup_dict_args(class_name).and_then(|a| a.first())
    }

    /// Look up a dictionary variable for a constraint class.
    ///
    /// Searches from innermost to outermost scope.
    #[must_use]
    pub fn lookup_dict(&self, class_name: Symbol) -> Option<&Var> {
        for scope in self.dict_scope.iter().rev() {
            if let Some(var) = scope.get(&class_name) {
                return Some(var);
            }
        }
        None
    }

    /// Find an in-scope dictionary whose class has the given class as a superclass.
    ///
    /// For example, if we need an `Eq` dictionary but only have `Ord` in scope,
    /// this will find the `Ord` dictionary since `Ord` has `Eq` as a superclass.
    ///
    /// Returns `(subclass_name, dict_var)` if found.
    #[must_use]
    pub fn lookup_superclass_dict(&self, needed_class: Symbol) -> Option<(Symbol, &Var)> {
        for scope in self.dict_scope.iter().rev() {
            for (class_name, dict_var) in scope {
                // Check if this class has the needed class as a superclass
                if let Some(class_info) = self.class_registry.lookup_class(*class_name) {
                    if class_info.superclasses.contains(&needed_class) {
                        return Some((*class_name, dict_var));
                    }
                }
            }
        }
        None
    }

    /// Get all dictionary variables that match the given constraints.
    ///
    /// Returns dictionary variables in the same order as the constraints.
    #[must_use]
    pub fn lookup_dicts_for_constraints(&self, constraints: &[Constraint]) -> Vec<Option<Var>> {
        constraints
            .iter()
            .map(|c| self.lookup_dict(c.class).cloned())
            .collect()
    }

    /// Set the class registry for dictionary construction.
    pub fn set_class_registry(&mut self, registry: ClassRegistry) {
        self.class_registry = registry;
    }

    /// Get a reference to the class registry.
    #[must_use]
    pub fn class_registry(&self) -> &ClassRegistry {
        &self.class_registry
    }

    /// Try to resolve a dictionary for a constraint.
    ///
    /// Resolution order:
    /// 1. Direct lookup: Check if we have an in-scope dictionary for the class
    /// 2. Superclass extraction: Check if we have a dictionary for a subclass
    ///    (e.g., have Ord, need Eq - extract Eq from Ord)
    /// 3. Instance construction: For concrete types, construct from an instance
    ///
    /// Returns the dictionary expression and any bindings that need to be added.
    pub fn resolve_dictionary(
        &mut self,
        constraint: &Constraint,
        span: Span,
    ) -> Option<core::Expr> {
        // 0. A fully concrete constraint names its instance directly. The
        // in-scope lookup below matches by CLASS alone and can hand back a
        // dictionary for a DIFFERENT type: readMarkdown's `ToSources a`
        // context dict is the Text instance, but its readWithM call site
        // instantiates `ToSources Sources` — passing the Text dict made
        // `toSources` walk a Sources ADT as UTF-8 bytes. When every
        // constraint argument is pinned and the instance is constructible,
        // build the instance dictionary; fall through on failure.
        if !constraint.args.is_empty() && constraint.args.iter().all(|ty| !has_type_variables(ty)) {
            let mut dict_ctx =
                DictContext::new_with_var_map(&self.class_registry, self.var_map.clone());
            dict_ctx.set_scope_dicts(self.scope_dicts_for_construction());
            dict_ctx.set_scope_super_dicts(self.scope_super_dicts_for_construction());
            if let Some(dict_expr) = dict_ctx.get_dictionary(constraint, span) {
                let bindings = dict_ctx.take_bindings();
                let mut result = dict_expr;
                for bind in bindings.into_iter().rev() {
                    result = core::Expr::Let(Box::new(bind), Box::new(result), span);
                }
                return Some(result);
            }
        }

        // 0b. EXPERIMENT (BHC_MONAD_WITNESS): the builtin monad instances are
        // registered under the BARE type constructor (`Con StateT`), because
        // codegen matches them by name rather than by shape. A use site's type
        // is applied — `StateT Int IO` — and `types_match` never matches a
        // bare `Con` pattern against an `App`, so step 0 above always misses.
        // Retry at the head constructor.
        if crate::dictionary::monad_witness_enabled()
            && matches!(
                constraint.class.as_str(),
                "Monad" | "Applicative" | "Functor"
            )
        {
            fn head_con(ty: &Ty) -> Option<&TyCon> {
                match ty {
                    Ty::Con(tc) => Some(tc),
                    Ty::App(f, _) => head_con(f),
                    _ => None,
                }
            }
            /// The layer directly beneath a transformer, as a method-name
            /// suffix. `ExceptT e (StateT s m)` -> `_st`; anything else has no
            /// distinct representation to select.
            fn inner_layer_suffix(ty: &Ty) -> Option<&'static str> {
                // `T a m` is App(App(Con T, a), m): the inner monad is the
                // outer argument of the two-argument application.
                let Ty::App(f, inner) = ty else {
                    return None;
                };
                let is_except_t = matches!(f.as_ref(), Ty::App(g, _)
                    if matches!(head_con(g), Some(tc) if tc.name.as_str() == "ExceptT"));
                if !is_except_t {
                    return None;
                }
                match head_con(inner)?.name.as_str() {
                    "StateT" => Some("_st"),
                    _ => None,
                }
            }
            if let Some(tc) = constraint.args.first().and_then(head_con) {
                let head_constraint =
                    Constraint::new(constraint.class, Ty::Con(tc.clone()), constraint.span);
                if head_constraint.args != constraint.args {
                    let mut dict_ctx =
                        DictContext::new_with_var_map(&self.class_registry, self.var_map.clone());
                    dict_ctx.set_scope_dicts(self.scope_dicts_for_construction());
                    dict_ctx.set_scope_super_dicts(self.scope_super_dicts_for_construction());
                    // Resolution matches on the bare head, which throws away
                    // the layer beneath a transformer — but ExceptT's methods
                    // differ between `ExceptT e (StateT s m)` and plain
                    // `ExceptT e m`. Recover it from the full type here, where
                    // it is still available, and let it pick the method
                    // representation (`ExceptT_st.pure` vs `ExceptT.pure`).
                    dict_ctx.set_transformer_variant(
                        constraint.args.first().and_then(inner_layer_suffix),
                    );
                    if let Some(dict_expr) = dict_ctx.get_dictionary(&head_constraint, span) {
                        let bindings = dict_ctx.take_bindings();
                        let mut result = dict_expr;
                        for bind in bindings.into_iter().rev() {
                            result = core::Expr::Let(Box::new(bind), Box::new(result), span);
                        }
                        return Some(result);
                    }
                }
            }
        }

        // 1. Next, try to find an in-scope dictionary variable directly
        if let Some(dict_var) = self.lookup_dict(constraint.class) {
            return Some(core::Expr::Var(dict_var.clone(), span));
        }

        // 2. Try superclass extraction: if we have Ord but need Eq, extract Eq from Ord
        if let Some((subclass, dict_var)) = self.lookup_superclass_dict(constraint.class) {
            // We found a dictionary for a class that has our needed class as a superclass
            // Extract the superclass dictionary
            if let Some(superclass_expr) = crate::dictionary::select_superclass(
                dict_var,
                subclass,
                constraint.class,
                &self.class_registry,
                span,
            ) {
                return Some(superclass_expr);
            }
        }

        // 3. If not in scope, try to construct from an instance
        // (only works for concrete types — all args must be concrete)
        if !constraint.args.is_empty() && constraint.args.iter().all(|ty| !has_type_variables(ty)) {
            // Create a DictContext with var_map so method_reference uses correct names
            let mut dict_ctx =
                DictContext::new_with_var_map(&self.class_registry, self.var_map.clone());
            dict_ctx.set_scope_dicts(self.scope_dicts_for_construction());
            dict_ctx.set_scope_super_dicts(self.scope_super_dicts_for_construction());
            let dict_expr = dict_ctx.get_dictionary(constraint, span)?;

            // If the dictionary construction generated bindings, wrap the
            // expression in let bindings
            let bindings = dict_ctx.take_bindings();
            if bindings.is_empty() {
                return Some(dict_expr);
            }

            // Wrap in let bindings (innermost first)
            let mut result = dict_expr;
            for bind in bindings.into_iter().rev() {
                result = core::Expr::Let(Box::new(bind), Box::new(result), span);
            }
            return Some(result);
        }

        // 4. Partially-concrete multi-parameter constraint with NO in-scope
        // dictionary: construct from a structurally matching instance. The
        // canonical case is `Stream Sources m Char` at readWithM's runParserT
        // call — `s`/`t` are pinned but `m` is the caller's own type variable,
        // so steps 0 and 3 (all-concrete guards) never fire, and there is no
        // in-scope `Stream` dict for step 1. Instance matching binds the
        // instance's `m` to our variable; `construct_dictionary` already
        // null-placeholders superclass/context slots it cannot resolve at a
        // polymorphic type. Ordering matters: this runs AFTER the in-scope
        // lookup, so constrained functions that carry their own dictionary
        // keep passing it along (the validated fast path).
        if constraint.args.len() > 1 && constraint.args.iter().any(|ty| !has_type_variables(ty)) {
            let mut dict_ctx =
                DictContext::new_with_var_map(&self.class_registry, self.var_map.clone());
            dict_ctx.set_scope_dicts(self.scope_dicts_for_construction());
            dict_ctx.set_scope_super_dicts(self.scope_super_dicts_for_construction());
            if let Some(dict_expr) = dict_ctx.get_dictionary(constraint, span) {
                let bindings = dict_ctx.take_bindings();
                let mut result = dict_expr;
                for bind in bindings.into_iter().rev() {
                    result = core::Expr::Let(Box::new(bind), Box::new(result), span);
                }
                return Some(result);
            }
        }

        None
    }

    /// Resolve a class method call at a concrete type.
    ///
    /// When a class method (like `(+)` from `Num`) is called at a concrete type
    /// (like `Int`), we need to:
    /// 1. Construct the dictionary for that instance (e.g., `Num Int`)
    /// 2. Select the method from the dictionary
    ///
    /// Returns the method selection expression with any necessary let bindings.
    pub fn resolve_method_at_concrete_type(
        &mut self,
        method_name: Symbol,
        class_name: Symbol,
        concrete_type: &Ty,
        span: Span,
    ) -> Option<core::Expr> {
        // Create a constraint for the concrete type
        let constraint = Constraint::new(class_name, concrete_type.clone(), span);

        // Construct the dictionary with var_map for correct method names
        let mut dict_ctx =
            DictContext::new_with_var_map(&self.class_registry, self.var_map.clone());
        dict_ctx.set_scope_dicts(self.scope_dicts_for_construction());
        dict_ctx.set_scope_super_dicts(self.scope_super_dicts_for_construction());
        let dict_expr = dict_ctx.get_dictionary(&constraint, span)?;
        let bindings = dict_ctx.take_bindings();

        // Create a fresh variable to hold the dictionary
        let dict_var = self.fresh_var(&format!("$d{}", class_name.as_str()), Ty::Error, span);

        // Select the method from the dictionary
        let method_expr = crate::dictionary::select_method(
            &dict_var,
            class_name,
            method_name,
            &self.class_registry,
            span,
        )?;

        // Build the let expression:
        // let $dict = <dict_expr> in <method_expr>
        let dict_bind = Bind::NonRec(dict_var, Box::new(dict_expr));

        // If there are additional bindings from nested dictionary construction,
        // wrap them around the whole thing
        let mut result = core::Expr::Let(Box::new(dict_bind), Box::new(method_expr), span);
        for bind in bindings.into_iter().rev() {
            result = core::Expr::Let(Box::new(bind), Box::new(result), span);
        }

        Some(result)
    }

    /// Resolve a class method call at multiple concrete types (multi-param classes).
    ///
    /// Similar to `resolve_method_at_concrete_type` but takes multiple types
    /// for multi-param type classes like `instance Convertible Int String`.
    pub fn resolve_method_at_concrete_types(
        &mut self,
        method_name: Symbol,
        class_name: Symbol,
        concrete_types: &[Ty],
        span: Span,
    ) -> Option<core::Expr> {
        let constraint = Constraint::new_multi(class_name, concrete_types.to_vec(), span);

        let mut dict_ctx =
            DictContext::new_with_var_map(&self.class_registry, self.var_map.clone());
        dict_ctx.set_scope_dicts(self.scope_dicts_for_construction());
        dict_ctx.set_scope_super_dicts(self.scope_super_dicts_for_construction());
        let dict_expr = dict_ctx.get_dictionary(&constraint, span)?;
        let bindings = dict_ctx.take_bindings();

        let dict_var = self.fresh_var(&format!("$d{}", class_name.as_str()), Ty::Error, span);

        let method_expr = crate::dictionary::select_method(
            &dict_var,
            class_name,
            method_name,
            &self.class_registry,
            span,
        )?;

        let dict_bind = Bind::NonRec(dict_var, Box::new(dict_expr));
        let mut result = core::Expr::Let(Box::new(dict_bind), Box::new(method_expr), span);
        for bind in bindings.into_iter().rev() {
            result = core::Expr::Let(Box::new(bind), Box::new(result), span);
        }

        Some(result)
    }

    /// Get the parameter count for a class (1 for single-param, 2+ for multi-param).
    #[must_use]
    pub fn class_param_count(&self, class_name: Symbol) -> usize {
        self.class_registry
            .lookup_class(class_name)
            .map_or(1, |c| c.param_count)
    }

    /// Select a method via superclass dictionary extraction.
    ///
    /// When a function has a `MyOrd a =>` constraint but calls `myEqual` (a method
    /// of superclass `MyEq`), we need to:
    /// 1. Extract the `MyEq` dictionary from the `MyOrd` dictionary
    /// 2. Select `myEqual` from the extracted `MyEq` dictionary
    ///
    /// Returns the method expression if successful, or None if no superclass path exists.
    pub fn select_method_via_superclass(
        &mut self,
        needed_class: Symbol,
        method_name: Symbol,
        span: Span,
    ) -> Option<core::Expr> {
        // `return` is not a slot in the builtin Monad layout ([>>=, >>]) —
        // it IS Applicative's `pure`, one more superclass hop away. Left
        // unmapped, the selection fails and the fallback lowers `return` as
        // IDENTITY — a do-block continuation then returns a raw value where
        // an action is expected.
        let (needed_class, method_name) = match method_name.as_str() {
            "return" => (Symbol::intern("Applicative"), Symbol::intern("pure")),
            "liftA" | "liftM" | "<$>" => (Symbol::intern("Functor"), Symbol::intern("fmap")),
            _ => (needed_class, method_name),
        };

        // Find an in-scope dictionary whose class reaches `needed_class`
        // through the superclass graph (TRANSITIVELY — `MyC ⊃ Monad ⊃
        // Applicative` needs two hops for `pure`), recording the field-index
        // path. Compose the selections DIRECTLY (`$sel_j ($sel_i $d)`): a
        // Let-bound intermediate got lowered as a lazy thunk that failed to
        // capture the enclosing dict param and forced to null.
        let mut found: Option<(Var, Vec<usize>)> = None;
        'outer: for scope in self.dict_scope.iter().rev() {
            for (class_name, dict_var) in scope {
                // Only SINGLE-PARAM classes: a multi-param class's monad
                // superclass slot is a known placeholder (`Stream s m t`'s
                // `Monad m` is deliberately unconstructed — the builtin
                // fast-path world), and selecting through it crashed parsec's
                // `runPT` at runtime.
                let single_param = self
                    .class_registry
                    .lookup_class(*class_name)
                    .is_some_and(|c| c.param_count == 1);
                if !single_param && !self.superclass_hop_matches(*class_name, needed_class) {
                    continue;
                }
                if self.occurrence_refutes_hop(*class_name, needed_class, span) {
                    continue;
                }
                if let Some(path) = self.superclass_field_path(*class_name, needed_class, 0) {
                    found = Some((dict_var.clone(), path));
                    break 'outer;
                }
            }
        }
        let (dict_var, path) = found?;
        if path.is_empty() {
            // The dict IS the needed class — plain method selection.
            return self.select_method_from_dict(&dict_var, needed_class, method_name, span);
        }
        let info = self.class_registry.lookup_class(needed_class)?;
        let method_index = info.methods.iter().position(|m| *m == method_name)?;
        let field_index = info.superclasses.len() + method_index;
        // Walk the superclass chain, binding each intermediate dictionary
        // in a Let — the codegen thunk for the Let CAPTURES the enclosing
        // dict param and the method application is emitted correctly (the
        // pushed baseline reaching pandoc's titleBlock uses this form).
        // Direct `$sel_j ($sel_i $d)` composition makes codegen DROP the
        // method application entirely.
        let mut hops: Vec<(Var, core::Expr)> = Vec::new();
        let mut cur = core::Expr::Var(dict_var, span);
        for idx in path {
            let sel = Var {
                name: Symbol::intern(&format!("$sel_{idx}")),
                id: VarId::new(idx),
                ty: Ty::Error,
            };
            let super_expr =
                core::Expr::App(Box::new(core::Expr::Var(sel, span)), Box::new(cur), span);
            let temp = self.fresh_var("$super", Ty::Error, span);
            cur = core::Expr::Var(temp.clone(), span);
            hops.push((temp, super_expr));
        }
        let sel_var = Var {
            name: Symbol::intern(&format!("$sel_{field_index}")),
            id: VarId::new(field_index),
            ty: Ty::Error,
        };
        let mut result = core::Expr::App(
            Box::new(core::Expr::Var(sel_var, span)),
            Box::new(cur),
            span,
        );
        for (temp, super_expr) in hops.into_iter().rev() {
            result = core::Expr::Let(
                Box::new(Bind::NonRec(temp, Box::new(super_expr))),
                Box::new(result),
                span,
            );
        }
        Some(result)
    }

    /// Field-index path from `from`'s dictionary to (a dictionary of) `to`
    /// through superclass slots; empty when `from == to`.
    fn superclass_field_path(&self, from: Symbol, to: Symbol, depth: usize) -> Option<Vec<usize>> {
        if from == to {
            return Some(Vec::new());
        }
        if depth > 8 {
            return None;
        }
        let info = self.class_registry.lookup_class(from)?;
        for (i, sup) in info.superclasses.iter().enumerate() {
            if let Some(mut rest) = self.superclass_field_path(*sup, to, depth + 1) {
                let mut path = vec![i];
                path.append(&mut rest);
                return Some(path);
            }
        }
        None
    }

    /// Select a method from a dictionary.
    ///
    /// Given a dictionary variable and a method name, returns an expression
    /// that extracts that method from the dictionary.
    #[must_use]
    pub fn select_method_from_dict(
        &self,
        dict_var: &Var,
        class: Symbol,
        method_name: Symbol,
        span: Span,
    ) -> Option<core::Expr> {
        crate::dictionary::select_method(dict_var, class, method_name, &self.class_registry, span)
    }

    /// Check if a symbol is a class method.
    ///
    /// Returns the class name if the symbol is a method of some class.
    #[must_use]
    pub fn is_class_method(&self, method_name: Symbol) -> Option<Symbol> {
        for (class_name, class_info) in &self.class_registry.classes {
            if class_info.methods.contains(&method_name) {
                return Some(*class_name);
            }
        }
        None
    }

    /// Check if a class belongs to the monad family (Functor, Applicative, Monad).
    ///
    /// These classes use codegen fast paths for builtin monads (IO, `StateT`, etc.)
    /// but need dictionary dispatch for user-defined monads. This is distinct from
    /// `is_user_class` because these classes ARE builtin but their instances can be
    /// user-defined.
    #[must_use]
    pub fn is_monad_family_class(&self, class_name: Symbol) -> bool {
        crate::dictionary::MONAD_FAMILY_CLASSES.contains(&class_name.as_str())
    }

    /// Check if a type is a builtin monad that uses codegen fast paths.
    ///
    /// Returns true for IO, `StateT`, `ReaderT`, `ExceptT`, `WriterT`, and Identity
    /// (all of which have hardcoded codegen). Returns false for user-defined
    /// monads that need dictionary dispatch.
    #[must_use]
    pub fn is_builtin_monad_type(ty: &Ty) -> bool {
        let type_name = match ty {
            Ty::Con(tc) => Some(tc.name.as_str()),
            Ty::App(f, _) => match f.as_ref() {
                Ty::Con(tc) => Some(tc.name.as_str()),
                _ => None,
            },
            _ => None,
        };
        matches!(
            type_name,
            Some("IO" | "StateT" | "ReaderT" | "ExceptT" | "WriterT" | "Identity")
        )
    }

    /// Push a monad type onto the context stack.
    ///
    /// Used when lowering the lambda argument of `>>=`/`>>` for a non-builtin monad,
    /// so that `return`/`pure` inside the lambda body can resolve via dictionary dispatch.
    pub fn push_monad_type(&mut self, ty: Ty) {
        self.monad_type_stack.push(ty);
    }

    /// Pop the most recent monad type from the context stack.
    pub fn pop_monad_type(&mut self) {
        self.monad_type_stack.pop();
    }

    /// Get the current monad type from the context stack, if any.
    #[must_use]
    pub fn current_monad_type(&self) -> Option<&Ty> {
        self.monad_type_stack.last()
    }

    /// The type of the instance whose method bodies are currently being lowered,
    /// used to resolve a bare monad-family method in a point-free method body.
    pub(crate) fn current_instance_type(&self) -> Option<&Ty> {
        self.current_instance_type.as_ref()
    }

    /// The declared type of the top-level binding currently being lowered
    /// (see the field doc).
    pub(crate) fn current_binding_sig(&self) -> Option<&Ty> {
        self.current_binding_sig.as_ref()
    }

    /// Whether `class` has an instance at `head` that arrived through a module
    /// interface — i.e. this module is a CONSUMER of the instance rather than
    /// the module implementing it (see `imported_instance_heads`).
    pub(crate) fn has_imported_instance(&self, class: Symbol, head: Symbol) -> bool {
        self.imported_instance_heads.contains(&(class, head))
    }

    /// Narrow the binding signature to a local (`let`/`where`) binding while
    /// its right-hand side is lowered, returning the previous value to hand
    /// back to [`Self::restore_current_binding_sig`]. A `None` argument leaves
    /// the enclosing definition's signature in place, which is still better
    /// context than nothing.
    pub(crate) fn set_current_binding_sig(&mut self, ty: Option<Ty>) -> Option<Ty> {
        match ty {
            Some(ty) => self.current_binding_sig.replace(ty),
            None => self.current_binding_sig.clone(),
        }
    }

    /// Restore the signature saved by [`Self::set_current_binding_sig`].
    pub(crate) fn restore_current_binding_sig(&mut self, saved: Option<Ty>) {
        self.current_binding_sig = saved;
    }

    /// Install the type-synonym definitions threaded from typeck.
    pub fn set_type_aliases(&mut self, aliases: FxHashMap<Symbol, (Vec<bhc_types::TyVar>, Ty)>) {
        self.type_aliases = aliases;
    }

    /// Expand type synonyms in `ty` (recursively, depth-capped): the head of
    /// an application spine naming a registered alias with enough arguments
    /// is replaced by the alias body with its parameters substituted.
    pub(crate) fn expand_type_aliases(&self, ty: &Ty) -> Ty {
        fn go(aliases: &FxHashMap<Symbol, (Vec<bhc_types::TyVar>, Ty)>, ty: &Ty, depth: u32) -> Ty {
            if depth > 16 {
                return ty.clone();
            }
            // Collect the application spine.
            let mut head = ty;
            let mut args: Vec<&Ty> = Vec::new();
            while let Ty::App(f, a) = head {
                args.push(a.as_ref());
                head = f.as_ref();
            }
            args.reverse();
            if let Ty::Con(tc) = head {
                // `String` is a synonym of the language itself, declared in no
                // module, so it never reaches the alias map. Left unexpanded it
                // matches neither `[Char]` in an occurrence type nor the
                // `Stream [tok] m tok` instance head, and a parser declared
                // `ParsecT String () m Char` ends up passed with NO dictionaries.
                if tc.name.as_str() == "String"
                    && args.is_empty()
                    && !aliases.contains_key(&tc.name)
                {
                    return Ty::List(Box::new(Ty::Con(bhc_types::TyCon::new(
                        Symbol::intern("Char"),
                        bhc_types::Kind::Star,
                    ))));
                }
                if let Some((params, rhs)) = aliases.get(&tc.name) {
                    if params.len() <= args.len() {
                        let mut s = bhc_types::Subst::new();
                        for (p, a) in params.iter().zip(&args) {
                            s.insert(p, (*a).clone());
                        }
                        let expanded = s.apply(rhs);
                        let rebuilt = args[params.len()..].iter().fold(expanded, |acc, a| {
                            Ty::App(Box::new(acc), Box::new((*a).clone()))
                        });
                        return go(aliases, &rebuilt, depth + 1);
                    }
                }
            }
            match ty {
                Ty::App(f, a) => Ty::App(
                    Box::new(go(aliases, f, depth + 1)),
                    Box::new(go(aliases, a, depth + 1)),
                ),
                Ty::Fun(a, b) => Ty::Fun(
                    Box::new(go(aliases, a, depth + 1)),
                    Box::new(go(aliases, b, depth + 1)),
                ),
                Ty::List(t) => Ty::List(Box::new(go(aliases, t, depth + 1))),
                Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| go(aliases, t, depth + 1)).collect()),
                other => other.clone(),
            }
        }
        go(&self.type_aliases, ty, 0)
    }

    pub(crate) fn current_instance_class(&self) -> Option<Symbol> {
        self.current_instance_class
    }

    pub(crate) fn superclass_bind_depth(&self) -> usize {
        self.superclass_bind_depth
    }

    pub(crate) fn enter_superclass_bind(&mut self) {
        self.superclass_bind_depth += 1;
    }

    pub(crate) fn exit_superclass_bind(&mut self) {
        self.superclass_bind_depth = self.superclass_bind_depth.saturating_sub(1);
    }

    /// Check if a class name is a user-defined class (not a builtin like Eq, Ord, Show, etc.).
    ///
    /// This is used to determine whether dictionary-passing should be used for a class.
    /// Builtin classes (Eq, Ord, Num, Show, etc.) are dispatched via hardcoded codegen,
    /// while user-defined classes use the dictionary-passing transformation.
    #[must_use]
    pub fn is_user_class(&self, class_name: Symbol) -> bool {
        let name_str = class_name.as_str();
        !crate::dictionary::BUILTIN_CLASS_NAMES.contains(&name_str)
            && self.class_registry.lookup_class(class_name).is_some()
    }

    /// Whether this constraint becomes a runtime dictionary parameter on the
    /// binding that declares it — and therefore must be supplied at every call
    /// site.
    ///
    /// Callee and caller MUST agree here. A binding that takes a dictionary its
    /// callers do not pass has every later argument shifted one slot; a caller
    /// that passes one the binding never bound has the same problem in reverse.
    /// Both sides go through this predicate for exactly that reason.
    ///
    /// Under `BHC_MONAD_WITNESS`, a binding polymorphic in its MONAD needs a
    /// witness too: codegen otherwise picks `return` from the ambient
    /// transformer layer, which for such a binding is `IO`, whose `return` is
    /// identity. Only when the class parameter really is a variable — a
    /// concrete `Monad IO` constraint has nothing to dispatch.
    pub(crate) fn constraint_is_dict_passed(&self, c: &Constraint) -> bool {
        self.is_user_class(c.class)
            || (crate::dictionary::monad_witness_enabled()
                && matches!(c.class.as_str(), "Monad" | "Applicative")
                && matches!(c.args.first(), Some(Ty::Var(_))))
    }

    /// Try to derive an instance for a user-defined typeclass (`DeriveAnyClass`).
    ///
    /// Creates an empty instance (no method bindings) that relies entirely on
    /// default method implementations from the class definition. This mirrors
    /// GHC's `DeriveAnyClass` extension.
    fn try_derive_any_class(
        &self,
        type_name: Symbol,
        params: &[TyVar],
        class_name: Symbol,
        _span: Span,
    ) -> Option<DerivedInstance> {
        if !self.is_user_class(class_name) {
            return None;
        }

        // Build instance type (e.g., `Color` or `Maybe a`)
        let base = Ty::Con(TyCon::new(type_name, Kind::Star));
        let instance_type = if params.is_empty() {
            base
        } else {
            params.iter().fold(base, |acc, param| {
                Ty::App(Box::new(acc), Box::new(Ty::Var(param.clone())))
            })
        };

        let instance = InstanceInfo {
            class: class_name,
            instance_types: vec![instance_type],
            methods: FxHashMap::default(),
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![],
        })
    }

    /// Register a type class definition in the class registry.
    fn register_class_def(&mut self, class_def: &bhc_hir::ClassDef) {
        use crate::dictionary::AssocTypeInfo;

        let mut method_types = FxHashMap::default();
        let mut method_names = Vec::new();

        // Collect method signatures
        for method_sig in &class_def.methods {
            method_names.push(method_sig.name);
            method_types.insert(method_sig.name, method_sig.ty.clone());
        }

        // Collect default method DefIds
        let mut defaults = FxHashMap::default();
        for default_def in &class_def.defaults {
            defaults.insert(default_def.name, default_def.id);
        }

        // Collect associated type declarations
        let assoc_types: Vec<AssocTypeInfo> = class_def
            .assoc_types
            .iter()
            .map(|assoc| AssocTypeInfo {
                name: assoc.name,
                params: assoc.params.clone(),
                kind: assoc.kind.clone(),
                default: assoc.default.clone(),
            })
            .collect();

        let class_info = ClassInfo {
            name: class_def.name,
            param_count: class_def.params.len(),
            methods: method_names,
            method_types,
            superclasses: class_def.supers.clone(),
            superclass_params: class_def.super_params.clone(),
            defaults,
            assoc_types,
        };

        self.class_registry.register_class(class_info);
    }

    /// Register a type class instance definition in the class registry.
    fn register_instance_def(&mut self, instance_def: &bhc_hir::InstanceDef) {
        // Collect method implementations
        let mut methods = FxHashMap::default();
        for method_def in &instance_def.methods {
            methods.insert(method_def.name, method_def.id);

            // Register the method implementation as a variable, but only
            // if not already registered (the first pass may have registered
            // it with a $instance_ prefix name for codegen detection).
            if self.lookup_var(method_def.id).is_none() {
                let var = self.named_var(method_def.name, Ty::Error);
                self.register_var(method_def.id, var);
            }
        }

        // If the class isn't registered — an EXTERNAL class like Data.Default's
        // `Default`, which has no local `class` decl in this module, only this
        // instance — synthesize a minimal class registration from the instance's
        // method names. Without it, `is_class_method`/`is_user_class` don't
        // recognize `def`, so `def :: T` never dispatches by result type and
        // falls through to the unimplemented-external stub (a null value), which
        // then breaks e.g. `def{ field = x }` record updates. Mirrors the
        // synthesis in `register_imported_instances`.
        if self
            .class_registry
            .lookup_class(instance_def.class)
            .is_none()
        {
            let method_names: Vec<Symbol> = instance_def.methods.iter().map(|m| m.name).collect();
            self.class_registry
                .register_class(crate::dictionary::ClassInfo {
                    name: instance_def.class,
                    param_count: instance_def.types.len().max(1),
                    methods: method_names,
                    method_types: FxHashMap::default(),
                    superclasses: Vec::new(),
                    superclass_params: vec![],
                    defaults: FxHashMap::default(),
                    assoc_types: Vec::new(),
                });
        }

        // Collect associated type implementations
        let mut assoc_type_impls = FxHashMap::default();
        for assoc_impl in &instance_def.assoc_type_impls {
            assoc_type_impls.insert(assoc_impl.name, assoc_impl.rhs.clone());
        }

        // Use all instance types (supports multi-param type classes)
        let instance_types = instance_def.types.clone();
        let first_type = instance_types.first().cloned().unwrap_or(Ty::Error);

        // For superclass instances, use the CLASS's superclass list (not the
        // instance's constraints, which may be empty).
        //
        // Which instance type satisfies a given superclass comes from the
        // class's recorded parameter mapping: `class Monad m => Stream s m t`
        // says its superclass is about parameter 1, so `instance Stream
        // Sources m Char` must offer `m` — not `Sources`. Falling back to the
        // first type when the mapping is unknown keeps single-parameter
        // classes working exactly as before.
        let superclass_instances =
            if let Some(class_info) = self.class_registry.lookup_class(instance_def.class) {
                class_info
                    .superclasses
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        class_info
                            .superclass_params
                            .get(i)
                            .and_then(|params| params.first())
                            .and_then(|&p| instance_types.get(p))
                            .cloned()
                            .unwrap_or_else(|| first_type.clone())
                    })
                    .collect()
            } else {
                // Builtin class not in registry — fall back to instance constraints
                instance_def
                    .constraints
                    .iter()
                    .map(|_| first_type.clone())
                    .collect()
            };

        let instance_info = InstanceInfo {
            class: instance_def.class,
            instance_types,
            methods,
            superclass_instances,
            assoc_type_impls,
            instance_constraints: instance_def.constraints.clone(),
        };

        self.class_registry.register_instance(instance_info);
    }

    /// Select a `MonadTrans` method — in practice `lift` — from the instance
    /// for the transformer this occurrence lifts INTO, matched by head
    /// constructor.
    ///
    /// `lift` is otherwise lowered from the ambient transformer layer, and a
    /// user transformer is not one of those layers: a `ParsecT` action falls
    /// back to `TransformerLayer::IO`, where lift is IDENTITY. So `lift
    /// getCommonState` inside a parser yields the bare action and parsec's
    /// bind then runs THAT as a parser. parsec's own
    /// `$instance_lift_ParsecT` was compiled and present the whole time;
    /// nothing ever reached it.
    ///
    /// The transformer comes from the OCCURRENCE's own type — `m a -> ParsecT
    /// s u m a` — because an inline parser passed as an argument has no
    /// binding of its own to ask, and the enclosing one returns in the
    /// CALLER's monad. Matching on the head is what makes this work where
    /// `select_method_by_result_type` cannot: `MonadTrans`'s instance head is
    /// a constructor applied to its own variables, so the occurrence type is
    /// never fully concrete.
    ///
    /// The builtin transformers are deliberately left alone — their `lift` has
    /// a codegen fast path that knows each layer's representation.
    pub(crate) fn select_monad_trans_method(
        &self,
        method: Symbol,
        span: Span,
    ) -> Option<core::Expr> {
        fn head_con(ty: &Ty) -> Option<&bhc_types::TyCon> {
            match ty {
                Ty::Con(tc) => Some(tc),
                Ty::App(f, _) => head_con(f),
                _ => None,
            }
        }
        fn result_of(t: &Ty) -> &Ty {
            match t {
                Ty::Fun(_, r) => result_of(r),
                other => other,
            }
        }
        let occ = self
            .resolved_expr_ty_opt(span)
            .or_else(|| self.expr_ty_opt(span));
        let head = occ
            .as_ref()
            .and_then(|t| head_con(result_of(t)))
            .map(|c| c.name)
            .or_else(|| {
                self.current_binding_sig()
                    .and_then(|s| head_con(result_of(s)))
                    .map(|c| c.name)
            })?;
        if matches!(
            head.as_str(),
            "StateT" | "ReaderT" | "ExceptT" | "WriterT" | "IO"
        ) {
            return None;
        }
        let instances = self
            .class_registry
            .instances
            .get(&Symbol::intern("MonadTrans"))?;
        let inst = instances.iter().find(|i| {
            i.instance_types
                .first()
                .and_then(head_con)
                .is_some_and(|c| c.name == head)
        })?;
        let def_id = inst.methods.get(&method).copied()?;
        let var = self.lookup_var(def_id).cloned()?;
        // Referenced bare. `lift :: Monad m => m a -> t m a` carries its own
        // `Monad m`, but the compiled instance method does not take that
        // dictionary as a leading argument — applying one made parsec's
        // `lift` a partial application of a non-closure.
        Some(core::Expr::Var(var, span))
    }

    /// Resolve a class-method reference by the RESULT type recorded for its
    /// use site. Nullary/result-position methods (`def :: Default a => a`)
    /// have no argument to drive specialization and no dictionary in scope;
    /// the type checker's span-keyed `expr_types` carries the resolved type
    /// (`WriterOptions` at `writeHtml5String def doc`), which picks the
    /// instance directly. Only fires when that type is fully concrete —
    /// a type variable would unify-match an arbitrary instance.
    pub(crate) fn select_method_by_result_type(
        &mut self,
        class_name: Symbol,
        method: Symbol,
        method_def_id: DefId,
        span: Span,
    ) -> Option<bhc_core::Expr> {
        fn ty_is_concrete(ty: &Ty) -> bool {
            match ty {
                Ty::Var(_) | Ty::Error => false,
                Ty::Con(_) | Ty::Prim(_) => true,
                Ty::App(f, a) => ty_is_concrete(f) && ty_is_concrete(a),
                Ty::Fun(a, b) => ty_is_concrete(a) && ty_is_concrete(b),
                Ty::List(t) => ty_is_concrete(t),
                Ty::Tuple(ts) => ts.iter().all(ty_is_concrete),
                _ => false,
            }
        }
        let dbg = std::env::var("BHC_DBG_DISPATCH").is_ok();
        // Prefer the fully-substituted resolved channel: the single-pass
        // `expr_types` map can leave nested variables unsubstituted
        // (`getB`'s occurrence read as `(t -> t') -> t''`), and dispatch on
        // an unresolved type silently fails to the stub path.
        let ty = match self
            .resolved_expr_ty_opt(span)
            .or_else(|| self.expr_ty_opt(span))
        {
            Some(t) => t,
            None => {
                if dbg {
                    eprintln!("[dispatch] {method}: no expr type at {span:?}");
                }
                return None;
            }
        };
        if dbg {
            eprintln!("[dispatch] {method}: occ ty = {ty:?}");
        }
        // Direct: for nullary/def-like methods the recorded type IS the
        // class parameter's instantiation.
        if ty_is_concrete(&ty) {
            if let Some(expr) = self.method_at_instance_type(class_name, method, &ty, span) {
                return Some(expr);
            }
        }
        // Scheme-match: for methods whose class parameter appears in neither
        // an argument nor the result head (`getOption :: (ReaderOptions -> b)
        // -> ParsecT s st m b`), the recorded type is the full OCCURRENCE
        // type. Match the method's declared scheme against it and read the
        // class param's instantiation off the implicit class constraint
        // (`st := ParserState`), then resolve the instance at that type.
        let scheme = match self.lookup_scheme(method_def_id) {
            Some(s) => s.clone(),
            None => {
                if dbg {
                    eprintln!("[dispatch] {method}: no scheme for {method_def_id:?}");
                }
                return None;
            }
        };
        if dbg {
            eprintln!(
                "[dispatch] {method}: scheme ty = {:?}, constraints = {:?}",
                scheme.ty, scheme.constraints
            );
        }
        let constraint_arg = scheme
            .constraints
            .iter()
            .find(|c| c.class == class_name)
            .and_then(|c| c.args.first())
            .cloned()?;
        // `types_match` is all-or-nothing, and typeck routinely records an
        // occurrence only partially resolved: the nullary `tag :: m Int` used
        // at `MyM` comes back as `MyM ?594`, whose element position defeats the
        // match even though the class parameter — the only part that selects an
        // instance — is pinned to `MyM`. Rejecting that dropped the method to
        // the stub path and aborted at runtime. Fall back to the lenient
        // one-way matcher, which records what it can and ignores positions it
        // cannot align; the concreteness check on `inst_ty` just below is what
        // actually gates dispatch.
        let subst = bhc_types::types_match(&scheme.ty, &ty).unwrap_or_else(|| {
            if dbg {
                eprintln!("[dispatch] {method}: types_match failed, trying lenient match");
            }
            let mut s = bhc_types::Subst::new();
            crate::expr::match_ty(&scheme.ty, &ty, &mut s);
            s
        });
        let inst_ty = subst.apply(&constraint_arg);
        if dbg {
            eprintln!("[dispatch] {method}: inst_ty = {inst_ty:?}");
        }
        if !ty_is_concrete(&inst_ty) {
            return None;
        }
        self.method_at_instance_type(class_name, method, &inst_ty, span)
    }

    /// Resolve `method` of `class_name` at a concrete instance type: prefer
    /// the instance's own implementation var; otherwise construct the
    /// dictionary (which applies class DEFAULTS to a partial dict) and
    /// select the method from it.
    fn method_at_instance_type(
        &mut self,
        class_name: Symbol,
        method: Symbol,
        inst_ty: &Ty,
        span: Span,
    ) -> Option<bhc_core::Expr> {
        let own_method = {
            let (instance, _subst) = self.class_registry.resolve_instance(class_name, inst_ty)?;
            instance.methods.get(&method).copied()
        };
        if let Some(def_id) = own_method {
            let var = self.lookup_var(def_id)?.clone();
            return Some(bhc_core::Expr::Var(var, span));
        }
        // Instance omits the method (class default): go through dictionary
        // construction, which fills the slot by applying the default fn to a
        // partial dict.
        self.resolve_method_at_concrete_type(method, class_name, inst_ty, span)
    }

    /// Register classes and instances loaded from module interfaces so
    /// class-method calls on concrete types specialize to
    /// `$instance_{method}_{TypeEnc}` variables — codegen resolves those
    /// names against module-qualified externs (e.g. `toSources` on Text in
    /// a dependent of Text.Pandoc.Sources dispatches to the extern
    /// `Text.Pandoc.Sources.$instance_toSources_Text` instead of a stub).
    /// Synthetic DefIds start far above real HIR DefIds.
    #[allow(clippy::type_complexity)]
    pub fn register_imported_instances(
        &mut self,
        classes: &[(
            Symbol,
            Vec<Symbol>,
            Vec<Symbol>,
            Vec<Symbol>,
            Vec<(Symbol, usize)>,
            usize,
            Vec<Vec<usize>>,
        )],
        instances: &[(Symbol, Vec<Ty>, Vec<Symbol>, Vec<Constraint>)],
    ) {
        let mut next_default_id: usize = 800_000;
        for (
            class_name,
            method_names,
            superclass_names,
            defaulted_names,
            method_arities,
            param_count,
            superclass_param_indices,
        ) in classes
        {
            if self.class_registry.lookup_class(*class_name).is_some() {
                continue; // local declaration wins
            }
            // Class-body defaults: the defining module exports the default
            // as a top-level dict-taking fn named after the method; register
            // a var with that bare name so codegen resolves it against the
            // module-qualified extern from `interface_symbols`
            // (`Text.Pandoc.Parsing.Capabilities.getOption`). Dictionary
            // construction then applies it to the partial dict when an
            // instance omits the method — the cross-module analogue of the
            // same-module `construct_dictionary` defaults path.
            let mut defaults = FxHashMap::default();
            for method in defaulted_names {
                let def_id = DefId::new(next_default_id);
                next_default_id += 1;
                let var = self.named_var(*method, Ty::Error);
                self.register_var(def_id, var);
                defaults.insert(*method, def_id);
            }
            // Arrow-skeleton method types: only the ARROW COUNT of the
            // declared method type survives the interface (as `(name, n)`
            // pairs); rebuild `Fun(Error, … Error)` chains so consumers that
            // need the arity (`eta_expand_point_free_method`) can count
            // arrows the same way as for locally-declared classes.
            let mut method_types = FxHashMap::default();
            for (mname, arrows) in method_arities {
                let mut t = Ty::Error;
                for _ in 0..*arrows {
                    t = Ty::Fun(Box::new(Ty::Error), Box::new(t));
                }
                method_types.insert(*mname, bhc_types::Scheme::mono(t));
            }
            self.class_registry
                .register_class(crate::dictionary::ClassInfo {
                    name: *class_name,
                    // The REAL parameter count from the interface: the
                    // superclass-selection guard keys on it, and hardcoding 1
                    // let dispatch walk through a multi-param class's
                    // deliberately-null superclass slot (Stream's Monad).
                    param_count: (*param_count).max(1),
                    methods: method_names.clone(),
                    method_types,
                    // Carry the class's superclasses: `select_method` places
                    // methods after `superclasses.len()` slots, so an importing
                    // module must agree with the defining module on the count
                    // or it selects the wrong dictionary field.
                    superclasses: superclass_names.clone(),
                    superclass_params: superclass_param_indices.clone(),
                    defaults,
                    assoc_types: Vec::new(),
                });
        }

        let mut next_id: usize = 900_000;
        for (class, types, method_names, constraints) in instances {
            // Remember that this instance came from an interface, so a method
            // use site can tell itself apart from the instance's own module.
            if let Some(head) = types.first().and_then(head_type_con_name) {
                self.imported_instance_heads.insert((*class, head));
            }
            // Externally-defined class (Default): no interface declares it,
            // so synthesize a minimal registration from the instance's own
            // method list — otherwise `def` isn't recognized as a class
            // method and never specializes.
            if self.class_registry.lookup_class(*class).is_none() {
                self.class_registry
                    .register_class(crate::dictionary::ClassInfo {
                        name: *class,
                        param_count: 1,
                        methods: method_names.clone(),
                        method_types: FxHashMap::default(),
                        superclasses: Vec::new(),
                        superclass_params: vec![],
                        defaults: FxHashMap::default(),
                        assoc_types: Vec::new(),
                    });
            }
            let inst_type_name = if types.is_empty() {
                "Unknown".to_string()
            } else {
                types
                    .iter()
                    .map(type_name_for_instance)
                    .collect::<Vec<_>>()
                    .join("_")
            };
            let mut methods = FxHashMap::default();
            for method in method_names {
                let def_id = DefId::new(next_id);
                next_id += 1;
                let instance_name =
                    Symbol::intern(&format!("$instance_{}_{}", method, inst_type_name));
                let var = self.named_var(instance_name, Ty::Error);
                self.register_var(def_id, var);
                methods.insert(*method, def_id);
            }
            // Superclass instances are at the same type (single-param classes:
            // `Monad N` needs `Applicative N`, which needs `Functor N`). Without
            // these, dictionary construction for an imported `Applicative`/`Monad`
            // instance fails on the superclass slot (construct_dictionary reads
            // `superclass_instances.get(i)?`) and `pure`/`>>=` never dispatch —
            // the cross-module analogue of the same-type superclass wiring the
            // derivers set locally.
            let superclass_instances: Vec<Ty> =
                match (types.first(), self.class_registry.lookup_class(*class)) {
                    (Some(ty), Some(cls)) => vec![ty.clone(); cls.superclasses.len()],
                    _ => Vec::new(),
                };
            self.class_registry
                .register_instance(crate::dictionary::InstanceInfo {
                    class: *class,
                    instance_types: types.clone(),
                    methods,
                    superclass_instances,
                    assoc_type_impls: FxHashMap::default(),
                    // An instance's OWN constraints give its methods a
                    // dictionary parameter each. Dropping them here (as this
                    // did, unconditionally) left consumers building the
                    // dictionary with a BARE method while the defining module
                    // had compiled it expecting the dictionary — parsec's
                    // `instance Monad m => Stream [tok] m tok` then had
                    // `tokenPrimEx` apply the stream into `uncons`'s dictionary
                    // slot and every token read failed.
                    instance_constraints: constraints.clone(),
                });
        }
    }

    /// Lower a HIR module to Core.
    pub fn lower_module(&mut self, module: &HirModule) -> LowerResult<CoreModule> {
        // Propagate extension flags
        self.generalized_newtype_deriving = module.generalized_newtype_deriving;

        // First pass: collect all top-level definitions and create Core variables
        // We use named_var here to preserve the original names for external visibility
        for item in &module.items {
            match item {
                Item::Value(value_def) => {
                    // Look up the type from the type checker
                    let ty = self.lookup_type(value_def.id);
                    let var = self.named_var(value_def.name, ty);
                    self.register_var(value_def.id, var);
                }
                Item::Class(class_def) => {
                    // Register the class in the class registry FIRST so that
                    // is_class_method() works when lowering value definitions
                    // that reference class methods.
                    self.register_class_def(class_def);

                    // Register variables for class methods so that references
                    // to them (via DefRef) can be resolved during lowering.
                    for method_sig in &class_def.methods {
                        let ty = self.lookup_type(method_sig.id);
                        let var = self.named_var(method_sig.name, ty);
                        self.register_var(method_sig.id, var);
                    }

                    // Also register variables for default method implementations
                    for default_def in &class_def.defaults {
                        let ty = self.lookup_type(default_def.id);
                        let var = self.named_var(default_def.name, ty);
                        self.register_var(default_def.id, var);
                    }
                }
                Item::Instance(instance_def) => {
                    // Register the instance in the class registry FIRST so that
                    // dictionary construction works when lowering value definitions.
                    self.register_instance_def(instance_def);

                    // Pre-register instance method variables so they can be
                    // referenced during the lowering pass.
                    // Use $instance_{method}_{TypeName} naming convention
                    // so codegen can detect and dispatch manual instance methods.
                    // For multi-param classes, join all type names with "_".
                    let inst_type_name = if instance_def.types.is_empty() {
                        "Unknown".to_string()
                    } else {
                        instance_def
                            .types
                            .iter()
                            .map(type_name_for_instance)
                            .collect::<Vec<_>>()
                            .join("_")
                    };
                    for method_def in &instance_def.methods {
                        let ty = self.lookup_type(method_def.id);
                        let instance_name = Symbol::intern(&format!(
                            "$instance_{}_{}",
                            method_def.name, inst_type_name
                        ));
                        let var = self.named_var(instance_name, ty);
                        self.register_var(method_def.id, var);
                    }
                }
                Item::Foreign(foreign) => {
                    // Pre-register foreign import variables so they're available
                    // when lowering expressions that reference them.
                    let ty = foreign.ty.ty.clone();
                    let var = self.named_var(foreign.name, ty);
                    self.register_var(foreign.id, var);
                }
                _ => {}
            }
        }

        // Second pass: lower all items
        let mut bindings = Vec::new();
        let mut deriv_ctx = DerivingContext::new();

        // Pre-scan which data types derive Eq/Ord (stock-style) so a derived
        // comparison can recurse into a field of a sibling user ADT via that
        // type's own `$derived_eq_/$derived_compare_` — only safe for types
        // whose binding is generated here. Newtypes are intentionally excluded
        // (they derive via the identity path).
        {
            let mut local_eq = rustc_hash::FxHashSet::default();
            let mut local_ord = rustc_hash::FxHashSet::default();
            for item in &module.items {
                if let Item::Data(data_def) = item {
                    for clause in &data_def.deriving {
                        if matches!(
                            clause.strategy,
                            bhc_hir::DerivingStrategy::Stock
                                | bhc_hir::DerivingStrategy::Default
                                | bhc_hir::DerivingStrategy::Newtype
                        ) {
                            match clause.class.as_str() {
                                "Eq" => {
                                    local_eq.insert(data_def.name);
                                }
                                "Ord" => {
                                    // `$derived_eq_` is keyed off an explicit
                                    // `Eq` clause (always present for a valid
                                    // `Ord` deriver), handled by the arm above.
                                    local_ord.insert(data_def.name);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            deriv_ctx.set_local_derives(local_eq, local_ord);
        }

        for item in &module.items {
            match item {
                Item::Value(value_def) => {
                    if let Some(bind) = self.lower_value_def(value_def)? {
                        bindings.push(bind);
                    }
                }
                Item::Data(data_def) => {
                    // Register data constructors with their metadata
                    // The tag is the 0-based position in the constructor list
                    for (tag, con) in data_def.cons.iter().enumerate() {
                        let var = self.named_var(con.name, Ty::Error);
                        self.register_var(con.id, var);

                        // Calculate arity and field names based on field type
                        let (arity, field_names) = match &con.fields {
                            bhc_hir::ConFields::Positional(fields) => (fields.len() as u32, vec![]),
                            bhc_hir::ConFields::Named(fields) => {
                                // Register field selector functions
                                for field in fields {
                                    // The accessor is emitted as a separate value
                                    // binding (id == field.id) and its var is
                                    // already registered by the pre-pass over all
                                    // items. Do NOT overwrite it with a fresh var
                                    // here: a value that forward-references the
                                    // accessor (a field used above its `data`
                                    // declaration, e.g. Text.Parsec's
                                    // `unknownError s = … statePos s` above
                                    // `data State`) is lowered before this Data
                                    // item and would then reference the pre-pass
                                    // var while the accessor binding uses the
                                    // fresh one — an undeclared, stubbed selector.
                                    if self.lookup_var(field.id).is_none() {
                                        let selector_var = self.named_var(field.name, Ty::Error);
                                        self.register_var(field.id, selector_var);
                                    }
                                    // Also register field metadata for later lookup
                                    self.register_field_selector(
                                        field.name,
                                        FieldSelectorInfo {
                                            field_name: field.name,
                                            con_id: con.id,
                                            con_name: con.name,
                                            type_name: data_def.name,
                                            field_index: fields
                                                .iter()
                                                .position(|f| f.id == field.id)
                                                .unwrap_or(0),
                                            total_fields: fields.len(),
                                        },
                                    );
                                }
                                let names: Vec<Symbol> = fields.iter().map(|f| f.name).collect();
                                (fields.len() as u32, names)
                            }
                        };

                        // Record declared field types (already `bhc_types::Ty`)
                        // so codegen can recover a pattern-bound field's type for
                        // `show`/`print` dispatch (otherwise erased).
                        let field_tys: Vec<Ty> = match &con.fields {
                            bhc_hir::ConFields::Positional(tys) => tys.clone(),
                            bhc_hir::ConFields::Named(fs) => {
                                fs.iter().map(|f| f.ty.clone()).collect()
                            }
                        };
                        self.constructor_field_types.insert(con.name, field_tys);

                        // Register constructor metadata
                        // Only count user-defined class constraints for dict fields.
                        // Builtin classes (Show, Eq, etc.) use codegen dispatch, not dicts.
                        let user_existential: Vec<Symbol> = con
                            .existential_context
                            .iter()
                            .filter(|c| self.is_user_class(c.class))
                            .map(|c| c.class)
                            .collect();
                        let existential_dict_count = user_existential.len() as u32;
                        let existential_classes = user_existential;
                        // Arity includes dict fields for existential constructors
                        let total_arity = arity + existential_dict_count;
                        self.register_constructor(
                            con.id,
                            ConstructorInfo {
                                name: con.name,
                                type_name: data_def.name,
                                tag: tag as u32,
                                arity: total_arity,
                                field_names,
                                is_newtype: false,
                                existential_dict_count,
                                existential_classes,
                            },
                        );
                    }

                    // Process deriving clauses with strategy dispatch
                    if !data_def.deriving.is_empty() {
                        let derived_instances: Vec<_> = data_def
                            .deriving
                            .iter()
                            .filter_map(|clause| {
                                let class_name = clause.class;
                                match &clause.strategy {
                                    bhc_hir::DerivingStrategy::Stock
                                    | bhc_hir::DerivingStrategy::Default => deriv_ctx
                                        .derive_for_data(data_def, class_name)
                                        .or_else(|| {
                                            self.try_derive_any_class(
                                                data_def.name,
                                                &data_def.params,
                                                class_name,
                                                data_def.span,
                                            )
                                        }),
                                    bhc_hir::DerivingStrategy::Anyclass
                                    | bhc_hir::DerivingStrategy::Via(_) => self
                                        .try_derive_any_class(
                                            data_def.name,
                                            &data_def.params,
                                            class_name,
                                            data_def.span,
                                        )
                                        .or_else(|| {
                                            deriv_ctx.derive_empty_instance(
                                                data_def.name,
                                                &data_def.params,
                                                class_name,
                                            )
                                        }),
                                    bhc_hir::DerivingStrategy::Newtype => {
                                        // Newtype strategy on a data type: fall back to stock
                                        deriv_ctx.derive_for_data(data_def, class_name).or_else(
                                            || {
                                                self.try_derive_any_class(
                                                    data_def.name,
                                                    &data_def.params,
                                                    class_name,
                                                    data_def.span,
                                                )
                                            },
                                        )
                                    }
                                }
                            })
                            .collect();
                        for derived in derived_instances {
                            self.class_registry.register_instance(derived.instance);
                            bindings.extend(derived.bindings);
                        }
                    }
                }
                Item::Newtype(newtype_def) => {
                    // Register the newtype constructor
                    let var = self.named_var(newtype_def.con.name, Ty::Error);
                    self.register_var(newtype_def.con.id, var);

                    // Register constructor metadata for codegen (newtype = identity)
                    let arity = match &newtype_def.con.fields {
                        bhc_hir::ConFields::Positional(fields) => fields.len() as u32,
                        bhc_hir::ConFields::Named(fields) => fields.len() as u32,
                    };
                    let field_names = match &newtype_def.con.fields {
                        bhc_hir::ConFields::Positional(_) => vec![],
                        bhc_hir::ConFields::Named(fields) => {
                            fields.iter().map(|f| f.name).collect()
                        }
                    };
                    self.register_constructor(
                        newtype_def.con.id,
                        ConstructorInfo {
                            name: newtype_def.con.name,
                            type_name: newtype_def.name,
                            tag: 0,
                            arity,
                            field_names,
                            is_newtype: true,
                            existential_dict_count: 0,
                            existential_classes: vec![],
                        },
                    );

                    // Process deriving clauses with strategy dispatch
                    if !newtype_def.deriving.is_empty() {
                        let derived_instances: Vec<_> = newtype_def
                            .deriving
                            .iter()
                            .filter_map(|clause| {
                                let class_name = clause.class;
                                match &clause.strategy {
                                    bhc_hir::DerivingStrategy::Stock => {
                                        // Stock: use built-in newtype derivation
                                        deriv_ctx
                                            .derive_for_newtype(newtype_def, class_name)
                                            .or_else(|| {
                                                self.try_derive_any_class(
                                                    newtype_def.name,
                                                    &newtype_def.params,
                                                    class_name,
                                                    newtype_def.span,
                                                )
                                            })
                                    }
                                    bhc_hir::DerivingStrategy::Newtype
                                    | bhc_hir::DerivingStrategy::Via(_) => {
                                        // GND / DerivingVia: empty instance (inner type's
                                        // dictionary works directly since newtypes are erased)
                                        deriv_ctx
                                            .derive_empty_instance(
                                                newtype_def.name,
                                                &newtype_def.params,
                                                class_name,
                                            )
                                            .or_else(|| {
                                                self.try_derive_any_class(
                                                    newtype_def.name,
                                                    &newtype_def.params,
                                                    class_name,
                                                    newtype_def.span,
                                                )
                                            })
                                    }
                                    bhc_hir::DerivingStrategy::Anyclass => self
                                        .try_derive_any_class(
                                            newtype_def.name,
                                            &newtype_def.params,
                                            class_name,
                                            newtype_def.span,
                                        )
                                        .or_else(|| {
                                            deriv_ctx.derive_empty_instance(
                                                newtype_def.name,
                                                &newtype_def.params,
                                                class_name,
                                            )
                                        }),
                                    bhc_hir::DerivingStrategy::Default => {
                                        // Default heuristic: try stock first, then anyclass
                                        deriv_ctx
                                            .derive_for_newtype(newtype_def, class_name)
                                            .or_else(|| {
                                                self.try_derive_any_class(
                                                    newtype_def.name,
                                                    &newtype_def.params,
                                                    class_name,
                                                    newtype_def.span,
                                                )
                                            })
                                    }
                                }
                            })
                            .collect();
                        for derived in derived_instances {
                            self.class_registry.register_instance(derived.instance);
                            bindings.extend(derived.bindings);
                        }
                    }
                }
                Item::TypeAlias(_) => {
                    // Type aliases don't produce bindings
                }
                Item::Class(class_def) => {
                    // Class already registered in first pass (register_class_def)

                    // Lower default method implementations
                    // Default methods need the class constraint, so we lower them specially
                    for default_def in &class_def.defaults {
                        if let Some(bind) = self.lower_default_method(class_def, default_def)? {
                            bindings.push(bind);
                        }
                    }
                }
                Item::Instance(instance_def) => {
                    // Instance already registered in first pass (register_instance_def)

                    // Check if this instance has constraints that need
                    // dictionary parameters (e.g., `Describable a =>
                    // Describable (Box a)`). Monad-family/value classes count
                    // too: `instance Monoid a => Monoid (Future s a)` has
                    // `mempty = return mempty`, whose inner `mempty` must
                    // select from the CONSTRAINT dictionary — without the dict
                    // lambda it mis-dispatched back to the Future instance and
                    // looped forever. The filter MUST be the shared
                    // `constraint_needs_dict_param` predicate so the lambda
                    // arity matches `construct_dictionary`'s `constraint_dicts`.
                    let inst_constraints: Vec<Constraint> = instance_def
                        .constraints
                        .iter()
                        .filter(|c| self.class_registry.constraint_needs_dict_param(c.class))
                        .cloned()
                        .collect();

                    // Lower instance method bodies to Core bindings.
                    // Each method in the instance provides an implementation that
                    // the evaluator needs to find.
                    //
                    // Record the instance type so a bare monad-family method used
                    // as a value in a point-free method body (parsec's
                    // `instance Alternative (ParsecT …) where (<|>) = mplus`)
                    // resolves to this type's instance method.
                    let prev_instance_type = self.current_instance_type.take();
                    self.current_instance_type = instance_def.types.first().cloned();
                    let prev_instance_class = self.current_instance_class.take();
                    self.current_instance_class = Some(instance_def.class);
                    for method_def in &instance_def.methods {
                        if inst_constraints.is_empty() {
                            // No instance constraints — lower normally
                            // (`lower_value_def` eta-expands point-free
                            // bodies to the declared arity).
                            if let Some(bind) = self.lower_value_def(method_def)? {
                                bindings.push(bind);
                            }
                        } else {
                            // Instance has user-class constraints — wrap method body
                            // with dict lambdas so dictionary construction can apply them.
                            if let Some(bind) = self.lower_instance_method_with_constraints(
                                method_def,
                                &inst_constraints,
                            )? {
                                bindings.push(bind);
                            }
                        }
                    }
                    self.current_instance_type = prev_instance_type;
                    self.current_instance_class = prev_instance_class;
                }
                Item::Fixity(_) => {
                    // Fixity declarations are only used during parsing
                }
                Item::Foreign(foreign) => {
                    // Use the var pre-registered in the first pass
                    let var = self.lookup_var(foreign.id).cloned().unwrap_or_else(|| {
                        let ty = foreign.ty.ty.clone();
                        self.named_var(foreign.name, ty)
                    });

                    // Map calling convention
                    let convention = match foreign.convention {
                        bhc_hir::ForeignConvention::CCall => core::ForeignConv::CCall,
                        bhc_hir::ForeignConvention::StdCall => core::ForeignConv::StdCall,
                        bhc_hir::ForeignConvention::JavaScript => core::ForeignConv::CCall,
                    };

                    // Map safety
                    let safety = match foreign.safety {
                        bhc_hir::ForeignSafety::Safe => core::ForeignSafety::Safe,
                        bhc_hir::ForeignSafety::Unsafe => core::ForeignSafety::Unsafe,
                        bhc_hir::ForeignSafety::Interruptible => core::ForeignSafety::Interruptible,
                    };

                    // Collect the foreign import for codegen
                    self.foreign_imports.push(core::ForeignImport {
                        haskell_name: foreign.name,
                        c_name: foreign.foreign_name,
                        var,
                        convention,
                        safety,
                    });
                }
                Item::StandaloneDeriving(sd) => {
                    // Look up the data type by name in the module items
                    // and derive the requested class for it.
                    let class_name = sd.class;
                    let type_name = sd.type_name;

                    // Search for matching data or newtype definition
                    let mut found = false;
                    for other_item in &module.items {
                        match other_item {
                            Item::Data(data_def) if data_def.name == type_name => {
                                if let Some(derived) =
                                    deriv_ctx.derive_for_data(data_def, class_name).or_else(|| {
                                        self.try_derive_any_class(
                                            data_def.name,
                                            &data_def.params,
                                            class_name,
                                            data_def.span,
                                        )
                                    })
                                {
                                    self.class_registry.register_instance(derived.instance);
                                    bindings.extend(derived.bindings);
                                }
                                found = true;
                                break;
                            }
                            Item::Newtype(newtype_def) if newtype_def.name == type_name => {
                                if let Some(derived) = deriv_ctx
                                    .derive_for_newtype(newtype_def, class_name)
                                    .or_else(|| {
                                        self.try_derive_any_class(
                                            newtype_def.name,
                                            &newtype_def.params,
                                            class_name,
                                            newtype_def.span,
                                        )
                                    })
                                {
                                    self.class_registry.register_instance(derived.instance);
                                    bindings.extend(derived.bindings);
                                }
                                found = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                    if !found {
                        // Type not found — silently ignore (may be imported)
                    }
                }
                Item::PatternSynonym(_) => {
                    // Pattern synonyms are fully handled during AST→HIR lowering.
                    // No Core bindings needed.
                }
                Item::TypeFamily(_)
                | Item::TypeFamilyInst(_)
                | Item::DataFamily(_)
                | Item::DataFamilyInst(_) => {
                    // Type/data families are purely type-level; no Core bindings needed.
                    // Data family instance constructors are handled through the normal
                    // constructor registration path during codegen.
                }
            }
        }

        // Check for errors
        if self.has_errors() {
            return Err(LowerError::Multiple(self.take_errors()));
        }

        // Print any warnings (non-fatal diagnostics)
        let warnings = self.take_warnings();
        for warning in &warnings {
            eprintln!("{warning}");
        }

        // Collect constructor metadata for codegen
        let constructors: Vec<CoreConstructor> = self
            .constructor_map
            .values()
            .map(|info| CoreConstructor {
                name: info.name.as_str().to_string(),
                tag: info.tag,
                arity: info.arity,
                type_name: Some(info.type_name.as_str().to_string()),
                is_newtype: info.is_newtype,
                field_types: self
                    .constructor_field_types
                    .get(&info.name)
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect();

        let foreign_imports = std::mem::take(&mut self.foreign_imports);

        lazify_recursive_parser_calls(&mut bindings);

        Ok(CoreModule {
            name: module.name,
            bindings,
            exports: vec![],
            foreign_imports,
            overloaded_strings: module.overloaded_strings,
            constructors,
        })
    }

    /// Lower a value definition to a Core binding.
    fn lower_value_def(&mut self, value_def: &ValueDef) -> LowerResult<Option<Bind>> {
        // A composition-valued binding with fewer patterns than its type has
        // arrows (`many1TillChar p = fmap T.pack . many1Till p`) must be
        // eta-expanded at the HIR level: once the composition is LOWERED, the
        // partial `fmap T.pack` inside it is a dead value the method-dispatch
        // machinery can no longer see (it lowered to a broken builtin
        // partial). Rewriting `rhs` to `fmap T.pack (many1Till p $eta)`
        // BEFORE lowering lets the ordinary App-chain dispatch handle it.
        let rewritten;
        let value_def = match self.hir_eta_expand_composition(value_def) {
            Some(v) => {
                rewritten = v;
                &rewritten
            }
            None => value_def,
        };
        let join_rewritten;
        let value_def = match self.hir_rewrite_join(value_def) {
            Some(v) => {
                join_rewritten = v;
                &join_rewritten
            }
            None => value_def,
        };
        let mut var = self
            .lookup_var(value_def.id)
            .cloned()
            .ok_or_else(|| LowerError::Internal("missing variable for value def".into()))?;
        // Codegen derives this binding's transformer layer from its TYPE, and
        // a monad written through a synonym hides the transformer behind the
        // synonym's own constructor: `type S = StateT Int IO; go :: S Int`
        // arrives as `App(Con S, Int)`, no layer is detected, and `return`
        // compiles at the ambient IO layer — where it is IDENTITY — so
        // `evalStateT` is handed a raw value instead of a state function.
        // `current_binding_sig` is expanded just below for the same reason.
        var.ty = self.expand_type_aliases(&var.ty);

        // Record the binding's declared type so occurrence-type gaps inside
        // the body can be repaired against the signature (see
        // `current_binding_sig`). Saved/restored because value defs nest.
        let saved_binding_sig = self.current_binding_sig.take();
        // Expand type synonyms ONCE here, not per occurrence: the fallback in
        // dictionary resolution reads this on a hot path.
        let declared = self.lookup_scheme(value_def.id).map(|s| s.ty.clone());
        self.current_binding_sig = declared.map(|t| self.expand_type_aliases(&t));

        // Check if the definition has user-defined class constraints.
        // Only user-defined classes use dictionary-passing; builtin classes
        // (Eq, Ord, Num, Show, etc.) are dispatched via hardcoded codegen.
        let scheme = self.lookup_scheme(value_def.id);
        let user_constraints: Vec<_> = scheme
            .map(|s| {
                s.constraints
                    .iter()
                    .filter(|c| self.constraint_is_dict_passed(c))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // If there are user-defined class constraints, create dictionary variables
        // and push them into scope BEFORE compiling the body.
        let dict_vars: Vec<(Symbol, Var)> = user_constraints
            .iter()
            .map(|c| {
                let dict_var = self.make_dict_var(c);
                (c.class, dict_var)
            })
            .collect();

        // Push a new dictionary scope and register all dictionaries
        if !dict_vars.is_empty() {
            self.push_dict_scope();
            for ((class_name, dict_var), constraint) in dict_vars.iter().zip(&user_constraints) {
                if constraint.args.is_empty() {
                    self.register_dict(*class_name, dict_var.clone());
                } else {
                    self.register_dict_at(*class_name, constraint.args.clone(), dict_var.clone());
                }
            }
        }

        // Push the declared RESULT type into each equation's right-hand side.
        // A binding's own signature is the only place the concrete stream type
        // appears in `myOLS mb = try (maybe anyOrderedListMarker … mb)`:
        // nothing CALLS this parser at a known type inside the module, so
        // without the hint `anyOrderedListMarker`'s `Stream` dictionary
        // resolves to a null placeholder and the parser is run through it.
        if let Some(sig) = self.current_binding_sig.clone() {
            for eq in &value_def.equations {
                let mut result = &sig;
                let mut peeled = 0;
                while peeled < eq.pats.len() {
                    let Ty::Fun(_, ret) = result else { break };
                    result = ret.as_ref();
                    peeled += 1;
                }
                if peeled == eq.pats.len() {
                    let result = result.clone();
                    crate::expr::propagate_expected_ty(self, &eq.rhs, &result, 0);
                }
            }
        }

        // Compile equations to a single expression (now with dictionaries in scope)
        let mut body = self.compile_equations(value_def)?;

        // Pop the dictionary scope
        if !dict_vars.is_empty() {
            self.pop_dict_scope();
        }

        // Align Core arity with the declared type for point-free bindings
        // (`lookupEnv = fmap … . liftIO . …`): the interface records the Core
        // lambda count as the symbol's arity, and importers apply/CAF-call
        // based on it — a 0-lambda function-typed binding makes every
        // importer evaluate it with no arguments (see
        // `eta_expand_point_free_method`).
        body = self.eta_expand_point_free_method(body, value_def);

        // If there are constraints, wrap the body in dictionary lambdas.
        // For example, a function `f :: Num a => a -> a` becomes:
        //   f = \$dNum -> \x -> ... (using $dNum for Num operations)
        if !dict_vars.is_empty() {
            // Add dictionary parameters in reverse order so the first
            // constraint gets the outermost lambda
            for (_, dict_var) in dict_vars.into_iter().rev() {
                body = core::Expr::Lam(dict_var, Box::new(body), value_def.span);
            }
        }

        self.current_binding_sig = saved_binding_sig;
        Ok(Some(Bind::NonRec(var, Box::new(body))))
    }

    /// Lower an instance method that has instance-level constraints.
    ///
    /// For example, in:
    /// ```text
    /// instance Describable a => Describable (Box a) where
    ///   describe (Box x) = "Box(" ++ describe x ++ ")"
    /// ```
    ///
    /// The `describe` method needs a `Describable a` dictionary to call `describe x`.
    /// We wrap the method body: `\$dDescribable -> \(Box x) -> "Box(" ++ describe x ++ ")"`
    /// so that dictionary construction can apply the constraint dict.
    fn lower_instance_method_with_constraints(
        &mut self,
        method_def: &ValueDef,
        inst_constraints: &[Constraint],
    ) -> LowerResult<Option<Bind>> {
        let var = self
            .lookup_var(method_def.id)
            .cloned()
            .ok_or_else(|| LowerError::Internal("missing variable for instance method".into()))?;

        // Create dict variables for each instance constraint
        let dict_vars: Vec<(Symbol, Var)> = inst_constraints
            .iter()
            .map(|c| {
                let dict_var = self.make_dict_var(c);
                (c.class, dict_var)
            })
            .collect();

        // Push dict scope so the method body can reference constraint dicts
        self.push_dict_scope();
        for (class_name, dict_var) in &dict_vars {
            self.register_dict(*class_name, dict_var.clone());
        }

        // Compile the method body (now with constraint dicts in scope)
        let mut body = self.compile_equations(method_def)?;

        self.pop_dict_scope();

        body = self.eta_expand_point_free_method(body, method_def);

        // Wrap body with dict lambdas (outermost first)
        for (_, dict_var) in dict_vars.into_iter().rev() {
            body = core::Expr::Lam(dict_var, Box::new(body), method_def.span);
        }

        Ok(Some(Bind::NonRec(var, Box::new(body))))
    }

    /// HIR-level eta expansion for COMPOSITION-valued bindings whose type
    /// has more arrows than the equation has patterns (see `lower_value_def`).
    /// Returns a rewritten ValueDef, or None when not applicable.
    /// Rewrite `join m` to `m >>= \x -> x` at the HIR level.
    ///
    /// `join` is an external constrained function with no implementation; as
    /// a bare occurrence it stubs at runtime (parsec's `notFollowedBy' p =
    /// try $ join $ do … <|> …` — the stub's garbage return propagated as a
    /// null through pandoc's many1TillChar). Rewriting to an ordinary bind
    /// application BEFORE lowering rides the whole existing `>>=` dispatch
    /// machinery (concrete fast paths, generic-m superclass selection,
    /// operand-type dispatch) — a Core-level synthesis via manual dictionary
    /// selection was tried and mis-selected.
    fn hir_rewrite_join(&mut self, value_def: &ValueDef) -> Option<ValueDef> {
        // Cheap pre-scan: does any equation reference a var named `join`?
        fn mentions_join(ctx: &LowerContext, e: &bhc_hir::Expr) -> bool {
            use bhc_hir::Expr as E;
            match e {
                E::Var(r) => ctx
                    .lookup_var(r.def_id)
                    .is_some_and(|v| v.name.as_str() == "join"),
                E::App(f, a, _) => mentions_join(ctx, f) || mentions_join(ctx, a),
                E::Lam(_, b, _) => mentions_join(ctx, b),
                E::Let(_, b, _) => mentions_join(ctx, b),
                E::Case(s, alts, _) => {
                    mentions_join(ctx, s) || alts.iter().any(|a| mentions_join(ctx, &a.rhs))
                }
                E::If(c, t, f, _) => {
                    mentions_join(ctx, c) || mentions_join(ctx, t) || mentions_join(ctx, f)
                }
                E::Tuple(es, _) | E::List(es, _) => es.iter().any(|e| mentions_join(ctx, e)),
                _ => false,
            }
        }
        if !value_def
            .equations
            .iter()
            .any(|eq| mentions_join(self, &eq.rhs))
        {
            return None;
        }
        let bind_id = self.find_var_id_by_name(">>=")?;

        fn rewrite(ctx: &mut LowerContext, e: &mut bhc_hir::Expr, bind_id: DefId) {
            use bhc_hir::Expr as E;
            // Recurse first so nested joins rewrite too.
            match e {
                E::App(f, a, _) => {
                    rewrite(ctx, f, bind_id);
                    rewrite(ctx, a, bind_id);
                }
                E::Lam(_, b, _) => rewrite(ctx, b, bind_id),
                E::Let(_, b, _) => rewrite(ctx, b, bind_id),
                E::Case(s, alts, _) => {
                    rewrite(ctx, s, bind_id);
                    for alt in alts {
                        rewrite(ctx, &mut alt.rhs, bind_id);
                    }
                }
                E::If(c, t, f, _) => {
                    rewrite(ctx, c, bind_id);
                    rewrite(ctx, t, bind_id);
                    rewrite(ctx, f, bind_id);
                }
                E::Tuple(es, _) | E::List(es, _) => {
                    for x in es {
                        rewrite(ctx, x, bind_id);
                    }
                }
                _ => {}
            }
            // Then rewrite `join m` heads — both the direct application and
            // the `$`-applied form `join $ m` (join as `$`'s left argument),
            // which is parsec's actual shape (`try $ join $ do ... <|> ...`).
            if let E::App(f, m, span) = e {
                let is_join = matches!(f.as_ref(), E::Var(r) if ctx
                    .lookup_var(r.def_id)
                    .is_some_and(|v| v.name.as_str() == "join"));
                let is_dollar_join = matches!(f.as_ref(), E::App(g, h, _)
                    if matches!(g.as_ref(), E::Var(r) if ctx
                        .lookup_var(r.def_id)
                        .is_some_and(|v| v.name.as_str() == "$"))
                    && matches!(h.as_ref(), E::Var(r) if ctx
                        .lookup_var(r.def_id)
                        .is_some_and(|v| v.name.as_str() == "join")));
                if is_join || is_dollar_join {
                    let span = *span;
                    let n = ctx.fresh_counter;
                    ctx.fresh_counter += 1;
                    let x_id = DefId::new(1_700_000 + n as usize);
                    let x_name = Symbol::intern("$join_x");
                    let ident = E::Lam(
                        vec![bhc_hir::Pat::Var(x_name, x_id, span)],
                        Box::new(E::Var(bhc_hir::DefRef { def_id: x_id, span })),
                        span,
                    );
                    let bind_var = E::Var(bhc_hir::DefRef {
                        def_id: bind_id,
                        span,
                    });
                    let inner = std::mem::replace(m.as_mut(), E::Tuple(vec![], span));
                    *e = E::App(
                        Box::new(E::App(Box::new(bind_var), Box::new(inner), span)),
                        Box::new(ident),
                        span,
                    );
                }
            }
        }

        let mut vd = value_def.clone();
        for eq in &mut vd.equations {
            let mut rhs = eq.rhs.clone();
            rewrite(self, &mut rhs, bind_id);
            eq.rhs = rhs;
        }
        Some(vd)
    }

    /// Find a registered variable's `DefId` by name (linear scan; used for
    /// low-frequency rewrites only).
    fn find_var_id_by_name(&self, name: &str) -> Option<DefId> {
        self.var_map
            .iter()
            .find(|(_, v)| v.name.as_str() == name)
            .map(|(id, _)| *id)
    }

    fn hir_eta_expand_composition(&mut self, value_def: &ValueDef) -> Option<ValueDef> {
        fn arrows(mut t: &Ty) -> usize {
            let mut n = 0;
            while let Ty::Fun(_, r) = t {
                n += 1;
                t = r.as_ref();
            }
            n
        }
        if value_def.equations.len() != 1 {
            return None;
        }
        let eq = value_def.equations.first()?;
        // The RHS must be a `.`-composition chain (head of the App spine is
        // the composition operator).
        fn is_dot_var(ctx: &LowerContext, e: &bhc_hir::Expr) -> bool {
            if let bhc_hir::Expr::Var(dr) = e {
                ctx.lookup_var(dr.def_id)
                    .is_some_and(|v| v.name.as_str() == ".")
            } else {
                false
            }
        }
        fn is_composition(ctx: &LowerContext, e: &bhc_hir::Expr) -> bool {
            if let bhc_hir::Expr::App(f, _, _) = e {
                if let bhc_hir::Expr::App(g, _, _) = f.as_ref() {
                    return is_dot_var(ctx, g);
                }
            }
            false
        }
        if std::env::var("BHC_DBG_HETA").is_ok() {
            let head_desc = if let bhc_hir::Expr::App(f, _, _) = &eq.rhs {
                if let bhc_hir::Expr::App(g, _, _) = f.as_ref() {
                    if let bhc_hir::Expr::Var(dr) = g.as_ref() {
                        format!(
                            "inner-head var {:?} -> {:?}",
                            dr.def_id,
                            self.lookup_var(dr.def_id).map(|v| v.name.as_str())
                        )
                    } else {
                        "inner-head not var".to_string()
                    }
                } else {
                    "f not app".to_string()
                }
            } else {
                "rhs not app".to_string()
            };
            eprintln!(
                "[heta] {}: pats={} comp={} {}",
                value_def.name,
                eq.pats.len(),
                is_composition(self, &eq.rhs),
                head_desc
            );
        }
        if !is_composition(self, &eq.rhs) {
            return None;
        }
        let arrow_count = value_def
            .sig
            .as_ref()
            .map(|s| arrows(&s.ty))
            .filter(|n| *n > 0)
            .or_else(|| {
                self.lookup_scheme(value_def.id)
                    .map(|s| arrows(&s.ty))
                    .filter(|n| *n > 0)
            })
            .unwrap_or(0);
        let npats = eq.pats.len();
        if arrow_count <= npats {
            return None;
        }
        let missing = arrow_count - npats;
        // The scheme's arrow types, so the fresh params carry their CONCRETE
        // types (`$heta :: ParsecT [Char] () Identity Char`) — downstream
        // argument-type inference pins dictionary resolution from them.
        let param_tys: Vec<Ty> = {
            let mut tys = Vec::new();
            let mut t = self
                .lookup_scheme(value_def.id)
                .map(|s| s.ty.clone())
                .or_else(|| value_def.sig.as_ref().map(|s| s.ty.clone()))
                .unwrap_or(Ty::Error);
            while let Ty::Fun(a, r) = t {
                tys.push(*a);
                t = *r;
            }
            tys
        };
        // Fresh params: synthetic DefIds in a range distinct from real HIR
        // ids and the other synthetic ranges.
        let mut new_pats = eq.pats.clone();
        let mut rhs = eq.rhs.clone();
        for i in 0..missing {
            let pty = param_tys.get(npats + i).cloned().unwrap_or(Ty::Error);
            if std::env::var("BHC_DBG_HETA").is_ok() {
                eprintln!("[heta] {} param {}: {:?}", value_def.name, npats + i, pty);
            }
            let v = self.fresh_var("$heta", pty, eq.span);
            let def_id = DefId::new(1_700_000 + v.id.index());
            self.register_var(def_id, v.clone());
            new_pats.push(bhc_hir::Pat::Var(v.name, def_id, eq.span));
            let arg = bhc_hir::Expr::Var(bhc_hir::DefRef {
                def_id,
                span: eq.span,
            });
            rhs = if i == 0 {
                // Unroll the composition chain: `(f . g) x` → `f (g x)`.
                fn unroll(
                    ctx: &LowerContext,
                    e: bhc_hir::Expr,
                    arg: bhc_hir::Expr,
                    span: bhc_span::Span,
                ) -> bhc_hir::Expr {
                    if let bhc_hir::Expr::App(f, g, _) = &e {
                        if let bhc_hir::Expr::App(dot, lhs, _) = f.as_ref() {
                            if is_dot_var(ctx, dot) {
                                let inner = unroll(ctx, g.as_ref().clone(), arg, span);
                                return bhc_hir::Expr::App(
                                    Box::new(lhs.as_ref().clone()),
                                    Box::new(inner),
                                    span,
                                );
                            }
                        }
                    }
                    bhc_hir::Expr::App(Box::new(e), Box::new(arg), span)
                }
                unroll(self, rhs, arg, eq.span)
            } else {
                bhc_hir::Expr::App(Box::new(rhs), Box::new(arg), eq.span)
            };
        }
        let mut new_eq = eq.clone();
        new_eq.pats = new_pats;
        new_eq.rhs = rhs;
        let mut out = value_def.clone();
        out.equations = vec![new_eq];
        Some(out)
    }

    /// Eta-expand a point-free instance-method body to its type's arrow count.
    ///
    /// A point-free method binding (`lookupEnv = IO.lookupEnv`) compiles to a
    /// 0-lambda Core bind, so the interface records arity 0 and an importing
    /// module EVALUATES the dictionary slot as a CAF — calling a function
    /// that expects its value arguments with none, which reads garbage
    /// registers (pandoc's PandocMonad PandocIO dict crashed in
    /// `bhc_text_char_count`). Eta-expanding makes the Core arity match how
    /// call sites must apply it. Arrow count comes from the explicit
    /// signature, the typeck scheme, or the RHS occurrence type, in that
    /// order; a 0-arrow (value-typed) method is left alone — it IS a CAF.
    fn eta_expand_point_free_method(
        &mut self,
        body: core::Expr,
        method_def: &ValueDef,
    ) -> core::Expr {
        // Count existing value lambdas: a PARTIALLY point-free binding
        // (`many1TillChar p = fmap T.pack . many1Till p` — one lambda, two
        // arrows) needs the REMAINING arity expanded under them, or its
        // recorded arity under-counts and callers mis-apply.
        let existing = {
            let mut n = 0;
            let mut e = &body;
            while let core::Expr::Lam(_, b, _) = e {
                n += 1;
                e = b.as_ref();
            }
            n
        };
        fn arrows(mut t: &Ty) -> usize {
            let mut n = 0;
            while let Ty::Fun(_, r) = t {
                n += 1;
                t = r.as_ref();
            }
            n
        }
        let dbg = std::env::var("BHC_DBG_ETA").is_ok();
        if dbg {
            let sig_a = method_def.sig.as_ref().map(|s| arrows(&s.ty));
            let sch_a = self.lookup_scheme(method_def.id).map(|s| arrows(&s.ty));
            let occ = method_def
                .equations
                .first()
                .map(|e| e.rhs.span())
                .and_then(|sp| {
                    self.resolved_expr_ty_opt(sp)
                        .or_else(|| self.expr_ty_opt(sp))
                });
            eprintln!(
                "[eta] {}: sig={sig_a:?} scheme={sch_a:?} occ={occ:?}",
                method_def.name
            );
        }
        let arrow_count = method_def
            .sig
            .as_ref()
            .map(|s| arrows(&s.ty))
            .filter(|n| *n > 0)
            .or_else(|| {
                self.lookup_scheme(method_def.id)
                    .map(|s| arrows(&s.ty))
                    .filter(|n| *n > 0)
            })
            .or_else(|| {
                let rhs_span = method_def.equations.first()?.rhs.span();
                self.resolved_expr_ty_opt(rhs_span)
                    .or_else(|| self.expr_ty_opt(rhs_span))
                    .map(|t| arrows(&t))
                    .filter(|n| *n > 0)
            })
            .or_else(|| {
                // The class's declared method signature (local classes carry
                // real schemes; imported classes carry arrow-skeletons from
                // the interface's method arities).
                let class = self.is_class_method(method_def.name)?;
                let info = self.class_registry.lookup_class(class)?;
                info.method_types
                    .get(&method_def.name)
                    .map(|s| arrows(&s.ty))
                    .filter(|n| *n > 0)
            })
            .unwrap_or(0);
        if arrow_count <= existing {
            return body;
        }
        let missing = arrow_count - existing;
        // Applying an argument to a `.`-composition chain unrolls it
        // (`(f . g . h) x` → `f (g (h x))`): the inner partial applications
        // (`map (*2)`, `filter pos`, builtin Text partials) become SATURATED
        // calls that codegen's builtin paths handle — left as a chain, each
        // builtin-as-value falls to a runtime stub.
        fn apply_unrolling_composition(
            body: core::Expr,
            arg: core::Expr,
            span: Span,
        ) -> core::Expr {
            // A `where`-group wraps the composition in a Let — apply inside.
            if let core::Expr::Let(bind, inner, lspan) = body {
                return core::Expr::Let(
                    bind,
                    Box::new(apply_unrolling_composition(*inner, arg, span)),
                    lspan,
                );
            }
            if let core::Expr::App(f, g, _) = &body {
                if let core::Expr::App(dot, lhs, _) = f.as_ref() {
                    if let core::Expr::Var(v, _) = dot.as_ref() {
                        if v.name.as_str() == "." {
                            let inner = apply_unrolling_composition(g.as_ref().clone(), arg, span);
                            return core::Expr::App(
                                Box::new(lhs.as_ref().clone()),
                                Box::new(inner),
                                span,
                            );
                        }
                    }
                }
            }
            core::Expr::App(Box::new(body), Box::new(arg), span)
        }
        let params: Vec<Var> = (0..missing)
            .map(|i| self.fresh_var(&format!("$eta{i}"), Ty::Error, method_def.span))
            .collect();
        // Descend the EXISTING lambdas; expand the missing arity under them.
        fn expand_under(
            body: core::Expr,
            params: &[Var],
            span: Span,
            unroll: &dyn Fn(core::Expr, core::Expr, Span) -> core::Expr,
        ) -> core::Expr {
            if let core::Expr::Lam(v, inner, lspan) = body {
                return core::Expr::Lam(
                    v,
                    Box::new(expand_under(*inner, params, span, unroll)),
                    lspan,
                );
            }
            let mut b = body;
            for (i, p) in params.iter().enumerate() {
                let arg = core::Expr::Var(p.clone(), span);
                b = if i == 0 {
                    unroll(b, arg, span)
                } else {
                    core::Expr::App(Box::new(b), Box::new(arg), span)
                };
            }
            for p in params.iter().rev() {
                b = core::Expr::Lam(p.clone(), Box::new(b), span);
            }
            b
        }
        expand_under(body, &params, method_def.span, &apply_unrolling_composition)
    }

    /// Lower a default method implementation from a class definition.
    ///
    /// Default methods are special because they implicitly have the class constraint.
    /// For example, in:
    /// ```text
    /// class Eq a where
    ///   (==) :: a -> a -> Bool
    ///   (/=) :: a -> a -> Bool
    ///   x /= y = not (x == y)  -- default
    /// ```
    ///
    /// The default `/=` has an implicit `Eq a` constraint. When lowered, it becomes:
    /// ```text
    /// $default_neq = \$dEq -> \x -> \y -> not (($sel_0 $dEq) x y)
    /// ```
    /// where `$sel_0` selects `(==)` from the Eq dictionary.
    fn lower_default_method(
        &mut self,
        class_def: &bhc_hir::ClassDef,
        default_def: &ValueDef,
    ) -> LowerResult<Option<Bind>> {
        let var = self
            .lookup_var(default_def.id)
            .cloned()
            .ok_or_else(|| LowerError::Internal("missing variable for default method".into()))?;

        // Default methods have the class constraint.
        // Create a constraint for the class with its type parameter.
        // The type parameter is from the class definition.
        let class_constraint = if let Some(type_param) = class_def.params.first() {
            Constraint::new(
                class_def.name,
                Ty::Var(type_param.clone()),
                default_def.span,
            )
        } else {
            // Class with no type parameters - unusual but handle it
            Constraint::new(class_def.name, Ty::Error, default_def.span)
        };

        // Create dictionary variable for the class constraint
        let dict_var = self.make_dict_var(&class_constraint);

        // Push dictionary scope and register the class dictionary
        self.push_dict_scope();
        self.register_dict(class_def.name, dict_var.clone());

        // Compile the default method body (now with class dictionary in scope)
        let mut body = self.compile_equations(default_def)?;

        // Pop the dictionary scope
        self.pop_dict_scope();

        // Wrap body in dictionary lambda
        body = core::Expr::Lam(dict_var, Box::new(body), default_def.span);

        Ok(Some(Bind::NonRec(var, Box::new(body))))
    }

    /// Create a dictionary variable for a type class constraint.
    ///
    /// The naming convention is `$d<ClassName>` to avoid conflicts with
    /// user-defined variables.
    fn make_dict_var(&mut self, constraint: &bhc_types::Constraint) -> Var {
        let dict_name = format!("$d{}", constraint.class.as_str());
        self.fresh_var(&dict_name, Ty::Error, constraint.span)
    }

    /// Compile multiple equations into a single Core expression.
    ///
    /// For simple definitions like `f = e`, this just lowers the expression.
    /// For pattern-matching definitions like:
    /// ```haskell
    /// f 0 = 1
    /// f n = n * f (n-1)
    /// ```
    /// This compiles to a lambda with a case expression.
    fn compile_equations(&mut self, value_def: &ValueDef) -> LowerResult<core::Expr> {
        if value_def.equations.is_empty() {
            return Err(LowerError::Internal(
                "value definition with no equations".into(),
            ));
        }

        // Simple case: single equation with no patterns
        if value_def.equations.len() == 1 && value_def.equations[0].pats.is_empty() {
            let eq = &value_def.equations[0];
            return lower_expr(self, &eq.rhs);
        }

        // Complex case: multiple equations or patterns
        // Figure out how many arguments the function takes
        let arity = value_def.equations[0].pats.len();

        if arity == 0 {
            // Multiple equations with no arguments - this is an error
            // but we'll just use the first one
            return lower_expr(self, &value_def.equations[0].rhs);
        }

        // Generate fresh variables for each argument.
        // Try to extract parameter types from the function's type signature
        // so that type-directed method resolution can use them.
        let mut param_types: Vec<Ty> = Vec::new();
        if let Some(scheme) = self.lookup_scheme(value_def.id) {
            let mut ty = &scheme.ty;
            for _ in 0..arity {
                if let Ty::Fun(arg_ty, ret_ty) = ty {
                    param_types.push(arg_ty.as_ref().clone());
                    ty = ret_ty.as_ref();
                } else {
                    break;
                }
            }
        }
        let args: Vec<Var> = (0..arity)
            .map(|i| {
                let ty = param_types.get(i).cloned().unwrap_or(Ty::Error);
                self.fresh_var(&format!("arg{i}"), ty, value_def.span)
            })
            .collect();

        // Use decision tree compilation for all multi-equation pattern matching
        let case_expr = self.compile_pattern_match_expr(value_def, &args)?;

        // Wrap in lambdas
        let mut result = case_expr;
        for arg in args.into_iter().rev() {
            result = core::Expr::Lam(arg, Box::new(result), value_def.span);
        }

        Ok(result)
    }

    /// Make a tuple expression from variables.
    /// Compile pattern matching for multiple equations using decision trees.
    ///
    /// Returns a Core expression (case tree) that performs the pattern dispatch.
    fn compile_pattern_match_expr(
        &mut self,
        value_def: &ValueDef,
        args: &[Var],
    ) -> LowerResult<core::Expr> {
        use crate::pattern::compile_match_to_expr;
        compile_match_to_expr(self, value_def, args)
    }
}

impl Default for LowerContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a type contains type variables.
pub(crate) fn has_type_variables(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => true,
        Ty::Con(_) | Ty::Prim(_) | Ty::Error => false,
        Ty::App(f, a) | Ty::Fun(f, a) => has_type_variables(f) || has_type_variables(a),
        Ty::Tuple(tys) => tys.iter().any(has_type_variables),
        Ty::List(elem) => has_type_variables(elem),
        Ty::Forall(_, body) => has_type_variables(body),
        Ty::Nat(_) | Ty::TyList(_) => false,
    }
}

/// Extract a human-readable type name from a `Ty` for instance naming.
///
/// Handles flexible instance heads like `Ty::App(Box, a)` → `"Box_a"`,
/// `Ty::List(a)` → `"List_a"`, `Ty::Con(Int)` → `"Int"`, etc.
/// The name of a type's head constructor, walking through applications:
/// `ParsecT s u m` yields `ParsecT`. A variable-headed type has none.
fn head_type_con_name(ty: &Ty) -> Option<Symbol> {
    let mut head = ty;
    while let Ty::App(f, _) = head {
        head = f.as_ref();
    }
    match head {
        Ty::Con(con) => Some(con.name),
        _ => None,
    }
}

fn type_name_for_instance(ty: &Ty) -> String {
    match ty {
        Ty::Con(con) => con.name.as_str().to_string(),
        Ty::App(f, a) => {
            format!(
                "{}_{}",
                type_name_for_instance(f),
                type_name_for_instance(a)
            )
        }
        Ty::List(elem) => format!("List_{}", type_name_for_instance(elem)),
        // A type variable in an instance head is universally quantified — its
        // fresh VarId is assigned per-compilation and differs between the
        // producing and consuming modules. Encoding it by id (`v148` vs
        // `v90174`) made the instance symbol for a multi-parameter head like
        // `ParsecT s u m` mismatch across modules → the consumer stubbed
        // `$instance_pure_ParsecT_v9017x` while the producer defined
        // `$instance_pure_ParsecT_v14x`. Encode all instance-head type
        // variables canonically so both sides agree. (Single `Con` heads like
        // `MyIO` are unaffected — they have no variable in the head.)
        Ty::Var(_) => "v".to_string(),
        Ty::Fun(from, to) => {
            format!(
                "Fun_{}_{}",
                type_name_for_instance(from),
                type_name_for_instance(to)
            )
        }
        Ty::Tuple(elems) => {
            let names: Vec<String> = elems.iter().map(type_name_for_instance).collect();
            format!("Tup_{}", names.join("_"))
        }
        _ => "Unknown".to_string(),
    }
}

/// Break mutual-recursion cycles among top-level PARSER-SHAPED bindings by
/// making cycle-internal calls lazy.
///
/// pandoc's Markdown reader has `block` → `codeBlockFenced` → … →
/// `yamlMetaBlock'` → `parseBlocks` → `block`, where every member is a
/// dictionary-taking function (`\$dPandocMonad -> parser-value`). Building
/// one member's parser VALUE eagerly calls the next member around the cycle,
/// which never terminates (stack overflow in `bhc_force`'s blackhole CAS).
/// The nullary-CAF thunking in codegen cannot see these — they have a
/// (dict) lambda.
///
/// Nodes: top-level bindings whose lambda params are all dictionaries
/// (`$d…`-named), including nullary ones. For every SCC with a cycle,
/// wrap each APPLIED reference to a fellow member inside a member's body in
/// `Expr::Lazy` — codegen thunks it, and the apply/force paths evaluate on
/// first use, exactly like the nullary-CAF thunks.
fn lazify_recursive_parser_calls(bindings: &mut [Bind]) {
    use rustc_hash::{FxHashMap, FxHashSet};

    if std::env::var("BHC_NO_LAZIFY").is_ok() {
        return;
    }

    fn dict_param_count(mut e: &core::Expr) -> Option<usize> {
        let mut n = 0;
        while let core::Expr::Lam(v, body, _) = e {
            if !v.name.as_str().starts_with("$d") {
                return None;
            }
            n += 1;
            e = body;
        }
        Some(n)
    }

    // Collect candidate nodes.
    let mut node_names: FxHashSet<Symbol> = FxHashSet::default();
    let mut arities: FxHashMap<Symbol, usize> = FxHashMap::default();
    let mut exprs: Vec<(Symbol, &core::Expr)> = Vec::new();
    for bind in bindings.iter() {
        let pairs: Vec<(&Var, &core::Expr)> = match bind {
            Bind::NonRec(v, e) => vec![(v, e.as_ref())],
            Bind::Rec(ps) => ps.iter().map(|(v, e)| (v, e.as_ref())).collect(),
        };
        for (v, e) in pairs {
            if let Some(n) = dict_param_count(e) {
                node_names.insert(v.name);
                arities.insert(v.name, n);
                exprs.push((v.name, e));
            }
        }
    }
    if node_names.len() < 2 {
        return;
    }

    // Edges among candidates via free-variable names.
    let names_vec: Vec<Symbol> = exprs.iter().map(|(n, _)| *n).collect();
    let mut deps: Vec<Vec<usize>> = Vec::with_capacity(exprs.len());
    for (_, e) in &exprs {
        let fvs = crate::binding::collect_free_vars(e);
        let d: Vec<usize> = names_vec
            .iter()
            .enumerate()
            .filter(|(_, n)| fvs.contains(n))
            .map(|(j, _)| j)
            .collect();
        deps.push(d);
    }

    // Tarjan SCC (iterative).
    let n = exprs.len();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut counter = 0usize;
    let mut cycle_members: FxHashSet<Symbol> = FxHashSet::default();
    let mut call_stack: Vec<(usize, usize)> = Vec::new();
    for start in 0..n {
        if index[start] != usize::MAX {
            continue;
        }
        call_stack.push((start, 0));
        index[start] = counter;
        low[start] = counter;
        counter += 1;
        stack.push(start);
        on_stack[start] = true;
        while let Some(&mut (node, ref mut ei)) = call_stack.last_mut() {
            if *ei < deps[node].len() {
                let next = deps[node][*ei];
                *ei += 1;
                if index[next] == usize::MAX {
                    call_stack.push((next, 0));
                    index[next] = counter;
                    low[next] = counter;
                    counter += 1;
                    stack.push(next);
                    on_stack[next] = true;
                } else if on_stack[next] && index[next] < low[node] {
                    low[node] = index[next];
                }
            } else {
                call_stack.pop();
                if let Some(&(parent, _)) = call_stack.last() {
                    if low[node] < low[parent] {
                        low[parent] = low[node];
                    }
                }
                if low[node] == index[node] {
                    let mut comp = Vec::new();
                    while let Some(top) = stack.pop() {
                        on_stack[top] = false;
                        comp.push(top);
                        if top == node {
                            break;
                        }
                    }
                    let self_loop = comp.len() == 1 && deps[comp[0]].contains(&comp[0]);
                    if comp.len() > 1 || self_loop {
                        for m in comp {
                            cycle_members.insert(names_vec[m]);
                        }
                    }
                }
            }
        }
    }
    if cycle_members.is_empty() {
        return;
    }
    if std::env::var("BHC_DBG_CAFS").is_ok() {
        let mut names: Vec<&str> = cycle_members.iter().map(|s| s.as_str()).collect();
        names.sort_unstable();
        eprintln!("[lazify] parser cycle members: {:?}", names);
    }

    // Rewrite applied references to cycle members inside cycle members.
    fn spine_head_name(e: &core::Expr) -> Option<(Symbol, usize)> {
        fn go(e: &core::Expr, len: usize) -> Option<(Symbol, usize)> {
            match e {
                core::Expr::App(f, _, _) => go(f, len + 1),
                core::Expr::Var(v, _) => Some((v.name, len)),
                _ => None,
            }
        }
        go(e, 0)
    }
    fn rewrite(
        e: &mut core::Expr,
        members: &FxHashSet<Symbol>,
        arities: &FxHashMap<Symbol, usize>,
    ) {
        // Wrap an EXACT dict-arity cycle-member call — the "construct the
        // parser value" form. Over-applications (the CPS parser value applied
        // to state/continuations) must stay eager: wrapping them captures raw
        // char/int arguments in a thunk env, which the eval path derefs as
        // pointers (crashed forcing 0xd = '\r').
        if let core::Expr::App(..) = e {
            if let Some((h, len)) = spine_head_name(e) {
                if members.contains(&h) && arities.get(&h) == Some(&len) {
                    let span = match e {
                        core::Expr::App(_, _, s) => *s,
                        _ => unreachable!(),
                    };
                    let inner = std::mem::replace(
                        e,
                        core::Expr::Lit(core::Literal::Int(0), Ty::Error, span),
                    );
                    *e = core::Expr::Lazy(Box::new(inner), span);
                    return;
                }
            }
        }
        match e {
            core::Expr::App(f, a, _) => {
                rewrite(f, members, arities);
                rewrite(a, members, arities);
            }
            core::Expr::Lam(_, b, _) | core::Expr::TyLam(_, b, _) => rewrite(b, members, arities),
            core::Expr::Let(bind, body, _) => {
                match bind.as_mut() {
                    Bind::NonRec(_, r) => rewrite(r, members, arities),
                    Bind::Rec(ps) => {
                        for (_, r) in ps {
                            rewrite(r, members, arities);
                        }
                    }
                }
                rewrite(body, members, arities);
            }
            core::Expr::Case(scrut, alts, _, _) => {
                rewrite(scrut, members, arities);
                for alt in alts {
                    rewrite(&mut alt.rhs, members, arities);
                }
            }
            core::Expr::Lazy(b, _) | core::Expr::Cast(b, _, _) | core::Expr::Tick(_, b, _) => {
                rewrite(b, members, arities)
            }
            _ => {}
        }
    }
    for bind in bindings.iter_mut() {
        let pairs: Vec<(&Var, &mut Box<core::Expr>)> = match bind {
            Bind::NonRec(v, e) => vec![(&*v, e)],
            Bind::Rec(ps) => ps.iter_mut().map(|(v, e)| (&*v, e)).collect(),
        };
        for (v, e) in pairs {
            if cycle_members.contains(&v.name) {
                rewrite(e.as_mut(), &cycle_members, &arities);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_var() {
        let mut ctx = LowerContext::new();
        let v1 = ctx.fresh_var("x", Ty::Error, Span::default());
        let v2 = ctx.fresh_var("x", Ty::Error, Span::default());
        assert_ne!(v1.id, v2.id);
    }
}
