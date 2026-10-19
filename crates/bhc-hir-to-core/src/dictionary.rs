//! Type class dictionary construction and method selection.
//!
//! This module implements the dictionary-passing transformation for type classes.
//! During HIR-to-Core lowering, type class constraints are translated into
//! explicit dictionary parameters, and method calls become field selections
//! from dictionaries.
//!
//! ## Dictionary Representation
//!
//! A dictionary for a type class is represented as a tuple of method implementations.
//! For example, a class like:
//!
//! ```haskell
//! class Eq a where
//!   (==) :: a -> a -> Bool
//!   (/=) :: a -> a -> Bool
//! ```
//!
//! Has dictionaries represented as 2-tuples of functions.
//!
//! ## Instance Resolution
//!
//! When we encounter a constraint like `Eq Int`, we:
//! 1. Look up the `Eq Int` instance
//! 2. Construct a dictionary containing the method implementations
//! 3. Pass this dictionary where needed
//!
//! ## Superclass Dictionaries
//!
//! If a class has superclasses (e.g., `class Eq a => Ord a`), the dictionary
//! includes the superclass dictionaries as additional fields at the beginning.

use bhc_core::{self as core, Bind, Var, VarId};
use bhc_hir::DefId;
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Kind, Scheme, Subst, Ty, TyCon, TyVar};
use rustc_hash::FxHashMap;

/// Information about an associated type in a class.
#[derive(Clone, Debug)]
pub struct AssocTypeInfo {
    /// The name of the associated type.
    pub name: Symbol,
    /// Additional type parameters beyond the class parameters.
    pub params: Vec<TyVar>,
    /// The result kind (usually `*`).
    pub kind: Kind,
    /// Optional default type definition.
    pub default: Option<Ty>,
}

/// Information about a type class for dictionary construction.
#[derive(Clone, Debug)]
pub struct ClassInfo {
    /// The class name.
    pub name: Symbol,
    /// Number of type parameters (1 for single-param, 2+ for multi-param).
    pub param_count: usize,
    /// Method names in order (defines dictionary field order).
    pub methods: Vec<Symbol>,
    /// Method type signatures.
    pub method_types: FxHashMap<Symbol, Scheme>,
    /// Superclass names.
    pub superclasses: Vec<Symbol>,
    /// For each entry in `superclasses`, which of this class's parameters it
    /// constrains, by index. `class Monad m => Stream s m t` records `[[1]]`.
    ///
    /// Empty means unknown, which keeps the historical behaviour of assuming
    /// the instance's FIRST type — right for a single-parameter class, and
    /// the reason a multi-parameter one looked its monad superclass up at the
    /// stream type and found nothing.
    pub superclass_params: Vec<Vec<usize>>,
    /// Default method implementations (method name -> `DefId`).
    pub defaults: FxHashMap<Symbol, DefId>,
    /// Associated type declarations.
    pub assoc_types: Vec<AssocTypeInfo>,
}

/// Information about a type class instance.
#[derive(Clone, Debug)]
pub struct InstanceInfo {
    /// The class being instantiated.
    pub class: Symbol,
    /// The instance types (e.g., `[Int]` for `instance Eq Int`,
    /// or `[Int, String]` for `instance Convert Int String`).
    pub instance_types: Vec<Ty>,
    /// Method implementations (method name -> `DefId`).
    pub methods: FxHashMap<Symbol, DefId>,
    /// Superclass instance types for resolving superclass dictionaries.
    pub superclass_instances: Vec<Ty>,
    /// Associated type implementations (assoc type name -> concrete type).
    pub assoc_type_impls: FxHashMap<Symbol, Ty>,
    /// Instance constraints (e.g., `Describable a` in `instance Describable a => Describable (Box a)`).
    /// These constraints require dictionaries to be passed to instance methods.
    pub instance_constraints: Vec<Constraint>,
}

impl InstanceInfo {
    /// Create a new single-parameter instance (most common case).
    #[must_use]
    pub fn new_single(class: Symbol, instance_type: Ty, methods: FxHashMap<Symbol, DefId>) -> Self {
        Self {
            class,
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        }
    }

    /// Create a new multi-parameter instance.
    #[must_use]
    pub fn new_multi(
        class: Symbol,
        instance_types: Vec<Ty>,
        methods: FxHashMap<Symbol, DefId>,
    ) -> Self {
        Self {
            class,
            instance_types,
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        }
    }

    /// Create a new instance with associated types.
    #[must_use]
    pub fn new_with_assoc_types(
        class: Symbol,
        instance_types: Vec<Ty>,
        methods: FxHashMap<Symbol, DefId>,
        assoc_type_impls: FxHashMap<Symbol, Ty>,
    ) -> Self {
        Self {
            class,
            instance_types,
            methods,
            superclass_instances: vec![],
            assoc_type_impls,
            instance_constraints: vec![],
        }
    }

    /// Get the first instance type (for backward compatibility with single-param classes).
    #[must_use]
    pub fn first_type(&self) -> Option<&Ty> {
        self.instance_types.first()
    }

    /// Look up an associated type implementation.
    #[must_use]
    pub fn lookup_assoc_type(&self, name: Symbol) -> Option<&Ty> {
        self.assoc_type_impls.get(&name)
    }
}

/// Registry of classes and instances for dictionary construction.
#[derive(Clone, Debug, Default)]
pub struct ClassRegistry {
    /// Class information by name.
    pub classes: FxHashMap<Symbol, ClassInfo>,
    /// Instances indexed by (class name, instance type hash).
    /// Since we can't use Ty as a key directly, we use a list and linear search.
    pub instances: FxHashMap<Symbol, Vec<InstanceInfo>>,
    /// Declared schemes of class methods that carry constraints of their
    /// OWN, beyond the class's (`query :: Monoid c => (a -> c) -> b -> c` in
    /// pandoc's `Walkable a b`), keyed by (class, method). The scheme's
    /// constraints are exactly those method-level ones; its type shares their
    /// variables, so matching it against an occurrence instantiates them.
    pub method_level: FxHashMap<(Symbol, Symbol), Scheme>,
    /// Type parameters of LOCALLY declared classes, the variables their
    /// `method_types` are written in. Instantiating a method's declared type
    /// at an instance head needs them.
    pub class_params: FxHashMap<Symbol, Vec<TyVar>>,
    /// Functional dependencies by class, as (determining, determined)
    /// parameter indices: `MonadError e m | m -> e` is `[([1], [0])]`.
    /// A class absent here has no recorded dependencies.
    pub fundeps: ClassFunDeps,
}

/// Functional dependencies by class name, as (determining, determined)
/// parameter-index lists per dependency.
pub type ClassFunDeps = FxHashMap<Symbol, Vec<(Vec<usize>, Vec<usize>)>>;

/// Give `Monad`/`Applicative` constraints a runtime witness, so a
/// monad-polymorphic binding can dispatch `return`/`pure`.
///
/// Without one, codegen picks `return` from an AMBIENT transformer layer; a
/// binding polymorphic in its monad has no layer, falls back to
/// `TransformerLayer::IO`, and `lower_builtin_return` there is IDENTITY. So
/// `runParserT :: Monad m => ... -> m (Either ParseError a)` hands back a bare
/// `Right x`, and the caller's transformer bind reads that ADT's tag as a
/// function pointer (`blr 0x1`). See `VQ.hs`/`VR.hs` in the pandoc harness.
///
/// **On by default since 2026-08-28.** It was an opt-in experiment while two
/// defects made it worse than useless on real code: instance constraints were
/// dropped crossing a `.bhi` boundary, and a constrained occurrence could
/// select a dictionary for the wrong monad. With both fixed, a measurement on
/// DBs built from scratch under each setting showed it strictly better and
/// nowhere worse — pandoc sweep 197/221 with a byte-identical fail list under
/// both, e2e 211/0 under both, parsec probe battery 10/10 under both, and the
/// 24-probe parser ladder going 6 passing to 18. `readWithM`, the call
/// `readMarkdown` makes, runs correctly for the first time.
///
/// Set `BHC_MONAD_WITNESS=0` to opt back out.
pub(crate) fn monad_witness_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("BHC_MONAD_WITNESS") {
        Ok(v) => !matches!(v.as_str(), "0" | "off" | "false"),
        Err(_) => true,
    })
}

/// Builtin monad/value-family classes whose methods dispatch through the
/// resolved-type machinery (see `LowerContext::is_monad_family_class`).
pub(crate) const MONAD_FAMILY_CLASSES: &[&str] = &[
    "Functor",
    "Applicative",
    "Monad",
    "Alternative",
    "MonadPlus",
    "Semigroup",
    "Monoid",
    // `Default`'s `def` is a bare result-type-determined value method, dispatched
    // by the same occurrence-type path as `mempty` (see the `is_value_class` set
    // in expr.rs). Gating that path requires membership here.
    "Default",
];

/// Builtin classes that are NOT user classes even when a `ClassInfo` is
/// registered for them (see `LowerContext::is_user_class`). Their methods
/// resolve through dedicated machinery (Show/Num/... codegen fast paths or
/// the monad-family dispatch), not user-class dictionary passing.
pub(crate) const BUILTIN_CLASS_NAMES: &[&str] = &[
    "Eq",
    "Ord",
    "Show",
    "Read",
    "Num",
    "Integral",
    "Fractional",
    "Floating",
    "Real",
    "RealFrac",
    "RealFloat",
    "Enum",
    "Bounded",
    "Functor",
    "Applicative",
    "Monad",
    "MonadFail",
    "MonadIO",
    "MonadTrans",
    "MonadReader",
    "MonadState",
    "MonadError",
    "MonadWriter",
    "Foldable",
    "Traversable",
    "Semigroup",
    "Monoid",
    "IsString",
    "Generic",
    "NFData",
    // Registered as builtin monad-family classes so their methods
    // (`<|>`, `mplus`) dispatch to user instances rather than being
    // treated as user classes (which would leave a bare reference
    // undispatched).
    "Alternative",
    "MonadPlus",
    // `def` dispatches through the value-class occurrence path, not user-class
    // dictionary passing; keep it out of the user-class set.
    "Default",
];

/// Whether `name` is one of [`BUILTIN_CLASS_NAMES`]. Under the closure IO
/// model the mtl classes (`MonadState`, `MonadIO`, ...) are library classes and
/// get ordinary dictionary passing (spec/BHC-BRIEF-0005).
pub(crate) fn is_builtin_class_name(name: &str) -> bool {
    BUILTIN_CLASS_NAMES.contains(&name) && !bhc_types::is_library_transformer_name(name)
}

/// The representation suffix for a transformer whose methods depend on the
/// layer beneath it. Mirrors the copy in `resolve_dictionary`: `ExceptT e
/// (StateT s m)` needs `_st`, because its methods thread the state.
fn inner_layer_suffix(ty: &Ty) -> Option<&'static str> {
    fn head_con(ty: &Ty) -> Option<&bhc_types::TyCon> {
        match ty {
            Ty::Con(tc) => Some(tc),
            Ty::App(f, _) => head_con(f),
            _ => None,
        }
    }
    // `T a m` is App(App(Con T, a), m): the inner monad is the outer argument.
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

/// Whether two types share a head constructor, with a type variable matching
/// only another type variable.
fn same_head(a: &Ty, b: &Ty) -> bool {
    fn head(ty: &Ty) -> &Ty {
        match ty {
            Ty::App(f, _) => head(f),
            other => other,
        }
    }
    match (head(a), head(b)) {
        (Ty::Con(x), Ty::Con(y)) => x.name == y.name,
        (Ty::Var(_), Ty::Var(_)) => true,
        _ => false,
    }
}

impl ClassRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The parameter positions of `class` that its functional dependencies
    /// determine once the positions in `fixed` are known (the closure, not
    /// including `fixed` itself). `None` when the class has no recorded
    /// dependencies, so callers can keep their previous behaviour.
    #[must_use]
    pub fn determined_positions(&self, class: Symbol, fixed: &[usize]) -> Option<Vec<usize>> {
        let deps = self.fundeps.get(&class).filter(|d| !d.is_empty())?;
        let mut known: Vec<usize> = fixed.to_vec();
        loop {
            let before = known.len();
            for (from, to) in deps {
                if from.iter().all(|f| known.contains(f)) {
                    for t in to {
                        if !known.contains(t) {
                            known.push(*t);
                        }
                    }
                }
            }
            if known.len() == before {
                break;
            }
        }
        Some(known.into_iter().filter(|k| !fixed.contains(k)).collect())
    }

    /// Whether an instance-declaration constraint of this class becomes a
    /// runtime dictionary parameter on the instance's methods.
    ///
    /// This predicate MUST be used both when wrapping instance-method bodies
    /// in dictionary lambdas (instance lowering in `LowerContext`) and when
    /// applying `constraint_dicts` in `construct_dictionary`, so a method's
    /// dict-lambda arity always matches the dictionaries applied to it.
    /// User-defined classes count, and so do the builtin VALUE classes
    /// (Semigroup/Monoid): e.g.
    /// `instance (Semigroup a, Monoid a) => Monoid (Future s a)` needs the
    /// `Monoid a` dict so `mempty = return mempty` selects the inner `mempty`
    /// from the constraint dictionary instead of looping on its own instance.
    ///
    /// `Monad` and `Applicative` count when the monad witness is on, which it
    /// is by default — see `monad_witness_enabled`. The rest of the monadic
    /// family (Functor, Alternative, MonadPlus, ...) still does not: monadic
    /// ops on those constraint tyvars keep the builtin codegen fast path.
    ///
    /// Counting `Monad` used to be actively harmful, which is why it was long
    /// gated off. Wrapping e.g. parsec's `instance Monad m => Stream [tok] m
    /// tok` in a Monad dict lambda makes `uncons`'s `return` resolve via the
    /// dict — but the dict was a null placeholder, so every token read
    /// crashed. That had two causes, both since fixed: instance constraints
    /// were not serialized into `.bhi`, so a consumer rebuilt the dict with
    /// the wrong arity, and a constrained occurrence could select a dictionary
    /// for the wrong monad.
    #[must_use]
    pub fn constraint_needs_dict_param(&self, class_name: Symbol) -> bool {
        let name = class_name.as_str();
        if monad_witness_enabled() && matches!(name, "Monad" | "Applicative") {
            return true;
        }
        // Under the closure IO model `Functor` too: the transformers library's
        // `instance Functor m => Functor (StateT s m)` maps through the inner
        // functor's `fmap` (spec/BHC-BRIEF-0005).
        if bhc_types::io_closure_mode() && name == "Functor" {
            return true;
        }
        matches!(name, "Semigroup" | "Monoid")
            || (!is_builtin_class_name(name) && self.lookup_class(class_name).is_some())
    }

    /// Register a type class.
    pub fn register_class(&mut self, info: ClassInfo) {
        self.classes.insert(info.name, info);
    }

    /// Register a type class instance.
    pub fn register_instance(&mut self, info: InstanceInfo) {
        self.instances.entry(info.class).or_default().push(info);
    }

    /// Look up a class by name.
    #[must_use]
    pub fn lookup_class(&self, name: Symbol) -> Option<&ClassInfo> {
        self.classes.get(&name)
    }

    /// Resolve an instance for a class and a single concrete type.
    ///
    /// This is a convenience method for single-parameter type classes.
    /// Returns the instance info and the substitution binding type variables.
    #[must_use]
    pub fn resolve_instance(&self, class: Symbol, ty: &Ty) -> Option<(&InstanceInfo, Subst)> {
        self.resolve_instance_multi(class, std::slice::from_ref(ty))
    }

    /// Resolve an instance for a class with multiple type arguments.
    ///
    /// For example, `resolve_instance_multi("Convert", &[Int, String])` would
    /// find `instance Convert Int String`.
    ///
    /// Returns both the instance and a substitution mapping type variables
    /// in the instance head to concrete types. For example, matching
    /// `instance Eq a => Eq [a]` against `Eq [Int]` returns the substitution
    /// `{a -> Int}`.
    #[must_use]
    pub fn resolve_instance_multi(
        &self,
        class: Symbol,
        types: &[Ty],
    ) -> Option<(&InstanceInfo, Subst)> {
        let instances = self.instances.get(&class);
        let instances = instances?;

        // Most specific first. A VARIABLE-headed instance head matches
        // anything, so whichever such instance is registered first would
        // otherwise answer for every type: pandoc's `getCommonState` at
        // `ParsecT Sources ParserState m` picked up a `PandocMonad`
        // instance that is not ParsecT's, and the parser ran an action
        // belonging to another monad. Require the head constructors to
        // agree first, then fall back to the permissive match.
        //
        // `String` is tried a second time as `[Char]` before any permissive
        // match. It is the language's own synonym, so an occurrence carries
        // whichever spelling the source used, while the instance is declared at
        // the list: `Semigroup String` agreed with no head, fell through to the
        // permissive pass, and matched a variable-headed instance with no
        // methods — a dictionary whose `<>` was `error "No implementation for
        // method"`. Only `String`, and only as a second try: other synonyms
        // carry instances of their own (`instance Semigroup Inlines` is keyed at
        // `Inlines`), which an unconditional expansion would miss.
        let normalized: Vec<Ty> = types.iter().map(string_as_char_list).collect();
        let spellings: &[&[Ty]] = if normalized.as_slice() == types {
            &[types]
        } else {
            &[types, &normalized]
        };
        for require_head in [true, false] {
            for tys in spellings {
                for inst in instances {
                    if require_head && !instance_heads_agree(&inst.instance_types, tys) {
                        continue;
                    }
                    if let Some(subst) = types_match_multi(&inst.instance_types, tys) {
                        return Some((inst, subst));
                    }
                }
            }
        }
        None
    }

    /// Get all method names for a class in order.
    #[must_use]
    pub fn class_methods(&self, class: Symbol) -> Vec<Symbol> {
        self.classes
            .get(&class)
            .map(|c| c.methods.clone())
            .unwrap_or_default()
    }

    /// Get superclasses for a class.
    #[must_use]
    pub fn superclasses(&self, class: Symbol) -> Vec<Symbol> {
        self.classes
            .get(&class)
            .map(|c| c.superclasses.clone())
            .unwrap_or_default()
    }

    /// Get associated types for a class.
    #[must_use]
    pub fn class_assoc_types(&self, class: Symbol) -> Vec<Symbol> {
        self.classes
            .get(&class)
            .map(|c| c.assoc_types.iter().map(|at| at.name).collect())
            .unwrap_or_default()
    }

    /// Look up which class defines an associated type.
    #[must_use]
    pub fn lookup_assoc_type_class(&self, assoc_name: Symbol) -> Option<Symbol> {
        for (class_name, class_info) in &self.classes {
            for assoc in &class_info.assoc_types {
                if assoc.name == assoc_name {
                    return Some(*class_name);
                }
            }
        }
        None
    }

    /// Get information about an associated type.
    #[must_use]
    pub fn lookup_assoc_type(&self, assoc_name: Symbol) -> Option<&AssocTypeInfo> {
        for class_info in self.classes.values() {
            for assoc in &class_info.assoc_types {
                if assoc.name == assoc_name {
                    return Some(assoc);
                }
            }
        }
        None
    }

    /// Resolve an associated type for a given instance.
    ///
    /// Given an associated type name and concrete type arguments, looks up
    /// the matching instance and returns the concrete type that the associated
    /// type resolves to.
    ///
    /// For example, `resolve_assoc_type("Elem", &[List Int])` would return `Int`
    /// if there's an instance `instance Collection [a] where type Elem [a] = a`.
    #[must_use]
    pub fn resolve_assoc_type(&self, assoc_name: Symbol, types: &[Ty]) -> Option<Ty> {
        // Find which class defines this associated type
        let class_name = self.lookup_assoc_type_class(assoc_name)?;

        // Find the matching instance and get the substitution
        let (instance, subst) = self.resolve_instance_multi(class_name, types)?;

        // Look up the associated type implementation
        if let Some(ty) = instance.assoc_type_impls.get(&assoc_name) {
            // Apply the substitution to get the concrete type
            return Some(subst.apply(ty));
        }

        // Check for default (also apply substitution)
        let assoc_info = self.lookup_assoc_type(assoc_name)?;
        assoc_info.default.as_ref().map(|ty| subst.apply(ty))
    }
}

use bhc_types::types_match_multi;

/// Whether every instance-head argument names the same type CONSTRUCTOR as the
/// corresponding query argument. An instance head that is a bare variable does
/// not agree — it matches everything, and is only reached on the second,
/// permissive pass. See `resolve_instance_multi`.
fn instance_heads_agree(inst_tys: &[Ty], tys: &[Ty]) -> bool {
    fn head_con(ty: &Ty) -> Option<Symbol> {
        let mut cur = ty;
        while let Ty::App(f, _) = cur {
            cur = f;
        }
        match cur {
            Ty::Con(c) => Some(c.name),
            Ty::List(_) => Some(Symbol::intern("[]")),
            _ => None,
        }
    }
    inst_tys.len() == tys.len()
        && inst_tys
            .iter()
            .zip(tys)
            .all(|(i, t)| match (head_con(i), head_con(t)) {
                (Some(a), Some(b)) => a == b,
                // The query says nothing: any instance may still answer.
                (_, None) => true,
                // A variable-headed instance is the less specific choice.
                (None, Some(_)) => false,
            })
}

/// Context for dictionary construction during lowering.
pub struct DictContext<'a> {
    /// The class registry.
    pub registry: &'a ClassRegistry,
    /// Fresh variable counter.
    fresh_counter: u32,
    /// Generated dictionary bindings to be added to the module.
    pub dict_bindings: Vec<Bind>,
    /// Cache of already-constructed dictionaries (class, type) -> Var.
    dict_cache: FxHashMap<(Symbol, String), Var>,
    /// Optional variable map for looking up registered variable names.
    /// When set, `method_reference` uses the registered name instead of the
    /// instance method name, which is important for instance methods that
    /// have been renamed (e.g., `$instance_describe_Color`).
    var_map: Option<FxHashMap<DefId, Var>>,
    /// Which representation a builtin transformer's methods should use, chosen
    /// from the FULL instance type before construction starts.
    ///
    /// ExceptT's runtime representation depends on the layer beneath it — over
    /// StateT a computation threads the state, plain it does not — and a
    /// dictionary slot cannot read that from codegen's ambient transformer
    /// stack, because the wrapper for a given method NAME is generated once per
    /// module under whatever layer happened to be current. So the variant
    /// travels in the name: `ExceptT.pure` vs `ExceptT_st.pure`.
    transformer_variant: Option<&'static str>,
    /// Nesting of `get_dictionary`: an instance context that leads back to
    /// itself through ever-different types never terminates otherwise.
    depth: u32,
    /// Dictionaries in scope at the construction site, with the constraint
    /// arguments each is for.
    ///
    /// A superclass slot can only be built from an instance when the type is
    /// concrete. `readWithM :: Monad m => …` constructs `Stream Sources m
    /// Char` while `m` is still ITS OWN variable, so the `Monad m` superclass
    /// resolved to nothing and the slot became a null placeholder — and
    /// `runPT`, which now selects its methods from that dictionary, read a
    /// field off null. But `readWithM` HAS a `Monad m` dictionary in scope, for
    /// exactly that `m`. This is how it reaches the slot.
    scope_dicts: Vec<(Symbol, Vec<Ty>, Var)>,
    /// Dictionaries reachable from one in scope by SUPERCLASS selection, as
    /// (superclass, arguments, base dictionary, field-index path).
    ///
    /// `scope_dicts` above only covers the classes a binding is constrained by
    /// directly. `myRead :: PandocMonad m => …` builds `Stream Sources m Char`,
    /// whose `Monad m` slot has no instance to come from because `m` is still a
    /// variable — and no `Monad` dictionary is in scope either, only
    /// `PandocMonad`, which HAS Monad as a superclass. Without this the slot
    /// stayed a null placeholder and `runPT` read a field off it.
    scope_super_dicts: Vec<(Symbol, Vec<Ty>, Var, Vec<usize>)>,
}

impl<'a> DictContext<'a> {
    /// Create a new dictionary context.
    #[must_use]
    pub fn new(registry: &'a ClassRegistry) -> Self {
        Self {
            registry,
            fresh_counter: 1000, // Start high to avoid collisions
            dict_bindings: Vec::new(),
            dict_cache: FxHashMap::default(),
            var_map: None,
            transformer_variant: None,
            depth: 0,
            scope_dicts: Vec::new(),
            scope_super_dicts: Vec::new(),
        }
    }

    /// Create a new dictionary context with a variable map for name resolution.
    #[must_use]
    pub fn new_with_var_map(registry: &'a ClassRegistry, var_map: FxHashMap<DefId, Var>) -> Self {
        Self {
            registry,
            fresh_counter: 1000,
            dict_bindings: Vec::new(),
            dict_cache: FxHashMap::default(),
            var_map: Some(var_map),
            transformer_variant: None,
            depth: 0,
            scope_dicts: Vec::new(),
            scope_super_dicts: Vec::new(),
        }
    }

    /// Offer the dictionaries in scope at the construction site, so a
    /// superclass slot that cannot be built from an instance can still be
    /// filled. See `scope_dicts`.
    pub fn set_scope_dicts(&mut self, dicts: Vec<(Symbol, Vec<Ty>, Var)>) {
        self.scope_dicts = dicts;
    }

    /// The full arguments of the ONE instance of multi-parameter `class` whose
    /// last argument fits the last of `args`, when the instance pins every
    /// other argument (the class's fundep: `MonadError e m | m -> e`).
    /// `args` may be just that last argument (a superclass slot records only
    /// it). `None` when no such instance, more than one, or one that leaves
    /// an earlier argument a variable (`MonadError e (StateT s m)`).
    fn fill_by_unique_instance(&self, class: Symbol, args: &[Ty]) -> Option<Vec<Ty>> {
        let target = args.last()?;
        let mut fits = self
            .registry
            .instances
            .get(&class)?
            .iter()
            .filter_map(|inst| {
                let (last, _) = inst.instance_types.split_last()?;
                if inst.instance_types.len() < 2
                    || !instance_heads_agree(
                        std::slice::from_ref(last),
                        std::slice::from_ref(target),
                    )
                {
                    return None;
                }
                let s =
                    types_match_multi(std::slice::from_ref(last), std::slice::from_ref(target))?;
                Some(
                    inst.instance_types
                        .iter()
                        .map(|t| s.apply(t))
                        .collect::<Vec<_>>(),
                )
            });
        let filled = fits.next()?;
        if fits.next().is_some()
            || filled[..filled.len() - 1]
                .iter()
                .any(crate::context::has_type_variables)
        {
            return None;
        }
        Some(filled)
    }

    /// A dictionary in scope at the construction site for this very
    /// constraint, matched by HEAD so `m` lines up with `m` and never with a
    /// different monad.
    ///
    /// An instance's OWN constraint often stays a variable: building
    /// `PandocMonad (ParsecT s st m)` needs `PandocMonad m`, which no instance
    /// can supply — but the enclosing binding is constrained by exactly that
    /// and holds the dictionary. Without this the slot was a null placeholder
    /// and every method the instance defines through it read a field off null.
    fn dict_in_scope_by_head(&self, constraint: &Constraint, span: Span) -> Option<core::Expr> {
        let matches = |class: &Symbol, args: &Vec<Ty>| {
            *class == constraint.class
                && args.len() == constraint.args.len()
                && args
                    .iter()
                    .zip(&constraint.args)
                    .all(|(a, b)| same_head(a, b))
        };
        if let Some((_, _, var)) = self
            .scope_dicts
            .iter()
            .find(|(class, args, _)| matches(class, args))
        {
            return Some(core::Expr::Var(var.clone(), span));
        }
        // A superclass reached from a single-parameter holder records only
        // the holder's parameter: `PandocMonad m`'s `MonadError PandocError m`
        // is offered as `[m]`. Match it on the wanted LAST argument (the
        // position that parameter stands for), or the StateT instance's
        // `MonadError e m` context inside a `PandocMonad m =>` function was a
        // null placeholder and its `catchError` jumped through it.
        let matches_last = |class: &Symbol, args: &Vec<Ty>| {
            *class == constraint.class
                && args.len() == 1
                && constraint.args.len() > 1
                && constraint
                    .args
                    .last()
                    .is_some_and(|want| same_head(&args[0], want))
        };
        let (_, _, base, path) = self
            .scope_super_dicts
            .iter()
            .find(|(class, args, _, _)| matches(class, args))
            .or_else(|| {
                self.scope_super_dicts
                    .iter()
                    .find(|(class, args, _, _)| matches_last(class, args))
            })?;
        let mut cur = core::Expr::Var(base.clone(), span);
        for idx in path {
            cur = core::Expr::App(
                Box::new(core::Expr::Var(
                    Var {
                        name: Symbol::intern(&format!("$sel_{idx}")),
                        id: VarId::new(0),
                        ty: Ty::Error,
                    },
                    span,
                )),
                Box::new(cur),
                span,
            );
        }
        Some(cur)
    }

    /// Offer the dictionaries reachable from one in scope by superclass
    /// selection. See `scope_super_dicts`.
    pub fn set_scope_super_dicts(&mut self, dicts: Vec<(Symbol, Vec<Ty>, Var, Vec<usize>)>) {
        self.scope_super_dicts = dicts;
    }

    /// Select the representation for builtin transformer methods in the
    /// dictionaries this context builds. See `transformer_variant`.
    pub fn set_transformer_variant(&mut self, variant: Option<&'static str>) {
        self.transformer_variant = variant;
    }

    /// Take ownership of the generated dictionary bindings.
    ///
    /// These bindings define intermediate dictionaries that were constructed
    /// during dictionary resolution. They should be added to the enclosing
    /// let expression or module bindings.
    pub fn take_bindings(&mut self) -> Vec<Bind> {
        std::mem::take(&mut self.dict_bindings)
    }

    /// Generate a fresh variable.
    fn fresh_var(&mut self, prefix: &str, ty: Ty, _span: Span) -> Var {
        let name = Symbol::intern(&format!("{}_{}", prefix, self.fresh_counter));
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
            ty,
        }
    }

    /// Get or create a dictionary for a constraint with concrete type(s).
    ///
    /// For single-parameter classes like `Num Int`, this returns a variable
    /// bound to the `Num Int` dictionary instance.
    ///
    /// For multi-parameter classes like `Convert Int String`, all type
    /// arguments are used to resolve the instance.
    ///
    /// For polymorphic instances like `instance Eq a => Eq [a]`, matching
    /// against `Eq [Int]` will:
    /// 1. Find the instance with substitution `{a -> Int}`
    /// 2. Apply the substitution to get the superclass constraint `Eq Int`
    /// 3. Recursively construct the `Eq Int` dictionary
    pub fn get_dictionary(&mut self, constraint: &Constraint, span: Span) -> Option<core::Expr> {
        // Need at least one type argument
        if constraint.args.is_empty() {
            return None;
        }

        // Check cache first - use all type args for the key
        let cache_key = (constraint.class, format!("{:?}", constraint.args));
        if let Some(var) = self.dict_cache.get(&cache_key) {
            return Some(core::Expr::Var(var.clone(), span));
        }

        // Try to resolve the instance using all type arguments
        // This returns both the instance and a substitution mapping type variables
        // to concrete types (e.g., {a -> Int} when matching Eq [a] against Eq [Int])
        let (instance, subst) = self
            .registry
            .resolve_instance_multi(constraint.class, &constraint.args)?;
        let class = self.registry.lookup_class(constraint.class)?;

        // Construct the dictionary, applying the substitution to resolve
        // type variables in superclass constraints
        const MAX_DICT_DEPTH: u32 = 64;
        if self.depth >= MAX_DICT_DEPTH {
            if std::env::var_os("BHC_DBG_DICTDEPTH").is_some() {
                eprintln!(
                    "[dict-depth] giving up at {} {:?}",
                    constraint.class.as_str(),
                    constraint.args
                );
            }
            return None;
        }
        self.depth += 1;
        let dict_expr = self.construct_dictionary(class, instance, &subst, span);
        self.depth -= 1;
        let dict_expr = dict_expr?;

        // Create a variable for this dictionary and cache it
        let type_names: Vec<String> = constraint.args.iter().map(type_name).collect();
        let dict_var = self.fresh_var(
            &format!(
                "$dict{}_{}",
                constraint.class.as_str(),
                type_names.join("_")
            ),
            Ty::Error, // Dictionary type
            span,
        );

        // Add binding for the dictionary
        self.dict_bindings
            .push(Bind::NonRec(dict_var.clone(), Box::new(dict_expr)));

        self.dict_cache.insert(cache_key, dict_var.clone());
        Some(core::Expr::Var(dict_var, span))
    }

    /// Construct a dictionary expression for an instance.
    ///
    /// The dictionary is a tuple containing:
    /// 1. Superclass dictionaries (if any)
    /// 2. Method implementations
    ///
    /// The `subst` parameter maps type variables in the instance to concrete types.
    /// This is crucial for polymorphic instances: when we match `instance Eq a => Eq [a]`
    /// against `Eq [Int]`, the substitution is `{a -> Int}`, and we need to
    /// recursively construct the dictionary for `Eq Int` (not `Eq a`).
    ///
    /// When an instance omits methods that have defaults, we use a two-phase approach:
    /// 1. Build a "partial dict" with superclass dicts + instance methods + error placeholders
    /// 2. Build the final dict, applying default functions to the partial dict
    ///
    /// This works because default methods are lowered as `\$dClass -> body` where
    /// `body` uses `$sel_N $dClass` to call other class methods. Passing the partial
    /// dict gives the default access to the instance-provided methods.
    fn construct_dictionary(
        &mut self,
        class: &ClassInfo,
        instance: &InstanceInfo,
        subst: &Subst,
        span: Span,
    ) -> Option<core::Expr> {
        // Build superclass dictionaries (shared by both paths).
        //
        // One slot per superclass is ALWAYS reserved, so method fields land at
        // index `superclasses.len() + method_index` — exactly where
        // `select_method` reads them. When a superclass dictionary can't be
        // resolved at this call site — e.g. an imported multi-parameter class
        // whose superclass constrains a type parameter that isn't pinned to a
        // concrete instance here (`class Monad m => Stream s m t`, superclass
        // `Monad m`) — the slot is filled with a null placeholder rather than
        // abandoning the whole dictionary. The placeholder is only dereferenced
        // if a method actually uses that superclass; for an erased monad such
        // as Identity the superclass `>>=`/`return` never touch the dict.
        let mut super_fields: Vec<core::Expr> = Vec::new();
        for (i, superclass) in class.superclasses.iter().enumerate() {
            let super_dict = instance
                .superclass_instances
                .get(i)
                .and_then(|super_ty| {
                    let concrete_super_ty = subst.apply(super_ty);
                    let super_constraint =
                        Constraint::new(*superclass, concrete_super_ty.clone(), span);
                    self.get_dictionary(&super_constraint, span).or_else(|| {
                        // A multi-parameter superclass is recorded at its LAST
                        // argument alone: `PandocMonad PandocIO`'s `MonadError
                        // PandocError m` slot asked for `MonadError PandocIO`,
                        // which no two-argument instance matches, so the slot
                        // was a null placeholder and every `catchError` inside
                        // a `PandocMonad m =>` function jumped through it.
                        let args = self.fill_by_unique_instance(
                            *superclass,
                            std::slice::from_ref(&concrete_super_ty),
                        )?;
                        self.get_dictionary(&Constraint::new_multi(*superclass, args, span), span)
                    })
                })
                .or_else(|| {
                    // The instance's OWN constraints name the right type when
                    // the head above does not resolve: `instance Monad m =>
                    // Stream Sources m Char` carries `Monad m` in its context,
                    // and with the parameter mapping recorded that is the same
                    // `m` the superclass is about. Without this the slot stays
                    // a null placeholder and selecting through it segfaults.
                    let matching = instance
                        .instance_constraints
                        .iter()
                        .find(|c| c.class == *superclass)?;
                    let args: Vec<Ty> = matching.args.iter().map(|a| subst.apply(a)).collect();
                    self.get_dictionary(
                        &Constraint::new_multi(*superclass, args.clone(), span),
                        span,
                    )
                    .or_else(|| {
                        // Builtin monad instances are registered under the
                        // BARE constructor (`Con StateT`) because codegen
                        // matches them by name, so an applied type such as
                        // `StateT Int IO` never matches. Retry at the head,
                        // as `resolve_dictionary` does for the same reason.
                        fn head_con(ty: &Ty) -> Option<&bhc_types::TyCon> {
                            match ty {
                                Ty::Con(tc) => Some(tc),
                                Ty::App(f, _) => head_con(f),
                                _ => None,
                            }
                        }
                        let head = args.first().and_then(head_con)?;
                        let bare = vec![Ty::Con(head.clone())];
                        if bare == args {
                            return None;
                        }
                        // Matching on the bare head throws away the layer
                        // BENEATH a transformer, and ExceptT's runtime
                        // representation depends on it: over StateT its methods
                        // thread the state, so the plain variant yields a value
                        // the `_st` bind then unwraps one level too far. A
                        // parser under `ExceptT e (StateT s IO)` had its own
                        // `Either ParseError a` result taken for ExceptT's error
                        // layer, and the character inside was forced as a
                        // pointer.
                        let saved = self.transformer_variant;
                        self.set_transformer_variant(args.first().and_then(inner_layer_suffix));
                        let built = self
                            .get_dictionary(&Constraint::new_multi(*superclass, bare, span), span);
                        self.set_transformer_variant(saved);
                        built
                    })
                })
                .or_else(|| {
                    // Last resort: a dictionary in scope AT THE CONSTRUCTION
                    // SITE for this very superclass. `readWithM :: Monad m =>`
                    // builds `Stream Sources m Char` while `m` is still its own
                    // variable, so no instance can supply the `Monad m` slot —
                    // but its own `Monad m` dictionary is right there. Without
                    // this the slot stayed null, and `runPT`, which now selects
                    // its methods from that dictionary, read a field off null.
                    //
                    // Matched by head, so `m` lines up with `m` and never with
                    // a different monad.
                    let want: Option<Vec<Ty>> = instance
                        .instance_constraints
                        .iter()
                        .find(|c| c.class == *superclass)
                        .map(|c| c.args.iter().map(|a| subst.apply(a)).collect());
                    let matches = |class: &Symbol, args: &Vec<Ty>| {
                        *class == *superclass
                            && want.as_ref().is_none_or(|w| {
                                w.len() == args.len()
                                    && w.iter().zip(args).all(|(a, b)| same_head(a, b))
                            })
                    };
                    self.scope_dicts
                        .iter()
                        .find(|(class, args, _)| matches(class, args))
                        .map(|(_, _, var)| core::Expr::Var(var.clone(), span))
                        .or_else(|| {
                            // Nothing in scope IS this class, but something in
                            // scope may HAVE it as a superclass: `myRead ::
                            // PandocMonad m => …` has only a `PandocMonad`
                            // dictionary, and Monad sits above it. Compose the
                            // selections directly rather than binding the
                            // intermediate — a Let-bound hop gets lowered as a
                            // thunk that captures the enclosing frame.
                            let (_, _, base, path) = self
                                .scope_super_dicts
                                .iter()
                                .find(|(class, args, _, _)| matches(class, args))?;
                            let mut cur = core::Expr::Var(base.clone(), span);
                            for idx in path {
                                let sel = Var {
                                    name: Symbol::intern(&format!("$sel_{idx}")),
                                    id: VarId::new(0),
                                    ty: Ty::Error,
                                };
                                cur = core::Expr::App(
                                    Box::new(core::Expr::Var(sel, span)),
                                    Box::new(cur),
                                    span,
                                );
                            }
                            Some(cur)
                        })
                })
                .unwrap_or_else(|| crate::context::null_dict_placeholder(*superclass, &[], span));
            super_fields.push(super_dict);
        }

        // Build constraint dictionaries if the instance has constraints.
        // For example, `instance Describable a => Describable (Box a)` needs
        // a `Describable a` dictionary. With substitution `{a -> Int}`, we
        // recursively construct the `Describable Int` dictionary.
        // Arity alignment: the instance's methods were wrapped in exactly one
        // dict lambda per `constraint_needs_dict_param` constraint, so use the
        // SAME filter here and NEVER drop a slot — fall back to a null
        // placeholder when the dictionary cannot be resolved (e.g. when
        // constructing at a still-polymorphic type). Dropping a slot shifts
        // every value argument of every method by one.
        let constraint_dicts: Vec<core::Expr> = instance
            .instance_constraints
            .iter()
            .filter(|c| self.registry.constraint_needs_dict_param(c.class))
            .map(|c| {
                let concrete = Constraint {
                    class: c.class,
                    args: c.args.iter().map(|a| subst.apply(a)).collect(),
                    span,
                };
                // A constraint that stays a bare variable (`Semigroup a` for
                // `Semigroup (Future s a)` built inside that instance's own
                // body) is the dictionary in scope; no instance can know it.
                // Instance-body dictionaries are registered without argument
                // types, so an empty argument list matches too.
                let var_only = !concrete.args.is_empty()
                    && concrete.args.iter().all(|a| matches!(a, Ty::Var(_)));
                let scoped = var_only
                    .then(|| {
                        self.scope_dicts
                            .iter()
                            .find(|(class, args, _)| {
                                *class == concrete.class
                                    && (args.is_empty()
                                        || (args.len() == concrete.args.len()
                                            && args
                                                .iter()
                                                .zip(&concrete.args)
                                                .all(|(a, b)| same_head(a, b))))
                            })
                            .map(|(_, _, var)| core::Expr::Var(var.clone(), span))
                    })
                    .flatten();
                // A context constraint whose leading arguments are still
                // variables but whose last is pinned: `MonadError e m` for
                // `MonadError e (StateT s m)` built at `StateT () PandocIO`
                // with `e` unknown at the use site. The class's fundep fixes
                // `e` from `PandocIO`; without it the slot was a null
                // placeholder and `catchError` at StateT jumped through it.
                let needs_fill = concrete.args.len() > 1
                    && concrete.args[..concrete.args.len() - 1]
                        .iter()
                        .any(crate::context::has_type_variables);
                scoped
                    .or_else(|| {
                        if !needs_fill {
                            return None;
                        }
                        let args = self.fill_by_unique_instance(concrete.class, &concrete.args)?;
                        self.get_dictionary(
                            &Constraint::new_multi(concrete.class, args, span),
                            span,
                        )
                    })
                    .or_else(|| self.get_dictionary(&concrete, span))
                    .or_else(|| self.dict_in_scope_by_head(&concrete, span))
                    .unwrap_or_else(|| {
                        crate::context::null_dict_placeholder(concrete.class, &concrete.args, span)
                    })
            })
            .collect();

        // Check if any defaults are needed
        let needs_defaults = class
            .methods
            .iter()
            .any(|m| !instance.methods.contains_key(m) && class.defaults.contains_key(m));

        if needs_defaults {
            // Phase 1: Build partial dict with null placeholders for defaults.
            // These slots are never accessed at runtime because the final dict
            // replaces them with applied default methods. Using an int literal 0
            // (null pointer) avoids the codegen issue where `error` (noreturn)
            // would terminate the basic block during tuple construction.
            // If a default method calls another default, this will segfault —
            // a known limitation of E.41.
            let mut partial_fields = super_fields.clone();
            for method_name in &class.methods {
                if let Some(&method_def_id) = instance.methods.get(method_name) {
                    let method_ref = Self::at_instance_ty(
                        self.method_reference(method_def_id, span),
                        class,
                        instance,
                        *method_name,
                        subst,
                    );
                    partial_fields.push(apply_constraint_dicts(
                        method_ref,
                        &constraint_dicts,
                        span,
                    ));
                } else {
                    partial_fields.push(core::Expr::Lit(core::Literal::Int(0), Ty::Error, span));
                }
            }

            let partial_dict_var = self.fresh_var("$partial_dict", Ty::Error, span);
            self.dict_bindings.push(Bind::NonRec(
                partial_dict_var.clone(),
                Box::new(make_dict(partial_fields, span)),
            ));

            // Phase 2: Build final dict with applied defaults
            let mut final_fields = super_fields;
            for method_name in &class.methods {
                if let Some(&method_def_id) = instance.methods.get(method_name) {
                    let method_ref = Self::at_instance_ty(
                        self.method_reference(method_def_id, span),
                        class,
                        instance,
                        *method_name,
                        subst,
                    );
                    final_fields.push(apply_constraint_dicts(method_ref, &constraint_dicts, span));
                } else if let Some(&default_def_id) = class.defaults.get(method_name) {
                    // Apply the default function to the partial dict.
                    // Default is `\$dClass -> body`, so `default partial_dict`
                    // produces the method implementation.
                    let default_ref = Self::at_instance_ty(
                        self.method_reference(default_def_id, span),
                        class,
                        instance,
                        *method_name,
                        subst,
                    );
                    final_fields.push(core::Expr::App(
                        Box::new(default_ref),
                        Box::new(core::Expr::Var(partial_dict_var.clone(), span)),
                        span,
                    ));
                } else {
                    let error_msg = format!(
                        "No implementation for method '{}' in instance {} {}",
                        method_name.as_str(),
                        instance.class.as_str(),
                        instance
                            .instance_types
                            .iter()
                            .map(|t| format!("{t:?}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                    final_fields.push(missing_method_field(
                        self,
                        class.name,
                        &final_fields[..class.superclasses.len()],
                        &instance.methods,
                        &constraint_dicts,
                        *method_name,
                        &error_msg,
                        span,
                    ));
                }
            }

            Some(make_dict(final_fields, span))
        } else {
            // No defaults needed — straightforward dictionary construction
            let mut fields = super_fields;
            for method_name in &class.methods {
                let method_expr = if let Some(&method_def_id) = instance.methods.get(method_name) {
                    let method_ref = Self::at_instance_ty(
                        self.method_reference(method_def_id, span),
                        class,
                        instance,
                        *method_name,
                        subst,
                    );
                    apply_constraint_dicts(method_ref, &constraint_dicts, span)
                } else {
                    let error_msg = format!(
                        "No implementation for method '{}' in instance {} {}",
                        method_name.as_str(),
                        instance.class.as_str(),
                        instance
                            .instance_types
                            .iter()
                            .map(|t| format!("{t:?}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                    missing_method_field(
                        self,
                        class.name,
                        &fields[..class.superclasses.len()],
                        &instance.methods,
                        &constraint_dicts,
                        *method_name,
                        &error_msg,
                        span,
                    )
                };
                fields.push(method_expr);
            }
            Some(make_dict(fields, span))
        }
    }

    /// Create a reference to a method implementation.
    /// An instance method reference carrying its type AT THIS INSTANCE: the
    /// class method's signature with the class parameter replaced by the
    /// (matched) instance head.
    ///
    /// The reference otherwise keeps its binder's scheme type. For pandoc's
    /// `instance … => PandocMonad (t m)` that is `t m CommonState`; in a clone
    /// of an imported caller (no span-keyed types to consult) monomorphization
    /// mapped only the head `t` to the concrete monad and specialized the
    /// method at an ill-kinded type, so the generic method ran instead of one
    /// built for `StateT s PandocIO`. Single-parameter classes only.
    fn at_instance_ty(
        method_ref: core::Expr,
        class: &ClassInfo,
        instance: &InstanceInfo,
        method: Symbol,
        subst: &Subst,
    ) -> core::Expr {
        let core::Expr::Var(mut var, span) = method_ref else {
            return method_ref;
        };
        if class.param_count != 1 || instance.instance_types.len() != 1 {
            return core::Expr::Var(var, span);
        }
        let Some(scheme) = class.method_types.get(&method) else {
            return core::Expr::Var(var, span);
        };
        let param = scheme
            .constraints
            .iter()
            .find(|c| c.class == class.name)
            .and_then(|c| match c.args.first() {
                Some(Ty::Var(v)) => Some(v.clone()),
                _ => None,
            });
        let head = subst.apply(&instance.instance_types[0]);
        if let Some(param) = param {
            let mut at = Subst::new();
            at.insert(&param, head);
            var.ty = at.apply(&scheme.ty);
        } else if crate::context::transformer_headed(&head) {
            // An IMPORTED class records only each method's arrow count
            // (`Fun(Error, … Error)`). For a monad-like class at a transformer
            // instance the method's result is an action in that transformer,
            // which is all monomorphization reads: `n` erased arguments to
            // `head <erased>`.
            let mut arrows = 0;
            let mut t = &scheme.ty;
            while let Ty::Fun(_, r) = t {
                arrows += 1;
                t = r;
            }
            let mut ty = Ty::App(Box::new(head), Box::new(Ty::Error));
            for _ in 0..arrows {
                ty = Ty::Fun(Box::new(Ty::Error), Box::new(ty));
            }
            var.ty = ty;
        }
        core::Expr::Var(var, span)
    }

    fn method_reference(&self, def_id: DefId, span: Span) -> core::Expr {
        // Builtin monad instance methods are SYNTHETIC: their DefIds are
        // hardcoded at registration and name nothing the program declared. They
        // must be resolved by name, not through the var_map, because that map
        // is keyed by the program's own DefIds and these numbers collide with
        // whatever happens to land on them — 150-154 hit `Control.Exception`'s
        // `throwIO`/`bracket`, and 10060-10064 hit `ExceptT`/`runExceptT`, each
        // silently filling an IO dictionary with unrelated functions. There is
        // no DefId range that is safe to assume free, so check the name table
        // first instead of hunting for one.
        if let Some(name) = self.transformer_method_name(def_id) {
            // Apply the representation chosen for this instance, if any:
            // `ExceptT.pure` becomes `ExceptT_st.pure` when the layer beneath
            // is StateT. See `transformer_variant`.
            let name = match (self.transformer_variant, name.as_str().split_once('.')) {
                (Some(suffix), Some((con, method))) => {
                    Symbol::intern(&format!("{con}{suffix}.{method}"))
                }
                _ => name,
            };
            return core::Expr::Var(
                Var {
                    name,
                    id: VarId::new(def_id.index()),
                    ty: Ty::Error,
                },
                span,
            );
        }

        // Otherwise the var_map holds the registered variable name. This
        // handles real instance methods that have been renamed (e.g.
        // $instance_describe_Color).
        if let Some(var_map) = &self.var_map {
            if let Some(var) = var_map.get(&def_id) {
                return core::Expr::Var(var.clone(), span);
            }
        }

        let name = self
            .find_method_name(def_id)
            .unwrap_or_else(|| Symbol::intern(&format!("$method_{}", def_id.index())));
        let var = Var {
            name,
            id: VarId::new(def_id.index()),
            ty: Ty::Error,
        };
        core::Expr::Var(var, span)
    }

    /// Map transformer instance method `DefIds` to qualified names for codegen.
    fn transformer_method_name(&self, def_id: DefId) -> Option<Symbol> {
        let name = match def_id.index() {
            // IO instances. These MUST be qualified even though IO is the
            // default layer: `create_builtin_closure` caches one wrapper per
            // NAME per module, so an unqualified `pure` in the IO dictionary
            // and an unqualified `pure` reached from a StateT context would
            // share a single wrapper baked with whichever layer was lowered
            // first. The other layers are already qualified for the same
            // reason; IO fell through to `find_method_name`'s bare names.
            10060 => "IO.fmap",
            10061 => "IO.pure",
            10062 => "IO.<*>",
            10063 => "IO.>>=",
            10064 => "IO.>>",
            // ExceptT instances. Only the methods with a value form in
            // `lower_builtin_direct` are registered; `<*>`/`>>` deliberately
            // are NOT, so `missing_method_field` gives them a deferred-error
            // lambda rather than a slot that aborts with `unknown builtin`.
            10080 => "ExceptT.fmap",
            10081 => "ExceptT.pure",
            10083 => "ExceptT.>>=",
            // Identity instances
            10000 => "Identity",
            10001 => "runIdentity",
            10002 => "Identity.fmap",
            10003 => "Identity.pure",
            10004 => "Identity.<*>",
            10005 => "Identity.>>=",
            10006 => "Identity.>>",
            // ReaderT instances
            10020 => "ReaderT",
            10021 => "runReaderT",
            10022 => "ReaderT.fmap",
            10023 => "ReaderT.pure",
            10024 => "ReaderT.<*>",
            10025 => "ReaderT.>>=",
            10026 => "ReaderT.>>",
            10027 => "ReaderT.lift",
            10028 => "ReaderT.liftIO",
            10029 => "ask",
            10030 => "asks",
            10031 => "local",
            // StateT instances
            10040 => "StateT",
            10041 => "runStateT",
            10042 => "StateT.fmap",
            10043 => "StateT.pure",
            10044 => "StateT.<*>",
            10045 => "StateT.>>=",
            10046 => "StateT.>>",
            10047 => "StateT.lift",
            10048 => "StateT.liftIO",
            10049 => "get",
            10050 => "put",
            10051 => "modify",
            10053 => "gets",
            10054 => "evalStateT",
            10055 => "execStateT",
            _ => return None,
        };
        Some(Symbol::intern(name))
    }

    /// Find the method name for a given `DefId` by searching all registered instances.
    fn find_method_name(&self, def_id: DefId) -> Option<Symbol> {
        for instances in self.registry.instances.values() {
            for instance in instances {
                for (name, &id) in &instance.methods {
                    if id == def_id {
                        return Some(*name);
                    }
                }
            }
        }
        // Also check class defaults
        for class in self.registry.classes.values() {
            for (name, &id) in &class.defaults {
                if id == def_id {
                    return Some(*name);
                }
            }
        }
        None
    }
}

/// Extract a method from a dictionary.
///
/// Given a dictionary variable and a method name, generates the
/// expression to select that method from the dictionary.
///
/// # Arguments
///
/// * `dict_var` - The dictionary variable
/// * `class` - The class name (to determine field order)
/// * `method_name` - The method to extract
/// * `registry` - The class registry
/// * `span` - Source span
#[must_use]
pub fn select_method(
    dict_var: &Var,
    class: Symbol,
    method_name: Symbol,
    registry: &ClassRegistry,
    span: Span,
) -> Option<core::Expr> {
    let class_info = registry.lookup_class(class)?;

    // Calculate the field index
    // First come superclass dictionaries, then methods
    let superclass_count = class_info.superclasses.len();
    let method_index = class_info.methods.iter().position(|m| *m == method_name)?;

    let _total_fields = superclass_count + class_info.methods.len();

    // Note: even for single-method classes, the dictionary is always a proper
    // tuple (padded by make_dict), so $sel_0 is always used.

    let field_index = superclass_count + method_index;

    // Generate a selector expression
    Some(make_field_selector(dict_var, field_index, span))
}

/// Extract a superclass dictionary from a dictionary.
///
/// # Arguments
///
/// * `dict_var` - The dictionary variable
/// * `class` - The class name
/// * `superclass` - The superclass to extract
/// * `registry` - The class registry
/// * `span` - Source span
#[must_use]
pub fn select_superclass(
    dict_var: &Var,
    class: Symbol,
    superclass: Symbol,
    registry: &ClassRegistry,
    span: Span,
) -> Option<core::Expr> {
    let class_info = registry.lookup_class(class)?;

    // Find the superclass index
    let superclass_index = class_info
        .superclasses
        .iter()
        .position(|s| *s == superclass)?;

    // Generate a selector expression
    Some(make_field_selector(dict_var, superclass_index, span))
}

/// Create a tuple expression from a list of fields.
fn make_tuple(fields: Vec<core::Expr>, span: Span) -> core::Expr {
    if fields.is_empty() {
        // Unit
        let unit_var = Var {
            name: Symbol::intern("()"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        return core::Expr::Var(unit_var, span);
    }

    if fields.len() == 1 {
        // Single element - no tuple needed
        return fields.into_iter().next().unwrap();
    }

    // Build tuple constructor application
    let tuple_name = Symbol::intern(&format!("({})", ",".repeat(fields.len() - 1)));
    let tuple_var = Var {
        name: tuple_name,
        id: VarId::new(0),
        ty: Ty::Error,
    };

    let mut result = core::Expr::Var(tuple_var, span);
    for field in fields {
        result = core::Expr::App(Box::new(result), Box::new(field), span);
    }

    result
}

/// Apply constraint dictionaries to a method expression.
///
/// When an instance has constraints (e.g., `instance Describable a => Describable (Box a)`),
/// each method was lowered as `\$dDescribable -> body`. To build the dictionary,
/// we need to apply the resolved constraint dictionaries:
///   `method $dDescribable_Int` → fully applied method
fn apply_constraint_dicts(
    method_expr: core::Expr,
    constraint_dicts: &[core::Expr],
    span: Span,
) -> core::Expr {
    let mut result = method_expr;
    for dict in constraint_dicts {
        result = core::Expr::App(Box::new(result), Box::new(dict.clone()), span);
    }
    result
}

/// Build a dictionary tuple, padding single-element dictionaries to 2 elements.
///
/// Dictionaries are extracted with `$sel_N` which expects a proper tuple struct
/// (tag + fields). When a class has only 1 method and no superclasses, the
/// standard `make_tuple` would optimize away the tuple wrapper, causing
/// `$sel_0` to fail. This function pads 1-element dictionaries with a
/// `String` replaced by `[Char]` throughout a type.
pub(crate) fn string_as_char_list(ty: &Ty) -> Ty {
    match ty {
        Ty::Con(tc) if tc.name.as_str() == "String" => Ty::List(Box::new(Ty::Con(TyCon::new(
            Symbol::intern("Char"),
            bhc_types::Kind::Star,
        )))),
        Ty::App(f, a) => Ty::App(
            Box::new(string_as_char_list(f)),
            Box::new(string_as_char_list(a)),
        ),
        Ty::Fun(a, b) => Ty::Fun(
            Box::new(string_as_char_list(a)),
            Box::new(string_as_char_list(b)),
        ),
        Ty::List(e) => Ty::List(Box::new(string_as_char_list(e))),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(string_as_char_list).collect()),
        other => other.clone(),
    }
}

/// dummy field to ensure a valid 2-tuple is always created.
fn make_dict(fields: Vec<core::Expr>, span: Span) -> core::Expr {
    if fields.len() == 1 {
        // Pad to 2 elements so (,) creates a proper tuple for $sel_0
        let mut padded = fields;
        padded.push(core::Expr::Lit(core::Literal::Int(0), Ty::Error, span));
        return make_tuple(padded, span);
    }

    make_tuple(fields, span)
}

/// Create an error expression that will fail at runtime with a message.
///
/// This generates `error "message"` which will throw an exception at runtime.
/// Used for cases that should have been caught at type checking but weren't.
fn make_error_expr(msg: &str, span: Span) -> core::Expr {
    let error_var = Var {
        name: Symbol::intern("error"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let msg_lit = core::Expr::Lit(core::Literal::String(Symbol::intern(msg)), Ty::Error, span);
    core::Expr::App(
        Box::new(core::Expr::Var(error_var, span)),
        Box::new(msg_lit),
        span,
    )
}

/// Synthesize a missing `<$` from the instance's own `fmap`
/// (`\x p -> fmap (const x) p`), or defer a missing-method error to CALL time
/// by wrapping it in a lambda — building a dictionary must never throw.
#[allow(clippy::too_many_arguments)]
fn missing_method_field(
    ctx: &mut DictContext,
    class_name: Symbol,
    super_fields: &[core::Expr],
    class_methods: &FxHashMap<Symbol, bhc_hir::DefId>,
    constraint_dicts: &[core::Expr],
    method_name: Symbol,
    error_msg: &str,
    span: Span,
) -> core::Expr {
    // The builtin Monoid class carries no default bodies, so an instance that
    // defines only `mempty` (the usual modern shape — `<>` lives in the
    // Semigroup instance) got error closures in its `mappend`/`mconcat`
    // slots. `foldr mappend mempty` and `mconcat` over any user Monoid then
    // ran them, which is how pandoc's `query`/`foldMap` over `Min Int`
    // crashed. Fill them with the class defaults: `mappend = (<>)` via the
    // Semigroup superclass slot, `mconcat = foldr mappend mempty`.
    if class_name.as_str() == "Monoid" {
        let semigroup_append = || -> Option<core::Expr> {
            let sg = super_fields.first()?;
            if matches!(sg, core::Expr::Lit(..)) {
                return None; // unresolved superclass placeholder
            }
            let sel = Var {
                name: Symbol::intern("$sel_0"),
                id: VarId::new(0),
                ty: Ty::Error,
            };
            Some(core::Expr::App(
                Box::new(core::Expr::Var(sel, span)),
                Box::new(sg.clone()),
                span,
            ))
        };
        match method_name.as_str() {
            "mappend" => {
                if let Some(append) = semigroup_append() {
                    return append;
                }
            }
            "mconcat" => {
                let mappend = match class_methods.get(&Symbol::intern("mappend")) {
                    Some(&id) => {
                        let r = ctx.method_reference(id, span);
                        Some(apply_constraint_dicts(r, constraint_dicts, span))
                    }
                    None => semigroup_append(),
                };
                let mempty = class_methods.get(&Symbol::intern("mempty")).map(|&id| {
                    let r = ctx.method_reference(id, span);
                    apply_constraint_dicts(r, constraint_dicts, span)
                });
                if let (Some(mappend), Some(mempty)) = (mappend, mempty) {
                    let foldr_var = Var {
                        name: Symbol::intern("foldr"),
                        id: VarId::new(0),
                        ty: Ty::Error,
                    };
                    let xs = ctx.fresh_var("$mconcat_xs", Ty::Error, span);
                    let body = core::Expr::App(
                        Box::new(core::Expr::App(
                            Box::new(core::Expr::App(
                                Box::new(core::Expr::Var(foldr_var, span)),
                                Box::new(mappend),
                                span,
                            )),
                            Box::new(mempty),
                            span,
                        )),
                        Box::new(core::Expr::Var(xs.clone(), span)),
                        span,
                    );
                    return core::Expr::Lam(xs, Box::new(body), span);
                }
            }
            _ => {}
        }
    }
    if method_name.as_str() == "<$" {
        if let Some(&fmap_id) = class_methods.get(&Symbol::intern("fmap")) {
            let fmap_ref = ctx.method_reference(fmap_id, span);
            let fmap_applied = apply_constraint_dicts(fmap_ref, constraint_dicts, span);
            let xv = ctx.fresh_var("x", Ty::Error, span);
            let pv = ctx.fresh_var("p", Ty::Error, span);
            let iv = ctx.fresh_var("_i", Ty::Error, span);
            let const_l = core::Expr::Lam(iv, Box::new(core::Expr::Var(xv.clone(), span)), span);
            let body = core::Expr::App(
                Box::new(core::Expr::App(
                    Box::new(fmap_applied),
                    Box::new(const_l),
                    span,
                )),
                Box::new(core::Expr::Var(pv.clone(), span)),
                span,
            );
            return core::Expr::Lam(
                xv,
                Box::new(core::Expr::Lam(pv, Box::new(body), span)),
                span,
            );
        }
    }
    // Class defaults for sequencing. An instance that defines only `>>=`
    // (or only `<*>`) is the ordinary shape, and its `>>` / `*>` / `<*`
    // slots held the error closure below — so `p >> q` on any such user
    // monad crashed. `m >> k = m >>= \_ -> k`; `a *> b = (id <$ a) <*> b`
    // and `a <* b = fmap const a <*> b`, with `fmap` from the Functor
    // superclass slot.
    let lam2 = |ctx: &mut DictContext, body: &dyn Fn(&Var, &Var) -> core::Expr| {
        let a = ctx.fresh_var("a", Ty::Error, span);
        let b = ctx.fresh_var("b", Ty::Error, span);
        let inner = body(&a, &b);
        core::Expr::Lam(a, Box::new(core::Expr::Lam(b, Box::new(inner), span)), span)
    };
    let app = |f: core::Expr, x: core::Expr| core::Expr::App(Box::new(f), Box::new(x), span);
    let var = |v: &Var| core::Expr::Var(v.clone(), span);
    if class_name.as_str() == "Monad" && method_name.as_str() == ">>" {
        if let Some(&bind_id) = class_methods.get(&Symbol::intern(">>=")) {
            let bind_ref = ctx.method_reference(bind_id, span);
            let bind = apply_constraint_dicts(bind_ref, constraint_dicts, span);
            let ignored = ctx.fresh_var("_", Ty::Error, span);
            return lam2(ctx, &|m, k| {
                let cont = core::Expr::Lam(ignored.clone(), Box::new(var(k)), span);
                app(app(bind.clone(), var(m)), cont)
            });
        }
    }
    if class_name.as_str() == "Applicative" && matches!(method_name.as_str(), "*>" | "<*") {
        let functor_fmap = super_fields
            .first()
            .filter(|f| !matches!(f, core::Expr::Lit(..)))
            .map(|functor| {
                let sel0 = Var {
                    name: Symbol::intern("$sel_0"),
                    id: VarId::new(0),
                    ty: Ty::Error,
                };
                app(core::Expr::Var(sel0, span), functor.clone())
            });
        if let (Some(fmap), Some(&ap_id)) =
            (functor_fmap, class_methods.get(&Symbol::intern("<*>")))
        {
            let ap_ref = ctx.method_reference(ap_id, span);
            let ap = apply_constraint_dicts(ap_ref, constraint_dicts, span);
            let x = ctx.fresh_var("x", Ty::Error, span);
            let y = ctx.fresh_var("y", Ty::Error, span);
            // `*>` keeps the second result, `<*` the first.
            let keep = if method_name.as_str() == "*>" { &y } else { &x };
            let pick = core::Expr::Lam(
                x.clone(),
                Box::new(core::Expr::Lam(y.clone(), Box::new(var(keep)), span)),
                span,
            );
            return lam2(ctx, &|a, b| {
                app(
                    app(ap.clone(), app(app(fmap.clone(), pick.clone()), var(a))),
                    var(b),
                )
            });
        }
    }
    // Defer the error to call time: a raw error field would throw while the
    // dictionary itself is being BUILT.
    let dummy = ctx.fresh_var("_missing", Ty::Error, span);
    core::Expr::Lam(dummy, Box::new(make_error_expr(error_msg, span)), span)
}

/// Create a field selector expression for a tuple.
///
/// This generates a case expression that extracts the field at the given index.
fn make_field_selector(dict_var: &Var, field_index: usize, span: Span) -> core::Expr {
    // For now, create a simple selector function application
    // A more complete implementation would generate proper case expressions
    let selector_name = Symbol::intern(&format!("$sel_{field_index}"));
    let selector_var = Var {
        name: selector_name,
        id: VarId::new(field_index),
        ty: Ty::Error,
    };

    core::Expr::App(
        Box::new(core::Expr::Var(selector_var, span)),
        Box::new(core::Expr::Var(dict_var.clone(), span)),
        span,
    )
}

/// Get a simple name for a type (for variable naming).
fn type_name(ty: &Ty) -> String {
    match ty {
        Ty::Con(c) => c.name.as_str().to_string(),
        Ty::Var(v) => format!("t{}", v.id),
        Ty::App(f, _) => type_name(f),
        Ty::Fun(_, _) => "Fun".to_string(),
        Ty::Tuple(_) => "Tuple".to_string(),
        Ty::List(_) => "List".to_string(),
        _ => "Unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_types::{types_match, TyCon};

    #[test]
    fn test_types_match_simple() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        // Same types match with empty substitution
        assert!(types_match(&int_ty, &int_ty).is_some());
        // Different types don't match
        assert!(types_match(&int_ty, &bool_ty).is_none());
    }

    #[test]
    fn test_class_registry() {
        let mut registry = ClassRegistry::new();

        // Register Eq class
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
        registry.register_class(eq_class);

        // Register Eq Int instance
        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("=="), DefId::new(100));
        methods.insert(Symbol::intern("/="), DefId::new(101));

        let eq_int = InstanceInfo {
            class: Symbol::intern("Eq"),
            instance_types: vec![Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star))],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };
        registry.register_instance(eq_int);

        // Test lookup
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let result = registry.resolve_instance(Symbol::intern("Eq"), &int_ty);
        assert!(result.is_some());
        let (instance, subst) = result.unwrap();
        assert_eq!(instance.class, Symbol::intern("Eq"));
        // For concrete type matching, substitution should be empty
        assert!(subst.is_empty());

        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        let no_instance = registry.resolve_instance(Symbol::intern("Eq"), &bool_ty);
        assert!(no_instance.is_none());
    }

    #[test]
    fn test_multi_param_type_class() {
        let mut registry = ClassRegistry::new();

        // Register Convert class (multi-parameter: class Convert a b where convert :: a -> b)
        let convert_class = ClassInfo {
            name: Symbol::intern("Convert"),
            param_count: 2,
            methods: vec![Symbol::intern("convert")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        registry.register_class(convert_class);

        // Register instance Convert Int String
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));

        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("convert"), DefId::new(200));

        let convert_int_string = InstanceInfo::new_multi(
            Symbol::intern("Convert"),
            vec![int_ty.clone(), string_ty.clone()],
            methods.clone(),
        );
        registry.register_instance(convert_int_string);

        // Register instance Convert Int Bool
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        methods.insert(Symbol::intern("convert"), DefId::new(201));

        let convert_int_bool = InstanceInfo::new_multi(
            Symbol::intern("Convert"),
            vec![int_ty.clone(), bool_ty.clone()],
            methods,
        );
        registry.register_instance(convert_int_bool);

        // Test multi-param lookup: Convert Int String should resolve
        let result = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone(), string_ty.clone()],
        );
        assert!(result.is_some());
        let (instance, _subst) = result.unwrap();
        assert_eq!(instance.instance_types.len(), 2);

        // Test multi-param lookup: Convert Int Bool should resolve
        let result2 = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone(), bool_ty.clone()],
        );
        assert!(result2.is_some());

        // Test multi-param lookup: Convert String Int should NOT resolve
        let no_instance = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[string_ty.clone(), int_ty.clone()],
        );
        assert!(no_instance.is_none());

        // Test that wrong number of args fails
        let wrong_arity = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            std::slice::from_ref(&int_ty), // Only one arg instead of two
        );
        assert!(wrong_arity.is_none());
    }

    #[test]
    fn test_types_match_multi() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        // Same types match (returns Some with empty substitution)
        let result = types_match_multi(
            &[int_ty.clone(), string_ty.clone()],
            &[int_ty.clone(), string_ty.clone()],
        );
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());

        // Different types don't match
        assert!(types_match_multi(
            &[int_ty.clone(), string_ty.clone()],
            &[int_ty.clone(), bool_ty.clone()]
        )
        .is_none());

        // Different lengths don't match
        assert!(types_match_multi(
            &[int_ty.clone(), string_ty.clone()],
            std::slice::from_ref(&int_ty)
        )
        .is_none());

        // Empty matches empty
        assert!(types_match_multi(&[], &[]).is_some());
    }

    #[test]
    fn test_instance_info_constructors() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));
        let methods = FxHashMap::default();

        // Test single constructor
        let single =
            InstanceInfo::new_single(Symbol::intern("Eq"), int_ty.clone(), methods.clone());
        assert_eq!(single.instance_types.len(), 1);
        assert!(single.first_type().is_some());

        // Test multi constructor
        let multi = InstanceInfo::new_multi(
            Symbol::intern("Convert"),
            vec![int_ty.clone(), string_ty.clone()],
            methods,
        );
        assert_eq!(multi.instance_types.len(), 2);
        assert!(multi.first_type().is_some());
    }

    #[test]
    fn test_associated_types() {
        let mut registry = ClassRegistry::new();

        // Create a class with an associated type:
        // class Container c where
        //   type Element c
        //   empty :: c
        //   insert :: Element c -> c -> c
        let container_class = ClassInfo {
            name: Symbol::intern("Container"),
            param_count: 1,
            methods: vec![Symbol::intern("empty"), Symbol::intern("insert")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![AssocTypeInfo {
                name: Symbol::intern("Element"),
                params: vec![],
                kind: Kind::Star,
                default: None,
            }],
        };
        registry.register_class(container_class);

        // Register instance Container [a] where type Element [a] = a
        let a_tyvar = bhc_types::TyVar::new_star(0);
        let list_a = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(Ty::Var(a_tyvar.clone())),
        );

        let a_ty = Ty::Var(a_tyvar);

        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("empty"), DefId::new(500));
        methods.insert(Symbol::intern("insert"), DefId::new(501));

        let mut assoc_impls = FxHashMap::default();
        assoc_impls.insert(Symbol::intern("Element"), a_ty.clone());

        let list_instance = InstanceInfo {
            class: Symbol::intern("Container"),
            instance_types: vec![list_a.clone()],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: assoc_impls,
            instance_constraints: vec![],
        };
        registry.register_instance(list_instance);

        // Test associated type lookup
        let assoc_types = registry.class_assoc_types(Symbol::intern("Container"));
        assert_eq!(assoc_types.len(), 1);
        assert_eq!(assoc_types[0], Symbol::intern("Element"));

        // Test lookup_assoc_type_class
        let class_name = registry.lookup_assoc_type_class(Symbol::intern("Element"));
        assert_eq!(class_name, Some(Symbol::intern("Container")));

        // Test lookup_assoc_type
        let assoc_info = registry.lookup_assoc_type(Symbol::intern("Element"));
        assert!(assoc_info.is_some());
        assert_eq!(assoc_info.unwrap().name, Symbol::intern("Element"));

        // Test that non-existent associated types return None
        let no_assoc = registry.lookup_assoc_type(Symbol::intern("NotAnAssocType"));
        assert!(no_assoc.is_none());
    }

    #[test]
    fn test_polymorphic_instance_matching() {
        let mut registry = ClassRegistry::new();

        // Register Eq class
        let eq_class = ClassInfo {
            name: Symbol::intern("Eq"),
            param_count: 1,
            methods: vec![Symbol::intern("==")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        registry.register_class(eq_class);

        // Register a polymorphic instance: instance Eq a => Eq [a]
        // This means: if we have Eq a, we can derive Eq [a]
        let a_tyvar = bhc_types::TyVar::new_star(0);
        let a_ty = Ty::Var(a_tyvar.clone());

        // [a] type = List applied to a
        let list_a = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(a_ty.clone()),
        );

        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("=="), DefId::new(100));

        let eq_list_instance = InstanceInfo {
            class: Symbol::intern("Eq"),
            instance_types: vec![list_a.clone()],
            methods,
            // Superclass instance: the Eq a constraint gets stored here as just `a`
            // When we match Eq [Int], we substitute a -> Int to get Eq Int
            superclass_instances: vec![a_ty.clone()],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };
        registry.register_instance(eq_list_instance);

        // Now try to resolve Eq [Int]
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let list_int = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(int_ty.clone()),
        );

        let result = registry.resolve_instance(Symbol::intern("Eq"), &list_int);
        assert!(result.is_some(), "Should find Eq [Int] via Eq [a] instance");

        let (instance, subst) = result.unwrap();
        assert_eq!(instance.class, Symbol::intern("Eq"));

        // The substitution should map type variable 'a' (id=0) to Int
        assert!(!subst.is_empty(), "Substitution should not be empty");

        // Apply the substitution to the superclass instance type
        // This should give us Int (the type for the Eq a constraint)
        let superclass_ty = &instance.superclass_instances[0];
        let concrete_superclass_ty = subst.apply(superclass_ty);

        // The concrete superclass type should be Int
        match &concrete_superclass_ty {
            Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
            _ => panic!("Expected Int type, got {:?}", concrete_superclass_ty),
        }
    }

    #[test]
    fn test_polymorphic_multi_param_instance() {
        let mut registry = ClassRegistry::new();

        // Register a multi-param class: class Convert a b where convert :: a -> b
        let convert_class = ClassInfo {
            name: Symbol::intern("Convert"),
            param_count: 2,
            methods: vec![Symbol::intern("convert")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            superclass_params: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        registry.register_class(convert_class);

        // Register a polymorphic instance: instance Convert a [a]
        // This converts any type to a singleton list
        let a_tyvar = bhc_types::TyVar::new_star(0);
        let a_ty = Ty::Var(a_tyvar.clone());

        let list_a = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(a_ty.clone()),
        );

        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("convert"), DefId::new(200));

        let convert_a_lista = InstanceInfo::new_multi(
            Symbol::intern("Convert"),
            vec![a_ty.clone(), list_a.clone()],
            methods,
        );
        registry.register_instance(convert_a_lista);

        // Try to resolve Convert Int [Int]
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let list_int = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(int_ty.clone()),
        );

        let result = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone(), list_int.clone()],
        );
        assert!(
            result.is_some(),
            "Should find Convert Int [Int] via Convert a [a]"
        );

        let (instance, subst) = result.unwrap();
        assert_eq!(instance.class, Symbol::intern("Convert"));
        assert!(!subst.is_empty(), "Substitution should map a -> Int");

        // Apply substitution to 'a' should give Int
        let applied = subst.apply(&a_ty);
        match &applied {
            Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
            _ => panic!("Expected Int, got {:?}", applied),
        }

        // Try to resolve Convert Int [Bool] - should NOT match
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        let list_bool = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(bool_ty.clone()),
        );

        let no_match = registry
            .resolve_instance_multi(Symbol::intern("Convert"), &[int_ty.clone(), list_bool]);
        assert!(
            no_match.is_none(),
            "Convert Int [Bool] should NOT match Convert a [a]"
        );
    }
}
