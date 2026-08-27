//! Interface file generation from compiled modules.
//!
//! Generates `.bhi` interface files that capture the public API of a module,
//! enabling separate compilation and type checking without source code.

use crate::{
    ClassMethod, Constraint, DataConstructor, ExportedClass, ExportedInstance, ExportedType,
    ExportedValue, Kind, ModuleInterface, Type, TypeDefinition, TypeSignature,
};
use bhc_ast::Module as AstModule;
use bhc_types::Ty as TyckTy;
use std::collections::HashMap;

/// A resolved type synonym: (parameter placeholders, expanded RHS in interface form).
type AliasMap = HashMap<String, (Vec<String>, Type)>;

/// Generate a module interface from a parsed AST and type-checked module.
///
/// Extracts exported values, types, classes, and instances to produce a
/// `ModuleInterface` suitable for serialization to a `.bhi` file.
pub fn generate_interface(
    module_name: &str,
    ast: &AstModule,
    typed: &bhc_typeck::TypedModule,
    imported_class_params: &HashMap<String, usize>,
) -> ModuleInterface {
    let mut iface = ModuleInterface::new(module_name);

    // Compute a simple hash from the source for consistency checking
    iface.header.module_hash = compute_module_hash(module_name);

    // Build a type-synonym map from the module's typeck aliases (local AND
    // imported). Exported signatures are expanded against it so the `.bhi` is
    // self-contained: a consumer that does not import a synonym's defining
    // module can still unfold it (e.g. `Text.Parsec.String` uses `runP` whose
    // signature mentions `SourceName` from `Text.Parsec.Pos`, which it does not
    // import directly).
    let aliases = build_alias_map(typed);

    // Number of type parameters per class declared in this module, so a
    // multi-parameter instance head (`instance Stream S Int`, parsed as a single
    // App spine) can be split into its component types when serialized. Without
    // this, the interface records only one type and a consumer cannot complete a
    // functional dependency (`Stream S t | s -> t`), so the dictionary fails to
    // resolve at the call site.
    // Seed with IMPORTED classes' parameter counts (from the loaded
    // interfaces), then let local declarations override. Without the seed, a
    // module defining an instance of an imported multi-parameter class
    // (`instance Monad m => Stream Sources m Char` — Stream lives in parsec)
    // serialized a flattened one-element head that no consumer could match.
    let mut class_param_count: HashMap<String, usize> = imported_class_params.clone();
    for decl in &ast.decls {
        if let bhc_ast::Decl::ClassDecl(cls) = decl {
            class_param_count.insert(cls.name.name.as_str().to_string(), cls.params.len());
        }
    }

    // Extract exports from AST declarations
    for decl in &ast.decls {
        match decl {
            bhc_ast::Decl::TypeSig(sig) => {
                // Split off a leading constraint context `C a => t` (under an
                // optional `forall`) so it is serialized in the signature's
                // `constraints`. Without this, a consumer importing a
                // constrained function reconstructs a scheme with NO
                // constraints, so call-site dictionary resolution omits the
                // dictionary and every following argument shifts by one.
                let mut cursor = &sig.ty;
                if let bhc_ast::Type::Forall(_, inner, _) = cursor {
                    cursor = inner.as_ref();
                }
                let (sig_constraints, bare_ty): (Vec<Constraint>, &bhc_ast::Type) = match cursor {
                    bhc_ast::Type::Constrained(cs, inner, _) => (
                        cs.iter().map(convert_ast_constraint).collect(),
                        inner.as_ref(),
                    ),
                    other => (Vec::new(), other),
                };
                for name in &sig.names {
                    let exported_ty = expand_synonyms(convert_ast_type(bare_ty), &aliases, 0);
                    iface.add_value(ExportedValue {
                        name: name.name.as_str().to_string(),
                        signature: TypeSignature {
                            type_vars: Vec::new(),
                            constraints: sig_constraints.clone(),
                            ty: exported_ty,
                        },
                        inline: crate::InlineInfo::None,
                        arity: None,
                    });
                }
            }
            bhc_ast::Decl::DataDecl(data) => {
                let params: Vec<String> = data
                    .params
                    .iter()
                    .map(|p| p.name.name.as_str().to_string())
                    .collect();
                let constructors: Vec<DataConstructor> =
                    data.constrs.iter().map(convert_con_decl).collect();
                let kind = params_to_kind(params.len());
                iface.add_type(ExportedType {
                    name: data.name.name.as_str().to_string(),
                    params,
                    kind,
                    definition: Some(TypeDefinition::Data(constructors)),
                });
            }
            bhc_ast::Decl::Newtype(nt) => {
                let params: Vec<String> = nt
                    .params
                    .iter()
                    .map(|p| p.name.name.as_str().to_string())
                    .collect();
                let con = convert_con_decl(&nt.constr);
                let kind = params_to_kind(params.len());
                iface.add_type(ExportedType {
                    name: nt.name.name.as_str().to_string(),
                    params,
                    kind,
                    definition: Some(TypeDefinition::Newtype(con)),
                });
                // GeneralizedNewtypeDeriving: carry the monad-stack instances
                // (Functor/Applicative/Monad) so importing modules can dispatch
                // `fmap`/`pure`/`<*>`/`return`/`>>=` at this newtype. HIR->Core
                // emits the impls as external `$instance_{method}_{Name}`
                // symbols (matching the consumer's reconstruction), harvested as
                // interface instance methods. Only these classes take part in
                // transformer-newtype deriving, and only with a single type
                // parameter (the deriver's own precondition).
                if nt.params.len() == 1 {
                    for clause in &nt.deriving {
                        let methods: Vec<String> = match clause.class.name.as_str() {
                            "Functor" => vec!["fmap".to_string()],
                            "Applicative" => {
                                vec!["pure".to_string(), "<*>".to_string()]
                            }
                            "Monad" => {
                                vec!["return".to_string(), ">>=".to_string(), ">>".to_string()]
                            }
                            _ => continue,
                        };
                        iface.add_instance(ExportedInstance {
                            class: clause.class.name.as_str().to_string(),
                            types: vec![Type::Con(nt.name.name.as_str().to_string())],
                            constraints: Vec::new(),
                            methods,
                        });
                    }
                }
            }
            bhc_ast::Decl::TypeAlias(ta) => {
                let params: Vec<String> = ta
                    .params
                    .iter()
                    .map(|p| p.name.name.as_str().to_string())
                    .collect();
                let kind = params_to_kind(params.len());
                iface.add_type(ExportedType {
                    name: ta.name.name.as_str().to_string(),
                    params,
                    kind,
                    definition: Some(TypeDefinition::TypeSynonym(convert_ast_type(&ta.ty))),
                });
            }
            bhc_ast::Decl::ClassDecl(cls) => {
                let params: Vec<String> = cls
                    .params
                    .iter()
                    .map(|p| p.name.name.as_str().to_string())
                    .collect();
                let supers: Vec<Constraint> =
                    cls.context.iter().map(convert_ast_constraint).collect();
                // Methods with a default implementation in the class body
                // (a FunBind alongside the TypeSigs). Consumers use this to
                // apply the exported default fn when an instance omits the
                // method, instead of fabricating a nonexistent
                // `$instance_{method}_{Type}` extern.
                let default_names: std::collections::HashSet<&str> = cls
                    .methods
                    .iter()
                    .filter_map(|d| {
                        if let bhc_ast::Decl::FunBind(fb) = d {
                            Some(fb.name.name.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                // Extract method signatures from class body declarations
                let methods: Vec<ClassMethod> = cls
                    .methods
                    .iter()
                    .filter_map(|d| {
                        if let bhc_ast::Decl::TypeSig(sig) = d {
                            let aliases = &aliases;
                            let default_names = &default_names;
                            let class_params = &params;
                            let class_name = cls.name.name.as_str();
                            Some(sig.names.iter().map(move |name| ClassMethod {
                                name: name.name.as_str().to_string(),
                                signature: TypeSignature {
                                    type_vars: Vec::new(),
                                    // The method's implicit class constraint
                                    // (`HasReaderOptions st => ...`), sharing
                                    // the class param var with the sig type.
                                    // Consumers use it to extract the class
                                    // param's instantiation from a recorded
                                    // occurrence type when the param appears
                                    // in neither argument nor result head.
                                    constraints: vec![Constraint {
                                        class: class_name.to_string(),
                                        args: class_params
                                            .iter()
                                            .map(|p| Type::Var(p.clone()))
                                            .collect(),
                                    }],
                                    ty: expand_synonyms(convert_ast_type(&sig.ty), aliases, 0),
                                },
                                has_default: default_names.contains(name.name.as_str()),
                            }))
                        } else {
                            None
                        }
                    })
                    .flatten()
                    .collect();
                iface.add_class(ExportedClass {
                    name: cls.name.name.as_str().to_string(),
                    params,
                    superclasses: supers,
                    methods,
                });
            }
            bhc_ast::Decl::InstanceDecl(inst) => {
                let constraints: Vec<Constraint> =
                    inst.context.iter().map(convert_ast_constraint).collect();
                let param_count = class_param_count
                    .get(inst.class.name.as_str())
                    .copied()
                    .unwrap_or(1);
                let types: Vec<Type> = flatten_instance_head(&inst.ty, param_count)
                    .iter()
                    .map(|t| convert_ast_type(t))
                    .collect();
                let methods: Vec<String> = inst
                    .methods
                    .iter()
                    .filter_map(|m| {
                        if let bhc_ast::Decl::FunBind(fb) = m {
                            Some(fb.name.name.as_str().to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                iface.add_instance(ExportedInstance {
                    class: inst.class.name.as_str().to_string(),
                    types,
                    constraints,
                    methods,
                });
            }
            // Other declarations (FunBind without sig, fixity, foreign, etc.) — skip for MVP
            _ => {}
        }
    }

    // Record re-exports so the consumer side can resolve names this module
    // exports but does not declare. Whole-module re-exports (`module X`)
    // become `(X, "*")` entries chased into X's interface — a facade like
    // Text.Pandoc.Class carries none of its submodules' declarations locally,
    // so without this its interface would be empty. Name-level re-exports
    // (`module Foo (bar) where import Baz (bar)`) become `(bar, origin)`
    // entries, with origin `""` when no explicit import list names them (an
    // open import); consumers chase the origin or fall back to a stub
    // binding, mirroring the source loader's re-export stubs. `module M`
    // naming the module itself re-exports the local declarations, already
    // included above.
    if let Some(export_list) = &ast.exports {
        let mut local_values: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut local_types: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for decl in &ast.decls {
            match decl {
                bhc_ast::Decl::TypeSig(sig) => {
                    local_values.extend(sig.names.iter().map(|n| n.name.as_str()));
                }
                bhc_ast::Decl::FunBind(fb) => {
                    local_values.insert(fb.name.name.as_str());
                }
                bhc_ast::Decl::DataDecl(d) => {
                    local_types.insert(d.name.name.as_str());
                }
                bhc_ast::Decl::Newtype(nt) => {
                    local_types.insert(nt.name.name.as_str());
                }
                bhc_ast::Decl::TypeAlias(ta) => {
                    local_types.insert(ta.name.name.as_str());
                }
                bhc_ast::Decl::ClassDecl(cls) => {
                    local_types.insert(cls.name.name.as_str());
                    for m in &cls.methods {
                        if let bhc_ast::Decl::TypeSig(sig) = m {
                            local_values.extend(sig.names.iter().map(|n| n.name.as_str()));
                        }
                    }
                }
                _ => {}
            }
        }

        // Origin module(s) for a re-exported name. An explicit import list
        // naming it gives the definitive origin. Otherwise, every unqualified
        // open/hiding import (that doesn't hide the name) is a CANDIDATE,
        // joined with ';' for the consumer to try in order — a facade like
        // Org.Parsing re-exports `QuoteContext (..)` obtained via `import
        // Text.Pandoc.Parsing hiding (..)`, and with no candidates the
        // consumer can only stub the type, losing its (..) children.
        let import_origin = |name: &str, want_type: bool| -> String {
            let module_of = |import: &bhc_ast::ImportDecl| -> String {
                import
                    .module
                    .parts
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(".")
            };
            let item_matches = |item: &bhc_ast::Import| match item {
                bhc_ast::Import::Var(i, _) => !want_type && i.name.as_str() == name,
                bhc_ast::Import::Type(i, _, _) => want_type && i.name.as_str() == name,
                bhc_ast::Import::Pattern(_, _) => false,
            };
            let mut candidates: Vec<String> = Vec::new();
            for import in &ast.imports {
                match &import.spec {
                    Some(bhc_ast::ImportSpec::Only(items)) => {
                        if items.iter().any(item_matches) {
                            return module_of(import);
                        }
                    }
                    Some(bhc_ast::ImportSpec::Hiding(items)) => {
                        if !import.qualified && !items.iter().any(item_matches) {
                            candidates.push(module_of(import));
                        }
                    }
                    None => {
                        if !import.qualified {
                            candidates.push(module_of(import));
                        }
                    }
                }
            }
            candidates.join(";")
        };

        for exp in export_list {
            match exp {
                bhc_ast::Export::Module(m, _) => {
                    let name = m
                        .parts
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(".");
                    if name != module_name {
                        iface.reexports.insert(name, "*".to_string());
                    }
                }
                bhc_ast::Export::Var(ident, _) => {
                    let name = ident.name.as_str();
                    if !local_values.contains(name) {
                        iface
                            .reexports
                            .insert(name.to_string(), import_origin(name, false));
                    }
                }
                bhc_ast::Export::Type(ident, _, _) => {
                    let name = ident.name.as_str();
                    if !local_types.contains(name) {
                        iface
                            .reexports
                            .insert(format!("type:{name}"), import_origin(name, true));
                    }
                }
                bhc_ast::Export::Pattern(_, _) => {}
            }
        }
    }

    // Record import dependencies
    for import in &ast.imports {
        let dep_name = import
            .module
            .parts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(".");
        iface.dependencies.push(crate::InterfaceDependency {
            module: dep_name,
            hash: 0, // Hash not yet available for dependencies
        });
    }

    iface
}

/// Convert an AST constructor declaration to an interface DataConstructor.
fn convert_con_decl(con: &bhc_ast::ConDecl) -> DataConstructor {
    match &con.fields {
        bhc_ast::ConFields::Positional(types) => DataConstructor {
            name: con.name.name.as_str().to_string(),
            fields: types.iter().map(convert_ast_type).collect(),
            field_names: None,
        },
        bhc_ast::ConFields::Record(fields) => DataConstructor {
            name: con.name.name.as_str().to_string(),
            fields: fields.iter().map(|f| convert_ast_type(&f.ty)).collect(),
            field_names: Some(
                fields
                    .iter()
                    .map(|f| f.name.name.as_str().to_string())
                    .collect(),
            ),
        },
    }
}

/// Convert an AST type expression to an interface Type.
/// Type synonyms that bhc's typeck registers UNCONDITIONALLY for every module
/// (see the "standard Haskell type aliases" block in bhc-typeck). Every consumer
/// already has these in scope and can unfold them, so we must NOT expand them
/// into exported signatures — several (`Attr`, `Markup`, `Parsec`, …) also serve
/// as codegen dispatch keys by name. Only user/imported synonyms a consumer
/// might lack (e.g. parsec's `SourceName`) get expanded.
const UNCONDITIONAL_SYNONYMS: &[&str] = &[
    "String",
    "ShowS",
    "ReadS",
    "FilePath",
    "Attr",
    "Target",
    "Attribute",
    "Markup",
    "MarkupM",
    "Html",
    "NonEmpty",
    "Seq",
    "Parsec",
    "SyntaxMap",
    "Token",
    "SourceLine",
    "ColSpec",
    "ListAttributes",
    "ShortCaption",
    "Blocks",
    "Inlines",
    "Many",
];

/// Build a synonym map for expansion from the module's typeck aliases, EXCLUDING
/// the unconditional builtins (which every consumer already has).
fn build_alias_map(typed: &bhc_typeck::TypedModule) -> AliasMap {
    let mut map = HashMap::new();
    for (name, (params, rhs)) in &typed.type_aliases {
        let n = name.as_str();
        if UNCONDITIONAL_SYNONYMS.contains(&n) {
            continue;
        }
        if let Some(rhs_iface) = ty_to_iface(rhs) {
            let param_ids = params.iter().map(|p| format!("v{}", p.id)).collect();
            map.insert(n.to_string(), (param_ids, rhs_iface));
        }
    }
    map
}

/// Convert a typeck `Ty` to an interface `Type` (best-effort; returns `None` for
/// types with no interface representation, e.g. unboxed primitives or `Error`).
fn ty_to_iface(ty: &TyckTy) -> Option<Type> {
    Some(match ty {
        TyckTy::Var(tv) => Type::Var(format!("v{}", tv.id)),
        TyckTy::Con(tc) => Type::Con(tc.name.as_str().to_string()),
        TyckTy::App(f, x) => Type::App(Box::new(ty_to_iface(f)?), Box::new(ty_to_iface(x)?)),
        TyckTy::Fun(a, b) => Type::Fun(Box::new(ty_to_iface(a)?), Box::new(ty_to_iface(b)?)),
        TyckTy::Tuple(ts) => {
            let mut v = Vec::with_capacity(ts.len());
            for t in ts {
                v.push(ty_to_iface(t)?);
            }
            Type::Tuple(v)
        }
        TyckTy::List(e) => Type::List(Box::new(ty_to_iface(e)?)),
        TyckTy::Forall(_, t) => ty_to_iface(t)?,
        _ => return None,
    })
}

/// Recursively expand type synonyms in an interface `Type` so the emitted
/// signature is self-contained (references no synonym a consumer might lack).
/// Handles nullary (`type SourceName = String`) and parameterized
/// (`type Parser = Parsec String ()`) synonyms; bounded to avoid a runaway on a
/// (malformed) recursive synonym.
fn expand_synonyms(ty: Type, aliases: &AliasMap, depth: u32) -> Type {
    if depth > 60 || aliases.is_empty() {
        return ty;
    }
    match ty {
        Type::App(_, _) => {
            // Collect the application spine: head applied to args.
            let mut args: Vec<Type> = Vec::new();
            let mut head = ty;
            while let Type::App(f, x) = head {
                args.push(*x);
                head = *f;
            }
            args.reverse();
            let args: Vec<Type> = args
                .into_iter()
                .map(|a| expand_synonyms(a, aliases, depth + 1))
                .collect();
            if let Type::Con(name) = &head {
                if let Some((params, rhs)) = aliases.get(name) {
                    if args.len() >= params.len() {
                        let mut subst: HashMap<String, Type> = HashMap::new();
                        for (p, a) in params.iter().zip(args.iter()) {
                            subst.insert(p.clone(), a.clone());
                        }
                        let mut result = subst_vars(rhs.clone(), &subst);
                        for a in &args[params.len()..] {
                            result = Type::App(Box::new(result), Box::new(a.clone()));
                        }
                        return expand_synonyms(result, aliases, depth + 1);
                    }
                }
            }
            let head = expand_synonyms(head, aliases, depth + 1);
            args.into_iter()
                .fold(head, |acc, a| Type::App(Box::new(acc), Box::new(a)))
        }
        Type::Con(ref name) => {
            if let Some((params, rhs)) = aliases.get(name) {
                if params.is_empty() {
                    return expand_synonyms(rhs.clone(), aliases, depth + 1);
                }
            }
            ty
        }
        Type::Fun(a, b) => Type::Fun(
            Box::new(expand_synonyms(*a, aliases, depth + 1)),
            Box::new(expand_synonyms(*b, aliases, depth + 1)),
        ),
        Type::Tuple(ts) => Type::Tuple(
            ts.into_iter()
                .map(|t| expand_synonyms(t, aliases, depth + 1))
                .collect(),
        ),
        Type::List(e) => Type::List(Box::new(expand_synonyms(*e, aliases, depth + 1))),
        Type::Var(_) => ty,
    }
}

/// Substitute type variables (by placeholder name) in an interface `Type`.
fn subst_vars(ty: Type, subst: &HashMap<String, Type>) -> Type {
    match ty {
        Type::Var(ref name) => subst.get(name).cloned().unwrap_or(ty),
        Type::Con(_) => ty,
        Type::App(f, x) => Type::App(
            Box::new(subst_vars(*f, subst)),
            Box::new(subst_vars(*x, subst)),
        ),
        Type::Fun(a, b) => Type::Fun(
            Box::new(subst_vars(*a, subst)),
            Box::new(subst_vars(*b, subst)),
        ),
        Type::Tuple(ts) => Type::Tuple(ts.into_iter().map(|t| subst_vars(t, subst)).collect()),
        Type::List(e) => Type::List(Box::new(subst_vars(*e, subst))),
    }
}

/// Split a (possibly multi-parameter) instance head into its component types.
/// The parser represents `instance Stream S Int` as a single App spine
/// `App(Con(S), Con(Int))`; with `param_count == 2` this yields `[S, Int]`.
/// Mirrors `flatten_instance_type` in bhc-lower so the serialized instance types
/// match what the same-module path registers.
fn flatten_instance_head(ty: &bhc_ast::Type, param_count: usize) -> Vec<&bhc_ast::Type> {
    if param_count <= 1 {
        return vec![ty];
    }
    let mut spine = Vec::new();
    let mut current = ty;
    loop {
        if let bhc_ast::Type::App(f, x, _) = current {
            spine.push(x.as_ref());
            current = f.as_ref();
        } else {
            spine.push(current);
            break;
        }
    }
    spine.reverse();
    if spine.len() > param_count {
        spine.split_off(spine.len() - param_count)
    } else {
        spine
    }
}

fn convert_ast_type(ty: &bhc_ast::Type) -> Type {
    match ty {
        bhc_ast::Type::Var(tv, _) => Type::Var(tv.name.name.as_str().to_string()),
        bhc_ast::Type::Con(ident, _) => Type::Con(ident.name.as_str().to_string()),
        bhc_ast::Type::QualCon(_module, ident, _) => Type::Con(ident.name.as_str().to_string()),
        bhc_ast::Type::App(f, x, _) => {
            Type::App(Box::new(convert_ast_type(f)), Box::new(convert_ast_type(x)))
        }
        bhc_ast::Type::Fun(a, b, _) => {
            Type::Fun(Box::new(convert_ast_type(a)), Box::new(convert_ast_type(b)))
        }
        bhc_ast::Type::Tuple(ts, _) => Type::Tuple(ts.iter().map(convert_ast_type).collect()),
        bhc_ast::Type::List(t, _) => Type::List(Box::new(convert_ast_type(t))),
        bhc_ast::Type::Paren(t, _) => convert_ast_type(t),
        bhc_ast::Type::Forall(_, inner, _) => convert_ast_type(inner),
        bhc_ast::Type::Constrained(_, inner, _) => convert_ast_type(inner),
        bhc_ast::Type::Bang(inner, _) | bhc_ast::Type::Lazy(inner, _) => convert_ast_type(inner),
        bhc_ast::Type::InfixOp(lhs, op, rhs, _) => {
            // Desugar `a `Op` b` to `Op a b`
            let op_con = Type::Con(op.name.as_str().to_string());
            let app_l = Type::App(Box::new(op_con), Box::new(convert_ast_type(lhs)));
            Type::App(Box::new(app_l), Box::new(convert_ast_type(rhs)))
        }
        bhc_ast::Type::PromotedList(_, _) | bhc_ast::Type::NatLit(_, _) => {
            Type::Con("Unknown".to_string())
        }
    }
}

/// Convert an AST constraint to an interface Constraint.
fn convert_ast_constraint(constraint: &bhc_ast::Constraint) -> Constraint {
    Constraint {
        class: constraint.class.name.as_str().to_string(),
        args: constraint.args.iter().map(convert_ast_type).collect(),
    }
}

/// Compute a kind for a type constructor with the given number of parameters.
fn params_to_kind(n: usize) -> Kind {
    if n == 0 {
        Kind::Type
    } else {
        Kind::fun(Kind::Type, params_to_kind(n - 1))
    }
}

/// Compute a simple hash for a module name (placeholder for content-based hashing).
fn compute_module_hash(module_name: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    module_name.hash(&mut hasher);
    hasher.finish()
}
