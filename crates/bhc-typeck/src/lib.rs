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
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
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
use bhc_hir::{DefId, HirId, Module};
use bhc_intern::Symbol;
use bhc_span::FileId;
use bhc_types::{Scheme, Ty};
use indexmap::IndexMap;
use rustc_hash::FxHashMap;

/// The result of type checking a module.
///
/// Contains the original HIR along with type annotations for all
/// expressions and definitions.
#[derive(Debug)]
pub struct TypedModule {
    /// The original HIR module.
    pub hir: Module,
    /// Inferred types for each expression (indexed by `HirId`).
    pub expr_types: FxHashMap<HirId, Ty>,
    /// Type schemes for each definition (indexed by `DefId`).
    pub def_schemes: FxHashMap<DefId, Scheme>,
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
    let mut ctx = TyCtxt::new(file_id);
    ctx.overloaded_strings = hir.overloaded_strings;
    ctx.overloaded_lists = hir.overloaded_lists;
    ctx.scoped_type_variables = hir.scoped_type_variables;

    // Register built-in types
    ctx.register_builtins();

    // If we have definition mappings from the lowering pass, use them
    // to register builtins with the correct DefIds
    if let Some(def_map) = defs {
        ctx.register_lowered_builtins(def_map);
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
            (vec![], Ty::List(Box::new(Ty::Con(bhc_types::TyCon::new(Symbol::intern("Char"), Kind::Star))))),
        );

        // type ShowS = String -> String
        let string_ty = Ty::List(Box::new(Ty::Con(bhc_types::TyCon::new(Symbol::intern("Char"), Kind::Star))));
        ctx.type_aliases.insert(
            Symbol::intern("ShowS"),
            (vec![], Ty::Fun(Box::new(string_ty.clone()), Box::new(string_ty.clone()))),
        );

        // type ReadS a = String -> [(a, String)]
        let pair = Ty::Tuple(vec![Ty::Var(a.clone()), string_ty.clone()]);
        let list_pair = Ty::List(Box::new(pair));
        ctx.type_aliases.insert(
            Symbol::intern("ReadS"),
            (vec![a.clone()], Ty::Fun(Box::new(string_ty.clone()), Box::new(list_pair))),
        );

        // type FilePath = String
        ctx.type_aliases.insert(
            Symbol::intern("FilePath"),
            (vec![], string_ty),
        );

        // Pandoc type aliases (from pandoc-types)
        let text_ty = Ty::Con(bhc_types::TyCon::new(Symbol::intern("Text"), Kind::Star));

        // type Attr = (Text, [Text], [(Text, Text)])
        // Pandoc's type alias (XML.Light has a data type Attr, but that uses
        // a different mechanism — data types vs type aliases don't conflict).
        ctx.type_aliases.insert(
            Symbol::intern("Attr"),
            (vec![], Ty::Tuple(vec![
                text_ty.clone(),
                Ty::List(Box::new(text_ty.clone())),
                Ty::List(Box::new(Ty::Tuple(vec![text_ty.clone(), text_ty.clone()]))),
            ])),
        );

        // type Target = (Text, Text)
        ctx.type_aliases.insert(
            Symbol::intern("Target"),
            (vec![], Ty::Tuple(vec![text_ty.clone(), text_ty.clone()])),
        );

        // type ColSpec = (Alignment, ColWidth)
        ctx.type_aliases.insert(
            Symbol::intern("ColSpec"),
            (vec![], Ty::Tuple(vec![
                Ty::Con(bhc_types::TyCon::new(Symbol::intern("Alignment"), Kind::Star)),
                Ty::Con(bhc_types::TyCon::new(Symbol::intern("ColWidth"), Kind::Star)),
            ])),
        );

        // type ListAttributes = (Int, ListNumberStyle, ListNumberDelim)
        ctx.type_aliases.insert(
            Symbol::intern("ListAttributes"),
            (vec![], Ty::Tuple(vec![
                Ty::Con(bhc_types::TyCon::new(Symbol::intern("Int"), Kind::Star)),
                Ty::Con(bhc_types::TyCon::new(Symbol::intern("ListNumberStyle"), Kind::Star)),
                Ty::Con(bhc_types::TyCon::new(Symbol::intern("ListNumberDelim"), Kind::Star)),
            ])),
        );

        // type ShortCaption = [Inline]
        ctx.type_aliases.insert(
            Symbol::intern("ShortCaption"),
            (vec![], Ty::List(Box::new(Ty::Con(bhc_types::TyCon::new(Symbol::intern("Inline"), Kind::Star))))),
        );

        // Many is a newtype: newtype Many a = Many (Seq a)
        // But for type-checking purposes, we treat Blocks/Inlines as [Block]/[Inline]
        // since BHC doesn't have a real Seq type. The Many newtype is isomorphic to lists
        // in our simplified model.
        let many_con = bhc_types::TyCon::new(Symbol::intern("Many"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
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
