//! Built-in types and data constructors.
//!
//! This module defines the primitive types that are always available
//! in BHC programs: `Int`, `Float`, `Char`, `Bool`, `String`, etc.
//!
//! Additionally, this module defines the `Tensor` type constructor for
//! shape-indexed tensors (M9 Dependent Types Preview):
//!
//! ```text
//! Tensor :: [Nat] -> * -> *
//! ```
//!
//! Example usage:
//! ```text
//! Tensor '[1024, 768] Float  -- A 1024x768 matrix of floats
//! ```
//!
//! ## Dynamic Tensors (M9 Phase 5)
//!
//! For gradual adoption, `DynTensor` provides a runtime-shaped escape hatch:
//!
//! ```text
//! DynTensor :: * -> *
//!
//! toDynamic :: forall shape a. Tensor shape a -> DynTensor a
//! fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
//! ```
//!
//! These types are registered into the type environment before
//! type checking user code.

use bhc_hir::DefId;
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_types::{dyn_tensor, Kind, Scheme, Ty, TyCon, TyList, TyVar};

use crate::env::TypeEnv;

/// Built-in types and their type constructors.
#[derive(Debug, Clone)]
pub struct Builtins {
    // Type constructors
    /// The `Int` type constructor.
    pub int_con: TyCon,
    /// The `Float` type constructor.
    pub float_con: TyCon,
    /// The `Char` type constructor.
    pub char_con: TyCon,
    /// The `Bool` type constructor.
    pub bool_con: TyCon,
    /// The `String` type constructor.
    pub string_con: TyCon,
    /// The `[]` (list) type constructor.
    pub list_con: TyCon,
    /// The `Maybe` type constructor.
    pub maybe_con: TyCon,
    /// The `Either` type constructor.
    pub either_con: TyCon,
    /// The `IO` type constructor.
    pub io_con: TyCon,
    /// The `Tensor` type constructor (M9).
    /// Kind: `[Nat] -> * -> *`
    pub tensor_con: TyCon,

    // M9 Phase 5: Dynamic tensor types
    /// The `DynTensor` type constructor.
    /// Kind: `* -> *`
    /// An existentially-quantified tensor with runtime-only shape.
    pub dyn_tensor_con: TyCon,
    /// The `ShapeWitness` type constructor.
    /// Kind: `[Nat] -> *`
    /// A singleton type for reifying shapes at runtime.
    pub shape_witness_con: TyCon,

    // Convenient type values
    /// The `Int` type.
    pub int_ty: Ty,
    /// The `Float` type.
    pub float_ty: Ty,
    /// The `Char` type.
    pub char_ty: Ty,
    /// The `Bool` type.
    pub bool_ty: Ty,
    /// The `String` type.
    pub string_ty: Ty,
}

impl Default for Builtins {
    fn default() -> Self {
        Self::new()
    }
}

impl Builtins {
    /// Create the built-in types.
    #[must_use]
    pub fn new() -> Self {
        // Type constructors with kind *
        let int_con = TyCon::new(Symbol::intern("Int"), Kind::Star);
        let float_con = TyCon::new(Symbol::intern("Float"), Kind::Star);
        let char_con = TyCon::new(Symbol::intern("Char"), Kind::Star);
        let bool_con = TyCon::new(Symbol::intern("Bool"), Kind::Star);
        let string_con = TyCon::new(Symbol::intern("String"), Kind::Star);

        // Type constructors with kind * -> *
        let list_con = TyCon::new(Symbol::intern("[]"), Kind::star_to_star());
        let maybe_con = TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star());
        let io_con = TyCon::new(Symbol::intern("IO"), Kind::star_to_star());

        // Type constructors with kind * -> * -> *
        let either_kind = Kind::Arrow(
            Box::new(Kind::Star),
            Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
        );
        let either_con = TyCon::new(Symbol::intern("Either"), either_kind);

        // M9: Tensor type constructor with kind [Nat] -> * -> *
        // This enables shape-indexed tensors: Tensor '[1024, 768] Float
        let tensor_kind = Kind::Arrow(
            Box::new(Kind::List(Box::new(Kind::Nat))), // [Nat]
            Box::new(Kind::Arrow(
                Box::new(Kind::Star), // element type
                Box::new(Kind::Star), // result type
            )),
        );
        let tensor_con = TyCon::new(Symbol::intern("Tensor"), tensor_kind);

        // M9 Phase 5: Dynamic tensor types
        // DynTensor :: * -> *
        let dyn_tensor_con = dyn_tensor::dyn_tensor_tycon();

        // ShapeWitness :: [Nat] -> *
        let shape_witness_con = dyn_tensor::shape_witness_tycon();

        // Convenient types
        let int_ty = Ty::Con(int_con.clone());
        let float_ty = Ty::Con(float_con.clone());
        let char_ty = Ty::Con(char_con.clone());
        let bool_ty = Ty::Con(bool_con.clone());
        let string_ty = Ty::Con(string_con.clone());

        Self {
            int_con,
            float_con,
            char_con,
            bool_con,
            string_con,
            list_con,
            maybe_con,
            either_con,
            io_con,
            tensor_con,
            dyn_tensor_con,
            shape_witness_con,
            int_ty,
            float_ty,
            char_ty,
            bool_ty,
            string_ty,
        }
    }

    /// Register built-in data constructors in the environment.
    pub fn register_data_cons(&self, env: &mut TypeEnv) {
        // Bool constructors
        // True :: Bool
        // False :: Bool
        let true_id = DefId::new(BUILTIN_TRUE_ID);
        let false_id = DefId::new(BUILTIN_FALSE_ID);
        env.register_data_con(
            true_id,
            Symbol::intern("True"),
            Scheme::mono(self.bool_ty.clone()),
        );
        env.register_data_con(
            false_id,
            Symbol::intern("False"),
            Scheme::mono(self.bool_ty.clone()),
        );

        // Maybe constructors
        // Nothing :: forall a. Maybe a
        // Just :: forall a. a -> Maybe a
        let a = TyVar::new_star(BUILTIN_TYVAR_A);
        let maybe_a = Ty::App(
            Box::new(Ty::Con(self.maybe_con.clone())),
            Box::new(Ty::Var(a.clone())),
        );

        let nothing_id = DefId::new(BUILTIN_NOTHING_ID);
        let just_id = DefId::new(BUILTIN_JUST_ID);
        env.register_data_con(
            nothing_id,
            Symbol::intern("Nothing"),
            Scheme::poly(vec![a.clone()], maybe_a.clone()),
        );
        env.register_data_con(
            just_id,
            Symbol::intern("Just"),
            Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), maybe_a)),
        );

        // List constructors
        // [] :: forall a. [a]
        // (:) :: forall a. a -> [a] -> [a]
        let list_a = Ty::List(Box::new(Ty::Var(a.clone())));

        let nil_id = DefId::new(BUILTIN_NIL_ID);
        let cons_id = DefId::new(BUILTIN_CONS_ID);
        env.register_data_con(
            nil_id,
            Symbol::intern("[]"),
            Scheme::poly(vec![a.clone()], list_a.clone()),
        );
        env.register_data_con(
            cons_id,
            Symbol::intern(":"),
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
            ),
        );

        // Either constructors
        // Left :: forall a b. a -> Either a b
        // Right :: forall a b. b -> Either a b
        let b = TyVar::new_star(BUILTIN_TYVAR_B);
        let either_ab = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Con(self.either_con.clone())),
                Box::new(Ty::Var(a.clone())),
            )),
            Box::new(Ty::Var(b.clone())),
        );

        let left_id = DefId::new(BUILTIN_LEFT_ID);
        let right_id = DefId::new(BUILTIN_RIGHT_ID);
        env.register_data_con(
            left_id,
            Symbol::intern("Left"),
            Scheme::poly(
                vec![a.clone(), b.clone()],
                Ty::fun(Ty::Var(a.clone()), either_ab.clone()),
            ),
        );
        env.register_data_con(
            right_id,
            Symbol::intern("Right"),
            Scheme::poly(
                vec![a, b.clone()],
                Ty::fun(Ty::Var(b), either_ab),
            ),
        );

        // Unit constructor
        // () :: ()
        let unit_id = DefId::new(BUILTIN_UNIT_ID);
        env.register_data_con(unit_id, Symbol::intern("()"), Scheme::mono(Ty::unit()));
    }

    /// Create a list type `[a]`.
    #[must_use]
    #[allow(dead_code)]
    pub fn list_of(elem: Ty) -> Ty {
        Ty::List(Box::new(elem))
    }

    /// Create a Maybe type `Maybe a`.
    #[must_use]
    #[allow(dead_code)]
    pub fn maybe_of(&self, elem: Ty) -> Ty {
        Ty::App(Box::new(Ty::Con(self.maybe_con.clone())), Box::new(elem))
    }

    /// Create an IO type `IO a`.
    #[must_use]
    #[allow(dead_code)]
    pub fn io_of(&self, elem: Ty) -> Ty {
        Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(elem))
    }

    /// Create an Either type `Either a b`.
    #[must_use]
    #[allow(dead_code)]
    pub fn either_of(&self, left: Ty, right: Ty) -> Ty {
        Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Con(self.either_con.clone())),
                Box::new(left),
            )),
            Box::new(right),
        )
    }

    /// Create a Tensor type `Tensor shape elem`.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape as a type-level list (e.g., `TyList::shape_from_dims(&[1024, 768])`)
    /// * `elem` - The element type (e.g., `float_ty`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use bhc_types::TyList;
    ///
    /// let builtins = Builtins::new();
    /// let shape = TyList::shape_from_dims(&[1024, 768]);
    /// let tensor_type = builtins.tensor_of(Ty::TyList(shape), builtins.float_ty.clone());
    /// // tensor_type represents: Tensor '[1024, 768] Float
    /// ```
    #[must_use]
    #[allow(dead_code)]
    pub fn tensor_of(&self, shape: Ty, elem: Ty) -> Ty {
        Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Con(self.tensor_con.clone())),
                Box::new(shape),
            )),
            Box::new(elem),
        )
    }

    /// Create a DynTensor type `DynTensor a`.
    ///
    /// # Arguments
    ///
    /// * `elem` - The element type (e.g., `float_ty`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builtins = Builtins::new();
    /// let dyn_float = builtins.dyn_tensor_of(builtins.float_ty.clone());
    /// // dyn_float represents: DynTensor Float
    /// ```
    #[must_use]
    #[allow(dead_code)]
    pub fn dyn_tensor_of(&self, elem: Ty) -> Ty {
        Ty::App(
            Box::new(Ty::Con(self.dyn_tensor_con.clone())),
            Box::new(elem),
        )
    }

    /// Create a ShapeWitness type `ShapeWitness shape`.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape as a type-level list
    #[must_use]
    #[allow(dead_code)]
    pub fn shape_witness_of(&self, shape: TyList) -> Ty {
        Ty::App(
            Box::new(Ty::Con(self.shape_witness_con.clone())),
            Box::new(Ty::TyList(shape)),
        )
    }

    /// Register dynamic tensor operations in the environment.
    ///
    /// This registers:
    /// - `toDynamic :: forall shape a. Tensor shape a -> DynTensor a`
    /// - `fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)`
    /// - `withDynShape :: forall a r. DynTensor a -> (forall shape. Tensor shape a -> r) -> r`
    /// - `dynShape :: forall a. DynTensor a -> [Int]`
    /// - `dynRank :: forall a. DynTensor a -> Int`
    /// - `MkShapeWitness :: forall shape. ShapeWitness shape` (data constructor)
    /// - `MkDynTensor :: forall shape a. Tensor shape a -> DynTensor a` (data constructor)
    pub fn register_dyn_tensor_ops(&self, env: &mut TypeEnv) {
        // Type variables for schemes
        let shape_var = TyVar::new(BUILTIN_TYVAR_SHAPE, Kind::nat_list());
        let a_var = TyVar::new_star(BUILTIN_TYVAR_A);
        let r_var = TyVar::new_star(BUILTIN_TYVAR_R);

        // Tensor shape a
        let tensor_shape_a = self.tensor_of(
            Ty::TyList(TyList::Var(shape_var.clone())),
            Ty::Var(a_var.clone()),
        );

        // DynTensor a
        let dyn_tensor_a = self.dyn_tensor_of(Ty::Var(a_var.clone()));

        // ShapeWitness shape
        let witness_shape = self.shape_witness_of(TyList::Var(shape_var.clone()));

        // Maybe (Tensor shape a)
        let maybe_tensor = self.maybe_of(tensor_shape_a.clone());

        // 1. toDynamic :: forall shape a. Tensor shape a -> DynTensor a
        let to_dynamic_ty = Ty::fun(tensor_shape_a.clone(), dyn_tensor_a.clone());
        let to_dynamic_scheme = Scheme::poly(
            vec![shape_var.clone(), a_var.clone()],
            to_dynamic_ty,
        );
        env.register_value(
            DefId::new(BUILTIN_TO_DYNAMIC_ID),
            Symbol::intern("toDynamic"),
            to_dynamic_scheme,
        );

        // 2. fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
        let from_dynamic_ty = Ty::fun(
            witness_shape.clone(),
            Ty::fun(dyn_tensor_a.clone(), maybe_tensor),
        );
        let from_dynamic_scheme = Scheme::poly(
            vec![shape_var.clone(), a_var.clone()],
            from_dynamic_ty,
        );
        env.register_value(
            DefId::new(BUILTIN_FROM_DYNAMIC_ID),
            Symbol::intern("fromDynamic"),
            from_dynamic_scheme,
        );

        // 3. dynShape :: forall a. DynTensor a -> [Int]
        let dyn_shape_ty = Ty::fun(dyn_tensor_a.clone(), Ty::List(Box::new(self.int_ty.clone())));
        let dyn_shape_scheme = Scheme::poly(vec![a_var.clone()], dyn_shape_ty);
        env.register_value(
            DefId::new(BUILTIN_DYN_SHAPE_ID),
            Symbol::intern("dynShape"),
            dyn_shape_scheme,
        );

        // 4. dynRank :: forall a. DynTensor a -> Int
        let dyn_rank_ty = Ty::fun(dyn_tensor_a.clone(), self.int_ty.clone());
        let dyn_rank_scheme = Scheme::poly(vec![a_var.clone()], dyn_rank_ty);
        env.register_value(
            DefId::new(BUILTIN_DYN_RANK_ID),
            Symbol::intern("dynRank"),
            dyn_rank_scheme,
        );

        // 5. withDynShape :: forall a r. DynTensor a -> (forall shape. Tensor shape a -> r) -> r
        // The continuation: forall shape. Tensor shape a -> r
        let inner_shape_var = TyVar::new(BUILTIN_TYVAR_SHAPE2, Kind::nat_list());
        let tensor_inner = self.tensor_of(
            Ty::TyList(TyList::Var(inner_shape_var.clone())),
            Ty::Var(a_var.clone()),
        );
        let continuation = Ty::Forall(
            vec![inner_shape_var],
            Box::new(Ty::fun(tensor_inner, Ty::Var(r_var.clone()))),
        );
        let with_dyn_shape_ty = Ty::fun(
            dyn_tensor_a,
            Ty::fun(continuation, Ty::Var(r_var.clone())),
        );
        let with_dyn_shape_scheme = Scheme::poly(
            vec![a_var.clone(), r_var],
            with_dyn_shape_ty,
        );
        env.register_value(
            DefId::new(BUILTIN_WITH_DYN_SHAPE_ID),
            Symbol::intern("withDynShape"),
            with_dyn_shape_scheme,
        );

        // 6. MkShapeWitness :: forall shape. ShapeWitness shape (data constructor)
        let mk_witness_scheme = Scheme::poly(vec![shape_var.clone()], witness_shape);
        env.register_data_con(
            DefId::new(BUILTIN_MK_SHAPE_WITNESS_ID),
            Symbol::intern("MkShapeWitness"),
            mk_witness_scheme,
        );

        // 7. MkDynTensor :: forall shape a. Tensor shape a -> DynTensor a (data constructor)
        let mk_dyn_tensor_ty = Ty::fun(tensor_shape_a, self.dyn_tensor_of(Ty::Var(a_var.clone())));
        let mk_dyn_tensor_scheme = Scheme::poly(
            vec![shape_var, a_var],
            mk_dyn_tensor_ty,
        );
        env.register_data_con(
            DefId::new(BUILTIN_MK_DYN_TENSOR_ID),
            Symbol::intern("MkDynTensor"),
            mk_dyn_tensor_scheme,
        );
    }
}

// Reserved DefId values for built-in constructors
// These are in a reserved range that won't conflict with user definitions
const BUILTIN_BASE: usize = 0xFFFF_0000;
const BUILTIN_TRUE_ID: usize = BUILTIN_BASE;
const BUILTIN_FALSE_ID: usize = BUILTIN_BASE + 1;
const BUILTIN_NOTHING_ID: usize = BUILTIN_BASE + 2;
const BUILTIN_JUST_ID: usize = BUILTIN_BASE + 3;
const BUILTIN_NIL_ID: usize = BUILTIN_BASE + 4;
const BUILTIN_CONS_ID: usize = BUILTIN_BASE + 5;
const BUILTIN_LEFT_ID: usize = BUILTIN_BASE + 6;
const BUILTIN_RIGHT_ID: usize = BUILTIN_BASE + 7;
const BUILTIN_UNIT_ID: usize = BUILTIN_BASE + 8;

// M9 Phase 5: Dynamic tensor operations
const BUILTIN_TO_DYNAMIC_ID: usize = BUILTIN_BASE + 9;
const BUILTIN_FROM_DYNAMIC_ID: usize = BUILTIN_BASE + 10;
const BUILTIN_DYN_SHAPE_ID: usize = BUILTIN_BASE + 11;
const BUILTIN_DYN_RANK_ID: usize = BUILTIN_BASE + 12;
const BUILTIN_WITH_DYN_SHAPE_ID: usize = BUILTIN_BASE + 13;
const BUILTIN_MK_SHAPE_WITNESS_ID: usize = BUILTIN_BASE + 14;
const BUILTIN_MK_DYN_TENSOR_ID: usize = BUILTIN_BASE + 15;

// Reserved TyVar IDs for built-in schemes
const BUILTIN_TYVAR_A: u32 = 0xFFFF_0000;
const BUILTIN_TYVAR_B: u32 = 0xFFFF_0001;
const BUILTIN_TYVAR_SHAPE: u32 = 0xFFFF_0002;
const BUILTIN_TYVAR_R: u32 = 0xFFFF_0003;
const BUILTIN_TYVAR_SHAPE2: u32 = 0xFFFF_0004;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtins_creation() {
        let builtins = Builtins::new();

        assert_eq!(builtins.int_con.name, Symbol::intern("Int"));
        assert!(builtins.int_con.kind.is_star());

        assert_eq!(builtins.maybe_con.name, Symbol::intern("Maybe"));
        assert!(!builtins.maybe_con.kind.is_star()); // * -> *
    }

    #[test]
    fn test_list_of() {
        let builtins = Builtins::new();
        let list_int = Builtins::list_of(builtins.int_ty.clone());

        match list_int {
            Ty::List(elem) => assert_eq!(*elem, builtins.int_ty),
            _ => panic!("expected list type"),
        }
    }

    #[test]
    fn test_register_data_cons() {
        let builtins = Builtins::new();
        let mut env = TypeEnv::new();
        builtins.register_data_cons(&mut env);

        // Check True is registered
        let true_info = env.lookup_data_con(Symbol::intern("True")).unwrap();
        assert_eq!(true_info.scheme.ty, builtins.bool_ty);

        // Check Just is registered with correct scheme
        let just_info = env.lookup_data_con(Symbol::intern("Just")).unwrap();
        assert!(!just_info.scheme.is_mono());
        assert_eq!(just_info.scheme.vars.len(), 1);
    }

    #[test]
    fn test_tensor_kind() {
        let builtins = Builtins::new();

        // Tensor has kind [Nat] -> * -> *
        assert_eq!(builtins.tensor_con.name, Symbol::intern("Tensor"));

        // Verify the kind structure
        match &builtins.tensor_con.kind {
            Kind::Arrow(arg, result) => {
                // First argument should be [Nat]
                match arg.as_ref() {
                    Kind::List(elem_kind) => {
                        assert!(matches!(elem_kind.as_ref(), Kind::Nat));
                    }
                    _ => panic!("expected [Nat] as first argument"),
                }
                // Result should be * -> *
                match result.as_ref() {
                    Kind::Arrow(elem, final_result) => {
                        assert!(elem.is_star());
                        assert!(final_result.is_star());
                    }
                    _ => panic!("expected * -> * as result"),
                }
            }
            _ => panic!("expected arrow kind"),
        }
    }

    #[test]
    fn test_tensor_of() {
        use bhc_types::TyList;

        let builtins = Builtins::new();

        // Create Tensor '[1024, 768] Float
        let shape = TyList::shape_from_dims(&[1024, 768]);
        let tensor_type = builtins.tensor_of(Ty::TyList(shape), builtins.float_ty.clone());

        // Verify structure
        match &tensor_type {
            Ty::App(f, elem) => {
                assert_eq!(**elem, builtins.float_ty);
                match f.as_ref() {
                    Ty::App(tensor, shape) => {
                        assert!(matches!(tensor.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("Tensor")));
                        assert!(matches!(shape.as_ref(), Ty::TyList(_)));
                    }
                    _ => panic!("expected Tensor applied to shape"),
                }
            }
            _ => panic!("expected application"),
        }
    }

    // === M9 Phase 5: DynTensor tests ===

    #[test]
    fn test_dyn_tensor_kind() {
        let builtins = Builtins::new();

        // DynTensor has kind * -> *
        assert_eq!(builtins.dyn_tensor_con.name, Symbol::intern("DynTensor"));
        assert_eq!(builtins.dyn_tensor_con.kind, Kind::star_to_star());
    }

    #[test]
    fn test_shape_witness_kind() {
        let builtins = Builtins::new();

        // ShapeWitness has kind [Nat] -> *
        assert_eq!(builtins.shape_witness_con.name, Symbol::intern("ShapeWitness"));
        match &builtins.shape_witness_con.kind {
            Kind::Arrow(from, to) => {
                assert_eq!(**from, Kind::nat_list());
                assert_eq!(**to, Kind::Star);
            }
            _ => panic!("expected arrow kind"),
        }
    }

    #[test]
    fn test_dyn_tensor_of() {
        let builtins = Builtins::new();

        // Create DynTensor Float
        let dyn_type = builtins.dyn_tensor_of(builtins.float_ty.clone());

        match &dyn_type {
            Ty::App(f, elem) => {
                assert!(matches!(f.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("DynTensor")));
                assert_eq!(**elem, builtins.float_ty);
            }
            _ => panic!("expected application"),
        }
    }

    #[test]
    fn test_shape_witness_of() {
        let builtins = Builtins::new();

        // Create ShapeWitness '[1024, 768]
        let shape = TyList::shape_from_dims(&[1024, 768]);
        let witness = builtins.shape_witness_of(shape);

        match &witness {
            Ty::App(f, shape_arg) => {
                assert!(matches!(f.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("ShapeWitness")));
                assert!(matches!(shape_arg.as_ref(), Ty::TyList(_)));
            }
            _ => panic!("expected application"),
        }
    }

    #[test]
    fn test_register_dyn_tensor_ops() {
        let builtins = Builtins::new();
        let mut env = TypeEnv::new();
        builtins.register_dyn_tensor_ops(&mut env);

        // Check toDynamic is registered
        let to_dyn_scheme = env.lookup_local(Symbol::intern("toDynamic"));
        assert!(to_dyn_scheme.is_some(), "toDynamic should be registered");
        let to_dyn = to_dyn_scheme.unwrap();
        assert!(!to_dyn.is_mono(), "toDynamic should be polymorphic");
        assert_eq!(to_dyn.vars.len(), 2); // shape, a

        // Check fromDynamic is registered
        let from_dyn_scheme = env.lookup_local(Symbol::intern("fromDynamic"));
        assert!(from_dyn_scheme.is_some(), "fromDynamic should be registered");
        let from_dyn = from_dyn_scheme.unwrap();
        assert_eq!(from_dyn.vars.len(), 2); // shape, a

        // Check dynShape is registered
        let dyn_shape_scheme = env.lookup_local(Symbol::intern("dynShape"));
        assert!(dyn_shape_scheme.is_some(), "dynShape should be registered");
        let dyn_shape = dyn_shape_scheme.unwrap();
        assert_eq!(dyn_shape.vars.len(), 1); // a

        // Check dynRank is registered
        let dyn_rank_scheme = env.lookup_local(Symbol::intern("dynRank"));
        assert!(dyn_rank_scheme.is_some(), "dynRank should be registered");

        // Check withDynShape is registered
        let with_dyn_scheme = env.lookup_local(Symbol::intern("withDynShape"));
        assert!(with_dyn_scheme.is_some(), "withDynShape should be registered");

        // Check MkDynTensor data constructor is registered
        let mk_dyn = env.lookup_data_con(Symbol::intern("MkDynTensor"));
        assert!(mk_dyn.is_some(), "MkDynTensor should be registered");

        // Check MkShapeWitness data constructor is registered
        let mk_witness = env.lookup_data_con(Symbol::intern("MkShapeWitness"));
        assert!(mk_witness.is_some(), "MkShapeWitness should be registered");
    }

    #[test]
    fn test_to_dynamic_type_structure() {
        let builtins = Builtins::new();
        let mut env = TypeEnv::new();
        builtins.register_dyn_tensor_ops(&mut env);

        // toDynamic :: forall shape a. Tensor shape a -> DynTensor a
        let scheme = env.lookup_local(Symbol::intern("toDynamic")).unwrap();

        // Should have 2 type variables: shape and a
        assert_eq!(scheme.vars.len(), 2);
        // First var should have kind [Nat] (shape)
        assert_eq!(scheme.vars[0].kind, Kind::nat_list());
        // Second var should have kind * (a)
        assert_eq!(scheme.vars[1].kind, Kind::Star);

        // Body should be a function type
        assert!(scheme.ty.is_fun());
    }

    #[test]
    fn test_from_dynamic_type_structure() {
        let builtins = Builtins::new();
        let mut env = TypeEnv::new();
        builtins.register_dyn_tensor_ops(&mut env);

        // fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
        let scheme = env.lookup_local(Symbol::intern("fromDynamic")).unwrap();

        // Should have 2 type variables
        assert_eq!(scheme.vars.len(), 2);

        // Body should be a function type (ShapeWitness shape -> ...)
        assert!(scheme.ty.is_fun());
    }
}
