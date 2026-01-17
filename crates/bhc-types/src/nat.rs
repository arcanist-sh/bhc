//! Type-level natural numbers.
//!
//! This module implements type-level natural numbers for shape-indexed tensors,
//! enabling compile-time dimension checking per H26-SPEC Section 7.
//!
//! ## Overview
//!
//! Type-level naturals are used to express tensor shapes at the type level:
//!
//! ```text
//! matmul :: Tensor '[m, k] Float -> Tensor '[k, n] Float -> Tensor '[m, n] Float
//! ```
//!
//! The `k` dimensions must match, which is enforced at compile time.
//!
//! ## Representation
//!
//! - `Lit(n)` - A concrete natural number
//! - `Var(v)` - A polymorphic dimension variable
//! - `Add(a, b)` - Sum of two naturals (m + k)
//! - `Mul(a, b)` - Product of two naturals (m * k)

use serde::{Deserialize, Serialize};

use crate::TyVar;

/// A type-level natural number.
///
/// Represents dimensions in shape-indexed tensor types. Supports both
/// concrete values (for static shapes) and variables (for polymorphic
/// operations like `matmul`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TyNat {
    /// A concrete natural number literal.
    ///
    /// Example: `1024` in `Tensor '[1024, 768] Float`
    Lit(u64),

    /// A type-level natural variable.
    ///
    /// Example: `n` in `Tensor '[n] Float`
    Var(TyVar),

    /// Addition of two naturals.
    ///
    /// Example: `m + k` for computing output dimensions
    Add(Box<TyNat>, Box<TyNat>),

    /// Multiplication of two naturals.
    ///
    /// Example: `m * k` for computing buffer sizes
    Mul(Box<TyNat>, Box<TyNat>),
}

impl TyNat {
    /// Creates a literal type-level natural.
    #[must_use]
    pub fn lit(n: u64) -> Self {
        Self::Lit(n)
    }

    /// Creates a zero type-level natural.
    #[must_use]
    pub fn zero() -> Self {
        Self::Lit(0)
    }

    /// Creates a one type-level natural.
    #[must_use]
    pub fn one() -> Self {
        Self::Lit(1)
    }

    /// Creates an addition of two naturals.
    #[must_use]
    pub fn add(a: TyNat, b: TyNat) -> Self {
        // Simplify if both are literals
        match (&a, &b) {
            (TyNat::Lit(x), TyNat::Lit(y)) => TyNat::Lit(x + y),
            _ => Self::Add(Box::new(a), Box::new(b)),
        }
    }

    /// Creates a multiplication of two naturals.
    #[must_use]
    pub fn mul(a: TyNat, b: TyNat) -> Self {
        // Simplify if both are literals
        match (&a, &b) {
            (TyNat::Lit(x), TyNat::Lit(y)) => TyNat::Lit(x * y),
            _ => Self::Mul(Box::new(a), Box::new(b)),
        }
    }

    /// Returns true if this is a concrete literal.
    #[must_use]
    pub fn is_lit(&self) -> bool {
        matches!(self, Self::Lit(_))
    }

    /// Returns the literal value if this is a concrete literal.
    #[must_use]
    pub fn as_lit(&self) -> Option<u64> {
        match self {
            Self::Lit(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns true if this natural contains no variables.
    #[must_use]
    pub fn is_ground(&self) -> bool {
        match self {
            Self::Lit(_) => true,
            Self::Var(_) => false,
            Self::Add(a, b) | Self::Mul(a, b) => a.is_ground() && b.is_ground(),
        }
    }

    /// Evaluates this natural if it is ground (contains no variables).
    ///
    /// Returns `None` if the natural contains variables.
    #[must_use]
    pub fn eval(&self) -> Option<u64> {
        match self {
            Self::Lit(n) => Some(*n),
            Self::Var(_) => None,
            Self::Add(a, b) => Some(a.eval()? + b.eval()?),
            Self::Mul(a, b) => Some(a.eval()? * b.eval()?),
        }
    }

    /// Collects all type variables occurring in this natural.
    #[must_use]
    pub fn free_vars(&self) -> Vec<TyVar> {
        let mut vars = Vec::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, vars: &mut Vec<TyVar>) {
        match self {
            Self::Lit(_) => {}
            Self::Var(v) => {
                if !vars.contains(v) {
                    vars.push(v.clone());
                }
            }
            Self::Add(a, b) | Self::Mul(a, b) => {
                a.collect_free_vars(vars);
                b.collect_free_vars(vars);
            }
        }
    }

    /// Substitutes a type variable with a natural in this expression.
    #[must_use]
    pub fn subst(&self, var: &TyVar, replacement: &TyNat) -> Self {
        match self {
            Self::Lit(n) => Self::Lit(*n),
            Self::Var(v) if v.id == var.id => replacement.clone(),
            Self::Var(v) => Self::Var(v.clone()),
            Self::Add(a, b) => Self::add(a.subst(var, replacement), b.subst(var, replacement)),
            Self::Mul(a, b) => Self::mul(a.subst(var, replacement), b.subst(var, replacement)),
        }
    }
}

impl std::fmt::Display for TyNat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lit(n) => write!(f, "{n}"),
            Self::Var(v) => write!(f, "n{}", v.id),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Kind;

    #[test]
    fn test_literal_naturals() {
        let n = TyNat::lit(42);
        assert!(n.is_lit());
        assert!(n.is_ground());
        assert_eq!(n.as_lit(), Some(42));
        assert_eq!(n.eval(), Some(42));
    }

    #[test]
    fn test_variable_naturals() {
        let v = TyVar::new(0, Kind::Nat);
        let n = TyNat::Var(v);
        assert!(!n.is_lit());
        assert!(!n.is_ground());
        assert_eq!(n.as_lit(), None);
        assert_eq!(n.eval(), None);
    }

    #[test]
    fn test_add_literals() {
        let a = TyNat::lit(10);
        let b = TyNat::lit(20);
        let sum = TyNat::add(a, b);
        // Should simplify to Lit(30)
        assert_eq!(sum, TyNat::lit(30));
    }

    #[test]
    fn test_add_with_variable() {
        let v = TyVar::new(0, Kind::Nat);
        let a = TyNat::Var(v.clone());
        let b = TyNat::lit(10);
        let sum = TyNat::add(a, b);
        // Should not simplify
        assert!(!sum.is_lit());
        assert!(!sum.is_ground());
    }

    #[test]
    fn test_mul_literals() {
        let a = TyNat::lit(3);
        let b = TyNat::lit(4);
        let prod = TyNat::mul(a, b);
        assert_eq!(prod, TyNat::lit(12));
    }

    #[test]
    fn test_free_vars() {
        let v1 = TyVar::new(0, Kind::Nat);
        let v2 = TyVar::new(1, Kind::Nat);
        let n = TyNat::add(TyNat::Var(v1.clone()), TyNat::Var(v2.clone()));
        let vars = n.free_vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&v1));
        assert!(vars.contains(&v2));
    }

    #[test]
    fn test_substitution() {
        let v = TyVar::new(0, Kind::Nat);
        let n = TyNat::add(TyNat::Var(v.clone()), TyNat::lit(5));
        let result = n.subst(&v, &TyNat::lit(10));
        assert_eq!(result.eval(), Some(15));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TyNat::lit(42)), "42");
        let v = TyVar::new(0, Kind::Nat);
        assert_eq!(format!("{}", TyNat::Var(v)), "n0");
    }
}
