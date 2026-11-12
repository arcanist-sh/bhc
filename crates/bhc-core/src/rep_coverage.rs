//! How many of a module's binders have a known representation.
//!
//! A typed Core (#95) and a uniform value representation (#94) both need
//! every binder to say what it holds. [`crate::Var::checked_ty`] is where
//! the checker's type is recorded; this counts how far that has got, so the
//! number can be watched instead of guessed.

use crate::{Bind, CoreModule, Expr, Rep, Var};

/// Counts of a module's binders by what is known of their representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RepCoverage {
    /// Binders whose type is not known.
    pub unknown: usize,
    /// Binders known to hold, or possibly hold, a heap object.
    pub lifted: usize,
    /// Binders known to hold a scalar.
    pub scalar: usize,
}

impl RepCoverage {
    /// Every binder counted.
    #[must_use]
    pub fn total(&self) -> usize {
        self.unknown + self.lifted + self.scalar
    }

    /// The share of binders whose representation is known, in percent.
    #[must_use]
    pub fn known_percent(&self) -> f64 {
        match self.total() {
            0 => 100.0,
            total => 100.0 * (self.lifted + self.scalar) as f64 / total as f64,
        }
    }

    fn count(&mut self, var: &Var) {
        match var.rep() {
            Rep::Unknown => self.unknown += 1,
            Rep::Lifted => self.lifted += 1,
            rep if rep.is_scalar() => self.scalar += 1,
            _ => self.unknown += 1,
        }
    }

    fn bind(&mut self, bind: &Bind) {
        match bind {
            Bind::NonRec(var, rhs) => {
                self.count(var);
                self.expr(rhs);
            }
            Bind::Rec(pairs) => {
                for (var, rhs) in pairs {
                    self.count(var);
                    self.expr(rhs);
                }
            }
        }
    }

    fn expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Var(..) | Expr::Lit(..) | Expr::Type(..) | Expr::Coercion(..) => {}
            Expr::App(f, a, _) => {
                self.expr(f);
                self.expr(a);
            }
            Expr::Lam(param, body, _) => {
                self.count(param);
                self.expr(body);
            }
            Expr::TyApp(inner, _, _)
            | Expr::TyLam(_, inner, _)
            | Expr::Lazy(inner, _)
            | Expr::Cast(inner, _, _)
            | Expr::Tick(_, inner, _) => self.expr(inner),
            Expr::Let(bind, body, _) => {
                self.bind(bind);
                self.expr(body);
            }
            Expr::Case(scrutinee, alts, _, _) => {
                self.expr(scrutinee);
                for alt in alts {
                    for binder in &alt.binders {
                        self.count(binder);
                    }
                    self.expr(&alt.rhs);
                }
            }
        }
    }
}

/// Count the binders of `module`: top-level and local bindings, lambda
/// parameters and the variables case alternatives bind.
#[must_use]
pub fn binder_rep_coverage(module: &CoreModule) -> RepCoverage {
    let mut coverage = RepCoverage::default();
    for bind in &module.bindings {
        coverage.bind(bind);
    }
    coverage
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VarId;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::{Kind, Ty, TyCon};

    fn var(name: &str, checked: Ty) -> Var {
        Var::new(Symbol::intern(name), VarId::new(1), Ty::Error).with_checked_ty(checked)
    }

    fn con(name: &str) -> Ty {
        Ty::Con(TyCon::new(Symbol::intern(name), Kind::Star))
    }

    #[test]
    fn counts_each_kind_of_binder() {
        let mut coverage = RepCoverage::default();
        let body = Expr::Var(var("x", Ty::Error), Span::default());
        // \(x :: Int) -> let (y :: [Int]) = x in x, with an untyped binding.
        let inner = Expr::Let(
            Box::new(Bind::NonRec(
                var("y", Ty::List(Box::new(con("Int")))),
                Box::new(body.clone()),
            )),
            Box::new(body),
            Span::default(),
        );
        let lam = Expr::Lam(var("x", con("Int")), Box::new(inner), Span::default());
        coverage.bind(&Bind::NonRec(var("f", Ty::Error), Box::new(lam)));
        assert_eq!(
            coverage,
            RepCoverage {
                unknown: 1,
                lifted: 1,
                scalar: 1
            }
        );
        assert_eq!(coverage.total(), 3);
        assert!((coverage.known_percent() - 200.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn representation_follows_the_checked_type() {
        assert_eq!(var("a", Ty::Error).rep(), Rep::Unknown);
        assert_eq!(var("a", con("Int")).rep(), Rep::Int);
        assert_eq!(var("a", con("Word8")).rep(), Rep::Int);
        assert_eq!(var("a", con("Char")).rep(), Rep::Char);
        assert_eq!(var("a", con("Double")).rep(), Rep::Double);
        assert_eq!(var("a", con("Float")).rep(), Rep::Float);
        assert_eq!(var("a", con("Bool")).rep(), Rep::Lifted);
        assert_eq!(var("a", con("Integer")).rep(), Rep::Lifted);
        assert_eq!(
            var("a", Ty::Fun(Box::new(con("Int")), Box::new(con("Int")))).rep(),
            Rep::Lifted
        );
        assert!(Rep::Double.is_scalar() && !Rep::Lifted.is_scalar() && !Rep::Unknown.is_scalar());
    }

    #[test]
    fn checked_type_is_not_part_of_identity() {
        assert_eq!(var("a", con("Int")), var("a", Ty::Error));
    }
}
