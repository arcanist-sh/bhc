//! How many of a module's binders have a known representation.
//!
//! A typed Core (#95) and a uniform value representation (#94) both need
//! every binder to say what it holds. [`crate::Var::checked_ty`] is where
//! the checker's type is recorded; this counts how far that has got, so the
//! number can be watched instead of guessed.

use crate::{Bind, CoreModule, Expr, Rep, TypeExpansions, Var};

/// Counts of a module's binders by what is known of their representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RepCoverage {
    /// Binders whose type is not known.
    pub unknown: usize,
    /// Binders at a type variable, whose word may be a pointer or a scalar.
    pub polymorphic: usize,
    /// Binders known to hold a heap object.
    pub lifted: usize,
    /// Binders known to hold a scalar.
    pub scalar: usize,
}

impl RepCoverage {
    /// Every binder counted.
    #[must_use]
    pub fn total(&self) -> usize {
        self.unknown + self.polymorphic + self.lifted + self.scalar
    }

    /// The share of binders whose type is known, in percent. A binder at a
    /// type variable counts: its type is known, and is what says it needs
    /// the uniform representation.
    #[must_use]
    pub fn known_percent(&self) -> f64 {
        match self.total() {
            0 => 100.0,
            total => 100.0 * (total - self.unknown) as f64 / total as f64,
        }
    }

    fn count(&mut self, var: &Var, expansions: &TypeExpansions) {
        match var.rep_in(expansions) {
            Rep::Unknown => self.unknown += 1,
            Rep::Polymorphic => self.polymorphic += 1,
            Rep::Lifted => self.lifted += 1,
            Rep::Int | Rep::Char | Rep::Double | Rep::Float => self.scalar += 1,
        }
    }

    fn bind(&mut self, bind: &Bind, expansions: &TypeExpansions) {
        match bind {
            Bind::NonRec(var, rhs) => {
                self.count(var, expansions);
                self.expr(rhs, expansions);
            }
            Bind::Rec(pairs) => {
                for (var, rhs) in pairs {
                    self.count(var, expansions);
                    self.expr(rhs, expansions);
                }
            }
        }
    }

    fn expr(&mut self, expr: &Expr, expansions: &TypeExpansions) {
        match expr {
            Expr::Var(..) | Expr::Lit(..) | Expr::Type(..) | Expr::Coercion(..) => {}
            Expr::App(f, a, _) => {
                self.expr(f, expansions);
                self.expr(a, expansions);
            }
            Expr::Lam(param, body, _) => {
                self.count(param, expansions);
                self.expr(body, expansions);
            }
            Expr::TyApp(inner, _, _)
            | Expr::TyLam(_, inner, _)
            | Expr::Lazy(inner, _)
            | Expr::Cast(inner, _, _)
            | Expr::Tick(_, inner, _) => self.expr(inner, expansions),
            Expr::Let(bind, body, _) => {
                self.bind(bind, expansions);
                self.expr(body, expansions);
            }
            Expr::Case(scrutinee, alts, _, _) => {
                self.expr(scrutinee, expansions);
                for alt in alts {
                    for binder in &alt.binders {
                        self.count(binder, expansions);
                    }
                    self.expr(&alt.rhs, expansions);
                }
            }
        }
    }
}

/// The binders of `module` whose representation is unknown, grouped by name
/// with any trailing counter removed (`arg0_200043` counts under `arg0`),
/// most frequent first. Says where the untyped binders come from.
#[must_use]
pub fn unknown_binder_names(module: &CoreModule) -> Vec<(String, usize)> {
    fn stem(name: &str) -> String {
        let trimmed = name.trim_end_matches(|c: char| c.is_ascii_digit());
        match trimmed.strip_suffix('_') {
            Some(stem) if trimmed.len() < name.len() && !stem.is_empty() => stem.to_string(),
            _ => name.to_string(),
        }
    }
    fn bind(b: &Bind, out: &mut Vec<String>) {
        match b {
            Bind::NonRec(var, rhs) => {
                note(var, out);
                expr(rhs, out);
            }
            Bind::Rec(pairs) => {
                for (var, rhs) in pairs {
                    note(var, out);
                    expr(rhs, out);
                }
            }
        }
    }
    fn note(var: &Var, out: &mut Vec<String>) {
        if var.rep() == Rep::Unknown {
            out.push(stem(var.name.as_str()));
        }
    }
    fn expr(e: &Expr, out: &mut Vec<String>) {
        match e {
            Expr::Var(..) | Expr::Lit(..) | Expr::Type(..) | Expr::Coercion(..) => {}
            Expr::App(f, a, _) => {
                expr(f, out);
                expr(a, out);
            }
            Expr::Lam(param, body, _) => {
                note(param, out);
                expr(body, out);
            }
            Expr::TyApp(inner, _, _)
            | Expr::TyLam(_, inner, _)
            | Expr::Lazy(inner, _)
            | Expr::Cast(inner, _, _)
            | Expr::Tick(_, inner, _) => expr(inner, out),
            Expr::Let(b, body, _) => {
                bind(b, out);
                expr(body, out);
            }
            Expr::Case(scrutinee, alts, _, _) => {
                expr(scrutinee, out);
                for alt in alts {
                    for binder in &alt.binders {
                        note(binder, out);
                    }
                    expr(&alt.rhs, out);
                }
            }
        }
    }
    let mut names = Vec::new();
    for b in &module.bindings {
        bind(b, &mut names);
    }
    names.sort();
    let mut counts: Vec<(String, usize)> = Vec::new();
    for name in names {
        match counts.last_mut() {
            Some((last, n)) if *last == name => *n += 1,
            _ => counts.push((name, 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts
}

/// Count the binders of `module`: top-level and local bindings, lambda
/// parameters and the variables case alternatives bind. `expansions` says
/// what the module's newtypes and synonyms stand for, so a binder of a
/// newtype over a scalar counts as the scalar it is.
#[must_use]
pub fn binder_rep_coverage(module: &CoreModule, expansions: &TypeExpansions) -> RepCoverage {
    let mut coverage = RepCoverage::default();
    for bind in &module.bindings {
        coverage.bind(bind, expansions);
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
        coverage.bind(
            &Bind::NonRec(var("f", Ty::Error), Box::new(lam)),
            &TypeExpansions::default(),
        );
        assert_eq!(
            coverage,
            RepCoverage {
                unknown: 1,
                polymorphic: 0,
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
        assert_eq!(
            var("a", Ty::Var(bhc_types::TyVar::new(3, Kind::Star))).rep(),
            Rep::Polymorphic
        );
        assert!(Rep::Double.is_scalar() && !Rep::Lifted.is_scalar() && !Rep::Unknown.is_scalar());
        assert!(!Rep::Polymorphic.is_scalar());
    }

    #[test]
    fn unknown_binders_are_grouped_by_name_stem() {
        let body = Expr::Var(var("x", Ty::Error), Span::default());
        let lam = |name: &str, checked: Ty, body: Expr| {
            Expr::Lam(var(name, checked), Box::new(body), Span::default())
        };
        let rhs = lam(
            "arg0_12",
            Ty::Error,
            lam("arg0_13", Ty::Error, lam("n", con("Int"), body)),
        );
        let module = CoreModule {
            name: Symbol::intern("M"),
            bindings: vec![Bind::NonRec(var("f", Ty::Error), Box::new(rhs))],
            exports: vec![],
            foreign_imports: vec![],
            overloaded_strings: false,
            constructors: vec![],
        };
        assert_eq!(
            unknown_binder_names(&module),
            vec![("arg0".to_string(), 2), ("f".to_string(), 1)]
        );
    }

    #[test]
    fn fresh_variables_take_a_given_type_or_a_dictionary_type() {
        use crate::{dictionary_ty, fresh_checked_ty, is_dictionary_name};
        assert_eq!(fresh_checked_ty("x", &con("Int")), con("Int"));
        assert_eq!(fresh_checked_ty("x", &Ty::Error), Ty::Error);
        assert_eq!(
            fresh_checked_ty("$dictFunctor_IO", &Ty::Error),
            dictionary_ty()
        );
        assert_eq!(fresh_checked_ty("$dMonad", &Ty::Error), dictionary_ty());
        assert!(!is_dictionary_name("$derived_eq_T"));
        assert!(!is_dictionary_name("$d"));
        assert_eq!(Rep::of_ty(&dictionary_ty()), Rep::Lifted);
    }

    #[test]
    fn representation_sees_through_newtypes() {
        use crate::TypeExpansions;
        use bhc_types::TyVar;
        let a = TyVar::new(9, Kind::Star);
        let mut table = TypeExpansions::default();
        // newtype Age = Age Int
        table.insert(Symbol::intern("Age"), (vec![], con("Int")));
        // newtype Wrap a = Wrap a
        table.insert(
            Symbol::intern("Wrap"),
            (vec![a.clone()], Ty::Var(a.clone())),
        );
        // newtype Name = Name String, through a synonym
        table.insert(
            Symbol::intern("Str"),
            (vec![], Ty::List(Box::new(con("Char")))),
        );
        table.insert(Symbol::intern("Name"), (vec![], con("Str")));
        // newtype Fix f = Fix (f (Fix f))
        let f = TyVar::new(10, Kind::Star);
        let fix_f = Ty::App(Box::new(con("Fix")), Box::new(Ty::Var(f.clone())));
        table.insert(
            Symbol::intern("Fix"),
            (
                vec![f.clone()],
                Ty::App(Box::new(Ty::Var(f)), Box::new(fix_f)),
            ),
        );
        // newtype Loop = Loop Loop
        table.insert(Symbol::intern("Loop"), (vec![], con("Loop")));

        let wrap = |t: Ty| Ty::App(Box::new(con("Wrap")), Box::new(t));
        assert_eq!(Rep::of_ty(&con("Age")), Rep::Lifted);
        assert_eq!(Rep::of_ty_in(&con("Age"), &table), Rep::Int);
        assert_eq!(Rep::of_ty_in(&wrap(con("Double")), &table), Rep::Double);
        assert_eq!(Rep::of_ty_in(&wrap(wrap(con("Char"))), &table), Rep::Char);
        assert_eq!(Rep::of_ty_in(&wrap(Ty::Var(a)), &table), Rep::Polymorphic);
        assert_eq!(Rep::of_ty_in(&con("Name"), &table), Rep::Lifted);
        assert_eq!(
            Rep::of_ty_in(
                &Ty::App(Box::new(con("Fix")), Box::new(con("Maybe"))),
                &table
            ),
            Rep::Lifted
        );
        // Partly applied, or unknown: unchanged.
        assert_eq!(Rep::of_ty_in(&con("Wrap"), &table), Rep::Lifted);
        assert_eq!(Rep::of_ty_in(&con("Bool"), &table), Rep::Lifted);
        // Terminates.
        assert_eq!(Rep::of_ty_in(&con("Loop"), &table), Rep::Lifted);
    }

    #[test]
    fn checked_type_is_not_part_of_identity() {
        assert_eq!(var("a", con("Int")), var("a", Ty::Error));
    }
}
