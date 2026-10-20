//! Witness dictionaries for the builtin value monads, `Maybe` and lists.
//!
//! A `Monad m =>` (or `MonadPlus m =>`) function takes its monad as a
//! dictionary (see `LowerContext::constraint_is_dict_passed`). The builtin
//! value monads have no instance declarations to build one from, so a call at
//! `Maybe` or a list passed nothing: `inc :: Monad m => Int -> m Int` returned
//! garbage at `Maybe`, and pandoc's `safeRead :: (MonadPlus m, Read a) => …`
//! could not fail. These dictionaries are written out in Core here, in the
//! layout `dictionary::select_method` reads: superclass dictionaries first,
//! then the methods in class-registry order.

use bhc_core::{self as core, Alt, AltCon, DataCon, Var, VarId};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Kind, Ty, TyCon};

use crate::context::LowerContext;

/// The builtin value monad a constraint's type names.
#[derive(Clone, Copy)]
enum ValueMonad {
    Maybe,
    List,
}

fn value_monad_of(ty: &Ty) -> Option<ValueMonad> {
    match ty {
        Ty::List(_) => Some(ValueMonad::List),
        Ty::App(f, _) => value_monad_of(f),
        Ty::Con(tc) => match tc.name.as_str() {
            "Maybe" => Some(ValueMonad::Maybe),
            "[]" | "List" => Some(ValueMonad::List),
            _ => None,
        },
        _ => None,
    }
}

/// The dictionary for `constraint` when it is one of the monad-family
/// classes at `Maybe` or a list, else `None`.
pub(crate) fn value_monad_dict(
    ctx: &mut LowerContext,
    constraint: &Constraint,
    span: Span,
) -> Option<core::Expr> {
    let monad = value_monad_of(constraint.args.first()?)?;
    let mut b = Builder { ctx, span, monad };
    match constraint.class.as_str() {
        "Functor" => Some(b.functor()),
        "Applicative" => Some(b.applicative()),
        "Monad" => Some(b.monad()),
        "Alternative" => Some(b.alternative()),
        "MonadPlus" => Some(b.monad_plus()),
        _ => None,
    }
}

struct Builder<'a> {
    ctx: &'a mut LowerContext,
    span: Span,
    monad: ValueMonad,
}

impl Builder<'_> {
    fn named(&self, name: &str) -> core::Expr {
        core::Expr::Var(
            Var {
                name: Symbol::intern(name),
                id: VarId::new(0),
                ty: Ty::Error,
            },
            self.span,
        )
    }

    fn fresh(&mut self, base: &str) -> Var {
        self.ctx.fresh_var(base, Ty::Error, self.span)
    }

    fn var(&self, v: &Var) -> core::Expr {
        core::Expr::Var(v.clone(), self.span)
    }

    fn app(&self, f: core::Expr, x: core::Expr) -> core::Expr {
        core::Expr::App(Box::new(f), Box::new(x), self.span)
    }

    fn app2(&self, f: core::Expr, x: core::Expr, y: core::Expr) -> core::Expr {
        self.app(self.app(f, x), y)
    }

    fn lam(&self, v: Var, body: core::Expr) -> core::Expr {
        core::Expr::Lam(v, Box::new(body), self.span)
    }

    fn tuple(&self, fields: Vec<core::Expr>) -> core::Expr {
        let name = format!("({})", ",".repeat(fields.len() - 1));
        fields
            .into_iter()
            .fold(self.named(&name), |acc, f| self.app(acc, f))
    }

    fn nothing(&self) -> core::Expr {
        self.named("Nothing")
    }

    fn just(&self, x: core::Expr) -> core::Expr {
        self.app(self.named("Just"), x)
    }

    fn nil(&self) -> core::Expr {
        self.named("[]")
    }

    fn singleton(&self, x: core::Expr) -> core::Expr {
        self.app2(self.named(":"), x, self.nil())
    }

    /// `case m of { Nothing -> on_nothing; Just x -> on_just x }`.
    fn case_maybe(
        &mut self,
        m: core::Expr,
        on_nothing: core::Expr,
        on_just: impl FnOnce(&mut Self, &Var) -> core::Expr,
    ) -> core::Expr {
        let maybe = TyCon::new(Symbol::intern("Maybe"), Kind::Star);
        let x = self.fresh("x");
        let just_rhs = on_just(self, &x);
        let alts = vec![
            Alt {
                con: AltCon::DataCon(DataCon {
                    name: Symbol::intern("Nothing"),
                    ty_con: maybe.clone(),
                    tag: 0,
                    arity: 0,
                }),
                binders: vec![],
                rhs: on_nothing,
            },
            Alt {
                con: AltCon::DataCon(DataCon {
                    name: Symbol::intern("Just"),
                    ty_con: maybe,
                    tag: 1,
                    arity: 1,
                }),
                binders: vec![x],
                rhs: just_rhs,
            },
        ];
        core::Expr::Case(Box::new(m), alts, Ty::Error, self.span)
    }

    /// `\a b -> body a b`.
    fn lam2(&mut self, body: impl FnOnce(&mut Self, &Var, &Var) -> core::Expr) -> core::Expr {
        let a = self.fresh("a");
        let b = self.fresh("b");
        let inner = body(self, &a, &b);
        self.lam(a, self.lam(b, inner))
    }

    /// `concatMap f xs`.
    fn concat_map(&self, f: core::Expr, xs: core::Expr) -> core::Expr {
        self.app2(self.named("concatMap"), f, xs)
    }

    fn fmap(&mut self) -> core::Expr {
        match self.monad {
            ValueMonad::Maybe => self.lam2(|b, f, m| {
                let nothing = b.nothing();
                b.case_maybe(b.var(m), nothing, |b, x| b.just(b.app(b.var(f), b.var(x))))
            }),
            ValueMonad::List => self.named("map"),
        }
    }

    fn replace(&mut self) -> core::Expr {
        match self.monad {
            ValueMonad::Maybe => self.lam2(|b, y, m| {
                let nothing = b.nothing();
                b.case_maybe(b.var(m), nothing, |b, _| b.just(b.var(y)))
            }),
            ValueMonad::List => self.lam2(|b, y, xs| {
                let ignored = b.fresh("_");
                let k = b.lam(ignored, b.var(y));
                b.app2(b.named("map"), k, b.var(xs))
            }),
        }
    }

    fn pure(&mut self) -> core::Expr {
        let x = self.fresh("x");
        let body = match self.monad {
            ValueMonad::Maybe => self.just(self.var(&x)),
            ValueMonad::List => self.singleton(self.var(&x)),
        };
        self.lam(x, body)
    }

    fn ap(&mut self) -> core::Expr {
        match self.monad {
            ValueMonad::Maybe => self.lam2(|b, mf, mx| {
                let nothing = b.nothing();
                b.case_maybe(b.var(mf), nothing, |b, f| {
                    let nothing = b.nothing();
                    b.case_maybe(b.var(mx), nothing, |b, x| b.just(b.app(b.var(f), b.var(x))))
                })
            }),
            ValueMonad::List => self.lam2(|b, fs, xs| {
                let f = b.fresh("f");
                let inner = b.app2(b.named("map"), b.var(&f), b.var(xs));
                let k = b.lam(f, inner);
                b.concat_map(k, b.var(fs))
            }),
        }
    }

    /// `*>` (and `>>`): run the first, keep the second.
    fn then(&mut self) -> core::Expr {
        match self.monad {
            ValueMonad::Maybe => self.lam2(|b, x, y| {
                let nothing = b.nothing();
                b.case_maybe(b.var(x), nothing, |b, _| b.var(y))
            }),
            ValueMonad::List => self.lam2(|b, xs, ys| {
                let ignored = b.fresh("_");
                let k = b.lam(ignored, b.var(ys));
                b.concat_map(k, b.var(xs))
            }),
        }
    }

    /// `<*`: run both, keep the first.
    fn before(&mut self) -> core::Expr {
        match self.monad {
            ValueMonad::Maybe => self.lam2(|b, x, y| {
                let nothing = b.nothing();
                b.case_maybe(b.var(x), nothing, |b, v| {
                    let nothing = b.nothing();
                    b.case_maybe(b.var(y), nothing, |b, _| b.just(b.var(v)))
                })
            }),
            ValueMonad::List => self.lam2(|b, xs, ys| {
                let v = b.fresh("v");
                let ignored = b.fresh("_");
                let keep = b.lam(ignored, b.singleton(b.var(&v)));
                let inner = b.concat_map(keep, b.var(ys));
                let k = b.lam(v, inner);
                b.concat_map(k, b.var(xs))
            }),
        }
    }

    fn bind(&mut self) -> core::Expr {
        match self.monad {
            ValueMonad::Maybe => self.lam2(|b, m, k| {
                let nothing = b.nothing();
                b.case_maybe(b.var(m), nothing, |b, x| b.app(b.var(k), b.var(x)))
            }),
            ValueMonad::List => self.lam2(|b, xs, k| b.concat_map(b.var(k), b.var(xs))),
        }
    }

    fn plus(&mut self) -> core::Expr {
        match self.monad {
            ValueMonad::Maybe => self.lam2(|b, x, y| {
                let y_e = b.var(y);
                b.case_maybe(b.var(x), y_e, |b, v| b.just(b.var(v)))
            }),
            ValueMonad::List => self.named("++"),
        }
    }

    fn zero(&self) -> core::Expr {
        match self.monad {
            ValueMonad::Maybe => self.nothing(),
            ValueMonad::List => self.nil(),
        }
    }

    // Layouts follow the class registry: Functor [fmap, <$]; Applicative
    // [Functor | pure, <*>, *>, <*]; Monad [Applicative | >>=, >>];
    // Alternative [Applicative | <|>, empty]; MonadPlus [Monad | mplus, mzero].

    fn functor(&mut self) -> core::Expr {
        let fields = vec![self.fmap(), self.replace()];
        self.tuple(fields)
    }

    fn applicative(&mut self) -> core::Expr {
        let fields = vec![
            self.functor(),
            self.pure(),
            self.ap(),
            self.then(),
            self.before(),
        ];
        self.tuple(fields)
    }

    fn monad(&mut self) -> core::Expr {
        let fields = vec![self.applicative(), self.bind(), self.then()];
        self.tuple(fields)
    }

    fn alternative(&mut self) -> core::Expr {
        let fields = vec![self.applicative(), self.plus(), self.zero()];
        self.tuple(fields)
    }

    fn monad_plus(&mut self) -> core::Expr {
        let fields = vec![self.monad(), self.plus(), self.zero()];
        self.tuple(fields)
    }
}
