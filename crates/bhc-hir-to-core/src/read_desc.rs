//! Type descriptions for the runtime's generic `read`.
//!
//! `read`, `readMaybe` and `reads` at a data type, a tuple, a nested list or
//! a `Maybe` have no parser of their own: there is no `Read` class compiled
//! from source yet. For those the lowering passes the runtime a description
//! of the target type (`bhc_read_generic`, whose module documents the
//! format) and the runtime parses against it.
//!
//! A type gets a description only when every part of it is one the runtime
//! reader builds correctly: the base types, lists, tuples, `Maybe`, `Either`
//! and data types of the module being compiled that derive `Read`. Anything
//! else answers `None` and `read` is lowered as before.

use bhc_hir::{ConFields, DataDef, NewtypeDef};
use bhc_intern::Symbol;
use bhc_types::Ty;
use rustc_hash::FxHashMap;

/// A constructor of a data type that derives `Read`.
#[derive(Clone, Debug)]
struct ReadCon {
    name: Symbol,
    tag: u32,
    /// Field names, for a constructor declared with record syntax.
    record: Option<Vec<Symbol>>,
    fields: Vec<Ty>,
}

/// A data type or newtype that derives `Read`.
#[derive(Clone, Debug)]
pub(crate) struct ReadShape {
    /// The type parameters' variable ids, in order.
    params: Vec<u32>,
    cons: Vec<ReadCon>,
    newtype: bool,
}

fn derives_read(deriving: &[bhc_hir::DerivingClause]) -> bool {
    deriving.iter().any(|c| c.class.as_str() == "Read")
}

fn read_con(con: &bhc_hir::ConDef, tag: u32) -> Option<ReadCon> {
    // A GADT or existential constructor has no derived `Read`.
    if con.gadt_return_ty.is_some() || !con.existential_vars.is_empty() {
        return None;
    }
    let (record, fields) = match &con.fields {
        ConFields::Positional(tys) => (None, tys.clone()),
        ConFields::Named(fs) => (
            Some(fs.iter().map(|f| f.name).collect()),
            fs.iter().map(|f| f.ty.clone()).collect(),
        ),
    };
    Some(ReadCon {
        name: con.name,
        tag,
        record,
        fields,
    })
}

impl ReadShape {
    /// The shape of a data type, if it derives `Read`.
    pub(crate) fn of_data(def: &DataDef) -> Option<Self> {
        if !derives_read(&def.deriving) || def.is_gadt {
            return None;
        }
        let cons: Option<Vec<ReadCon>> = def
            .cons
            .iter()
            .enumerate()
            .map(|(tag, con)| read_con(con, tag as u32))
            .collect();
        Some(Self {
            params: def.params.iter().map(|p| p.id).collect(),
            cons: cons?,
            newtype: false,
        })
    }

    /// The shape of a newtype, if it derives `Read`.
    pub(crate) fn of_newtype(def: &NewtypeDef) -> Option<Self> {
        if !derives_read(&def.deriving) {
            return None;
        }
        let con = read_con(&def.con, 0)?;
        (con.fields.len() == 1).then(|| Self {
            params: def.params.iter().map(|p| p.id).collect(),
            cons: vec![con],
            newtype: true,
        })
    }
}

/// A name the description can carry: an identifier, as the runtime reader
/// lexes one. Operator constructors (`:+:`) are read infix, which the reader
/// does not do.
fn is_identifier(name: &str, constructor: bool) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let first_ok = if constructor {
        first.is_uppercase()
    } else {
        first.is_lowercase() || first == '_'
    };
    first_ok && chars.all(|c| c.is_alphanumeric() || c == '_' || c == '\'')
}

fn substitute(ty: &Ty, subst: &FxHashMap<u32, Ty>) -> Ty {
    match ty {
        Ty::Var(v) => subst.get(&v.id).cloned().unwrap_or_else(|| ty.clone()),
        Ty::App(f, a) => Ty::App(
            Box::new(substitute(f, subst)),
            Box::new(substitute(a, subst)),
        ),
        Ty::Fun(a, r) => Ty::Fun(
            Box::new(substitute(a, subst)),
            Box::new(substitute(r, subst)),
        ),
        Ty::List(e) => Ty::List(Box::new(substitute(e, subst))),
        Ty::Tuple(es) => Ty::Tuple(es.iter().map(|e| substitute(e, subst)).collect()),
        other => other.clone(),
    }
}

struct Describer<'a> {
    shapes: &'a FxHashMap<Symbol, ReadShape>,
    /// Data types described so far, by their instantiated type. `None` while
    /// the type's own description is being built (a recursive reference).
    datas: Vec<(Ty, Option<String>)>,
}

impl Describer<'_> {
    fn shape_named(&self, name: &str) -> Option<&ReadShape> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        self.shapes
            .get(&Symbol::intern(name))
            .or_else(|| self.shapes.get(&Symbol::intern(bare)))
    }

    fn data(&mut self, ty: &Ty, name: &str, args: &[&Ty]) -> Option<String> {
        if let Some(index) = self.datas.iter().position(|(t, _)| t == ty) {
            return Some(format!("@{index}"));
        }
        let shape = self.shape_named(name)?.clone();
        if shape.params.len() != args.len() {
            return None;
        }
        let subst: FxHashMap<u32, Ty> = shape
            .params
            .iter()
            .copied()
            .zip(args.iter().map(|a| (*a).clone()))
            .collect();
        let index = self.datas.len();
        self.datas.push((ty.clone(), None));
        let mut text = String::from(if shape.newtype { "(N" } else { "(D" });
        for con in &shape.cons {
            if !is_identifier(con.name.as_str(), true) {
                return None;
            }
            text.push_str(&format!(" (C {} {}", con.name.as_str(), con.tag));
            text.push_str(if con.record.is_some() { " R" } else { " P" });
            for (i, field) in con.fields.iter().enumerate() {
                if let Some(names) = &con.record {
                    let field_name = names[i].as_str();
                    if !is_identifier(field_name, false) {
                        return None;
                    }
                    text.push(' ');
                    text.push_str(field_name);
                }
                text.push(' ');
                text.push_str(&self.ty(&substitute(field, &subst))?);
            }
            text.push(')');
        }
        text.push(')');
        self.datas[index].1 = Some(text);
        Some(format!("@{index}"))
    }

    fn ty(&mut self, ty: &Ty) -> Option<String> {
        match ty {
            Ty::List(elem) => self.list(elem),
            Ty::Tuple(elems) if elems.is_empty() => Some("u".to_string()),
            Ty::Tuple(elems) if elems.len() >= 2 => {
                let mut text = String::from("(t");
                for elem in elems {
                    text.push(' ');
                    text.push_str(&self.ty(elem)?);
                }
                text.push(')');
                Some(text)
            }
            Ty::Con(_) | Ty::App(..) => {
                let mut head = ty;
                let mut args: Vec<&Ty> = Vec::new();
                while let Ty::App(f, a) = head {
                    args.push(a.as_ref());
                    head = f.as_ref();
                }
                args.reverse();
                let Ty::Con(con) = head else {
                    return None;
                };
                let name = con.name.as_str();
                match (name, args.as_slice()) {
                    (
                        "Int" | "Int8" | "Int16" | "Int32" | "Int64" | "Word" | "Word8" | "Word16"
                        | "Word32" | "Word64",
                        [],
                    ) => Some("i".to_string()),
                    ("Integer", []) => Some("n".to_string()),
                    ("Double", []) => Some("d".to_string()),
                    ("Char", []) => Some("c".to_string()),
                    ("String", []) => Some("s".to_string()),
                    ("Bool", []) => Some("b".to_string()),
                    ("Ordering", []) => Some("o".to_string()),
                    ("()", []) => Some("u".to_string()),
                    ("[]" | "List", [elem]) => self.list(elem),
                    ("Maybe", [elem]) => Some(format!("(m {})", self.ty(elem)?)),
                    ("Either", [left, right]) => {
                        Some(format!("(e {} {})", self.ty(left)?, self.ty(right)?))
                    }
                    _ => self.data(ty, name, &args),
                }
            }
            _ => None,
        }
    }

    fn list(&mut self, elem: &Ty) -> Option<String> {
        if matches!(elem, Ty::Con(c) if c.name.as_str() == "Char") {
            return Some("s".to_string());
        }
        Some(format!("(l {})", self.ty(elem)?))
    }
}

/// The description of `ty` for the runtime reader, when every part of it is
/// one the reader handles.
pub(crate) fn read_description(shapes: &FxHashMap<Symbol, ReadShape>, ty: &Ty) -> Option<String> {
    let mut describer = Describer {
        shapes,
        datas: Vec::new(),
    };
    let mut text = describer.ty(ty)?;
    for (_, data) in describer.datas {
        text.push(' ');
        text.push_str(&data?);
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_types::{Kind, TyCon, TyVar};

    fn con(name: &str) -> Ty {
        Ty::Con(TyCon::new(Symbol::intern(name), Kind::Star))
    }

    fn shapes() -> FxHashMap<Symbol, ReadShape> {
        let mut shapes = FxHashMap::default();
        let positional = |name: &str, tag, fields: Vec<Ty>| ReadCon {
            name: Symbol::intern(name),
            tag,
            record: None,
            fields,
        };
        shapes.insert(
            Symbol::intern("Color"),
            ReadShape {
                params: vec![],
                cons: vec![positional("Red", 0, vec![]), positional("Blue", 1, vec![])],
                newtype: false,
            },
        );
        let a = Ty::Var(TyVar::new(7, Kind::Star));
        let tree_a = Ty::App(Box::new(con("Tree")), Box::new(a.clone()));
        shapes.insert(
            Symbol::intern("Tree"),
            ReadShape {
                params: vec![7],
                cons: vec![
                    positional("Leaf", 0, vec![]),
                    positional("Node", 1, vec![tree_a.clone(), a, tree_a]),
                ],
                newtype: false,
            },
        );
        shapes.insert(
            Symbol::intern("P"),
            ReadShape {
                params: vec![],
                cons: vec![ReadCon {
                    name: Symbol::intern("P"),
                    tag: 0,
                    record: Some(vec![Symbol::intern("name"), Symbol::intern("age")]),
                    fields: vec![Ty::List(Box::new(con("Char"))), con("Int")],
                }],
                newtype: false,
            },
        );
        shapes.insert(
            Symbol::intern("Op"),
            ReadShape {
                params: vec![],
                cons: vec![positional(":+:", 0, vec![con("Int"), con("Int")])],
                newtype: false,
            },
        );
        shapes
    }

    fn describe(ty: &Ty) -> Option<String> {
        read_description(&shapes(), ty)
    }

    #[test]
    fn base_and_container_types() {
        let pairs = Ty::List(Box::new(Ty::Tuple(vec![
            con("Int"),
            Ty::List(Box::new(con("Char"))),
        ])));
        assert_eq!(describe(&pairs).as_deref(), Some("(l (t i s))"));
        let maybe = Ty::App(Box::new(con("Maybe")), Box::new(con("Double")));
        assert_eq!(describe(&maybe).as_deref(), Some("(m d)"));
        assert_eq!(describe(&Ty::Tuple(vec![])).as_deref(), Some("u"));
    }

    #[test]
    fn data_types_are_numbered_and_recursion_is_a_reference() {
        assert_eq!(
            describe(&con("Color")).as_deref(),
            Some("@0 (D (C Red 0 P) (C Blue 1 P))")
        );
        let tree = Ty::App(Box::new(con("Tree")), Box::new(con("Color")));
        assert_eq!(
            describe(&tree).as_deref(),
            Some("@0 (D (C Leaf 0 P) (C Node 1 P @0 @1 @0)) (D (C Red 0 P) (C Blue 1 P))")
        );
        assert_eq!(
            describe(&con("P")).as_deref(),
            Some("@0 (D (C P 0 R name s age i))")
        );
    }

    #[test]
    fn unsupported_parts_give_no_description() {
        assert_eq!(describe(&con("Text")), None);
        assert_eq!(describe(&con("Float")), None);
        assert_eq!(describe(&con("Op")), None);
        assert_eq!(describe(&Ty::Var(TyVar::new(1, Kind::Star))), None);
        let list_of_unknown = Ty::List(Box::new(con("Unknown")));
        assert_eq!(describe(&list_of_unknown), None);
    }
}
