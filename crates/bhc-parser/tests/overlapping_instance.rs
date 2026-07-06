//! Regression test: an `instance {-# OVERLAPPING #-} ...` must parse as a
//! single `InstanceDecl` whose `where` methods belong to the instance — NOT
//! leak out as top-level bindings.
//!
//! The overlap pragma sits between `instance` and the head. If the parser does
//! not consume it, the head parse derails and error recovery reparses the
//! instance body's methods as top-level `FunBind`s. A method named like a class
//! method (e.g. `query`) then shadows the real class method in module scope and
//! silently mis-resolves overloaded uses (this was the pandoc-types `Walk`
//! `expected MetaValue, found [Block]` bug).

use bhc_ast::Decl;
use bhc_parser::parse_module;
use bhc_span::FileId;

fn parse(src: &str) -> bhc_ast::Module {
    let (module, diags) = parse_module(src, FileId::new(0));
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == bhc_diagnostics::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "unexpected parse errors: {errors:?}");
    module.expect("module should parse")
}

#[test]
fn overlapping_instance_methods_do_not_leak_to_top_level() {
    // Two instances of the same class; the overlapping one has a `name` method
    // with the SAME name as the class method. It must stay inside the instance.
    let src = "\
module T where

class C a where
  name :: a -> Int

instance {-# OVERLAPPING #-} C [Char] where
  name _ = 1

instance C a => C [a] where
  name _ = 2
";
    let module = parse(src);

    // Exactly one class decl and two instance decls; no top-level `name` FunBind.
    let classes = module
        .decls
        .iter()
        .filter(|d| matches!(d, Decl::ClassDecl(_)))
        .count();
    let instances: Vec<_> = module
        .decls
        .iter()
        .filter_map(|d| match d {
            Decl::InstanceDecl(i) => Some(i),
            _ => None,
        })
        .collect();
    let leaked_funbinds = module
        .decls
        .iter()
        .filter(|d| matches!(d, Decl::FunBind(fb) if fb.name.name.as_str() == "name"))
        .count();

    assert_eq!(classes, 1, "expected one class decl");
    assert_eq!(instances.len(), 2, "expected two instance decls");
    assert_eq!(
        leaked_funbinds, 0,
        "instance method `name` must not leak to top level"
    );
    // The overlapping instance must carry its method.
    assert!(
        instances.iter().any(|i| i.methods.len() == 1),
        "overlapping instance should contain its `name` method"
    );
}
