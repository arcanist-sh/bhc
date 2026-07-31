//! Regression: a multi-line (layout) `case` used as a TUPLE element must close
//! its layout block on the tuple's `,` (Haskell parse-error(t) rule), and a `,`
//! that separates GUARDS inside a case alternative must NOT.
//!
//! Text.Pandoc.Writers.Shared's `ensureValidXmlIdentifiers` does
//!   fixIdentifiers (ident, classes, kvs) =
//!     (case T.uncons ident of
//!        Nothing -> ident
//!        _ -> "id_" <> ident,      -- the `,` must end the `case`
//!      classes, kvs)
//! The `,` stayed among the case alternatives, the parser mis-recovered, and the
//! enclosing binding was silently DROPPED — so importers saw it as unbound
//! (Text.Pandoc.Writers.TEI). Conversely Text.Pandoc.Writers.RTF has
//!   case result of
//!     (imgdata, Just mime) | m' <- .., m' == .. -> ..   -- guard `,`, keep open
//! which must keep working (the case sits inside a `do`, not directly in a `(`).

use bhc_driver::Compiler;
use std::io::Write;

fn check_ok(source: &str) {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("temp");
    file.write_all(source.as_bytes()).expect("write");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8");
    let compiler = Compiler::with_defaults().expect("compiler");
    assert!(
        compiler.check_file(path).is_ok(),
        "expected the module to check"
    );
}

#[test]
fn multiline_case_as_tuple_element_closes_on_comma() {
    // The enclosing binding must not be dropped (its name is >1 char, which is
    // what surfaced the parser mis-recovery).
    check_ok(concat!(
        "module M where\n",
        "process :: Maybe Int -> (Int, Bool)\n",
        "process = go\n",
        "  where\n",
        "    go m =\n",
        "      (case m of\n",
        "         Nothing -> 0\n",
        "         Just x | x > 0 -> x\n",
        "         _ -> negate 1,\n",
        "       True)\n",
        "usesProcess :: Maybe Int -> (Int, Bool)\n",
        "usesProcess = process\n",
    ));
}

#[test]
fn guard_comma_in_case_alternative_is_not_closed() {
    // The `,` separating guards (`| x > 0, x < 10 ->`) with the `case` inside a
    // `do` (implicit parent) must NOT close the case block.
    check_ok(concat!(
        "module M where\n",
        "run :: Maybe Int -> Maybe Int\n",
        "run m =\n",
        "  (do case m of\n",
        "        Just x | x > 0, x < 10 -> Just x\n",
        "        _ -> Nothing)\n",
    ));
}
