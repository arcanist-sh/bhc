//! Regression coverage for the Writers.HTML / ODT / RST batch (2026-08-04,
//! second session). Each case mirrors a construct that silently dropped a
//! binding or mis-typed in a real pandoc module.

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
fn bang_operator_sections() {
    // `foldl' (!) h` + right section `(! x)` — `!` lexes as Bang, not
    // Operator (Writers.HTML's addAttrs / toList).
    check_ok(concat!(
        "module M where\n",
        "infixl 9 !\n",
        "(!) :: Int -> Int -> Int\n",
        "(!) a b = a + b\n",
        "f :: Int -> Int\n",
        "f h = foldl (!) h [1,2]\n",
        "g :: [Int] -> [Int]\n",
        "g = map (! 2)\n",
        "use :: Int\n",
        "use = f 3 + sum (g [1])\n",
    ));
}

#[test]
fn bang_infix_inside_parens() {
    // `(H.ol ! A.start x $ y) >> z` — the infix-in-parens op detection was
    // missing the Bang arm (Writers.HTML's footnoteSection).
    check_ok(concat!(
        "module M where\n",
        "infixl 9 !\n",
        "(!) :: Int -> Int -> Int\n",
        "(!) a b = a + b\n",
        "f :: Int -> Int\n",
        "f x = (x ! 1) + 2\n",
        "use :: Int\n",
        "use = f 1\n",
    ));
}

#[test]
fn inline_annotated_let_binding() {
    // `let mCss :: Maybe [Text] = lookupContext ..` (Writers.HTML's
    // pandocToHtml) — signature and binding in ONE let decl.
    check_ok(concat!(
        "module M where\n",
        "f :: Int -> Maybe [Int]\n",
        "f x = do\n",
        "  let m :: Maybe [Int] = Just [x]\n",
        "  m\n",
        "use :: Maybe [Int]\n",
        "use = f 1\n",
    ));
}

#[test]
fn guarded_pattern_binding() {
    // `(sampOrVar, cs') | "sample" `elem` cs = .. | otherwise = ..` — a
    // where-block PATTERN binding with guards (Writers.HTML's inlineToHtml).
    check_ok(concat!(
        "module M where\n",
        "f :: [Int] -> (Int, Int)\n",
        "f cs = (a, b)\n",
        "  where\n",
        "    (a, b)\n",
        "      | 1 `elem` cs = (1, 2)\n",
        "      | otherwise = (3, 4)\n",
        "use :: (Int, Int)\n",
        "use = f [1]\n",
    ));
}

#[test]
fn gadt_record_constructor_with_context() {
    // `data S a where S :: Show a => { fld :: a, other :: Int } -> S a` —
    // GADT record syntax (Readers.ODT.Generic.XMLConverter's
    // XMLConverterState). The accessors must register.
    check_ok(concat!(
        "module M where\n",
        "data S a where\n",
        "  S :: Show a => { fld :: a, other :: Int } -> S a\n",
        "get :: S Int -> Int\n",
        "get s = fld s + other s\n",
        "use :: Int\n",
        "use = get (S { fld = 1, other = 2 })\n",
    ));
}

#[test]
fn proc_notation_type_checks() {
    // Arrow `proc pat -> do .. -< ..` (Readers.ODT.StyleReader/ContentReader),
    // desugared permissively to `arr (\\pat -> ..)`.
    check_ok(concat!(
        "module M where\n",
        "f = proc target -> do\n",
        "    state <- getE -< ()\n",
        "    case lookup target state of\n",
        "      Just bs -> retV bs -<< ()\n",
        "      Nothing -> retV \"x\" -< ()\n",
        "getE = undefined\n",
        "retV = undefined\n",
        "use = f\n",
    ));
}

#[test]
fn identity_constructor_in_expression() {
    // `Identity` is builtin-registered as a VALUE (DefId 10000); an
    // expression use must fall back to that scheme (Readers.RST's
    // simpleTable/gridTable).
    check_ok(concat!(
        "module M where\n",
        "import Control.Monad.Identity (Identity (..))\n",
        "wrap :: Int -> Identity Int\n",
        "wrap = Identity\n",
        "use :: Int\n",
        "use = runIdentity (wrap 3)\n",
    ));
}
