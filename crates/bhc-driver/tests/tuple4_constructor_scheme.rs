//! Regression: tuple constructors of arity 4..7 must have the proper
//! `a1 -> .. -> an -> (a1, .., an)` scheme — the bound components tied to the
//! tuple's element types.
//!
//! Only `(,)` and `(,,)` were curated; arity-4+ tuple constructors were created
//! on the fly and fell to the arity fallback, which produced a result type
//! `Tuple4 a b c d` (a bare `Con`) whose field variables were UNRELATED to the
//! tuple components. A tuple pattern then failed to tie its bound variables to
//! the matched value's component types. This surfaced when a where/let-bound
//! MULTI-CLAUSE function desugars to `\args -> case (a, ..) of (p, ..) -> ..`
//! (a 4-tuple scrutinee + tuple patterns) — e.g.
//! `Text.Pandoc.Writers.AnnotatedTable`'s `annotateBodySection`, which failed
//! with `expected ColNumber, found [(RowSpan, ColSpec)]`.

use bhc_driver::Compiler;
use std::io::Write;

fn check_ok(source: &str) {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("create temp file");
    file.write_all(source.as_bytes()).expect("write source");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8 path");

    let compiler = Compiler::with_defaults().expect("compiler");
    let result = compiler.check_file(path);
    assert!(
        result.is_ok(),
        "expected a multi-clause where-helper destructuring a 4-tuple to check, got {result:?}"
    );
}

#[test]
fn multiclause_helper_destructures_four_tuple() {
    // A where-bound, un-signatured, multi-clause helper with a 4-tuple pattern
    // in each clause body. Pre-fix, the destructured `colnum` (component 0) was
    // conflated with a case-scrutinee slot via the mis-schemed 4-tuple con.
    check_ok(concat!(
        "module M where\n",
        "import Control.Monad.State (State)\n",
        "g :: Int -> [Char] -> [Bool] -> (Int, [Char], [Bool], [Bool])\n",
        "g = undefined\n",
        "wrapper :: [Char] -> [Char] -> [Int] -> State Int [Bool]\n",
        "wrapper h1 h2 items = go h1 h2 id items\n",
        "  where\n",
        "    go headHang bodyHang acc (_ : rs) = do\n",
        "      let (colnum, headHang', rowStub, cells') = g 0 headHang [True]\n",
        "      let (_, bodyHang', rowBody, _) = g colnum bodyHang cells'\n",
        "      return (acc rowStub)\n",
        "    go _ _ acc [] = return (acc [])\n",
    ));
}
