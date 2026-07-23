//! Regression: skylighting's external type aliases must expand during
//! unification. `SyntaxMap = Map Text Syntax`, `SourceLine = [Token]`,
//! `Token = (TokenType, Text)` are registered as builtin aliases; otherwise they
//! stay opaque and `M.elems (sm :: SyntaxMap)` / matching a `SourceLine` as a
//! list of pairs fail (`Text.Pandoc.Highlighting`: "expected SyntaxMap, found
//! (Map t t)" / "expected SourceLine, found [(t, Text)]").

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
        "expected skylighting-alias program to type-check, got {result:?}"
    );
}

#[test]
fn syntaxmap_and_sourceline_aliases_expand() {
    check_ok(concat!(
        "module M where\n",
        "import Skylighting.Types (SyntaxMap, SourceLine)\n",
        "import qualified Data.Map as M\n",
        "countLangs :: SyntaxMap -> Int\n",
        "countLangs sm = length (M.elems sm)\n",
        "firstTok :: SourceLine -> Maybe Char\n",
        "firstTok [] = Nothing\n",
        "firstTok ((_, _t) : _) = Nothing\n",
    ));
}
