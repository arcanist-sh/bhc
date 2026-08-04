//! Regression: a GUARD comma inside a `\case`/`case` block that sits DIRECTLY
//! inside parentheses must not close the implicit block.
//!
//! The comma-closes-implicit-block lexer heuristic (for `case`-as-tuple-element,
//! see `case_in_tuple_layout.rs`) fired on `t | not (p t), t /= "x" -> t` when
//! the case's immediate parent was an explicit `(` — exactly Readers.Ipynb's
//! `jsonMetaToPairs`. The `guard_active` flag (a `|` seen more recently than
//! `->`/`=`/`;`) now suppresses the close mid-guard.

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
fn guarded_lambda_case_in_parens_with_guard_comma() {
    // The second top-level binding (`use`) is essential: a dropped
    // `jsonMetaToPairs` only surfaces at another binding's use site.
    check_ok(concat!(
        "{-# LANGUAGE LambdaCase #-}\n",
        "{-# LANGUAGE OverloadedStrings #-}\n",
        "module M where\n",
        "import qualified Data.Map as M\n",
        "import qualified Data.Text as T\n",
        "import Data.Char (isDigit)\n",
        "import Data.Text (Text)\n",
        "jsonMetaToPairs :: M.Map Text Text -> [(Text, Text)]\n",
        "jsonMetaToPairs m = M.toList . M.map\n",
        "  (\\case\n",
        "      t\n",
        "        | not (T.all isDigit t)\n",
        "        , t /= \"true\"\n",
        "        , t /= \"false\"\n",
        "                 -> t\n",
        "      x          -> T.reverse x) $ m\n",
        "use :: M.Map Text Text -> Int\n",
        "use = length . jsonMetaToPairs\n",
    ));
}
