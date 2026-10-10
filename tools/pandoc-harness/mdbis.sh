#!/bin/bash
# mdbis.sh '<body of parseMarkdown after "do">' — patch pandoc's parseMarkdown
# in a scratch copy, compile it into db-bis, relink QR4 and print the outcome.
set -u
MY="$(cd "$(dirname "$0")" && pwd)"
export DB=$MY/db-bis SNAP=${SNAP:-$MY/snap}
SRC=$HOME/Development/pandoc-3.6.4/src/Text/Pandoc/Readers/Markdown.hs
OUT=$MY/mdbis-src/Markdown.hs
python3 - "$1" "$SRC" "$OUT" <<'PY'
import sys
body, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src).read()
start = s.index("parseMarkdown = do\n")
end = s.index("\n-- check for notes with no corresponding note references")
open(out, "w").write(s[:start] + "parseMarkdown = do\n" + body.rstrip() + "\n" + s[end:])
PY
"$SNAP/bhc" --package-db "$DB" -c "$OUT" --odir "$DB" --hidir "$DB" > "$MY/mdbis-src/compile.log" 2>&1 || {
  echo "COMPILE-FAIL"; grep -A5 -m1 "rror" "$MY/mdbis-src/compile.log"; exit 1; }
/opt/homebrew/opt/llvm/bin/llvm-objcopy --wildcard --localize-symbol='_show_desc_*' "$DB/Text/Pandoc/Readers/Markdown.o"
"$MY/flink.sh" "$MY/QR4.hs" 2>&1 | grep -E "RESULT|Segmentation|Abort|panicked|stub:|not a closure" | head -2
