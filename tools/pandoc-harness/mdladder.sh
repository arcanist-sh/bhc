#!/bin/bash
# mdladder.sh — walk UP pandoc's markdown reader, one parseMarkdown body per
# rung, and print what each one does to QR4. Refresh db-bis from parsec-db
# first (a compiler change invalidates it).
set -u
MY="$(cd "$(dirname "$0")" && pwd)"
if [ "${REFRESH:-1}" = 1 ]; then
  rm -rf "$MY/db-bis" && cp -R "$MY/parsec-db" "$MY/db-bis" && echo "db-bis refreshed"
fi
run() { # <label> <body>
  printf '%-22s ' "$1"
  "$MY/mdbis.sh" "$2" 2>&1 | tail -1
}
R='  return (Pandoc nullMeta [])'
run "empty"            "$R"
run "symbol"           "  _ <- symbol
$R"
run "inline"           "  _ <- inline
$R"
run "str;inline"       "  _ <- str
  _ <- inline
$R"
run "many1 inline"     "  _ <- many1 inline
$R"
run "inlines1"         "  _ <- inlines1
$R"
run "plain"            "  _ <- plain
$R"
run "block"            "  _ <- block
$R"
run "parseBlocks"      "  _ <- parseBlocks
$R"
run "full"             "  optional titleBlock
  blocks <- parseBlocks
  st <- getState
  checkNotes
  let doc = runF (do Pandoc _ bs <- B.doc <\$> blocks
                     meta <- stateMeta' st
                     return \$ Pandoc meta bs) st
  reportLogMessages
  return doc"
