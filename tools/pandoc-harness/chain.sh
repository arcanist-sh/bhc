#!/bin/bash
# chain.sh — build the pandoc package DB with bhc, in dependency order.
#
#   ./chain.sh snapshot   # copy bhc + static libs out of target/debug
#   ./chain.sh deps       # parsec, pandoc-types, THEN the override modules
#   ./chain.sh sweep      # the 221-module sweep, to convergence
#   ./chain.sh all        # all three
#
# Override SNAP/DB/LOGS/FAILS to keep builds side by side — a flag-on and a
# flag-off sweep must never share a fail list (that cost one bad comparison).
set -u
MY="$(cd "$(dirname "$0")" && pwd)"
BHCSRC=${BHCSRC:-$(cd "$MY/../.." && pwd)}
SNAP=${SNAP:-$MY/snap}
DB=${DB:-$MY/parsec-db}
LOGS=${LOGS:-$MY/logs}
FAILS=${FAILS:-$MY/fails.txt}
BHC=$SNAP/bhc
PANDOC=${PANDOC:-$HOME/Development/pandoc-3.6.4/src}
PTYPES=${PTYPES:-$HOME/Development/pandoc-types-1.23.1.2/src}
VENDOR=${VENDOR:-$HOME/Development/pandoc-vendor}
PARSEC=${PARSEC:-$MY/parsec-vendored/src}
MAXROUNDS=${MAXROUNDS:-6}

mkdir -p "$DB" "$LOGS"

logname() {
  echo "$1" | sed 's|.*/src/||; s|.*/pandoc-vendor/||; s|.*/blaze-vendored/||; s|/|.|g; s|\.hs$||'
}

compile1() { # <file> -> 0 pass / 1 fail; log in $LOGS/<Module>.log
  local f=$1 log
  log=$LOGS/$(logname "$f").log
  "$BHC" --package-db "$DB" -c "$f" --odir "$DB" --hidir "$DB" >"$log" 2>&1
}

snapshot() {
  mkdir -p "$SNAP"
  cp "$BHCSRC/target/debug/bhc" "$SNAP/" || exit 1
  cp "$BHCSRC"/target/debug/libbhc_{rts,text,base,containers}.a "$SNAP/" || exit 1
  echo "snapshot: $("$BHC" version 2>/dev/null | head -1) @ $(date +%H:%M:%S)"
}

vendor_parsec() {
  if [ ! -d "$PARSEC" ]; then
    mkdir -p "$MY/parsec-vendored"
    cp -R "$MY/parsec-3.1.17.0/src" "$PARSEC"
    # Open bhc bug: an OPTIONS_HADDOCK pragma breaks Prim's header parse.
    sed -i '' '/OPTIONS_HADDOCK/d' "$PARSEC/Text/Parsec/Prim.hs"
  fi
}

deps() {
  vendor_parsec
  local round f pass
  # parsec core (the facade needs Pos/Error/Prim/Char/Combinator; String and
  # Token for the Parser synonym)
  local PFILES=""
  for m in Pos Error Prim Char Combinator String Token; do
    PFILES="$PFILES $PARSEC/Text/Parsec/$m.hs"
  done
  PFILES="$PFILES $PARSEC/Text/Parsec.hs"
  local XFILES="$VENDOR/Text/Collate/Lang.hs $VENDOR/System/Console/GetOpt.hs $VENDOR/Data/Version.hs $VENDOR/Paths_pandoc.hs"
  local TFILES="$VENDOR/Text/Pandoc/Definition.hs $PTYPES/Text/Pandoc/Walk.hs $PTYPES/Text/Pandoc/Builder.hs $PTYPES/Text/Pandoc/Generic.hs $PTYPES/Text/Pandoc/JSON.hs"
  # LATE: a module in the DB overrides bhc's stub list for its WHOLE export
  # surface, and the vendored Data.Semigroup does not export Semigroup(..)'s
  # sconcat/stimes, which parsec's Prim imports. So these go in AFTER parsec
  # and pandoc-types, never before. (Text/DocLayout.hs is deliberately absent:
  # the vendored one is incomplete and would break modules importing charWidth.)
  local BZ=$MY/blaze-vendored
  local LATE="$BZ/Data/Semigroup.hs $BZ/Text/Blaze/Internal.hs $BZ/Text/Blaze.hs $BZ/Text/Blaze/Html.hs"
  LATE="$LATE $BZ/Text/Blaze/Html5.hs $BZ/Text/Blaze/Html5/Attributes.hs $BZ/Text/Blaze/XHtml5.hs $BZ/Text/Blaze/XHtml5/Attributes.hs"
  LATE="$LATE $BZ/Text/Blaze/XHtml1/Transitional.hs $BZ/Text/Blaze/XHtml1/Transitional/Attributes.hs $BZ/Text/Blaze/Html/Renderer/Text.hs"

  for round in 1 2; do
    pass=0
    for f in $XFILES $PFILES $TFILES; do
      compile1 "$f" && pass=$((pass + 1))
    done
    echo "deps round $round: $pass/$(echo $XFILES $PFILES $TFILES | wc -w | tr -d ' ')"
  done
  for f in $XFILES $PFILES $TFILES; do
    compile1 "$f" || echo "  DEP-FAIL: $(logname "$f")"
  done
  for f in $LATE; do
    compile1 "$f" || echo "  LATE-FAIL: $(logname "$f")"
  done
}

sweep() {
  [ -f "$MY/order.txt" ] || python3 "$MY/topo.py" "$PANDOC" >"$MY/order.txt"
  local total round pass prevfails="" fails
  total=$(wc -l <"$MY/order.txt" | tr -d ' ')
  for round in $(seq 1 "$MAXROUNDS"); do
    pass=0
    fails=""
    while IFS= read -r f; do
      if compile1 "$f"; then pass=$((pass + 1)); else fails="$fails $(logname "$f")"; fi
    done <"$MY/order.txt"
    echo "sweep round $round: $pass/$total"
    [ "$fails" = "$prevfails" ] && break
    prevfails="$fails"
  done
  echo "$fails" | tr ' ' '\n' | sed '/^$/d' >"$FAILS"
  echo "final: $pass/$total ($(wc -l <"$FAILS" | tr -d ' ') failing, list in $FAILS)"
}

case "${1:-all}" in
  snapshot) snapshot ;;
  deps) deps ;;
  sweep) sweep ;;
  all)
    snapshot
    deps
    sweep
    ;;
  *)
    echo "usage: chain.sh [snapshot|deps|sweep|all]"
    exit 2
    ;;
esac
