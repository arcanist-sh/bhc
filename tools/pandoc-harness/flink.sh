#!/bin/bash
# flink.sh — fast probe link loop against an already-built package DB.
#
#   ./flink.sh localize          # once per DB build
#   ./flink.sh Probe.hs [args]   # compile probe, link, run (~3 s)
#
# codegen emits `_show_desc_N` per module with external linkage, so a link
# needs those localized. Doing it for every DB object on every probe is nearly
# all of a ~4-minute link, and the DB objects only change when a module is
# recompiled — so localize the DB once and each probe only its own objects.
set -u
MY="$(cd "$(dirname "$0")" && pwd)"
SNAP=${SNAP:-$MY/snap}
DB=${DB:-$MY/parsec-db}
BHC=$SNAP/bhc
OBJCOPY=${OBJCOPY:-/opt/homebrew/opt/llvm/bin/llvm-objcopy}

# ONE input file per invocation: llvm-objcopy takes `input [output]`, so a
# batch of two would overwrite the second with the first.
localize() {
  find "$1" -name '*.o' -print0 | xargs -0 -P 8 -n 1 \
    "$OBJCOPY" --wildcard --localize-symbol='_show_desc_*' 2>/dev/null
  true
}

if [ "${1:-}" = localize ]; then
  localize "$DB"
  echo "localized $(find "$DB" -name '*.o' | wc -l | tr -d ' ') objects"
  exit 0
fi

src=$1
shift || true
base=$(basename "$src" .hs)
bin=$MY/bin_$base
odir=$MY/probe_$base
rm -rf "$odir"
mkdir -p "$odir"
"$BHC" --package-db "$DB" -c "$src" --odir "$odir" --hidir "$odir" \
  >"$odir/compile.log" 2>&1 || { tail -30 "$odir/compile.log"; exit 1; }
localize "$odir"

objs="$(find "$odir" -name '*.o') $(find "$DB" -name '*.o')"
libs="$SNAP/libbhc_rts.a $SNAP/libbhc_text.a $SNAP/libbhc_base.a $SNAP/libbhc_containers.a -lSystem"
if ! cc $objs $libs -o "$bin" 2>"$odir/link.err"; then
  # Modules that failed to compile leave undefined symbols; stub them so the
  # probe still runs, and trap loudly if one is actually called.
  grep -o '"_[^"]*", referenced' "$odir/link.err" \
    | sed 's/^"_//; s/", referenced//' | sort -u >"$odir/undef.txt"
  [ -s "$odir/undef.txt" ] || { grep -v 'no platform load command' "$odir/link.err" | head -30; exit 1; }
  {
    while IFS= read -r s; do
      printf '.globl "_%s"\n"_%s":\n  b _bhc_linkstub_trap\n' "$s" "$s"
    done <"$odir/undef.txt"
    printf '.globl _bhc_linkstub_trap\n_bhc_linkstub_trap:\n  b _abort\n'
  } >"$odir/linkstubs.s"
  echo "link: stubbed $(wc -l <"$odir/undef.txt" | tr -d ' ') undefined symbols (see $odir/undef.txt)"
  cc $objs "$odir/linkstubs.s" $libs -o "$bin" 2>"$odir/link2.err" \
    || { grep -v 'no platform load command' "$odir/link2.err" | head -30; exit 1; }
fi

echo "=== running $bin $* ==="
"$bin" "$@"
echo "=== exit: $? ==="
