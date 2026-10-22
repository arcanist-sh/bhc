#!/bin/bash
# resweep.sh — after a fix, recompile ONLY the modules in $FAILS (in topo
# order) and repeat while the fail list keeps shrinking. A full sweep round is
# ~8 minutes; this is seconds when a handful of modules failed.
set -u
MY="$(cd "$(dirname "$0")" && pwd)"
SNAP=${SNAP:-$MY/snap}
DB=${DB:-$MY/parsec-db}
LOGS=${LOGS:-$MY/logs}
FAILS=${FAILS:-$MY/fails.txt}
BHC=$SNAP/bhc

for round in 1 2 3; do
  fails=""
  while IFS= read -r f; do
    m=$(echo "$f" | sed 's|.*/src/||; s|/|.|g; s|\.hs$||')
    grep -qx "$m" "$FAILS" || continue
    if "$BHC" --package-db "$DB" -c "$f" --odir "$DB" --hidir "$DB" >"$LOGS/$m.log" 2>&1; then :; else
      fails="$fails $m"
    fi
  done <"$MY/order.txt"
  echo "$fails" | tr ' ' '\n' | sed '/^$/d' >"$FAILS.new"
  n_old=$(wc -l <"$FAILS" | tr -d ' ')
  n_new=$(wc -l <"$FAILS.new" | tr -d ' ')
  mv "$FAILS.new" "$FAILS"
  echo "resweep round $round: $n_old -> $n_new failing"
  [ "$n_new" -lt "$n_old" ] || break
done
cat "$FAILS"
