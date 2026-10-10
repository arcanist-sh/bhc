#!/bin/bash
# ladder.sh — run the pandoc parser ladder and CHECK VALUES (the only honest
# gate: a probe printing "OK " for "OK a" reads as a pass at a glance).
set -u
MY="$(cd "$(dirname "$0")" && pwd)"
SPECS="${SPECS:-\
MD1:len=6
TA:len=3
TB:len=3
OPT1:abbrevs=28}"
pass=0; fail=0
while IFS=: read -r p want; do
  [ -z "$p" ] && continue
  out=$(DB="${DB:-$MY/parsec-db}" SNAP="${SNAP:-$MY/snap}" ${LINKER:-$MY/flink.sh} "$MY/$p.hs" 2>&1)
  got=$(echo "$out" | sed -n '/^=== running/,/^=== exit/p' | sed '1d;$d' \
        | grep -vE '^note:|^thread caused|^link: stubbed' | head -1)
  if [ "$got" = "$want" ]; then v=PASS; pass=$((pass+1)); else v=FAIL; fail=$((fail+1)); fi
  printf '%-8s %s  got=%-30q want=%q\n' "$p" "$v" "$got" "$want"
done <<< "$SPECS"
echo "ladder: $pass pass, $fail fail"
