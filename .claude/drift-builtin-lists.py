#!/usr/bin/env python3
"""Diff the two ordered builtin name lists to map the bhc-lower <-> bhc-typeck drift.

Lowering (source of truth for name->DefId): bhc-lower/src/context.rs `builtin_funcs`
Typeck (guesses DefIds positionally):        bhc-typeck/src/builtins.rs `ops`
"""
import re, sys

LOWER = "crates/bhc-lower/src/context.rs"
TYPECK = "crates/bhc-typeck/src/builtins.rs"

def lines(path):
    with open(path) as f:
        return f.readlines()

# --- Lowering builtin_funcs: bare string-literal lines within the array ---
lo = lines(LOWER)
# find `let builtin_funcs = [` and its closing `];`
start = next(i for i,l in enumerate(lo) if "let builtin_funcs = [" in l)
end   = next(i for i,l in enumerate(lo) if i>start and l.strip()=="];")
lower_names = []
for l in lo[start+1:end]:
    m = re.match(r'\s*"((?:[^"\\]|\\.)+)"\s*,?\s*(//.*)?$', l)
    if m:
        lower_names.append(m.group(1))

# --- Typeck ops: first string of each 12-space-indented tuple open ---
ty = lines(TYPECK)
tstart = next(i for i,l in enumerate(ty) if re.match(r'\s*let ops: Vec', l))
# closing of the vec: first line that is exactly 8-space `];`
tend = next(i for i,l in enumerate(ty) if i>tstart and re.match(r'^        \];\s*$', l))
typeck_names = []
for l in ty[tstart+1:tend]:
    m = re.match(r'^            \("((?:[^"\\]|\\.)+)"\s*,', l)
    if m:
        typeck_names.append(m.group(1))

print(f"lowering builtin_funcs entries: {len(lower_names)}")
print(f"typeck ops entries:             {len(typeck_names)}")

# first divergence
n = min(len(lower_names), len(typeck_names))
first = next((i for i in range(n) if lower_names[i]!=typeck_names[i]), None)
if first is None:
    print(f"\nNo positional divergence in first {n} entries.")
else:
    print(f"\nFIRST DIVERGENCE at index {first}:")
    print(f"  lowering[{first}] = {lower_names[first]!r}")
    print(f"  typeck  [{first}] = {typeck_names[first]!r}")
    print(f"\nContext (indices {max(0,first-3)}..{first+8}):")
    print(f"  {'idx':>4}  {'lowering(real DefId)':<28} {'typeck(guessed scheme name)':<28} match")
    for i in range(max(0,first-3), min(n, first+9)):
        lname = lower_names[i]
        tname = typeck_names[i]
        mark = "OK" if lname==tname else "XX  <-- collision: DefId gets wrong scheme"
        print(f"  {i:>4}  {lname:<28} {tname:<28} {mark}")

# count total positional mismatches over the shared prefix
mismatch = sum(1 for i in range(n) if lower_names[i]!=typeck_names[i])
print(f"\nTotal positional mismatches over shared prefix ({n}): {mismatch}  ({100*mismatch//n}%)")

# names present in one list but not the other (set-level)
ls, tsn = set(lower_names), set(typeck_names)
print(f"\nIn lowering but NOT in typeck ops ({len(ls-tsn)}): permissive fresh-var path, no scheme collision")
print("   " + ", ".join(sorted(ls-tsn)[:40]))
print(f"\nIn typeck ops but NOT in lowering ({len(tsn-ls)}): dead schemes (never applied at a real DefId by index... unless collided)")
print("   " + ", ".join(sorted(tsn-ls)[:40]))
