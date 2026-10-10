#!/usr/bin/env python3
"""Emit pandoc's library modules in dependency order, one path per line.

Reads every `.hs` under the given source root, builds a graph from the imports
that name another module in the same tree, and topologically sorts it. Import
cycles (pandoc has a few) are broken by source order — the sweep iterates to
convergence anyway, so a near-order is enough; a good order just means fewer
rounds.
"""
import os
import re
import sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Development/pandoc-3.6.4/src"
)

paths = {}  # module name -> path
for dirpath, _dirs, files in os.walk(root):
    for f in files:
        if not f.endswith(".hs"):
            continue
        p = os.path.join(dirpath, f)
        mod = os.path.relpath(p, root)[:-3].replace(os.sep, ".")
        paths[mod] = p

IMPORT = re.compile(r"^import\s+(?:qualified\s+)?([A-Z][\w.]*)")
deps = defaultdict(set)
for mod, p in paths.items():
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = IMPORT.match(line)
            if m and m.group(1) in paths and m.group(1) != mod:
                deps[mod].add(m.group(1))

order = []
state = {}  # 0 = visiting, 1 = done


def visit(mod):
    s = state.get(mod)
    if s == 1 or s == 0:  # done, or a cycle we are already inside
        return
    state[mod] = 0
    for d in sorted(deps[mod]):
        visit(d)
    state[mod] = 1
    order.append(mod)


sys.setrecursionlimit(10000)
for mod in sorted(paths):
    visit(mod)

for mod in order:
    print(paths[mod])
