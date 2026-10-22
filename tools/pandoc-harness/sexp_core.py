#!/usr/bin/env python3
"""Parse a bhc `BHC_DUMP_BIND` Rust-Debug dump into compact one-line sexps."""
import re
import sys

text = open(sys.argv[1]).read()
m = re.search(r"=== CORE BIND [^\n]*===\s*", text)
text = text[m.end():] if m else text

TOK = re.compile(r'Symbol\("[^"]*"\)|[A-Za-z_$][A-Za-z0-9_.$\'<>*=+!?-]*|\d+|[(){}\[\],:]')
toks = TOK.findall(text)
pos = 0


def peek():
    return toks[pos] if pos < len(toks) else None


def eat():
    global pos
    t = toks[pos]
    pos += 1
    return t


def sym(t):
    m = re.match(r'Symbol\("([^"]*)"\)', t)
    return m.group(1) if m else t


def parse():
    """Return a compact string for the next value in the token stream."""
    t = eat()
    if t == "(":  # tuple
        items = []
        while peek() not in (")", None):
            if peek() == ",":
                eat()
                continue
            items.append(parse())
        if peek() == ")":
            eat()
        return "(" + " ".join(items) + ")"
    if t == "[":
        items = []
        while peek() not in ("]", None):
            if peek() == ",":
                eat()
                continue
            items.append(parse())
        if peek() == "]":
            eat()
        return "[" + " ".join(items) + "]"
    if t == "{":  # struct body: keep only the interesting fields
        fields = {}
        while peek() not in ("}", None):
            if peek() == ",":
                eat()
                continue
            key = eat()
            if peek() == ":":
                eat()
            fields[key] = parse()
        if peek() == "}":
            eat()
        if "name" in fields:
            n = fields["name"]
            return n + ("#" + fields["id"] if "id" in fields else "")
        return "{" + " ".join(f"{k}={v}" for k, v in fields.items()) + "}"
    if peek() == "(":  # constructor with payload
        eat()
        items = []
        while peek() not in (")", None):
            if peek() == ",":
                eat()
                continue
            items.append(parse())
        if peek() == ")":
            eat()
        head = sym(t)
        if head == "App":
            return "(@ " + " ".join(items) + ")"
        if head in ("Var", "Lit", "Span", "BytePos", "VarId", "DefId", "TyCon", "TyVar"):
            return items[0] if items else head
        return "(" + head + " " + " ".join(items) + ")"
    if peek() == "{":  # struct-like constructor
        head = sym(t)
        body = parse()
        return body if head in ("Var", "TyCon", "TyVar") else f"({head} {body})"
    return sym(t)


out = parse()
# Flatten nested App spines: (@ (@ f x) y) -> (@ f x y)
for _ in range(40):
    new = re.sub(r"\(@ \(@ ([^()]*)\) ", r"(@ \1 ", out)
    if new == out:
        break
    out = new
print(out)
