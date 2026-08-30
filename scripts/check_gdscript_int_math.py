#!/usr/bin/env python3
"""Poka-yoke: GDScript mini/maxi/clampi silently truncate floats to int.

Fail CI if those int helpers are called with float-looking arguments.
Legitimate uses (Array.size(), frame counts, indices) still pass.

Usage: python scripts/check_gdscript_int_math.py [root]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FUNCS = ("mini", "maxi", "clampi")
TYPED_ARRAY_TERNARY = re.compile(
    r"Array\[[^\]]+\]\s*=\s*[^\n;]+?\s+if\s+.+?\s+else\s+",
    re.MULTILINE,
)
# Godot often fails to infer `var x := obj.method() if c else y` (Julian mid 972).
INFERRED_METHOD_TERNARY = re.compile(
    r"^\s*var\s+\w+\s*:=\s*.+\.\w+\s*\(.*\)\s+if\s+.+\s+else\s+",
)
# Untyped `for x in […]` makes x a Variant — `var y := x.method()` fails in editor (Julian 977).
UNTYPED_FOR = re.compile(r"^\s*for\s+(\w+)\s+in\s+")
TYPED_FOR = re.compile(r"^\s*for\s+\w+\s*:\s*\w+")
INFERRED_FROM_LOOP_VAR = re.compile(r"^\s*var\s+\w+\s*:=\s*{var}\.")

CALL_RE = re.compile(r"\b(" + "|".join(FUNCS) + r")\s*\(")

FLOATISH_IDENT = re.compile(
    r"""(?x)
    \b(?:sel_t[01]|_sel_t[01]|_view_t[01]|_hop_s|_duration_s
         |CARET_PLAY_PAD_S|_anchor_x|_cur_x|_cam_dist|_pitch|_yaw
         |left_frac|t_focus|t_end|span|factor|_pivot)\b
    | \b[a-zA-Z_][\w]*_(?:s|t|x|y|frac|pad)\b
    """
)



def check_typed_array_ternary(root: Path) -> list[str]:
    """Godot rejects `var x: Array[String] = a if c else b` (untyped Array)."""
    errs: list[str] = []
    for path in sorted(root.rglob("*.gd")):
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            continue
        untyped_loop_vars: set[str] = set()
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if TYPED_FOR.match(line):
                # typed for shadows any prior untyped same name in this file scan
                m = re.match(r"^\s*for\s+(\w+)\s*:", line)
                if m:
                    untyped_loop_vars.discard(m.group(1))
            else:
                m = UNTYPED_FOR.match(line)
                if m:
                    untyped_loop_vars.add(m.group(1))
            if TYPED_ARRAY_TERNARY.search(line) or (
                "Array[" in line and " = " in line and " if " in line and line.rstrip().endswith("\\")
            ):
                errs.append(f"{path}:{i}: typed Array ternary (use .assign / if-block)")
            elif "Array[" in line and " = " in line and line.rstrip().endswith("\\"):
                lines = src.splitlines()
                if i < len(lines) and " else " in lines[i]:
                    errs.append(f"{path}:{i}: typed Array ternary (use .assign / if-block)")
            if INFERRED_METHOD_TERNARY.search(line):
                errs.append(
                    f"{path}:{i}: inferred := method ternary (use explicit type or if-block)"
                )
            for v in untyped_loop_vars:
                if re.search(rf"^\s*var\s+\w+\s*:=\s*{re.escape(v)}\.", line):
                    errs.append(
                        f"{path}:{i}: inferred := from untyped for-loop var '{v}' "
                        "(type the for-loop or the var)"
                    )
                    break
    return errs

def _arg_floatish(arg: str) -> bool:
    s = arg.strip()
    # Already coerced into int domain — clampi/mini on these is intentional.
    if re.match(r"^(int|floori|ceili|roundi)\s*\(", s):
        return False
    # Float literal at paren-depth 0 (ignore nested 15.0 inside int(round(...))).
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and c.isdigit():
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "_"):
                j += 1
            if j < len(s) and s[j] == ".":
                return True
            i = j
            continue
        i += 1
    return bool(FLOATISH_IDENT.search(s))


# Strip comments / strings roughly so we don't false-positive in docs.
STRING_OR_COMMENT = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|#[^\n]*'
)


def _calls_in(text: str) -> list[tuple[str, str, int]]:
    """Return (func, args_inside, line_no) for each mini/maxi/clampi call."""
    cleaned = STRING_OR_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
    out: list[tuple[str, str, int]] = []
    for m in CALL_RE.finditer(cleaned):
        func = m.group(1)
        i = m.end()  # past '('
        depth = 1
        start = i
        while i < len(cleaned) and depth:
            c = cleaned[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        args = cleaned[start : i - 1]
        line_no = cleaned.count("\n", 0, m.start()) + 1
        out.append((func, args, line_no))
    return out


def _split_args(args: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for c in args:
        if c == "(":
            depth += 1
            cur.append(c)
        elif c == ")":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    hits: list[str] = []
    for func, args, line_no in _calls_in(text):
        for arg in _split_args(args):
            if _arg_floatish(arg):
                hits.append(
                    f"{path}:{line_no}: {func}(...): float-looking arg `{arg}` "
                    f"— use minf/maxf/clampf (GDScript {func} truncates to int)"
                )
                break
    return hits


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "godot-demo")
    if not root.is_dir():
        print(f"missing dir: {root}", file=sys.stderr)
        return 2
    bad: list[str] = []
    for path in sorted(root.rglob("*.gd")):
        bad.extend(check_file(path))
    bad.extend(check_typed_array_ternary(root))
    if bad:
        print("GDScript footgun(s):", file=sys.stderr)
        for h in bad:
            print(f"  {h}", file=sys.stderr)
        print(
            "\nHint: mini/maxi/clampi cast to int; prefer minf/maxf/clampf for times/pixels.\n"
            "Typed Array[T] cannot be assigned from a ternary — use .assign() / if-block.\n"
            "Avoid `var x := obj.method() if c else y` — give an explicit type or use an if-block.",
            file=sys.stderr,
        )
        return 1
    print(f"ok: no GDScript int-math / ternary footguns under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
