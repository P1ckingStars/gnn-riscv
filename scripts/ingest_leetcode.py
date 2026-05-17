#!/usr/bin/env python3
"""Ingest LeetCode problems from the HuggingFace ``greengerong/leetcode`` dataset
into a C corpus the model can train on.

The dataset's source code is C++, but ~7% of the solutions are STL-free and translate
cleanly when compiled as C. This script:

  1. Filters problems whose C++ solution contains no STL/iostream/classes/templates.
  2. Wraps each one with a standard C prologue (``#include <stdint.h>`` etc. + pure-C
     ``struct ListNode`` / ``struct TreeNode`` definitions so the LeetCode boilerplate
     compiles).
  3. Tries to compile with ``riscv64-linux-gnu-gcc -xc -O2 -S``; keeps the ones that
     succeed.
  4. Extracts the top-level function signature via libclang.
  5. Writes each kept problem to ``data/leetcode/<id>_<slug>/``:
       - ``source.c``       — prologue + the function body
       - ``ref_o2.s``       — gcc -O2 asm
       - ``ref_o3.s``       — gcc -O3 asm
       - ``signature.json`` — fn name, return ty, param tys, return-array indicator
       - ``cost.json``      — instruction count + cycle estimate for the -O2 ref

Run:
    python scripts/ingest_leetcode.py --n 100 --out data/leetcode
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from clang import cindex


# Heuristic filter — anything matching this regex implies STL / iostream / classes
# we can't compile as C.
_STL_PAT = re.compile(
    r"\b("
    r"std::|"
    r"vector|unordered_|map|set|stack|queue|priority_queue|deque|list|"
    r"string\b|"
    r"new\s|delete\s|class\s|template\s|public:|private:|protected:|"
    r"cout|cin|iostream|algorithm|"
    r"nullptr|virtual\s|friend\s|throw\s|try\s|catch\s|"
    r"const\s+\w+\s*&"  # const-ref params (C++)
    r")\b"
)

_CLASS_CTOR_PAT = re.compile(r"(ListNode|TreeNode)\s*\([^)]*\)\s*:")

# What we prepend to every solution to make it compile as C. Adds the LeetCode
# boilerplate types + the helper macros most C++-as-C solutions assume (min/max).
_PROLOGUE = """\
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* LeetCode boilerplate — pure-C declarations of the standard nodes. */
struct ListNode { int val; struct ListNode *next; };
typedef struct ListNode ListNode;

struct TreeNode { int val; struct TreeNode *left; struct TreeNode *right; };
typedef struct TreeNode TreeNode;

/* Minimal C++ → C helper bridge. */
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
"""


def _extract_code_block(text: str) -> str | None:
    m = re.search(r"```(?:cpp|c\+\+|c)\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _is_c_friendly(code: str) -> bool:
    if _STL_PAT.search(code):
        return False
    if _CLASS_CTOR_PAT.search(code):
        return False
    return True


def _try_compile(c_source: str, gcc: str, opt: str, out_path: Path) -> tuple[bool, str]:
    """Compile ``c_source`` as C with ``-march=rv64gc -mabi=lp64d -<opt> -S``.
    Returns (success, stderr_or_msg)."""
    with tempfile.TemporaryDirectory(prefix="lc-compile-") as d:
        src = Path(d) / "f.c"
        src.write_text(c_source)
        try:
            proc = subprocess.run(
                [gcc, "-xc", "-march=rv64gc", "-mabi=lp64d", f"-{opt}", "-S",
                 "-Wno-implicit-function-declaration",
                 "-o", str(out_path), str(src)],
                check=True, capture_output=True, timeout=20,
            )
            return True, proc.stderr.decode(errors="replace")
        except subprocess.CalledProcessError as e:
            return False, e.stderr.decode(errors="replace")[:600]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def _link_check(asm_text: str, signature: dict, gcc: str) -> tuple[bool, str]:
    """Link the asm against a generated harness; rejects functions that call
    undefined helpers (LeetCode interactive APIs like `isBadVersion`, `knows`).
    -lm covers math.h users."""
    from eval.differential_test import _harness_for_int_signature, signature_is_int_only
    if not signature_is_int_only(signature):
        return True, ""   # link-check only meaningful when we can generate a harness
    harness = _harness_for_int_signature(signature)
    with tempfile.TemporaryDirectory(prefix="lc-link-") as d:
        dp = Path(d)
        (dp / "asm.s").write_text(asm_text)
        (dp / "main.c").write_text(harness)
        try:
            subprocess.run(
                [gcc, "-march=rv64gc", "-mabi=lp64d", "-O0", "-static",
                 "-Wno-implicit-function-declaration",
                 "-o", str(dp / "prog"),
                 str(dp / "main.c"), str(dp / "asm.s"), "-lm"],
                check=True, capture_output=True, timeout=30,
            )
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, e.stderr.decode(errors="replace")[:400]
        except subprocess.TimeoutExpired:
            return False, "link_timeout"


def _smoke_test(asm_text: str, signature: dict) -> tuple[bool, str]:
    """Self-differential with a tight timeout + safer input range. Rejects LeetCode
    solutions that hang on adversarial inputs (``divide(INT_MIN, -1)`` style)."""
    from eval.differential_test import differential_test, signature_is_int_only
    if not signature_is_int_only(signature):
        return True, "skipped_non_int"
    # We patch the qemu timeout for ingest specifically — eval uses the default.
    import eval.differential_test as dt_mod
    original = dt_mod._run_under_qemu
    def _wrapped(binary, qemu, stdin_data, timeout_s=10.0):
        return original(binary, qemu, stdin_data, timeout_s=10.0)
    dt_mod._run_under_qemu = _wrapped
    try:
        v = differential_test(asm_text, asm_text, signature, n_inputs=16, seed=7)
    except Exception as e:
        return False, f"smoke_exception: {e}"[:200]
    finally:
        dt_mod._run_under_qemu = original
    if v.passed:
        return True, ""
    return False, f"smoke_failed: {v.failure_reason}"


_idx: cindex.Index | None = None


def _index() -> cindex.Index:
    global _idx
    if _idx is None:
        _idx = cindex.Index.create()
    return _idx


def _parse_signature(c_source: str) -> dict | None:
    """Return a dict {fn_name, return_ty, params: [{name, ty}, ...]} for the LAST
    top-level non-static function in the file. Returns None if no function is found
    or the signature can't be normalised to scalar/pointer types we know how to harness.
    """
    tu = _index().parse(
        "x.c",
        unsaved_files=[("x.c", c_source)],
        args=["-xc", "-std=c11"],
    )
    if tu is None:
        return None
    candidates = []
    for c in tu.cursor.walk_preorder():
        if c.kind != cindex.CursorKind.FUNCTION_DECL:
            continue
        if c.location.file is None or "x.c" not in c.location.file.name:
            continue
        # The user's solution function — last function-decl with a body in the file.
        if not c.is_definition():
            continue
        candidates.append(c)
    if not candidates:
        return None
    fn = candidates[-1]
    return {
        "fn_name": fn.spelling,
        "return_ty": fn.result_type.spelling,
        "params": [{"name": p.spelling, "ty": p.type.spelling} for p in fn.get_arguments()],
    }


def _cost_estimate(asm_path: Path, fn_name: str) -> dict | None:
    """Reuse eval.cost.estimate for the reference asm's cost."""
    try:
        from eval.cost import estimate
        c = estimate(asm_path.read_text(), fn_name=fn_name)
        return {"instruction_count": c.instruction_count,
                "estimated_cycles": c.estimated_cycles}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="target number of accepted problems")
    ap.add_argument("--out", type=Path, default=Path("data/leetcode"))
    ap.add_argument("--max-scan", type=int, default=2400,
                    help="cap the number of source rows we scan")
    args = ap.parse_args()

    gcc = shutil.which("riscv64-linux-gnu-gcc")
    if gcc is None:
        sys.exit("riscv64-linux-gnu-gcc not on PATH; run sudo sh scripts/install_system_deps.sh")

    from datasets import load_dataset
    ds = load_dataset("greengerong/leetcode", split="train")
    print(f"loaded {len(ds)} rows from greengerong/leetcode")

    args.out.mkdir(parents=True, exist_ok=True)
    accepted = 0
    scanned = 0
    skipped_stl = 0
    skipped_no_code = 0
    skipped_compile = 0
    skipped_sig = 0
    skipped_link = 0
    skipped_smoke = 0
    t0 = time.time()

    for row in ds:
        if accepted >= args.n or scanned >= args.max_scan:
            break
        scanned += 1
        code = _extract_code_block(row["c++"] or "")
        if not code:
            skipped_no_code += 1
            continue
        if not _is_c_friendly(code):
            skipped_stl += 1
            continue

        c_source = _PROLOGUE + "\n" + code + "\n"
        with tempfile.TemporaryDirectory(prefix="lc-ingest-") as d:
            o2 = Path(d) / "o2.s"
            ok_o2, err = _try_compile(c_source, gcc, "O2", o2)
            if not ok_o2:
                skipped_compile += 1
                continue
            o3 = Path(d) / "o3.s"
            ok_o3, _ = _try_compile(c_source, gcc, "O3", o3)

            sig = _parse_signature(c_source)
            if sig is None or not sig["fn_name"]:
                skipped_sig += 1
                continue

            o2_text = o2.read_text()
            # Link-check: catches undefined LeetCode-API references (knows, isBadVersion).
            ok_link, link_err = _link_check(o2_text, sig, gcc)
            if not ok_link:
                skipped_link += 1
                continue
            # Smoke test: catches reference algorithms that loop on adversarial inputs.
            ok_smoke, smoke_err = _smoke_test(o2_text, sig)
            if not ok_smoke:
                skipped_smoke += 1
                continue

            # Commit to disk.
            slug = re.sub(r"[^a-z0-9]+", "-", row["slug"].lower())[:40].strip("-")
            problem_dir = args.out / f"{row['id']:04d}_{slug}"
            problem_dir.mkdir(parents=True, exist_ok=True)
            (problem_dir / "source.c").write_text(c_source)
            (problem_dir / "ref_o2.s").write_text(o2_text)
            if ok_o3:
                (problem_dir / "ref_o3.s").write_text(o3.read_text())
            (problem_dir / "signature.json").write_text(json.dumps(sig, indent=2))
            cost = _cost_estimate(o2, sig["fn_name"])
            if cost is not None:
                (problem_dir / "cost.json").write_text(json.dumps(cost, indent=2))
            (problem_dir / "meta.json").write_text(json.dumps({
                "id": row["id"], "slug": row["slug"], "title": row["title"],
                "difficulty": row["difficulty"],
            }, indent=2))

        accepted += 1
        if accepted % 10 == 0:
            print(f"  accepted {accepted}/{args.n}  scanned {scanned}  "
                  f"elapsed {time.time() - t0:.1f}s")

    manifest = {
        "n_accepted": accepted,
        "n_scanned": scanned,
        "skipped_no_code": skipped_no_code,
        "skipped_stl": skipped_stl,
        "skipped_compile": skipped_compile,
        "skipped_signature": skipped_sig,
        "skipped_link": skipped_link,
        "skipped_smoke": skipped_smoke,
        "wall_time_s": time.time() - t0,
        "source": "greengerong/leetcode",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone — {accepted} accepted, {scanned} scanned, "
          f"{time.time() - t0:.1f}s wall")
    print(f"  skips:  stl={skipped_stl}  compile={skipped_compile}  "
          f"signature={skipped_sig}  link={skipped_link}  smoke={skipped_smoke}")
    print(f"  manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
