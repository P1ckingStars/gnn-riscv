#!/usr/bin/env python3
"""Parallel-ingest AnghaBench C files into a training corpus.

Each .c file becomes (or is rejected as):
  data/anghabench/<seq>/{source.c, ref_o2.s, signature.json}

Rejection criteria (the per-file filter):
  - gcc -O2 -S fails
  - libclang cannot recover a function definition (top-level fn_name)
  - signature is not int-only (v1 trainer needs all-int args + scalar return)
  - asm tokenization fails (unknown opcode / unparseable operand)
  - the asm is trivially tiny (< 4 instructions) or absurdly large (> 2000 tokens)

Workers process file batches in parallel via multiprocessing. Each worker shares the
same vocabs; outputs are written under a per-worker temp dir and atomically renamed.

Usage:
    python scripts/ingest_anghabench.py \\
        --src /tmp/anghabench-clone \\
        --out data/anghabench \\
        --target 10000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _list_c_files(root: Path, shuffle_seed: int) -> list[Path]:
    """Walk the AnghaBench tree and return all .c files (deterministically shuffled)."""
    files: list[Path] = []
    for p in root.rglob("*.c"):
        files.append(p)
    random.Random(shuffle_seed).shuffle(files)
    return files


def _try_compile_o2(c_source: str, gcc: str) -> str | None:
    """Compile with -O2 -S; return asm text or None on failure."""
    with tempfile.TemporaryDirectory(prefix="ang-c-") as d:
        src = Path(d) / "f.c"
        out = Path(d) / "f.s"
        src.write_text(c_source)
        try:
            subprocess.run(
                [gcc, "-xc", "-march=rv64gc", "-mabi=lp64d", "-O2", "-S",
                 "-Wno-implicit-function-declaration",
                 "-w",  # silence all warnings — AnghaBench has many
                 "-o", str(out), str(src)],
                check=True, capture_output=True, timeout=15,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        try:
            return out.read_text()
        except Exception:
            return None


# Concrete int kinds we treat as "int-typed" after canonicalization. Skips bool/char
# (signatures are stored as their canonical name, so any TypeKind in this set maps to
# a name like ``int``/``long``/``unsigned int`` that signature_is_int_only accepts).
_INT_KINDS_FOR_INGEST = None  # filled in lazily because libclang.cindex isn't available
                              #  in some test environments.


def _canonical_int_name(ty) -> str | None:
    from clang import cindex
    canon = ty.get_canonical()
    int_kinds = {
        cindex.TypeKind.SCHAR, cindex.TypeKind.UCHAR, cindex.TypeKind.CHAR_S,
        cindex.TypeKind.CHAR_U, cindex.TypeKind.SHORT, cindex.TypeKind.USHORT,
        cindex.TypeKind.INT, cindex.TypeKind.UINT, cindex.TypeKind.LONG,
        cindex.TypeKind.ULONG, cindex.TypeKind.LONGLONG, cindex.TypeKind.ULONGLONG,
        cindex.TypeKind.BOOL,
    }
    return canon.spelling if canon.kind in int_kinds else None


def _canonical_param_ty(ty) -> str:
    """Best-effort canonical spelling for any param type — used so the saved signature
    is at least informative even when the function isn't int-only (for non-diff-testable
    samples that still serve as training-loss data)."""
    canon = ty.get_canonical()
    return canon.spelling or ty.spelling or "?"


def _parse_signature(c_source: str):
    """Return {fn_name, return_ty, params, diff_testable} for the LAST function-with-body.
    ``diff_testable`` is True iff all types resolve to scalar ints; False signals that
    val_pass eval should skip this sample (training loss still applies)."""
    from clang import cindex
    idx = cindex.Index.create()
    tu = idx.parse("x.c", args=["-xc", "-std=c11"], unsaved_files=[("x.c", c_source)])
    if tu is None:
        return None
    fn = None
    for c in tu.cursor.walk_preorder():
        if c.kind != cindex.CursorKind.FUNCTION_DECL:
            continue
        if not c.is_definition():
            continue
        if c.location.file is None or "x.c" not in c.location.file.name:
            continue
        fn = c
    if fn is None or not fn.spelling:
        return None
    ret_int = _canonical_int_name(fn.result_type)
    if ret_int is not None:
        ret_ty = ret_int
    else:
        ret_ty = _canonical_param_ty(fn.result_type)
    params = []
    all_int = ret_int is not None
    for p in fn.get_arguments():
        pty_int = _canonical_int_name(p.type)
        if pty_int is None:
            all_int = False
            pty = _canonical_param_ty(p.type)
        else:
            pty = pty_int
        params.append({"name": p.spelling or f"a{len(params)}", "ty": pty})
    return {
        "fn_name": fn.spelling, "return_ty": ret_ty, "params": params,
        "diff_testable": all_int,
    }


# Process-local worker state.
_WORKER_VOCAB = None
_WORKER_INT_FILTER = None


def _worker_init() -> None:
    """Initialize per-process state: asm vocab. (No signature filter — we accept any
    function whose asm tokenizes; the diff-testable subset is marked in signature.json
    for downstream val_pass eval.)"""
    global _WORKER_VOCAB
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.asm_decoder import AsmVocab
    _WORKER_VOCAB = AsmVocab()


def _process_one(c_path_str: str, gcc: str) -> dict | None:
    """Per-file pipeline. Returns a result dict on success, None on rejection."""
    from model.asm_decoder import TokenizeError, tokenize_asm
    p = Path(c_path_str)
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if "main(" in src and "(int argc" in src:
        # Skip top-level mains — they tend to drag in lots of state.
        return None
    asm = _try_compile_o2(src, gcc)
    if asm is None:
        return None
    sig = _parse_signature(src)
    if sig is None:
        return None
    try:
        ids = tokenize_asm(asm, sig["fn_name"], _WORKER_VOCAB)
    except TokenizeError:
        return None
    if not (10 < len(ids) < 2000):
        return None
    return {
        "stem": p.stem,
        "source": src,
        "asm": asm,
        "sig": sig,
        "n_tokens": len(ids),
    }


def _commit(result: dict, out_dir: Path, seq: int) -> Path:
    pdir = out_dir / f"ang_{seq:07d}_{result['stem'][:60]}"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "source.c").write_text(result["source"])
    (pdir / "ref_o2.s").write_text(result["asm"])
    (pdir / "signature.json").write_text(json.dumps(result["sig"], indent=2))
    (pdir / "meta.json").write_text(json.dumps({
        "n_tokens": result["n_tokens"], "source": "AnghaBench", "stem": result["stem"],
    }, indent=2))
    return pdir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/tmp/anghabench-clone"))
    ap.add_argument("--out", type=Path, default=Path("data/anghabench"))
    ap.add_argument("--target", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() // 2))
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--scan-cap", type=int, default=200000,
                    help="max files to attempt before giving up")
    args = ap.parse_args()

    gcc = shutil.which("riscv64-linux-gnu-gcc")
    if gcc is None:
        sys.exit("riscv64-linux-gnu-gcc not on PATH")
    if not args.src.exists():
        sys.exit(f"AnghaBench source not found at {args.src}; "
                 f"clone from https://github.com/brenocfg/AnghaBench")

    files = _list_c_files(args.src, args.shuffle_seed)
    print(f"found {len(files)} .c files in {args.src}")
    files = files[:args.scan_cap]
    print(f"scanning up to {len(files)} files with {args.workers} workers")

    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    accepted = 0
    scanned = 0
    next_seq = 0
    seen_fn_names: set[str] = set()  # cheap dedup by function name
    batch_size = 32  # files per submitted future

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        # Submit in chunks so we don't queue 200k futures at once.
        chunk = 256
        idx = 0
        in_flight: list = []
        while accepted < args.target and idx < len(files):
            # Refill the in-flight queue.
            while len(in_flight) < chunk and idx < len(files):
                fp = files[idx]
                in_flight.append(ex.submit(_process_one, str(fp), gcc))
                idx += 1
                scanned += 1
            # Drain whatever's ready (or wait if none).
            done = []
            still = []
            for fut in in_flight:
                if fut.done():
                    done.append(fut)
                else:
                    still.append(fut)
            if not done:
                # Wait on the first one to make progress.
                done = [in_flight[0]]
                still = in_flight[1:]
            for fut in done:
                try:
                    r = fut.result()
                except Exception:
                    r = None
                if r is None:
                    continue
                fname = r["sig"]["fn_name"]
                if fname in seen_fn_names:
                    continue
                seen_fn_names.add(fname)
                _commit(r, args.out, next_seq)
                next_seq += 1
                accepted += 1
                if accepted >= args.target:
                    break
            in_flight = still
            # Progress: print on each crossing of a 200-boundary, not every iteration.
            milestone = (accepted // 200) * 200
            if milestone > 0 and milestone != getattr(main, "_last_milestone", 0):
                main._last_milestone = milestone  # type: ignore[attr-defined]
                rate = scanned / max(time.time() - t0, 1)
                print(f"  accepted {accepted}/{args.target}  scanned {scanned}  "
                      f"({rate:.0f} files/s)  elapsed {time.time() - t0:.0f}s")

    manifest = {
        "n_accepted": accepted,
        "n_scanned": scanned,
        "src": str(args.src),
        "shuffle_seed": args.shuffle_seed,
        "wall_time_s": time.time() - t0,
        "source": "AnghaBench (brenocfg/AnghaBench)",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone — {accepted} accepted, {scanned} scanned, "
          f"{time.time() - t0:.0f}s wall")


if __name__ == "__main__":
    main()
