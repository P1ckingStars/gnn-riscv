"""Produce reference RV64GC assembly for a spec by compiling its C lowering with
gcc/clang. Uses the Linux-gnu toolchain (riscv64-linux-gnu-gcc) so the same C source
flows cleanly into ``eval.verify_io`` for linked execution under qemu-riscv64-static.

Returns the full ``-S`` output (directives + label + instructions). Use
``eval.cost.estimate`` to count instructions in the function body alone.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from spec.dsl import Spec
from spec.lower_to_c import to_c


class ToolMissing(RuntimeError):
    """Raised when a required external tool is not on PATH."""


def _require(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise ToolMissing(f"required tool '{tool}' not found on PATH")
    return p


def gcc_o2(spec: Spec, fn_name: str = "spec_fn") -> str:
    """Compile spec → C → asm via ``riscv64-linux-gnu-gcc -O2 -S``."""
    gcc = _require("riscv64-linux-gnu-gcc")
    src = to_c(spec, fn_name)
    with tempfile.TemporaryDirectory(prefix="gnnriscv-baseline-") as d:
        dp = Path(d)
        (dp / "f.c").write_text(src)
        out = dp / "f.s"
        subprocess.run(
            [gcc, "-march=rv64gc", "-mabi=lp64d", "-O2", "-S",
             "-o", str(out), str(dp / "f.c")],
            check=True, capture_output=True,
        )
        return out.read_text()


def clang_o2(spec: Spec, fn_name: str = "spec_fn") -> str:
    """Compile spec → C → asm via ``clang --target=riscv64-linux-gnu -march=rv64gc -O2 -S``."""
    clang = _require("clang")
    src = to_c(spec, fn_name)
    with tempfile.TemporaryDirectory(prefix="gnnriscv-baseline-") as d:
        dp = Path(d)
        (dp / "f.c").write_text(src)
        out = dp / "f.s"
        subprocess.run(
            [clang, "--target=riscv64-linux-gnu", "-march=rv64gc", "-mabi=lp64d",
             "-O2", "-S", "-o", str(out), str(dp / "f.c")],
            check=True, capture_output=True,
        )
        return out.read_text()
