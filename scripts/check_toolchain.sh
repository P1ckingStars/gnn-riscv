#!/usr/bin/env bash
# Verify external dependencies needed by the gnn-riscv pipeline.
# Reports OK / MISSING for each; exits non-zero if any required tool is absent.
#
# Toolchain policy: riscv64-linux-gnu-gcc (Linux ELF + glibc) is the primary
# compiler — pairs cleanly with qemu-riscv64-static for I/O-driven verification.
# The bare-metal unknown-elf toolchain and spike are optional cross-checks.

set -u

required=(
    "riscv64-linux-gnu-gcc:RV64 Linux cross-compiler (Ubuntu: apt install gcc-riscv64-linux-gnu)"
    "riscv64-linux-gnu-as:RV64 assembler (ships with the gcc package)"
    "qemu-riscv64-static:RV64 user-mode emulator (Ubuntu: apt install qemu-user-static)"
    "python3:Python 3.11+"
)

optional=(
    "qemu-riscv64:Non-static qemu (apt install qemu-user; not required if -static is present)"
    "riscv64-unknown-elf-gcc:Bare-metal RV64 toolchain (optional cross-check)"
    "spike:Reference RISC-V ISA simulator (build from source; optional cross-check)"
    "clang:Second baseline asm source (Ubuntu: apt install clang)"
)

py_required=(z3 torch torch_geometric numpy)

fail=0

check_bin () {
    local name="$1" desc="$2" required="$3"
    if command -v "$name" >/dev/null 2>&1; then
        printf "  [ OK     ] %-32s %s\n" "$name" "$(command -v "$name")"
    else
        if [[ "$required" == "required" ]]; then
            printf "  [ MISSING ] %-32s %s\n" "$name" "$desc"
            fail=1
        else
            printf "  [ OPT     ] %-32s %s\n" "$name" "$desc"
        fi
    fi
}

check_py () {
    local mod="$1"
    if python3 -c "import $mod" >/dev/null 2>&1; then
        local ver
        ver=$(python3 -c "import $mod; print(getattr($mod, '__version__', '?'))" 2>/dev/null)
        printf "  [ OK     ] python:%-25s %s\n" "$mod" "$ver"
    else
        printf "  [ MISSING ] python:%-25s pip install -e .[dev]\n" "$mod"
        fail=1
    fi
}

echo "== Binaries =="
for entry in "${required[@]}"; do
    check_bin "${entry%%:*}" "${entry#*:}" required
done
for entry in "${optional[@]}"; do
    check_bin "${entry%%:*}" "${entry#*:}" optional
done

echo
echo "== Python modules =="
for mod in "${py_required[@]}"; do
    check_py "$mod"
done

echo
if [[ "$fail" -ne 0 ]]; then
    echo "Toolchain check FAILED — install missing items above."
    echo "On Ubuntu: sudo bash scripts/install_system_deps.sh"
    exit 1
fi
echo "Toolchain check OK."
