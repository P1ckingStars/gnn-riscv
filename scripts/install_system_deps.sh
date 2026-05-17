#!/usr/bin/env bash
# One-shot installer for the system-level dependencies the project needs.
# Run as: sudo bash scripts/install_system_deps.sh
#
# Installs:
#   - gcc-riscv64-linux-gnu    (riscv64-linux-gnu-{gcc,as,ld,objdump,...} + glibc)
#   - qemu-user-static         (qemu-riscv64-static — Linux-user-mode simulator)
#   - clang                    (optional second baseline)
#
# Optional (not installed by default):
#   - gcc-riscv64-unknown-elf  (bare-metal cross-check; not needed when targeting
#                              Linux-user via qemu)
#   - spike (riscv-isa-sim): not in this Ubuntu's apt cache. Build from source at
#                            https://github.com/riscv-software-src/riscv-isa-sim if
#                            you want it as an ISA-spec cross-check.

set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "This script needs root — re-run with: sudo sh $0"
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    gcc-riscv64-linux-gnu \
    libc6-dev-riscv64-cross \
    qemu-user-static \
    clang

echo
echo "Done. Verify with: bash scripts/check_toolchain.sh"
