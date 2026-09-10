"""Keep PyTensor compiling on a macOS toolchain whose linker rejects `-ld64`.

PyTensor adds `-ld64` to every C compile on macOS 15 and later to force the classic Apple
linker (pytensor/link/c/cmodule.py, `GCC_compiler.compile_args`). The Command Line Tools
for Xcode 27 (September 2026) dropped that linker and its flag, so the compile step fails
with "ld: library 'd64' not found" and every PyMC model build or logp evaluation dies.
PyTensor 3.3.1 still adds the flag unconditionally and offers no configuration switch
(pymc-devs/pytensor issue 1347). This module probes the active compiler once and strips the
flag only when the linker refuses it, so an older toolchain keeps its classic linker and a
newer one compiles with its own. Remove when PyTensor stops adding the flag.
"""
from __future__ import annotations

import functools
import subprocess
import sys
import tempfile
from pathlib import Path

LD64 = "-ld64"


@functools.cache
def linker_accepts_ld64(cxx: str) -> bool:
    """One trivial compile with `-ld64` under `cxx`; False when the linker rejects the flag."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cpp"
        src.write_text("int main() { return 0; }\\n")
        try:
            r = subprocess.run([cxx, str(src), LD64, "-o", str(Path(tmp) / "probe")],
                               capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return True                    # unknown compiler: leave PyTensor's flags alone
    return r.returncode == 0


def apply() -> bool:
    """Patch `GCC_compiler.compile_args` to drop `-ld64` when the linker rejects it.

    Idempotent; a no-op off macOS or when PyTensor is not installed. Returns whether the
    patch is in place.
    """
    if sys.platform != "darwin":
        return False
    try:
        import pytensor.link.c.cmodule as cm
    except ImportError:
        return False
    original = cm.GCC_compiler.compile_args
    if getattr(original, "_ld64_shim", False):
        return True

    def compile_args(*args, **kwargs):
        flags = original(*args, **kwargs)
        if LD64 in flags:
            from pytensor.configdefaults import config
            if not linker_accepts_ld64(config.cxx or "clang++"):
                flags = [f for f in flags if f != LD64]
        return flags

    compile_args._ld64_shim = True
    cm.GCC_compiler.compile_args = staticmethod(compile_args)
    return True
