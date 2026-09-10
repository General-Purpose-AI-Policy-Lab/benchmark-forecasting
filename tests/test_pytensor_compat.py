"""The PyTensor linker shim: the flag goes only when the linker refuses it."""
from __future__ import annotations

import sys

import pytest

from benchmark_forecasting import pytensor_compat


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS toolchain only")
def test_shim_strips_ld64_only_when_rejected(monkeypatch):
    import pytensor.link.c.cmodule as cm

    assert pytensor_compat.apply() is True                 # idempotent, already applied
    assert getattr(cm.GCC_compiler.compile_args, "_ld64_shim", False)
    monkeypatch.setattr(pytensor_compat, "linker_accepts_ld64", lambda cxx: False)
    assert "-ld64" not in cm.GCC_compiler.compile_args()
    monkeypatch.setattr(pytensor_compat, "linker_accepts_ld64", lambda cxx: True)
    flags = cm.GCC_compiler.compile_args()
    # PyTensor adds the flag on macOS 15+; with an accepting linker it stays.
    import platform
    assert ("-ld64" in flags) == (int(platform.mac_ver()[0].split(".")[0]) >= 15)


def test_probe_reports_a_real_compiler_result():
    pytensor_compat.compiler_accepts_flag.cache_clear()
    assert pytensor_compat.compiler_accepts_flag("definitely-not-a-compiler", "-O1") is True
    # A flag every compiler takes compiles; a library that does not exist fails to link.
    assert pytensor_compat.compiler_accepts_flag("clang++", "-O1") is True
    assert pytensor_compat.compiler_accepts_flag("clang++", "-lno_such_library_zz") is False
    assert isinstance(pytensor_compat.linker_accepts_ld64("clang++"), bool)
