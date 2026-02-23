"""Cython compilation support for marimo notebooks.

Usage::

    from marimo_cython import cy

    @cy.compile
    def fib(n: cython.int) -> cython.int:
        ...

    # or
    mod = cy.compile_module("def fib(...): ...")
    mod = cy.compile_file("fast_math.pyx")
"""

from __future__ import annotations

from marimo_cython import cy
from marimo_cython.compiler import (
    CompileOptions,
    CythonCompileError,
    CythonModule,
    clear_cache,
    compile,
    compile_file,
    compile_module,
)

__all__ = [
    "cy",
    "compile",
    "compile_file",
    "compile_module",
    "clear_cache",
    "CompileOptions",
    "CythonCompileError",
    "CythonModule",
]
