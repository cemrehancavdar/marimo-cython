import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import cython
    import marimo as mo
    import numpy as np

    from marimo_cython import cy

    return cy, cython, mo, np


@app.cell
def _(mo):
    mo.md("""
    # marimo-cython demo

    Three ways to use Cython in marimo:

    1. `@cy.compile` — decorator for pure Python Cython functions
    2. `cy.compile_module(source)` — compile a string (full .pyx syntax)
    3. `cy.compile_file("path.pyx")` — compile a .pyx file
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. `@cy.compile` — the simplest case
    """)
    return


@app.cell
def _(cy, cython, mo):
    import time as _time

    @cy.compile
    def fib(n: cython.int) -> cython.int:
        a: cython.int = 0
        b: cython.int = 1
        i: cython.int
        for i in range(n):
            a, b = b, a + b
        return a

    def py_fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    # Benchmark
    _n = 80
    _rounds = 10_000

    _s = _time.perf_counter()
    for _ in range(_rounds):
        py_fib(_n)
    _py_time = _time.perf_counter() - _s

    _s = _time.perf_counter()
    for _ in range(_rounds):
        fib(_n)
    _cy_time = _time.perf_counter() - _s

    mo.md(f"""
    `fib(30)` = {fib(30)}

    | | {_rounds:,} calls of fib({_n}) |
    |---|---|
    | Python | {_py_time:.3f}s |
    | Cython | {_cy_time:.3f}s |
    | **Speedup** | **{_py_time / _cy_time:.0f}x** |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Using C library functions
    """)
    return


@app.cell
def _(cy, cython, mo):
    from cython.cimports.libc.math import sqrt

    @cy.compile
    def hypotenuse(a: cython.double, b: cython.double) -> cython.double:
        return abs(sqrt(a * a + b * b))

    mo.md(f"""
    `from cython.cimports.libc.math import sqrt` at cell level gives
    the decorated function access to C's `sqrt`.

    `hypotenuse(3, 4)` = {hypotenuse(3.0, 4.0)}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. `compile_module` — full .pyx syntax
    """)
    return


@app.cell
def _(cy, mo, np):
    mathlib = cy.compile_module(
        """
    from libc.math cimport sqrt, fabs

    def norm(double[:] v):
        cdef Py_ssize_t i
        cdef double s = 0.0
        for i in range(v.shape[0]):
            s += v[i] * v[i]
        return sqrt(s)

    def l1_norm(double[:] v):
        cdef Py_ssize_t i
        cdef double s = 0.0
        for i in range(v.shape[0]):
            s += fabs(v[i])
        return s
    """,
        boundscheck=False,
        wraparound=False,
    )

    _v = np.array([3.0, 4.0], dtype=np.float64)

    mo.md(f"""
    `compile_module` accepts full Cython `.pyx` syntax — `cdef`, `cimport`,
    typed arguments — things that can't appear in a Python function body.

    - `norm([3, 4])` = {mathlib.norm(_v)}
    - `l1_norm([3, 4])` = {mathlib.l1_norm(_v)}

    The returned module works across cells like any Python object.
    """)
    return (mathlib,)


@app.cell
def _(mathlib, mo, np):
    _w = np.array([1.0, -2.0, 3.0], dtype=np.float64)

    mo.md(f"""
    ## 4. Cross-cell usage

    Using `mathlib` compiled in the cell above:

    `mathlib.norm([1, -2, 3])` = {mathlib.norm(_w):.4f}
    """)
    return


if __name__ == "__main__":
    app.run()
