# marimo-cython

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/cemrehancavdar/marimo-cython/blob/main/examples/mandelbrot.py)

Cython compilation support for [marimo](https://marimo.io) notebooks. Write Cython-accelerated functions directly in notebook cells and run them at native speed — no build system boilerplate required.

This is the marimo equivalent of Jupyter's `%%cython` magic, redesigned for marimo's reactive DAG model.

## Installation

Requires Python 3.10+.

```bash
uv add marimo-cython
```

Or with pip:

```bash
pip install marimo-cython
```

Runtime dependencies: `cython>=3.0` and `setuptools` (for C extension building).

## Quick start

```python
import cython
from marimo_cython import cy

@cy.compile
def fib(n: cython.int) -> cython.int:
    a: cython.int = 0
    b: cython.int = 1
    i: cython.int
    for i in range(n):
        a, b = b, a + b
    return a

fib(50)  # runs compiled native code
```

## API

Three entry points, all available via `cy.*` or as direct imports from `marimo_cython`:

### `@cy.compile` — decorator

Compiles a pure-Python Cython function. The function must use Cython's [pure Python syntax](https://cython.readthedocs.io/en/latest/src/tutorial/pure.html) (`cython.int`, `cython.double`, typed memoryviews, etc.).

```python
@cy.compile
def fib(n: cython.int) -> cython.int:
    a: cython.int = 0
    b: cython.int = 1
    i: cython.int
    for i in range(n):
        a, b = b, a + b
    return a
```

With options:

```python
@cy.compile(boundscheck=False, wraparound=False, cdivision=True)
def mandelbrot(
    out: cython.int[:, :],
    xmin: cython.double,
    xmax: cython.double,
    ymin: cython.double,
    ymax: cython.double,
    max_iter: cython.int,
) -> None:
    rows: cython.Py_ssize_t = out.shape[0]
    cols: cython.Py_ssize_t = out.shape[1]
    # ...
```

#### Using C library functions

To call C library functions (like `sqrt`, `fabs`, etc.) from a decorated function, use `from cython.cimports.*` at cell/module level above the function. Cython provides runtime Python stubs for these, so they are valid Python imports:

```python
from cython.cimports.libc.math import sqrt, fabs

@cy.compile(boundscheck=False)
def fast_sqrt(x: cython.double) -> cython.double:
    return sqrt(fabs(x))
```

The decorator automatically picks up `from cython.cimports.*` imports written above the function in the same cell or file, and includes them in the compiled module.

As a fallback, the `cimports` parameter accepts raw `cimport` strings (which use Cython syntax, not valid Python):

```python
@cy.compile(cimports=["from libc.math cimport sqrt, fabs"])
def fast_sqrt(x: cython.double) -> cython.double:
    return sqrt(fabs(x))
```

### `cy.compile_module(source)` — compile a string

Compiles a Cython source string into a module. Supports full `.pyx` syntax including `cdef`, `cimport`, and typed function signatures.

```python
mod = cy.compile_module("""
from libc.math cimport sqrt

def norm(double[:] v):
    cdef Py_ssize_t i
    cdef double s = 0.0
    for i in range(v.shape[0]):
        s += v[i] * v[i]
    return sqrt(s)
""", boundscheck=False, wraparound=False)

mod.norm(np.array([3.0, 4.0]))  # => 5.0
```

### `cy.compile_file(path)` — compile a `.pyx` file

```python
mod = cy.compile_file("solver.pyx", cplus=True, libraries=["lapack"])
mod.solve(matrix)
```

### `cy.clear_cache()` — remove compiled artifacts

```python
cy.clear_cache()  # removes .marimo_cython_cache/ and all contents
cy.clear_cache(cache_dir="custom_cache/")  # custom cache directory
```

Returns the number of files removed.

## Compile options

All three entry points accept the same keyword arguments. These map to the `CompileOptions` dataclass:

### C/C++ compilation

| Option | Type | Default | Description |
|---|---|---|---|
| `cplus` | `bool` | `False` | Compile as C++ instead of C |
| `compile_args` | `list[str]` | `[]` | Extra compiler flags (e.g. `["-O3", "-march=native"]`) |
| `link_args` | `list[str]` | `[]` | Extra linker flags |
| `libraries` | `list[str]` | `[]` | Libraries to link against |
| `include_dirs` | `list[str]` | `[]` | Additional include directories |
| `library_dirs` | `list[str]` | `[]` | Additional library directories |
| `runtime_library_dirs` | `list[str]` | `[]` | Runtime library search paths |
| `define_macros` | `list[tuple]` | `[]` | Preprocessor macros |
| `undef_macros` | `list[str]` | `[]` | Macros to undefine |
| `extra_objects` | `list[str]` | `[]` | Extra object files to link |

### Cython compiler directives

| Option | Type | Default | Description |
|---|---|---|---|
| `boundscheck` | `bool` | `None` | Array bounds checking |
| `wraparound` | `bool` | `None` | Negative indexing support |
| `cdivision` | `bool` | `None` | C-style division (no zero-division check) |
| `nonecheck` | `bool` | `None` | Check for None on extension types |
| `initializedcheck` | `bool` | `None` | Check memoryview initialization |
| `overflowcheck` | `bool` | `None` | Integer overflow checking |
| `infer_types` | `bool` | `None` | Automatic type inference |
| `profile` | `bool` | `None` | Enable profiling hooks |
| `linetrace` | `bool` | `None` | Enable line tracing |
| `embedsignature` | `bool` | `None` | Embed function signatures in docstrings |
| `emit_code_comments` | `bool` | `None` | Emit source comments in generated C |
| `freethreading_compatible` | `bool` | `None` | Mark as free-threading compatible |
| `language_level` | `int\|str` | `3` | Cython language level |
| `compiler_directives` | `dict` | `{}` | Catch-all for any Cython directive |

`None` means "use Cython's default". Set to `True`/`False` to override.

### Other options

| Option | Type | Default | Description |
|---|---|---|---|
| `module_name` | `str` | auto | Name for the compiled module |
| `cimports` | `list[str]\|str` | `[]` | Fallback: raw `cimport` statements (for `@cy.compile`) |
| `annotate` | `bool\|str` | `False` | Generate Cython annotation HTML |
| `nthreads` | `int` | `0` | Parallel cythonization threads |
| `force` | `bool` | `False` | Force recompilation (ignore cache) |
| `cache_dir` | `str\|Path` | `.marimo_cython_cache/` | Build artifact directory |

## How it works

### Pipeline

All three APIs converge into the same path: normalize source, compute cache key, build or load.

```
Source → Normalize → Cache check → Cythonize (.pyx → .c) → Build (.c → .so) → Load
```

### Source extraction (`@cy.compile`)

The decorator uses `inspect.getsource(fn)` to get the function source. This works in marimo because marimo populates `linecache` for exec'd cell code (the same mechanism Python uses for tracebacks — not a private API). Decorator lines are stripped before compilation.

### Cimport context recovery

`inspect.getsource` only returns the function body. `from cython.cimports.*` imports written above the function in the same cell are not included. To recover them, `_collect_context_cimports` AST-parses all lines above the function definition, finds `ImportFrom` nodes matching `cython.cimports.*`, and prepends them to the source.

### Auto `import cython`

`import cython` is auto-prepended if missing (needed for `cython.int`, `cython.double` annotations). Detection uses `ast.parse`, not regex. Critically, `from cython.cimports.*` does NOT count as a cython import — it imports C symbols, not the `cython` module. Getting this wrong causes `cython.double` to fail with "Unknown type declaration".

### Content-addressed caching

The cache key is a SHA-256 of: source (post-normalization), all C/C++ flags, all compiler directives, `EXT_SUFFIX` (Python ABI), and `Cython.__version__`. Module name becomes `{prefix}_{hash[:16]}`. On hit, the `.so` loads directly via `importlib`.

### Build

On cache miss: write `.pyx` → `cythonize()` to `.c` → `setuptools` `build_ext` to `.so` → load with `importlib`. Build temp uses a short-lived `tempfile.mkdtemp` to avoid path-length issues. Intermediate `.c`/`.cpp` files are cleaned after success; `.pyx` is kept for debugging.

### Other details

- **Numpy**: auto-detected from source; `numpy.get_include()` added to `include_dirs` automatically
- **Linker warnings**: `-Wl,-w` suppresses CPython 3.14's stale `Modules/_hacl` path warnings
- **`CythonModule`**: wrapper that delegates attribute access to the compiled module and supports marimo rich display (`_mime_()`) for annotation HTML
- **Cache dir**: `.marimo_cython_cache/` by default (add to `.gitignore`)

## Annotations

Compile with `annotate=True` to get Cython's annotation HTML rendered in marimo cell output:

```python
mod = cy.compile_module(source, annotate=True)
mod  # displays annotation HTML in marimo
```

## Cross-cell interaction

Compiled modules and functions work across marimo cells like any other Python object. Define a module in one cell and call its functions from another:

```python
# Cell 1
linalg = cy.compile_module("""
from libc.math cimport sqrt
def norm(double[:] v):
    ...
""")

# Cell 2 — uses linalg from Cell 1
result = linalg.norm(my_array)
```

## Running the demos

```bash
uv sync --extra dev
uv run marimo edit demo.py
```

The demo covers the core API: fibonacci benchmark, cimport patterns, `compile_module`, and cross-cell interaction.

### Examples

The `examples/` directory contains standalone notebooks:

- **`examples/mandelbrot.py`** — Interactive Mandelbrot set with a resolution slider. Runs Cython and Python side-by-side with progressive rendering (Cython result appears instantly, Python fills in when done). PEP 723 compatible — run with `uv run --sandbox examples/mandelbrot.py`.

```bash
uv sync --extra dev
uv run marimo run examples/mandelbrot.py
```

## Development

```bash
uv sync --extra dev
uv run ruff check src/
uv run pytest
```

## License

MIT
