"""Cython compilation engine for marimo notebooks.

Three ways to use Cython in marimo:

1. @cy.compile decorator — write pure Python Cython, get compiled version:

    @cy.compile
    def fib(n: cython.int) -> cython.int:
        a: cython.int = 0
        b: cython.int = 1
        i: cython.int
        for i in range(n):
            a, b = b, a + b
        return a

    fib(10)  # runs compiled Cython

2. cy.compile_module(source) — compile a string of Cython code:

    mod = cy.compile_module('''
    import cython
    def fib(n: cython.int) -> cython.int: ...
    ''')
    mod.fib(10)

3. cy.compile_file("path.pyx") — compile a .pyx file:

    mod = cy.compile_file("fast_math.pyx")
    mod.fib(10)
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
import re
import shutil
import sys
import sysconfig
import textwrap
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, overload

import Cython
from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".marimo_cython_cache"

# Suppress linker warnings about stale search paths (e.g. CPython 3.14's
# 'Modules/_hacl' path that leaks into extension builds).
_SUPPRESS_LD_WARNINGS = ["-Wl,-w"]

_ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
if _ext_suffix is None:
    raise RuntimeError(
        "sysconfig.get_config_var('EXT_SUFFIX') returned None. "
        "This means Python cannot determine the file extension for C extensions "
        "(e.g. '.cpython-314-darwin.so' on macOS, '.pyd' on Windows). "
        "Your Python installation may be incomplete or misconfigured."
    )
EXT_SUFFIX: str = _ext_suffix

F = TypeVar("F", bound=Callable[..., object])
T = TypeVar("T", bound=Callable[..., object])


class CythonCompileError(Exception):
    """Raised when Cython compilation or C/C++ build fails."""


class CythonModule:
    """Wrapper around a compiled Cython module.

    Delegates attribute access to the underlying module so you can call
    functions directly: ``mod.my_function(args)``.

    If compiled with ``annotate=True``, displaying this object in marimo
    renders the Cython annotation HTML.
    """

    def __init__(self, module: types.ModuleType, annotation_html: str | None = None) -> None:
        self._module = module
        self._annotation_html = annotation_html

    def __getattr__(self, name: str) -> object:
        return getattr(self._module, name)

    def __dir__(self) -> list[str]:
        return [*dir(self._module), "_module", "_annotation_html"]

    def __repr__(self) -> str:
        return f"<CythonModule '{self._module.__name__}'>"

    def _mime_(self) -> tuple[str, str]:
        """Marimo rich display."""
        if self._annotation_html is not None:
            return ("text/html", self._annotation_html)
        return ("text/plain", repr(self))


@dataclass(frozen=True)
class CompileOptions:
    """All compilation options in one place.

    Covers: Extension params, cythonize params, and compiler directives.
    Passed through @cy.compile, cy.compile_module, and cy.compile_file.
    """

    # --- Module identity ---
    module_name: str | None = None

    # --- C/C++ compilation (Extension params) ---
    cplus: bool = False
    compile_args: list[str] = field(default_factory=list)
    link_args: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    library_dirs: list[str] = field(default_factory=list)
    runtime_library_dirs: list[str] = field(default_factory=list)
    define_macros: list[tuple[str, str | None]] = field(default_factory=list)
    undef_macros: list[str] = field(default_factory=list)
    extra_objects: list[str] = field(default_factory=list)

    # --- Cython compiler directives ---
    # Language level
    language_level: int | str = 3

    # Performance (the most commonly tuned)
    boundscheck: bool | None = None
    wraparound: bool | None = None
    cdivision: bool | None = None
    nonecheck: bool | None = None
    initializedcheck: bool | None = None
    overflowcheck: bool | None = None
    infer_types: bool | None = None

    # Profiling / debugging
    profile: bool | None = None
    linetrace: bool | None = None
    embedsignature: bool | None = None
    emit_code_comments: bool | None = None

    # GIL
    freethreading_compatible: bool | None = None

    # Catch-all for any directive not listed above
    compiler_directives: dict[str, Any] = field(default_factory=dict)

    # --- Module-level cimports (for @cy.compile) ---
    cimports: list[str] | str = field(default_factory=list)

    # --- Cythonize options ---
    annotate: bool | str = False
    nthreads: int = 0
    force: bool = False

    # --- Cache ---
    cache_dir: str | Path | None = None

    def build_directives(self) -> dict[str, Any]:
        """Merge explicit directive fields + compiler_directives dict."""
        directives: dict[str, Any] = {"language_level": self.language_level}

        # Named directive fields — only include if explicitly set (not None)
        _named = {
            "boundscheck": self.boundscheck,
            "wraparound": self.wraparound,
            "cdivision": self.cdivision,
            "nonecheck": self.nonecheck,
            "initializedcheck": self.initializedcheck,
            "overflowcheck": self.overflowcheck,
            "infer_types": self.infer_types,
            "profile": self.profile,
            "linetrace": self.linetrace,
            "embedsignature": self.embedsignature,
            "emit_code_comments": self.emit_code_comments,
            "freethreading_compatible": self.freethreading_compatible,
        }
        for k, v in _named.items():
            if v is not None:
                directives[k] = v

        # User-provided dict overrides everything
        directives.update(self.compiler_directives)
        return directives

    def cache_key(self, source: str) -> str:
        """Deterministic hash from source + all options that affect output.

        Includes everything that changes the compiled binary:
        - Source code (already has cimports prepended by the caller)
        - C/C++ compilation flags and linked libraries
        - Cython compiler directives
        - Python ABI (via EXT_SUFFIX, e.g. '.cpython-314-darwin.so')
        - Cython version

        Excludes things that don't affect output:
        - nthreads (parallelism only, same result)
        - sys.executable (venv path irrelevant if ABI matches)
        - cimports (already baked into source by caller)
        """
        parts = [
            source,
            str(self.cplus),
            str(self.compile_args),
            str(self.link_args),
            str(self.libraries),
            str(self.include_dirs),
            str(self.library_dirs),
            str(self.runtime_library_dirs),
            str(self.define_macros),
            str(self.undef_macros),
            str(self.extra_objects),
            str(self.build_directives()),
            EXT_SUFFIX,
            Cython.__version__,  # type: ignore[attr-defined]
        ]
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def _options_from_kwargs(**kwargs: Any) -> CompileOptions:
    """Build CompileOptions from keyword arguments, validating field names."""
    known = {f.name for f in CompileOptions.__dataclass_fields__.values()}
    unknown = set(kwargs) - known
    if unknown:
        msg = f"Unknown compile options: {unknown}. Valid options: {sorted(known)}"
        raise TypeError(msg)
    return CompileOptions(**kwargs)


# ---------------------------------------------------------------------------
# Internal: caching, building
# ---------------------------------------------------------------------------


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    path = Path(cache_dir) if cache_dir else Path(DEFAULT_CACHE_DIR)
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_extension(cache_dir: Path, module_name: str) -> Path | None:
    """Find the compiled extension file (.so/.pyd) for a module in the cache."""
    ext_path = cache_dir / f"{module_name}{EXT_SUFFIX}"
    if ext_path.exists():
        return ext_path
    for p in cache_dir.rglob(f"{module_name}{EXT_SUFFIX}"):
        return p
    return None


def _load_module(module_name: str, ext_path: Path) -> types.ModuleType:
    """Load a compiled extension module from disk."""
    # Remove stale entry so Python doesn't return a cached version
    # (e.g. after force=True recompilation or source change).
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, ext_path)
    if spec is None or spec.loader is None:
        msg = f"Failed to create module spec from {ext_path}"
        raise CythonCompileError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_extension(extensions: list[Extension], build_dir: Path) -> None:
    # Use a short temp dir for build_temp to avoid path-length issues.
    # setuptools derives .o paths from the .c source path relative to build_temp;
    # when both are long absolute paths, the resulting nested path can exceed
    # OS limits or fail to be auto-created by the compiler.
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="_cy_build_")
    try:
        dist = Distribution({"ext_modules": extensions})
        dist.parse_config_files()
        cmd = build_ext(dist)
        cmd.build_lib = str(build_dir)
        cmd.build_temp = temp_dir
        cmd.inplace = False
        cmd.ensure_finalized()
        cmd.run()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _read_annotation(cache_dir: Path, module_name: str) -> str | None:
    html_path = cache_dir / f"{module_name}.html"
    if not html_path.exists():
        return None
    raw = html_path.read_text(encoding="utf-8")
    return _clean_annotation_html(raw)


# Patterns for annotation HTML cleaning
_STYLE_RE = re.compile(r"<style.*?</style>", re.DOTALL)
_BODY_RE = re.compile(r"<body[^>]*>(.+)</body>", re.DOTALL)
_RAW_OUTPUT_RE = re.compile(r'<p>Raw output: <a href="[^"]*">[^<]*</a></p>')


def _clean_annotation_html(html: str) -> str:
    """Strip the full-document wrapper from Cython annotation HTML.

    Cython generates a complete HTML document with <!DOCTYPE>, <html>, <head>,
    <body> tags. When embedding in marimo cell output, we only want the CSS
    styles and the body content — not a nested document.

    Also removes the "Raw output" link to the generated .c file, which is
    not meaningful in a notebook context.

    Mirrors Jupyter's CythonMagics.clean_annotated_html().
    """
    chunks: list[str] = []

    # Extract <style> blocks
    chunks.extend(_STYLE_RE.findall(html))

    # Extract body content
    body_match = _BODY_RE.search(html)
    if body_match is None:
        # Not a full HTML document — return as-is
        return html
    body = body_match.group(1)

    # Remove "Raw output: <a href="...">...</a>" link
    body = _RAW_OUTPUT_RE.sub("", body)

    chunks.append(body)
    return "\n".join(chunks)


def _auto_numpy_includes(source: str) -> list[str]:
    if "numpy" in source or "cimport numpy" in source:
        try:
            import numpy

            return [numpy.get_include()]
        except ImportError:
            logger.warning("Source references numpy but numpy is not installed")
    return []


def _has_cython_import(source: str) -> bool:
    """Check whether source already contains ``import cython`` or ``from cython import ...``.

    Only matches imports that make the ``cython`` name available for type
    annotations (``cython.int``, ``cython.double``, etc.).  ``from
    cython.cimports.*`` does NOT qualify — it imports C symbols, not the
    ``cython`` module itself.

    Uses ``ast.parse`` so it ignores comments and string literals.
    Returns False on syntax errors (e.g. raw .pyx with cdef syntax).
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # .pyx files with cdef syntax won't parse — fall back to False
        # so the caller prepends the import (harmless duplicate).
        return False

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            # ``import cython`` or ``import cython as cy``
            if any(alias.name == "cython" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            # ``from cython import ...`` but NOT ``from cython.cimports.*``
            if node.module == "cython":
                return True
    return False


def _ensure_cython_import(source: str) -> str:
    """Prepend 'import cython' if the source doesn't already import it."""
    if _has_cython_import(source):
        return source
    return "import cython\n" + source


def _cleanup_build_artifacts(cache_dir: Path, module_name: str) -> None:
    """Remove intermediate .c/.cpp files after successful build.

    Keeps only the .pyx source (for debugging) and the .so/.pyd binary.
    Annotation .html is kept if it exists.
    Build temp dirs are cleaned up in _build_extension itself.
    """
    for ext in (".c", ".cpp"):
        c_file = cache_dir / f"{module_name}{ext}"
        if c_file.exists():
            c_file.unlink()


# ---------------------------------------------------------------------------
# Public: cache management
# ---------------------------------------------------------------------------


def clear_cache(cache_dir: str | Path | None = None) -> int:
    """Remove all compiled artifacts from the cache directory.

    Returns the number of files removed.
    """
    path = Path(cache_dir) if cache_dir else Path(DEFAULT_CACHE_DIR)
    path = path.resolve()
    if not path.exists():
        return 0

    count = sum(1 for _ in path.rglob("*") if _.is_file())
    shutil.rmtree(path)
    logger.info("Cleared cache: %s (%d files)", path, count)
    return count


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def _normalize_cimports(cimports: list[str] | str) -> list[str]:
    """Normalize cimports to a list of import strings."""
    if isinstance(cimports, str):
        cimports = [cimports]
    return [line.strip() for line in cimports if line.strip()]


def _prepend_cimports(source: str, cimports: list[str] | str) -> str:
    """Prepend module-level cimport statements to source."""
    lines = _normalize_cimports(cimports)
    if not lines:
        return source
    return "\n".join(lines) + "\n" + source


def _prepare_source(source: str, opts: CompileOptions) -> tuple[str, str, str]:
    """Prepare source code and compute module identity.

    Returns (source, module_name, cache_key).
    """
    source = textwrap.dedent(source).strip()
    source = _ensure_cython_import(source)
    source = _prepend_cimports(source, opts.cimports)

    key = opts.cache_key(source)
    prefix = opts.module_name or "_cy"
    mod_name = f"{prefix}_{key}"

    return source, mod_name, key


def _build_and_load(
    source: str, mod_name: str, opts: CompileOptions, cache_path: Path
) -> CythonModule:
    """Cythonize, compile, and load a module from source. Cleans up intermediates."""
    # Auto-detect numpy includes
    inc_dirs = list(opts.include_dirs) + _auto_numpy_includes(source)

    # Write .pyx source
    pyx = cache_path / f"{mod_name}.pyx"
    pyx.write_text(source, encoding="utf-8")

    # Build Extension with all params
    ext = Extension(
        name=mod_name,
        sources=[str(pyx)],
        include_dirs=inc_dirs,
        library_dirs=list(opts.library_dirs),
        runtime_library_dirs=list(opts.runtime_library_dirs),
        libraries=list(opts.libraries),
        define_macros=list(opts.define_macros),
        undef_macros=list(opts.undef_macros),
        extra_objects=list(opts.extra_objects),
        extra_compile_args=list(opts.compile_args),
        extra_link_args=list(opts.link_args) + _SUPPRESS_LD_WARNINGS,
        language="c++" if opts.cplus else "c",
    )

    # Cythonize
    try:
        extensions = cythonize(
            [ext],
            compiler_directives=opts.build_directives(),
            annotate=opts.annotate,
            nthreads=opts.nthreads,
            force=True,  # We handle caching ourselves
            quiet=True,
        )
    except Exception as exc:
        raise CythonCompileError(f"Cython transpilation failed:\n{exc}") from exc

    # Build native extension
    try:
        _build_extension(extensions, cache_path)
    except Exception as exc:
        raise CythonCompileError(f"C/C++ compilation failed:\n{exc}") from exc

    ext_path = _find_extension(cache_path, mod_name)
    if ext_path is None:
        raise CythonCompileError(
            f"Build succeeded but {mod_name}{EXT_SUFFIX} not found in {cache_path}"
        )

    # Move to cache root if nested (setuptools may put it in a subdirectory)
    target = cache_path / f"{mod_name}{EXT_SUFFIX}"
    if ext_path != target:
        ext_path.rename(target)
        ext_path = target

    # Clean intermediate .c/.cpp and temp build dirs
    _cleanup_build_artifacts(cache_path, mod_name)

    mod = _load_module(mod_name, ext_path)
    html = _read_annotation(cache_path, mod_name) if opts.annotate else None
    logger.info("Compiled: %s", mod_name)
    return CythonModule(mod, html)


def _compile_source(source: str, opts: CompileOptions) -> CythonModule:
    """Compile a Cython source string into a loaded module."""
    source, mod_name, _key = _prepare_source(source, opts)
    cache_path = _resolve_cache_dir(opts.cache_dir)

    # Cache hit
    if not opts.force:
        ext_path = _find_extension(cache_path, mod_name)
        if ext_path is not None:
            logger.info("Cache hit: %s", ext_path)
            mod = _load_module(mod_name, ext_path)
            html = _read_annotation(cache_path, mod_name) if opts.annotate else None
            return CythonModule(mod, html)

    return _build_and_load(source, mod_name, opts, cache_path)


# ---------------------------------------------------------------------------
# Public API: compile (decorator), compile_module, compile_file
# ---------------------------------------------------------------------------


@overload
def compile(fn: F, /, **kwargs: Any) -> F: ...
@overload
def compile(fn: None = None, /, **kwargs: Any) -> Callable[[F], F]: ...


def compile(fn: F | None = None, /, **kwargs: Any) -> F | Callable[[F], F]:
    """Decorator that compiles a pure-Python-Cython function with Cython.

    The decorated function must use Cython pure Python syntax (cython.int,
    @cython.cfunc, etc.). The function source is extracted, compiled as a
    Cython module, and the compiled version replaces the original.

    Accepts ALL CompileOptions as keyword arguments::

        @cy.compile
        def fib(n: cython.int) -> cython.int: ...

        @cy.compile(boundscheck=False, wraparound=False)
        def fast_sum(data: cython.double[:]) -> cython.double: ...

        @cy.compile(cplus=True, compile_args=["-O3", "-march=native"])
        def heavy(x: cython.double) -> cython.double: ...

        @cy.compile(compiler_directives={"cdivision": True, "profile": True})
        def custom(x: cython.int) -> cython.int: ...

        @cy.compile(cimports=["from libc.math cimport sqrt, fabs"])
        def fast_sqrt(x: cython.double) -> cython.double:
            return sqrt(fabs(x))
    """

    def decorator(func: T) -> T:
        opts = _options_from_kwargs(module_name=func.__name__, **kwargs)
        source = _extract_function_source(func)
        mod = _compile_source(source, opts)
        compiled_fn = getattr(mod, func.__name__)
        compiled_fn.__doc__ = func.__doc__
        compiled_fn.__module__ = func.__module__
        compiled_fn.__qualname__ = func.__qualname__
        return compiled_fn  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator


def compile_module(source: str, **kwargs: Any) -> CythonModule:
    """Compile a Cython source string into a module.

    Automatically prepends ``import cython`` if not present.
    Accepts ALL CompileOptions as keyword arguments.

    Examples::

        # Basic
        mod = cy.compile_module('''
        def fib(n: cython.int) -> cython.int: ...
        ''')

        # With directives
        mod = cy.compile_module('''
        def fast(data):
            ...
        ''', boundscheck=False, wraparound=False, cdivision=True)

        # C++ mode with libraries
        mod = cy.compile_module(src, cplus=True, libraries=["stdc++"])

        # Full .pyx syntax (not just pure Python)
        mod = cy.compile_module('''
        cdef double square(double x):
            return x * x

        def py_square(double x):
            return square(x)
        ''')
    """
    opts = _options_from_kwargs(**kwargs)
    return _compile_source(source, opts)


def compile_file(path: str | Path, **kwargs: Any) -> CythonModule:
    """Compile a .pyx or .py Cython file into a module.

    Accepts ALL CompileOptions as keyword arguments.

    Examples::

        mod = cy.compile_file("fast_math.pyx")
        mod = cy.compile_file("solver.pyx", cplus=True, libraries=["lapack"])
        mod = cy.compile_file("tight_loop.pyx", boundscheck=False, wraparound=False)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cython source file not found: {p}")
    source = p.read_text(encoding="utf-8")
    if "module_name" not in kwargs:
        kwargs["module_name"] = p.stem
    opts = _options_from_kwargs(**kwargs)
    return _compile_source(source, opts)


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def _extract_function_source(fn: Callable[..., object]) -> str:
    """Extract a function's source code, including cython.cimports from context.

    Works in marimo cells because marimo populates ``linecache`` for exec'd code,
    which is a fundamental requirement for tracebacks — unlikely to change.

    Also scans lines above the function in the same file/cell for
    ``from cython.cimports.*`` imports and prepends them, since
    inspect.getsource only returns the function body itself.
    """
    try:
        source = inspect.getsource(fn)
    except OSError:
        name = getattr(fn, "__name__", repr(fn))
        raise OSError(
            f"Cannot retrieve source for '{name}'. "
            "inspect.getsource() failed — this can happen if the function is "
            "defined dynamically (exec/eval) without populating linecache. "
            "Use cy.compile_module(source_string) instead."
        ) from None

    source = textwrap.dedent(source)

    # Strip decorator line(s) — everything before the 'def' line
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            source = "".join(lines[i:])
            break

    # Collect cython.cimports from lines above the function in the source file.
    # inspect.getsource only returns the function — cimports written at cell/module
    # level are not included and need to be recovered from the surrounding context.
    context_cimports = _collect_context_cimports(fn)
    if context_cimports:
        source = "\n".join(context_cimports) + "\n" + source

    return source


def _collect_context_cimports(fn: Callable[..., object]) -> list[str]:
    """Collect ``from cython.cimports.*`` imports from lines above fn in its source file.

    Uses ``ast.parse`` to reliably detect imports regardless of formatting
    (single-line, multi-line parenthesized, aliased, etc.) and ignoring
    comments and string literals.

    Returns import statements reconstructed from the AST, e.g.::

        ["from cython.cimports.libc.math import sqrt, fabs"]
    """
    import ast
    import linecache

    try:
        source_file = inspect.getfile(fn)
        _, start_line = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return []

    all_lines = linecache.getlines(source_file)
    if not all_lines:
        return []

    # Parse lines above the function definition
    above_source = "".join(all_lines[: start_line - 1])
    try:
        tree = ast.parse(above_source)
    except SyntaxError:
        return []

    cimports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if not module.startswith("cython.cimports."):
            continue

        # Reconstruct the import statement from AST
        names = ", ".join(
            f"{alias.name} as {alias.asname}" if alias.asname else alias.name
            for alias in node.names
        )
        cimports.append(f"from {module} import {names}")

    return cimports
