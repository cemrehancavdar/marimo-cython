"""Tests for marimo_cython compiler module."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import cython  # noqa: F401 — needed for type annotations in compiled test functions
import pytest

from marimo_cython.compiler import (
    CompileOptions,
    CythonCompileError,
    CythonModule,
    _clean_annotation_html,
    _collect_context_cimports,
    _ensure_cython_import,
    _has_cython_import,
    _normalize_cimports,
    _options_from_kwargs,
    _prepend_cimports,
    clear_cache,
    compile,
    compile_file,
    compile_module,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMP_CACHE = Path(tempfile.mkdtemp(prefix="marimo_cython_test_"))


@pytest.fixture(autouse=True)
def _clean_test_cache() -> None:
    """Each test gets a fresh cache directory."""
    if TEMP_CACHE.exists():
        shutil.rmtree(TEMP_CACHE)
    TEMP_CACHE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CompileOptions
# ---------------------------------------------------------------------------


class TestCompileOptions:
    def test_defaults(self) -> None:
        opts = CompileOptions()
        assert opts.cplus is False
        assert opts.boundscheck is None
        assert opts.language_level == 3

    def test_build_directives_excludes_none(self) -> None:
        opts = CompileOptions(boundscheck=False, wraparound=False)
        d = opts.build_directives()
        assert d["boundscheck"] is False
        assert d["wraparound"] is False
        assert "cdivision" not in d
        assert "nonecheck" not in d

    def test_build_directives_catch_all_overrides(self) -> None:
        opts = CompileOptions(
            boundscheck=True,
            compiler_directives={"boundscheck": False, "custom_flag": 42},
        )
        d = opts.build_directives()
        # compiler_directives dict overrides named fields
        assert d["boundscheck"] is False
        assert d["custom_flag"] == 42

    def test_cache_key_deterministic(self) -> None:
        opts = CompileOptions(boundscheck=False)
        k1 = opts.cache_key("def f(): pass")
        k2 = opts.cache_key("def f(): pass")
        assert k1 == k2

    def test_cache_key_differs_on_source(self) -> None:
        opts = CompileOptions()
        k1 = opts.cache_key("def f(): pass")
        k2 = opts.cache_key("def g(): pass")
        assert k1 != k2

    def test_cache_key_differs_on_options(self) -> None:
        src = "def f(): pass"
        k1 = CompileOptions(boundscheck=False).cache_key(src)
        k2 = CompileOptions(boundscheck=True).cache_key(src)
        assert k1 != k2

    def test_cache_key_differs_on_cplus(self) -> None:
        src = "def f(): pass"
        k1 = CompileOptions(cplus=False).cache_key(src)
        k2 = CompileOptions(cplus=True).cache_key(src)
        assert k1 != k2


# ---------------------------------------------------------------------------
# _options_from_kwargs validation
# ---------------------------------------------------------------------------


class TestOptionsFromKwargs:
    def test_valid_kwargs(self) -> None:
        opts = _options_from_kwargs(boundscheck=False, cplus=True)
        assert opts.boundscheck is False
        assert opts.cplus is True

    def test_unknown_kwarg_raises(self) -> None:
        with pytest.raises(TypeError, match="Unknown compile options"):
            _options_from_kwargs(nonexistent_option=True)

    def test_error_message_lists_valid_options(self) -> None:
        with pytest.raises(TypeError, match="boundscheck"):
            _options_from_kwargs(bad_option=True)


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------


class TestSourceHelpers:
    def test_ensure_cython_import_prepends(self) -> None:
        result = _ensure_cython_import("def f(): pass")
        assert result.startswith("import cython\n")

    def test_ensure_cython_import_noop_if_present(self) -> None:
        src = "import cython\ndef f(): pass"
        result = _ensure_cython_import(src)
        assert result.count("import cython") == 1

    def test_ensure_cython_import_noop_from_import(self) -> None:
        src = "from cython import int as cint\ndef f(): pass"
        result = _ensure_cython_import(src)
        assert not result.startswith("import cython\n")

    def test_has_cython_import_ignores_comment(self) -> None:
        # Regex would false-positive on this; AST correctly ignores it
        assert _has_cython_import("# import cython\ndef f(): pass") is False

    def test_has_cython_import_ignores_string(self) -> None:
        assert _has_cython_import('s = "import cython"\ndef f(): pass') is False

    def test_has_cython_import_cimports_does_not_count(self) -> None:
        # from cython.cimports.* imports C symbols, not the cython module itself.
        # cython.double etc. still need ``import cython`` separately.
        assert _has_cython_import("from cython.cimports.libc.math import sqrt") is False

    def test_has_cython_import_pyx_syntax_returns_false(self) -> None:
        # .pyx files with cdef won't parse — should return False (not crash)
        assert _has_cython_import("cdef double x = 1.0") is False

    def test_normalize_cimports_string(self) -> None:
        result = _normalize_cimports("from libc.math cimport sqrt")
        assert result == ["from libc.math cimport sqrt"]

    def test_normalize_cimports_list(self) -> None:
        result = _normalize_cimports(["a", "b", ""])
        assert result == ["a", "b"]

    def test_normalize_cimports_empty_string(self) -> None:
        assert _normalize_cimports("") == []

    def test_normalize_cimports_empty_list(self) -> None:
        assert _normalize_cimports([]) == []

    def test_prepend_cimports(self) -> None:
        result = _prepend_cimports("def f(): pass", ["from libc.math cimport sqrt"])
        assert result.startswith("from libc.math cimport sqrt\n")

    def test_prepend_cimports_noop_empty(self) -> None:
        src = "def f(): pass"
        assert _prepend_cimports(src, []) == src


# ---------------------------------------------------------------------------
# _collect_context_cimports (AST-based)
# ---------------------------------------------------------------------------


def _make_module_with_fn(tmp_path: Path, source: str) -> Callable[..., object]:
    """Write source to a .py file, import it, and return the 'target' function."""
    import importlib.util

    mod_file = tmp_path / "_ctx_test.py"
    mod_file.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_ctx_test", mod_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.target  # type: ignore[attr-defined]


class TestCollectContextCimports:
    def test_single_cimport(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            "from cython.cimports.libc.math import sqrt\n\ndef target(x):\n    return x\n",
        )
        result = _collect_context_cimports(fn)
        assert result == ["from cython.cimports.libc.math import sqrt"]

    def test_multiple_cimports(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            (
                "from cython.cimports.libc.math import sqrt\n"
                "from cython.cimports.libc.math import fabs\n"
                "\ndef target(x):\n    return x\n"
            ),
        )
        result = _collect_context_cimports(fn)
        assert result == [
            "from cython.cimports.libc.math import sqrt",
            "from cython.cimports.libc.math import fabs",
        ]

    def test_aliased_import(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            "from cython.cimports.libc.math import sqrt as c_sqrt\n"
            "\ndef target(x):\n    return x\n",
        )
        result = _collect_context_cimports(fn)
        assert result == ["from cython.cimports.libc.math import sqrt as c_sqrt"]

    def test_multi_name_import(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            "from cython.cimports.libc.math import sqrt, fabs, pow\n"
            "\ndef target(x):\n    return x\n",
        )
        result = _collect_context_cimports(fn)
        assert result == ["from cython.cimports.libc.math import sqrt, fabs, pow"]

    def test_parenthesized_multiline_import(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            (
                "from cython.cimports.libc.math import (\n"
                "    sqrt,\n"
                "    fabs,\n"
                ")\n"
                "\ndef target(x):\n    return x\n"
            ),
        )
        result = _collect_context_cimports(fn)
        assert result == ["from cython.cimports.libc.math import sqrt, fabs"]

    def test_ignores_regular_imports(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            "import os\nfrom math import sqrt\n\ndef target(x):\n    return x\n",
        )
        result = _collect_context_cimports(fn)
        assert result == []

    def test_ignores_commented_cimport(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            "# from cython.cimports.libc.math import sqrt\n\ndef target(x):\n    return x\n",
        )
        result = _collect_context_cimports(fn)
        assert result == []

    def test_ignores_cimport_in_string(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            (
                'EXAMPLE = "from cython.cimports.libc.math import sqrt"\n'
                "\ndef target(x):\n    return x\n"
            ),
        )
        result = _collect_context_cimports(fn)
        assert result == []

    def test_ignores_imports_after_function(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            ("def target(x):\n    return x\n\nfrom cython.cimports.libc.math import sqrt\n"),
        )
        result = _collect_context_cimports(fn)
        # Only looks at lines *above* the function
        assert result == []

    def test_returns_empty_for_builtin(self) -> None:
        # Built-in functions don't have source files
        result = _collect_context_cimports(len)  # type: ignore[arg-type]
        assert result == []

    def test_mixed_imports(self, tmp_path: Path) -> None:
        fn = _make_module_with_fn(
            tmp_path,
            (
                "import os\n"
                "from cython.cimports.libc.math import sqrt\n"
                "from math import pi\n"
                "from cython.cimports.libc.math import fabs\n"
                "\ndef target(x):\n    return x\n"
            ),
        )
        result = _collect_context_cimports(fn)
        assert result == [
            "from cython.cimports.libc.math import sqrt",
            "from cython.cimports.libc.math import fabs",
        ]


# ---------------------------------------------------------------------------
# Annotation HTML cleaning
# ---------------------------------------------------------------------------


class TestAnnotationHtml:
    def test_passthrough_non_html_document(self) -> None:
        fragment = "<div>hello</div>"
        assert _clean_annotation_html(fragment) == fragment

    def test_strips_document_wrapper(self) -> None:
        html = (
            "<!DOCTYPE html><html><head>"
            "<style>.cython { color: red; }</style>"
            "</head><body><div>content</div></body></html>"
        )
        result = _clean_annotation_html(html)
        assert "<!DOCTYPE" not in result
        assert "<html>" not in result
        assert "<style>" in result
        assert "<div>content</div>" in result

    def test_removes_raw_output_link(self) -> None:
        html = (
            '<html><body><p>Raw output: <a href="mod.c">mod.c</a></p><div>code</div></body></html>'
        )
        result = _clean_annotation_html(html)
        assert "Raw output" not in result
        assert "<div>code</div>" in result


# ---------------------------------------------------------------------------
# compile_module
# ---------------------------------------------------------------------------


class TestCompileModule:
    def test_basic_function(self) -> None:
        mod = compile_module(
            """
            def square(x: cython.double) -> cython.double:
                return x * x
            """,
            cache_dir=str(TEMP_CACHE),
        )
        assert isinstance(mod, CythonModule)
        assert mod.square(7.0) == 49.0

    def test_cdef_syntax(self) -> None:
        mod = compile_module(
            """
            cdef double _square(double x):
                return x * x

            def square(double x):
                return _square(x)
            """,
            cache_dir=str(TEMP_CACHE),
        )
        assert mod.square(5.0) == 25.0

    def test_cache_hit(self) -> None:
        src = """
        def add(a: cython.double, b: cython.double) -> cython.double:
            return a + b
        """
        mod1 = compile_module(src, cache_dir=str(TEMP_CACHE))
        mod2 = compile_module(src, cache_dir=str(TEMP_CACHE))
        # Both should work (second from cache)
        assert mod1.add(1.0, 2.0) == 3.0
        assert mod2.add(1.0, 2.0) == 3.0

    def test_force_recompile(self) -> None:
        src = """
        def val() -> cython.int:
            return 42
        """
        mod1 = compile_module(src, cache_dir=str(TEMP_CACHE))
        mod2 = compile_module(src, cache_dir=str(TEMP_CACHE), force=True)
        assert mod1.val() == 42
        assert mod2.val() == 42

    def test_boundscheck_wraparound(self) -> None:
        mod = compile_module(
            """
            def first(lst: cython.double[:]) -> cython.double:
                return lst[0]
            """,
            boundscheck=False,
            wraparound=False,
            cache_dir=str(TEMP_CACHE),
        )
        import numpy as np

        arr = np.array([3.14], dtype=np.float64)
        assert mod.first(arr) == pytest.approx(3.14)

    def test_cimport_libc(self) -> None:
        mod = compile_module(
            """
            from libc.math cimport sqrt

            def my_sqrt(double x):
                return sqrt(x)
            """,
            cache_dir=str(TEMP_CACHE),
        )
        assert mod.my_sqrt(9.0) == pytest.approx(3.0)

    def test_module_name(self) -> None:
        mod = compile_module(
            """
            def noop():
                pass
            """,
            module_name="custom_name",
            cache_dir=str(TEMP_CACHE),
        )
        assert "custom_name" in repr(mod)

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(CythonCompileError, match="Cython transpilation failed"):
            compile_module(
                """
                this is not valid cython or python!!!
                """,
                cache_dir=str(TEMP_CACHE),
            )

    def test_repr(self) -> None:
        mod = compile_module(
            "def f(): pass",
            cache_dir=str(TEMP_CACHE),
        )
        assert repr(mod).startswith("<CythonModule '")

    def test_dir_includes_module_attrs(self) -> None:
        mod = compile_module(
            "def my_func(): return 1",
            cache_dir=str(TEMP_CACHE),
        )
        d = dir(mod)
        assert "my_func" in d
        assert "_module" in d

    def test_mime_plain_text(self) -> None:
        mod = compile_module(
            "def f(): pass",
            cache_dir=str(TEMP_CACHE),
        )
        mime, content = mod._mime_()
        assert mime == "text/plain"
        assert "CythonModule" in content


# ---------------------------------------------------------------------------
# @compile decorator
# ---------------------------------------------------------------------------


class TestCompileDecorator:
    def test_bare_decorator(self) -> None:
        @compile
        def fib(n: cython.int) -> cython.int:  # noqa: F821
            a: cython.int = 0  # noqa: F821
            b: cython.int = 1  # noqa: F821
            i: cython.int  # noqa: F821
            for i in range(n):
                a, b = b, a + b
            return a

        assert fib(10) == 55
        assert fib(0) == 0

    def test_decorator_with_kwargs(self) -> None:
        @compile(boundscheck=False, wraparound=False, cache_dir=str(TEMP_CACHE))
        def double_it(x: cython.double) -> cython.double:  # noqa: F821
            return x * 2.0

        assert double_it(3.5) == 7.0

    def test_preserves_metadata(self) -> None:
        @compile(cache_dir=str(TEMP_CACHE))
        def documented(x: cython.int) -> cython.int:  # noqa: F821
            """This function is documented."""
            return x

        assert documented.__doc__ == "This function is documented."
        assert (
            documented.__qualname__
            == "TestCompileDecorator.test_preserves_metadata.<locals>.documented"
        )


# ---------------------------------------------------------------------------
# compile_file
# ---------------------------------------------------------------------------


class TestCompileFile:
    def test_compile_pyx_file(self, tmp_path: Path) -> None:
        pyx = tmp_path / "mathlib.pyx"
        pyx.write_text(
            "def add(double a, double b):\n    return a + b\n",
            encoding="utf-8",
        )
        mod = compile_file(str(pyx), cache_dir=str(TEMP_CACHE))
        assert mod.add(2.0, 3.0) == 5.0

    def test_auto_module_name_from_stem(self, tmp_path: Path) -> None:
        pyx = tmp_path / "mymod.pyx"
        pyx.write_text("def val(): return 1\n", encoding="utf-8")
        mod = compile_file(str(pyx), cache_dir=str(TEMP_CACHE))
        assert "mymod" in repr(mod)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            compile_file("nonexistent.pyx", cache_dir=str(TEMP_CACHE))


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clear_populated_cache(self) -> None:
        compile_module("def f(): pass", cache_dir=str(TEMP_CACHE))
        count = clear_cache(cache_dir=str(TEMP_CACHE))
        assert count > 0
        assert not TEMP_CACHE.exists()

    def test_clear_nonexistent_cache(self) -> None:
        nonexistent = TEMP_CACHE / "does_not_exist"
        count = clear_cache(cache_dir=str(nonexistent))
        assert count == 0


# ---------------------------------------------------------------------------
# Build artifact cleanup
# ---------------------------------------------------------------------------


class TestBuildCleanup:
    def test_no_c_files_after_build(self) -> None:
        compile_module("def f(): pass", cache_dir=str(TEMP_CACHE))
        c_files = list(TEMP_CACHE.glob("*.c"))
        assert len(c_files) == 0, f"Intermediate .c files not cleaned: {c_files}"

    def test_no_build_temp_after_build(self) -> None:
        compile_module("def f(): pass", cache_dir=str(TEMP_CACHE))
        # Build temp is created in system temp dir and cleaned up automatically,
        # so it should never appear in the cache directory.
        temp_dirs = list(TEMP_CACHE.glob("_build_temp*"))
        assert len(temp_dirs) == 0, f"Build temp dirs found in cache: {temp_dirs}"

    def test_pyx_and_so_remain(self) -> None:
        compile_module("def f(): pass", cache_dir=str(TEMP_CACHE))
        pyx_files = list(TEMP_CACHE.glob("*.pyx"))
        so_files = list(TEMP_CACHE.glob("*" + ".so")) + list(TEMP_CACHE.glob("*.pyd"))
        assert len(pyx_files) > 0, "Source .pyx file should be kept"
        assert len(so_files) > 0, "Compiled .so/.pyd file should be kept"
