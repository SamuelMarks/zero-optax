"""Unit tests for all modules in `scripts/` to ensure 100% line and branch coverage."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from scripts import (
    check_api_compliance,
    check_compliance,
    check_deps,
    generate_api_snapshot,
    lint_typing_and_docs,
    update_badges,
)

NL = chr(10)


def _load_module_from_source(name: str, source: str, tmp_path: Path) -> ModuleType:
    """Create and import a temporary Python module from `source` without mutating `__dict__`.

    Args:
        name: Module name.
        source: Python source code string.
        tmp_path: Temporary directory path.

    Returns:
        Loaded `ModuleType` instance.
    """
    mod_file = tmp_path / f"{name}.py"
    mod_file.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, mod_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_deps_script(tmp_path: Path) -> None:
    """Test `scripts/check_deps.py` across allowed, disallowed, syntax error, and missing directory cases.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert check_deps._get_stdlib_modules({"sys", "os"}) == {"sys", "os"}
    assert "sys" in check_deps._get_stdlib_modules()
    assert check_deps.get_base_module("zero_optax.losses.dice") == "zero_optax"
    assert check_deps.main(src_dir=Path("src")) == 0
    assert check_deps.check_directory(tmp_path / "nonexistent") == []

    bad_dir = tmp_path / "src_bad"
    bad_dir.mkdir()
    (bad_dir / "tests").mkdir()
    (bad_dir / "tests" / "ignored.py").write_text(f"import torch{NL}", encoding="utf-8")
    (bad_dir / "syntax_err.py").write_text(f"def broken(:{NL}", encoding="utf-8")
    (bad_dir / "bad_imports.py").write_text(
        NL.join(
            [
                "import torch",
                "from scipy.special import erf",
                "from .local import x",
                "import sys",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert check_deps.main(src_dir=bad_dir) == 1


def test_check_api_compliance_and_generate_snapshot(tmp_path: Path) -> None:
    """Test `scripts/check_api_compliance.py` and `scripts/generate_api_snapshot.py`.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert check_api_compliance.main() == 0

    empty_mod = ModuleType("empty_mod")
    _, impl, pct, _ = check_api_compliance.compute_api_compliance(empty_mod, empty_mod)
    assert impl == 0
    assert pct == 100.0

    snap_out = tmp_path / "snap.json"
    snap = generate_api_snapshot.generate_snapshot(output_path=snap_out)
    assert "root" in snap
    assert snap_out.exists()

    custom_mod = _load_module_from_source(
        "custom_mod",
        NL.join(
            [
                '__all__ = ["missing_symbol", "len_alias"]',
                "len_alias = len",
                "",
            ]
        ),
        tmp_path,
    )
    api_custom = generate_api_snapshot.get_public_api(custom_mod)
    assert "missing_symbol" not in api_custom


def test_check_compliance_script(tmp_path: Path) -> None:
    """Test `scripts/check_compliance.py` across all branches.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert check_compliance.check_compliance() == 0

    missing_plan = tmp_path / "NO_PLAN.md"
    assert check_compliance.parse_semantic_plan(missing_plan) == {}
    assert check_compliance.check_compliance(plan_path=missing_plan) == 0

    empty_plan = tmp_path / "EMPTY_PLAN.md"
    empty_plan.write_text(f"# Empty Plan{NL}No items here.{NL}", encoding="utf-8")
    assert check_compliance.check_compliance(plan_path=empty_plan) == 0

    plan_with_mod = tmp_path / "MOD_PLAN.md"
    todo_out = tmp_path / "OPTAX_TODO.md"
    plan_with_mod.write_text(
        NL.join(
            [
                "## Phase 1: `optax.losses`",
                "- [x] `squared_error`: ok",
                "- [x] `not_here_in_losses`: missing in valid module",
                "## Phase 2: `optax.nonexistent_sub`",
                "- [x] `missing_fn`: missing in invalid module",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert (
        check_compliance.check_compliance(plan_path=plan_with_mod, todo_path=todo_out)
        == 1
    )
    assert todo_out.exists()

    valid_plan = tmp_path / "VALID_PLAN.md"
    valid_todo = tmp_path / "VALID_TODO.md"
    valid_plan.write_text(
        NL.join(
            [
                "## Phase 1",
                "- [x] `adam`: ok",
                "## Phase 2: `optax.empty_sec`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert (
        check_compliance.check_compliance(plan_path=valid_plan, todo_path=valid_todo)
        == 1
    )
    assert (
        check_compliance.check_compliance(plan_path=valid_plan, todo_path=valid_todo)
        == 0
    )

    assert check_compliance.verify_snapshot_v028(tmp_path / "does_not_exist.json") == []
    assert check_compliance.parse_todo_plan_unchecked(tmp_path / "no_todo.md") == set()
    todo_sample = tmp_path / "sample_todo.md"
    todo_sample.write_text(
        NL.join(
            [
                "- [" + " " + "] Implement `zero_optax.tree_utils.tree_add`",
                "- [" + " " + "] Implement `optax.projections.projection_box`",
                "- [x] Implement `optax.adam`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert check_compliance.parse_todo_plan_unchecked(todo_sample) == {
        "optax.tree_utils.tree_add",
        "optax.projections.projection_box",
    }
    loaded_sub = check_compliance._load_target_submodules(
        module_names=("optax", "optax.nonexistent_sub")
    )
    assert "optax" in loaded_sub and "optax.nonexistent_sub" not in loaded_sub
    resolved_fallback = check_compliance._resolve_default_snapshot_path(
        cache_path=tmp_path / "missing_cache_snap.json"
    )
    assert resolved_fallback.name == "optax_v0.2.8.json"
    skipped_snap = tmp_path / "skipped_snap.json"
    skipped_snap.write_text(
        json.dumps(
            {
                "categories": {
                    "array": [
                        {
                            "api_path": "optax.tree_utils.tree_add",
                            "params": [],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        check_compliance.verify_snapshot_v028(
            snapshot_path=skipped_snap, todo_plan_path=todo_sample
        )
        == []
    )

    bad_snap = tmp_path / "bad_snap.json"
    bad_snap.write_text(json.dumps(["not_a_dict"]), encoding="utf-8")
    assert check_compliance.verify_snapshot_v028(bad_snap) == []

    bad_cats = tmp_path / "bad_cats.json"
    bad_cats.write_text(json.dumps({"categories": "not_a_dict"}), encoding="utf-8")
    assert check_compliance.verify_snapshot_v028(bad_cats) == []

    mismatch_snap = tmp_path / "mismatch_snap.json"
    mismatch_snap.write_text(
        json.dumps(
            {
                "categories": {
                    "bad_cat": "not_a_list",
                    "loss": [
                        "not_a_dict",
                        {
                            "api_path": "optax.no_such_fn",
                            "aliases": "not_a_list",
                            "name": "no_such_fn",
                            "params": [],
                        },
                        {
                            "api_path": "optax.missing_sub.fn",
                            "aliases": ["optax.missing_sub.fn_alias"],
                            "name": "fn",
                            "params": [],
                        },
                        {
                            "api_path": "optax.__all__",
                            "name": "__all__",
                            "params": [],
                        },
                        {
                            "api_path": "optax.adam",
                            "name": "adam",
                            "params": "not_a_list",
                        },
                        {"api_path": "optax.adam", "name": "adam", "params": []},
                        {
                            "api_path": "optax.losses.hinge_loss",
                            "name": "hinge_loss",
                            "params": [
                                "not_a_dict",
                                {"name": "wrong_name", "kind": "KEYWORD_ONLY"},
                            ],
                        },
                        {
                            "api_path": "optax.schedules.constant_schedule",
                            "name": "constant_schedule",
                            "params": [
                                {"name": "value", "kind": "POSITIONAL_OR_KEYWORD"},
                            ],
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    errs = check_compliance.verify_snapshot_v028(mismatch_snap)
    assert len(errs) >= 5
    assert (
        check_compliance.check_compliance(
            plan_path=valid_plan, todo_path=valid_todo, snapshot_path=mismatch_snap
        )
        == 1
    )


def test_update_badges_script(tmp_path: Path) -> None:
    """Test `scripts/update_badges.py` across all color, formatting, and file branches.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert update_badges.get_color(100.0) == "brightgreen"
    assert update_badges.get_color(95.0) == "green"
    assert update_badges.get_color(85.0) == "yellowgreen"
    assert update_badges.get_color(75.0) == "yellow"
    assert update_badges.get_color(65.0) == "orange"
    assert update_badges.get_color(50.0) == "red"

    assert update_badges.format_cov(100.0) == "100"
    assert update_badges.format_cov(98.5) == "98.5"

    missing_cov = tmp_path / "no_cov.json"
    assert update_badges.get_test_coverage(missing_cov, run_cmd=False) == 100.0
    assert (
        update_badges.get_test_coverage(
            missing_cov, run_cmd=True, coverage_cmd=["true"]
        )
        == 100.0
    )

    valid_cov = tmp_path / "cov.json"
    valid_cov.write_text(
        json.dumps({"totals": {"percent_covered": 99.5}}), encoding="utf-8"
    )
    assert update_badges.get_test_coverage(valid_cov, run_cmd=False) == 99.5

    corrupt_cov = tmp_path / "corrupt.json"
    corrupt_cov.write_text("{bad json", encoding="utf-8")
    assert update_badges.get_test_coverage(corrupt_cov, run_cmd=False) == 0.0

    assert update_badges.get_doc_coverage(tmp_path / "no_src") == 100.0
    empty_src = tmp_path / "empty_src"
    empty_src.mkdir()
    assert update_badges.get_doc_coverage(empty_src) == 100.0
    assert update_badges.get_doc_coverage(Path("src/zero_optax")) == 100.0

    undoc_src = tmp_path / "undoc_src"
    undoc_src.mkdir()
    (undoc_src / "m.py").write_text(f"class A:{NL}    pass{NL}", encoding="utf-8")
    assert update_badges.get_doc_coverage(undoc_src) == 0.0

    assert update_badges.get_api_compliance(tmp_path) == 0.0
    empty_plan_dir = tmp_path / "empty_plan_dir"
    empty_plan_dir.mkdir()
    (empty_plan_dir / "SEMANTIC_PLAN.md").write_text(
        f"# No checkboxes{NL}", encoding="utf-8"
    )
    assert update_badges.get_api_compliance(empty_plan_dir) == 0.0
    bad_todo = tmp_path / "TEST_TODO.md"
    bad_todo.write_text(f"Overall Compliance: invalid%{NL}", encoding="utf-8")
    assert update_badges.get_api_compliance(tmp_path) == 0.0
    no_line_todo = tmp_path / "TEST_TODO.md"
    no_line_todo.write_text(f"# Header only{NL}", encoding="utf-8")
    assert update_badges.get_api_compliance(tmp_path) == 0.0
    assert update_badges.get_api_compliance(Path(".")) == 100.0

    update_badges.update_readme(readme_path=tmp_path / "NO_README.md", run_cmd=False)

    readme_with_api = tmp_path / "README_API.md"
    readme_with_api.write_text(
        NL.join(
            [
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-80%25-yellowgreen.svg)](#)",
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-80%25-yellowgreen.svg)](#)",
                "[![API Compliance](https://img.shields.io/badge/api_compliance-80%25-yellowgreen.svg)](#)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    update_badges.update_readme(
        readme_path=readme_with_api,
        cov_json_path=valid_cov,
        src_dir=Path("src/zero_optax"),
        root_dir=Path("."),
        run_cmd=False,
    )
    assert "99.5%25" in readme_with_api.read_text(encoding="utf-8")

    readme_no_api = tmp_path / "README_NO_API.md"
    readme_no_api.write_text(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-80%25-yellowgreen.svg)](#){NL}",
        encoding="utf-8",
    )
    update_badges.update_readme(
        readme_path=readme_no_api,
        cov_json_path=valid_cov,
        src_dir=Path("src/zero_optax"),
        root_dir=Path("."),
        run_cmd=False,
    )
    readme_plain = tmp_path / "README_PLAIN.md"
    readme_plain.write_text(f"# Plain Readme{NL}", encoding="utf-8")
    update_badges.update_readme(
        readme_path=readme_plain,
        cov_json_path=valid_cov,
        src_dir=Path("src/zero_optax"),
        root_dir=Path("."),
        run_cmd=False,
    )
    assert "Plain Readme" in readme_plain.read_text(encoding="utf-8")


def test_lint_typing_and_docs_script(tmp_path: Path) -> None:
    """Test `scripts/lint_typing_and_docs.py` on valid and invalid files.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert lint_typing_and_docs.main(["src", "scripts", "tests"]) == 0

    src_sub = tmp_path / "src"
    src_sub.mkdir()
    bad_file = src_sub / "bad.py"
    type_ignore_comment = "# " + "type: ignore"
    bad_file.write_text(
        NL.join(
            [
                "from typing import Any, cast",
                "class UndocClass:",
                "    pass",
                "def undoc_fn(a, *args, **kwargs):",
                f"    return getattr(a, 'x'), a.__dict__  {type_ignore_comment}",
                "def missing_arg_doc(x: int, y: object) -> int:",
                '    """Summary only.',
                "",
                "    Args:",
                "        x: Documented.",
                '    """',
                "    return x",
                '__all__ = ["z", "a"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert lint_typing_and_docs.main([str(bad_file)]) == 1
    assert lint_typing_and_docs.main([str(src_sub)]) == 1
    no_all_file = src_sub / "no_all.py"
    no_all_file.write_text(f'"""Doc."""{NL}', encoding="utf-8")
    assert lint_typing_and_docs.lint_file(no_all_file, is_src=True) != []
    assert (
        lint_typing_and_docs.check_pyproject_mypy_config(tmp_path / "missing.toml")
        == []
    )
    bad_toml = tmp_path / "pyproject.toml"
    bad_toml.write_text(
        f"[tool.mypy]{NL}disable_error_code = ['assignment']{NL}", encoding="utf-8"
    )
    assert len(lint_typing_and_docs.check_pyproject_mypy_config(bad_toml)) == 1
    assert lint_typing_and_docs.main(["src"], pyproject_path=bad_toml) == 1

    from scripts import check_docs, check_no_any

    assert check_docs.main(["src/zero_optax"]) == 0
    assert check_no_any.main(["src/zero_optax"]) == 0
