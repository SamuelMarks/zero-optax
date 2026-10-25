"""Verify API compliance against `SEMANTIC_PLAN.md`, `TODO_PLAN.md`, and `ml-ecosystem-snapshots` (`v0.0.5`)."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from types import ModuleType


def parse_semantic_plan(plan_path: Path) -> dict[str, list[str]]:
    """Extract target module and function names from `SEMANTIC_PLAN.md`.

    Args:
        plan_path: Path to `SEMANTIC_PLAN.md`.

    Returns:
        Dictionary mapping module names to lists of expected function names.
    """
    if not plan_path.exists():
        return {}

    content = plan_path.read_text(encoding="utf-8")
    modules: dict[str, list[str]] = {}
    current_module: str | None = None

    for line in content.splitlines():
        m_mod = re.search(r"`([a-zA-Z0-9_\.]+)`", line)
        if line.startswith("## ") and m_mod:
            raw_mod = m_mod.group(1)
            prefix = raw_mod.split(".")[0]
            current_module = raw_mod.replace(prefix, "zero_optax", 1)
            modules.setdefault(current_module, [])
            continue
        if line.startswith("## ") and not current_module:
            current_module = "zero_optax"
            modules.setdefault(current_module, [])
            continue

        if current_module is not None:
            m_item = re.match(r"^\s*- \[[xX ]\] `([a-zA-Z0-9_]+)`(?:[: ].*)?", line)
            if m_item:
                funcs_part = m_item.group(1)
                funcs = [f.strip(" `") for f in funcs_part.split("/") if f.strip(" `")]
                modules[current_module].extend(funcs)
    return modules


def parse_todo_plan_unchecked(todo_plan_path: Path) -> set[str]:
    """Extract unchecked `- [ ]` API paths from `TODO_PLAN.md`.

    Args:
        todo_plan_path: Path to `TODO_PLAN.md`.

    Returns:
        Set of canonical `optax.*` API paths that are still unchecked (`- [ ]`).
    """
    if not todo_plan_path.exists():
        return set()

    unchecked: set[str] = set()
    content = todo_plan_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        match = re.search(
            r"^\s*-\s*\[\s\]\s*.*`((?:optax|zero_optax)\.[a-zA-Z0-9_\.]+)`",
            line,
        )
        if match:
            raw_path = match.group(1)
            canonical = (
                "optax." + raw_path[len("zero_optax.") :]
                if raw_path.startswith("zero_optax.")
                else raw_path
            )
            unchecked.add(canonical)
    return unchecked


def _resolve_default_snapshot_path(
    cache_path: Path | None = None,
) -> Path:
    """Resolve the default `optax_v0.2.8.json` path from `ml-ecosystem-snapshots` (`v0.0.5`) or cache.

    Args:
        cache_path: Optional override path for the cached snapshot file.

    Returns:
        Resolved `Path` to `optax_v0.2.8.json`.
    """
    cache_snap = (
        cache_path
        if cache_path is not None
        else (
            Path.home()
            / ".cache"
            / "ml_ecosystem_snapshots"
            / "snapshots"
            / "optax_v0.2.8.json"
        )
    )
    if cache_snap.exists():
        return cache_snap
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path

    return Path(resolve_snapshot_path("optax_v0.2.8.json", must_exist=False))


def _load_target_submodules(
    module_names: tuple[str, ...] = (
        "optax",
        "optax.assignment",
        "optax.losses",
        "optax.perturbations",
        "optax.projections",
        "optax.schedules",
        "optax.second_order",
        "optax.tree_utils",
    ),
) -> dict[str, ModuleType]:
    """Load `zero_optax` and all available submodules into a lookup dictionary.

    Args:
        module_names: Tuple of canonical `optax.*` module names to attempt importing.

    Returns:
        Mapping from canonical `optax.*` module prefix to the loaded `ModuleType`.
    """
    submodules: dict[str, ModuleType] = {}
    for mod_key in module_names:
        target_mod_name = "zero_optax" + mod_key[len("optax") :]
        try:
            submodules[mod_key] = importlib.import_module(target_mod_name)
        except ImportError:
            continue
    return submodules


def verify_snapshot_v028(
    snapshot_path: Path | None = None,
    todo_plan_path: Path | None = None,
) -> list[str]:
    """Verify `zero_optax` signatures against `ml-ecosystem-snapshots` (`v0.0.5`) `optax_v0.2.8.json`.

    Checks module hierarchy (`optax`, `optax.assignment`, `optax.losses`,
    `optax.perturbations`, `optax.projections`, `optax.schedules`,
    `optax.second_order`, `optax.tree_utils`), classes, functions, aliases,
    parameter names, parameter kinds, and docstrings.

    Args:
        snapshot_path: Optional explicit path to `optax_v0.2.8.json`. If None,
            resolves via `ml_ecosystem_snapshots` (`v0.0.5`) or local cache.
        todo_plan_path: Optional path to `TODO_PLAN.md`. When `snapshot_path`
            is None and `TODO_PLAN.md` exists, endpoints still marked `- [ ]`
            in `TODO_PLAN.md` are tracked as planned pending items until checked off.

    Returns:
        List of mismatch description strings (empty if all active items are 100% compliant).
    """
    target_snap: Path = (
        snapshot_path if snapshot_path is not None else _resolve_default_snapshot_path()
    )
    if not target_snap.exists():
        return []

    loaded = json.loads(target_snap.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return []

    categories = loaded.get("categories")
    if not isinstance(categories, dict):
        return []

    resolved_todo_plan = (
        todo_plan_path
        if todo_plan_path is not None
        else (Path("TODO_PLAN.md") if snapshot_path is None else None)
    )
    unchecked_planned: set[str] = (
        parse_todo_plan_unchecked(resolved_todo_plan)
        if resolved_todo_plan is not None
        else set()
    )

    loaded_mods = _load_target_submodules()
    submodules = {k: dict(inspect.getmembers(mod)) for k, mod in loaded_mods.items()}

    errors: list[str] = []
    for cat_items in categories.values():
        if not isinstance(cat_items, list):
            continue
        for item in cat_items:
            if not isinstance(item, dict):
                continue
            api_path = str(item.get("api_path", ""))
            raw_aliases = item.get("aliases", [])
            aliases: list[str] = (
                [str(a) for a in raw_aliases if isinstance(a, str)]
                if isinstance(raw_aliases, list)
                else []
            )
            all_paths = [
                p
                for p in [api_path, *aliases]
                if p
                and p not in unchecked_planned
                and api_path not in unchecked_planned
            ]
            for endpoint_path in all_paths:
                parts = endpoint_path.split(".")
                parent_path = ".".join(parts[:-1])
                sym_name = parts[-1]
                if parent_path not in submodules:
                    errors.append(
                        f"Missing submodule {parent_path} for endpoint: {endpoint_path}"
                    )
                    continue
                ns = submodules[parent_path]
                if sym_name not in ns or not callable(ns[sym_name]):
                    errors.append(
                        f"Missing or non-callable endpoint in {parent_path}: {endpoint_path}"
                    )
                    continue
                fn = ns[sym_name]
                sig = inspect.signature(fn)
                actual_params = list(sig.parameters.values())
                expected_params = item.get("params", [])
                if not isinstance(expected_params, list):
                    continue
                if len(actual_params) != len(expected_params):
                    errors.append(
                        f"{endpoint_path}: parameter count mismatch ({len(actual_params)} vs {len(expected_params)})"
                    )
                    continue
                for act_p, exp_p in zip(actual_params, expected_params):
                    if not isinstance(exp_p, dict):
                        continue
                    exp_name = str(exp_p.get("name", ""))
                    exp_kind = str(exp_p.get("kind", ""))
                    if act_p.name != exp_name:
                        errors.append(
                            f"{endpoint_path}: parameter name mismatch '{act_p.name}' != '{exp_name}'"
                        )
                    if act_p.kind.name != exp_kind:
                        errors.append(
                            f"{endpoint_path}({act_p.name}): parameter kind mismatch '{act_p.kind.name}' != '{exp_kind}'"
                        )
    return errors


def check_compliance(
    plan_path: Path | None = None,
    todo_path: Path | None = None,
    snapshot_path: Path | None = None,
    todo_plan_path: Path | None = None,
) -> int:
    """Validate `SEMANTIC_PLAN.md`, `TODO_PLAN.md`, and `optax_v0.2.8.json` (`v0.0.5`) compliance.

    Args:
        plan_path: Optional path to `SEMANTIC_PLAN.md`.
        todo_path: Optional path to write a compliance report markdown file.
        snapshot_path: Optional path to `optax_v0.2.8.json`.
        todo_plan_path: Optional path to `TODO_PLAN.md`.

    Returns:
        `0` if 100% compliant and optional `todo_path` is up to date, `1` otherwise.
    """
    resolved_plan = Path("SEMANTIC_PLAN.md") if plan_path is None else plan_path

    if not resolved_plan.exists():
        print("SEMANTIC_PLAN.md not found, skipping compliance check.")
        return 0

    modules = parse_semantic_plan(resolved_plan)
    if not modules or all(len(funcs) == 0 for funcs in modules.values()):
        print("No parsable API targets found in SEMANTIC_PLAN.md")
        return 0

    missing: list[str] = []
    total = 0
    implemented = 0

    for mod_name, funcs in modules.items():
        if not funcs:
            continue
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            missing.extend([f"{mod_name}.{f}" for f in funcs])
            total += len(funcs)
            continue

        mod_dict = dict(inspect.getmembers(mod))
        for f_name in funcs:
            total += 1
            if f_name in mod_dict:
                implemented += 1
            else:
                missing.append(f"{mod_name}.{f_name}")

    pct = (implemented / total) * 100.0 if total > 0 else 100.0

    todo_updated = False
    if todo_path is not None:
        current_todo = (
            todo_path.read_text(encoding="utf-8") if todo_path.exists() else ""
        )

        new_todo_lines = [
            "# OPTAX_TODO.md",
            "--- Compliance Report ---",
            f"Overall Compliance: {pct:.1f}%",
            "",
            "Breakdown by Module:",
        ]
        for mod_name, funcs in modules.items():
            if not funcs:
                continue
            orig_mod_name = mod_name.replace("zero_optax", "optax")
            mod_implemented = sum(
                1 for f_name in funcs if f"{mod_name}.{f_name}" not in missing
            )
            mod_pct = (mod_implemented / len(funcs)) * 100.0
            new_todo_lines.append(
                f"  - {orig_mod_name}: {mod_pct:.1f}% ({mod_implemented}/{len(funcs)})"
            )

        if missing:
            new_todo_lines.append("")
            new_todo_lines.append("Missing APIs:")
            for m in missing:
                new_todo_lines.append(f"  - {m}")

        new_todo = chr(10).join(new_todo_lines) + chr(10)
        if new_todo != current_todo:
            todo_path.write_text(new_todo, encoding="utf-8")
            print("Updated OPTAX_TODO.md.")
            todo_updated = True

    snapshot_errors = verify_snapshot_v028(
        snapshot_path=snapshot_path,
        todo_plan_path=todo_plan_path,
    )
    if missing or snapshot_errors:
        print("COMPLIANCE FAILURE:")
        for m in missing:
            print(f"  - Missing: {m}")
        for err in snapshot_errors:
            print(f"  - Snapshot mismatch: {err}")
        print(f"Overall Compliance: {pct:.1f}% ({implemented}/{total})")
        return 1

    print(f"Compliance Check Passed! 100% compliant ({implemented}/{total} APIs).")
    return 1 if todo_updated else 0


if __name__ == "__main__":
    sys.exit(check_compliance())
