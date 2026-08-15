from __future__ import annotations

import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    # 1. Start from the directory containing this file
    current = Path(__file__).resolve().parent
    # Walk up to find a directory containing pyproject.toml or .git
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # 2. Otherwise (installed package), return current working directory
    return Path.cwd().resolve()


ROOT = _find_repo_root()
GENERATED_ROOT_NAMES = {
    "outputs",
    "release",
    "reports",
    "runs",
    "artifacts",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".web_cache",
    "target",
}
PROTECTED_ROOT_NAMES = {"src", "tests", "configs", "data", "crates", ".git", ".agent"}
SECRET_PATTERNS = (
    re.compile(r"\b(?:hf|sk)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"(?i)\b(?:authorization|token|api[_-]?key|secret|password)\b"
        r"\s*[:=]\s*[^\s,;]+"
    ),
)


def resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def redact_sensitive(value: Any) -> str:
    redacted = str(value)
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def validate_structure_limits(
    value: Any,
    *,
    max_depth: int = 32,
    max_items: int = 100_000,
    max_string_length: int = 4 * 1024 * 1024,
) -> None:
    """Reject pathologically deep or oversized decoded JSON/YAML structures."""
    stack = [(value, 0)]
    item_count = 0
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            raise ValueError(f"structured payload exceeds maximum depth {max_depth}")
        item_count += 1
        if item_count > max_items:
            raise ValueError(f"structured payload exceeds maximum items {max_items}")
        if isinstance(current, str) and len(current) > max_string_length:
            raise ValueError(
                f"structured payload string exceeds {max_string_length} characters"
            )
        if isinstance(current, dict):
            stack.extend((key, depth + 1) for key in current)
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list | tuple):
            stack.extend((item, depth + 1) for item in current)


def assert_not_protected_path(path: Path, *, purpose: str = "operation") -> Path:
    resolved = resolve_path(path)
    repo = ROOT.resolve()
    if resolved == repo:
        raise ValueError(f"refusing {purpose} on repository root: {resolved}")
    try:
        rel = resolved.relative_to(repo)
    except ValueError:
        return resolved
    if not rel.parts:
        raise ValueError(f"refusing {purpose} on repository root: {resolved}")
    if rel.parts[0] in PROTECTED_ROOT_NAMES:
        raise ValueError(f"refusing {purpose} inside protected directory: {resolved}")
    return resolved


def assert_generated_path(path: Path, *, purpose: str = "operation") -> Path:
    resolved = assert_not_protected_path(path, purpose=purpose)
    repo = ROOT.resolve()
    try:
        rel = resolved.relative_to(repo)
    except ValueError as exc:
        raise ValueError(
            f"refusing {purpose} outside repository generated roots: {resolved}"
        ) from exc
    if not rel.parts or rel.parts[0] not in GENERATED_ROOT_NAMES:
        raise ValueError(f"refusing {purpose} outside generated roots: {resolved}")
    return resolved


def _safe_rmtree(path: Path) -> None:
    try:
        for entry in os.scandir(path):
            entry_path = Path(entry.path)
            if entry.is_symlink():
                entry_path.unlink()
            elif entry.is_dir(follow_symlinks=False):
                _safe_rmtree(entry_path)
            else:
                entry_path.unlink()
        path.rmdir()
    except FileNotFoundError:
        pass


def safe_prepare_output_dir(path: Path, *, force: bool = False) -> Path:
    resolved = assert_not_protected_path(path, purpose="dataset output")

    if force:
        quarantine = resolved.with_name(f".{resolved.name}.old-{uuid.uuid4()}")
        try:
            resolved.replace(quarantine)
            # Inspect the renamed path safely
            st = os.lstat(quarantine)
            if stat.S_ISLNK(st.st_mode):
                os.unlink(quarantine)
            elif stat.S_ISDIR(st.st_mode):
                _safe_rmtree(quarantine)
            else:
                os.unlink(quarantine)
        except FileNotFoundError:
            pass
    return resolved


def safe_remove_generated_path(path: Path) -> None:
    resolved = assert_generated_path(path, purpose="delete")
    if resolved.is_symlink():
        resolved.unlink()
    elif resolved.is_dir():
        _safe_rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def resolve_generated_run(output_base: str | Path, run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or Path(run_id).is_absolute():
        raise ValueError(f"invalid run id: {run_id!r}")
    if any(part in {"", ".", ".."} for part in Path(run_id).parts):
        raise ValueError(f"invalid run id: {run_id!r}")
    base = assert_generated_path(Path(output_base), purpose="web run access")
    run_dir = resolve_path(base / run_id)
    if not is_relative_to(run_dir, base):
        raise ValueError("run path escapes output base")
    return run_dir


def assert_file_in_generated_roots(
    path: Path, *, suffixes: set[str] | None = None
) -> Path:
    resolved = assert_generated_path(path, purpose="file access")
    if suffixes is not None and resolved.suffix.lower() not in suffixes:
        raise ValueError(f"file suffix not allowed: {resolved.suffix}")
    return resolved
