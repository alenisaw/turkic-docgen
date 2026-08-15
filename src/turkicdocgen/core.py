from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoreResult:
    returncode: int
    stdout: str
    stderr: str


def rust_core_binary() -> str | None:
    discovered = shutil.which("turkicdocgen-core")
    if discovered:
        return discovered
    local_binary = Path(
        "target/debug/turkicdocgen-core.exe"
        if sys.platform == "win32"
        else "target/debug/turkicdocgen-core"
    )
    if local_binary.exists():
        return str(local_binary)
    return None


def _rust_core_command() -> list[str] | None:
    cargo = shutil.which("cargo")
    if cargo and Path("crates/turkicdocgen-core/Cargo.toml").exists():
        return [cargo, "run", "-q", "-p", "turkicdocgen-core", "--"]
    binary = rust_core_binary()
    if not binary:
        return None
    return [binary]


def run_rust_core(args: list[str], *, require: bool = True) -> CoreResult | None:
    command = _rust_core_command()
    if command is None:
        if require:
            raise RuntimeError(
                "Rust core binary not found; run `cargo build --workspace` or install turkicdocgen-core"
            )
        return None
    completed = subprocess.run(
        [*command, *args],
        capture_output=True,
        encoding="utf-8",
        text=True,
        check=False,
    )
    result = CoreResult(
        int(getattr(completed, "returncode", 1)),
        str(getattr(completed, "stdout", "")),
        str(getattr(completed, "stderr", "")),
    )
    if require and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            detail or f"turkicdocgen-core failed with code {result.returncode}"
        )
    return result


def rust_dataset_summary(manifest: Path, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    run_rust_core(["dataset-summary", "--manifest", str(manifest), "--out", str(out)])
    return json.loads(out.read_text(encoding="utf-8"))


def rust_leakage_report(left: Path, right: Path, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    result = run_rust_core(
        [
            "leakage-check",
            "--left",
            str(left),
            "--right",
            str(right),
            "--out",
            str(out),
        ],
        require=False,
    )
    if result is None:
        raise RuntimeError(
            "Rust core binary not found; run `cargo build --workspace` or install turkicdocgen-core"
        )
    if not out.exists():
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(detail or "turkicdocgen-core did not write leakage report")
    return json.loads(out.read_text(encoding="utf-8"))


def rust_validate_manifest(
    manifest: Path,
    *,
    images_root: Path | None = None,
) -> list[str]:
    args = ["manifest-check", "--manifest", str(manifest)]
    if images_root is not None:
        args.extend(["--images-root", str(images_root)])
    result = run_rust_core(args, require=False)
    if result is None:
        raise RuntimeError(
            "Rust core binary not found; run `cargo build --workspace` or install turkicdocgen-core"
        )
    if result.returncode == 0:
        return []
    text = result.stderr.strip() or result.stdout.strip()
    return [line for line in text.splitlines() if line.strip()]


def rust_validate_schema_manifest(
    manifest: Path,
    *,
    images_root: Path | None = None,
) -> list[str]:
    args = ["schema-manifest-check", "--manifest", str(manifest)]
    if images_root is not None:
        args.extend(["--images-root", str(images_root)])
    result = run_rust_core(args, require=False)
    if result is None:
        raise RuntimeError(
            "Rust core binary not found; run `cargo build --workspace` or install turkicdocgen-core"
        )
    if result.returncode == 0:
        return []
    text = result.stderr.strip() or result.stdout.strip()
    return [line for line in text.splitlines() if line.strip()]


def rust_check_glyph_coverage(font: Path, text: str) -> dict | None:
    result = run_rust_core(
        ["glyph-check", "--font", str(font), "--text", text],
        require=False,
    )
    if result is None or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
