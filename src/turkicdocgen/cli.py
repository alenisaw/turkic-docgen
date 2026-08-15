from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from turkicdocgen import __version__
from turkicdocgen.core import run_rust_core
from turkicdocgen.dataset import (
    GenerationOptions,
    generate_dataset_from_options,
    validate_output,
)
from turkicdocgen.hf.release import (
    export_hf_release,
    publish_hf_release,
    validate_hf_release,
)
from turkicdocgen.profiles import dataset_family, profile_count
from turkicdocgen.safety import safe_remove_generated_path

app = typer.Typer(
    name="turkicdocgen",
    help="Zone-first synthetic document dataset factory for Turkic OCR.",
)
console = Console()

OCR_CORE_PROFILES = tuple(sorted(dataset_family()))
OLD_PROFILE_HINTS = {
    "smoke",
    "visual_check",
    "quality_gate",
    "release_smoke",
    "dataset_100k",
    "dataset_500k",
    "benchmark_1k",
    "benchmark_5k",
    "schema_first",
    "quality_gate_schema_first",
}
GENERATED_ROOTS = (
    Path("outputs"),
    Path("release"),
    Path("reports"),
    Path("runs"),
    Path("artifacts"),
    Path(".pytest_cache"),
    Path(".ruff_cache"),
    Path(".cache"),
    Path(".web_cache"),
    Path("target"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ensure_profile(profile: str) -> None:
    if profile in OCR_CORE_PROFILES:
        return
    if profile in OLD_PROFILE_HINTS or profile.startswith(
        ("qg", "dataset_", "benchmark_")
    ):
        console.print(
            f"[red]Legacy profile removed[/red]: {profile}. "
            f"Use one of: {', '.join(OCR_CORE_PROFILES)}"
        )
        raise typer.Exit(2)
    console.print(f"[red]Unknown profile[/red]: {profile}")
    console.print(f"Known dataset profiles: {', '.join(OCR_CORE_PROFILES)}")
    raise typer.Exit(1)


def default_output_dir(profile: str) -> Path:
    _ensure_profile(profile)
    return Path("outputs") / profile


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"turkicdocgen {__version__}")


@app.command("profiles")
def profiles_command() -> None:
    """List active dataset profiles."""
    table = Table(title="Dataset profiles")
    table.add_column("profile")
    table.add_column("count", justify="right")
    for profile in OCR_CORE_PROFILES:
        table.add_row(profile, str(profile_count(profile)))
    console.print(table)


def _run_generate(options: GenerationOptions, quiet: bool) -> None:
    try:
        manifest = generate_dataset_from_options(options)
    except (RuntimeError, ValueError) as exc:
        printable = str(exc).encode("ascii", errors="backslashreplace").decode("ascii")
        console.print(f"[red]Generation refused[/red]: {printable}")
        raise typer.Exit(1) from exc
    if options.benchmark_mode:
        if not quiet:
            console.print("[green]Benchmark completed successfully[/green]")
            console.print(f"Performance Report: {manifest}")
        return
    rows = _read_jsonl(manifest)
    if not quiet:
        console.print(f"[green]Generated dataset[/green] {len(rows)} samples")
        console.print(f"Manifest: {manifest}")


def _run_pipeline(options: GenerationOptions) -> None:
    manifest = generate_dataset_from_options(options)
    if options.benchmark_mode:
        console.print("[green]Benchmark completed successfully[/green]")
        console.print(f"Performance Report: {manifest}")
        return
    errors = validate_output(options.out)
    rows = _read_jsonl(manifest)
    summary = {
        "profile": options.profile,
        "rows": len(rows),
        "validation_errors": errors,
        "accepted": sum(1 for row in rows if row.get("qa_ok") is True),
        "rejected": sum(1 for row in rows if row.get("qa_ok") is not True),
    }
    report = options.out / "reports" / "pipeline_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from turkicdocgen.page_planning.content.audit import run_diversity_audit

    audit_report = run_diversity_audit(rows)
    (options.out / "reports" / "content_diversity.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if errors:
        console.print("[red]Pipeline validation failed[/red]")
        console.print(f"Summary: {report}")
        raise typer.Exit(1)
    console.print(f"[green]Pipeline passed[/green] ({len(rows)} dataset rows)")
    console.print(f"Summary: {report}")


@app.command()
def generate(
    out: Annotated[Path, typer.Option("--out")],
    profile: Annotated[str, typer.Option("--profile")] = "visual_300",
    count: Annotated[int | None, typer.Option("--count", min=1)] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    language: Annotated[str | None, typer.Option("--language")] = None,
    layout: Annotated[str | None, typer.Option("--layout")] = None,
    effect: Annotated[str | None, typer.Option("--effect")] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    shard_size: Annotated[int, typer.Option("--shard-size", min=1)] = 1000,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    shard_range: Annotated[str | None, typer.Option("--shard-range")] = None,
    retry_rejected: Annotated[bool, typer.Option("--retry-rejected")] = False,
    verify_only: Annotated[bool, typer.Option("--verify-only")] = False,
    benchmark_mode: Annotated[bool, typer.Option("--benchmark-mode")] = False,
) -> None:
    """Generate document images and ground truth."""
    _ensure_profile(profile)
    options = GenerationOptions(
        profile=profile,
        out=out,
        seed=seed,
        count=count,
        force=force,
        language=language,
        layout=layout,
        effect=effect,
        workers=workers,
        shard_size=shard_size,
        resume=resume,
        shard_range=shard_range,
        retry_rejected=retry_rejected,
        verify_only=verify_only,
        benchmark_mode=benchmark_mode,
    )
    _run_generate(options, quiet)


@app.command()
def validate(
    target: Annotated[Path, typer.Argument()],
) -> None:
    """Validate a dataset output directory."""
    errors = validate_output(target)
    if errors:
        console.print("[red]Validation failed[/red]")
        for error in errors[:80]:
            printable = error.encode("ascii", errors="backslashreplace").decode("ascii")
            console.print(f"- {printable}")
        raise typer.Exit(1)
    rows = _read_jsonl(target / "manifest.jsonl")
    console.print(f"[green]Validation passed[/green] ({len(rows)} dataset rows)")


@app.command()
def pipeline(
    out: Annotated[Path, typer.Option("--out")],
    profile: Annotated[str, typer.Option("--profile")] = "visual_300",
    count: Annotated[int | None, typer.Option("--count", min=1)] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    language: Annotated[str | None, typer.Option("--language")] = None,
    layout: Annotated[str | None, typer.Option("--layout")] = None,
    effect: Annotated[str | None, typer.Option("--effect")] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    shard_size: Annotated[int, typer.Option("--shard-size", min=1)] = 1000,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    shard_range: Annotated[str | None, typer.Option("--shard-range")] = None,
    retry_rejected: Annotated[bool, typer.Option("--retry-rejected")] = False,
    verify_only: Annotated[bool, typer.Option("--verify-only")] = False,
    benchmark_mode: Annotated[bool, typer.Option("--benchmark-mode")] = False,
) -> None:
    """Run generate, validate, and QA summary."""
    _ensure_profile(profile)
    options = GenerationOptions(
        profile=profile,
        out=out,
        seed=seed,
        count=count,
        force=force,
        language=language,
        layout=layout,
        effect=effect,
        workers=workers,
        shard_size=shard_size,
        resume=resume,
        shard_range=shard_range,
        retry_rejected=retry_rejected,
        verify_only=verify_only,
        benchmark_mode=benchmark_mode,
    )
    _run_pipeline(options)


@app.command("export-det")
def export_det(
    input_dir: Annotated[Path, typer.Option("--input")],
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    """Copy OCR detection JSONL."""
    src = input_dir / "ocr_det.jsonl"
    if not src.exists():
        console.print(f"[red]Missing detection export[/red]: {src}")
        raise typer.Exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    console.print(f"[green]Detection export written[/green]: {out}")


@app.command("export-rec")
def export_rec(
    input_dir: Annotated[Path, typer.Option("--input")],
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    """Copy OCR recognition JSONL."""
    src = input_dir / "ocr_rec.jsonl"
    if not src.exists():
        console.print(f"[red]Missing recognition export[/red]: {src}")
        raise typer.Exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    console.print(f"[green]Recognition export written[/green]: {out}")


@app.command("export-release")
def export_release_command(
    input_dir: Annotated[Path, typer.Option("--input")],
    out: Annotated[Path, typer.Option("--out")],
    hf_card: Annotated[bool, typer.Option("--hf-card/--no-hf-card")] = True,
    pretty_name: Annotated[str | None, typer.Option("--pretty-name")] = None,
    publish: Annotated[bool, typer.Option("--publish")] = False,
) -> None:
    """Create a self-contained local release folder."""
    try:
        export_hf_release(
            input_dir,
            out,
            hf_card=hf_card,
            pretty_name=pretty_name,
            publish=publish,
            force=True,
        )
    except ValueError as exc:
        console.print(f"[red]Release export failed[/red]: {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Release exported[/green]: {out}")
    console.print(f"Report: {out / 'provenance.json'}")


@app.command("validate-release")
def validate_release_command(
    input_dir: Annotated[Path, typer.Option("--input")],
) -> None:
    """Validate a self-contained local release folder."""
    errors = validate_hf_release(input_dir)
    if errors:
        console.print("[red]Release validation failed[/red]")
        for error in errors[:80]:
            console.print(f"- {error}")
        raise typer.Exit(1)
    console.print("[green]Release validation passed[/green]")


@app.command("publish-release")
def publish_release_command(
    input_dir: Annotated[Path, typer.Option("--input")],
    repo_id: Annotated[str, typer.Option("--repo-id")],
    private: Annotated[bool, typer.Option("--private/--public")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    num_workers: Annotated[int | None, typer.Option("--num-workers", min=1)] = None,
) -> None:
    """Validate and publish a local release folder through the Hugging Face CLI."""
    try:
        plan = publish_hf_release(
            input_dir,
            repo_id,
            private=private,
            dry_run=dry_run,
            num_workers=num_workers,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Release publication failed[/red]: {exc}")
        raise typer.Exit(1) from exc
    if dry_run:
        console.print("[yellow]Dry run only; no remote changes made.[/yellow]")
        for command in plan:
            console.print("hf " + " ".join(command))
        return
    console.print(f"[green]Release published[/green]: {repo_id}")


@app.command("cleanup-runs")
def cleanup_runs(
    dry_run: Annotated[bool, typer.Option("--dry-run/--delete")] = True,
) -> None:
    """Remove generated output roots without touching source directories."""
    removed: list[Path] = []
    for path in GENERATED_ROOTS:
        if not path.exists():
            continue
        if path.parts and path.parts[0] in {"src", "tests", "configs", "data"}:
            continue
        removed.append(path)
        if not dry_run:
            safe_remove_generated_path(path)
    verb = "would remove" if dry_run else "removed"
    if not removed:
        console.print("No generated roots found.")
    for path in removed:
        console.print(f"{verb}: {path}")


@app.command()
def status() -> None:
    """Show active profiles and generated roots."""
    console.print(f"Active profiles: {', '.join(OCR_CORE_PROFILES)}")
    existing = [str(path) for path in GENERATED_ROOTS if path.exists()]
    console.print(f"Generated roots: {', '.join(existing) if existing else 'none'}")


@app.command()
def start(
    profile: Annotated[str, typer.Option("--profile")] = "visual_300",
    out: Annotated[Path | None, typer.Option("--out")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = 42,
) -> None:
    """Print the recommended command for a profile."""
    _ensure_profile(profile)
    target = out or default_output_dir(profile)
    console.print(
        " ".join(
            [
                "turkicdocgen",
                "pipeline",
                "--profile",
                profile,
                "--seed",
                str(seed),
                "--out",
                str(target),
                "--force",
            ]
        )
    )


@app.command("web")
def web_panel(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 7860,
    input_dir: Annotated[Path | None, typer.Option("--input")] = None,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Launch the TurkicDocGen local web panel."""
    if input_dir is not None:
        os.environ["TURKICDOCGEN_WEB_INPUT"] = str(input_dir)
    suffix = f" input={input_dir}" if input_dir else ""
    if host not in ("127.0.0.1", "localhost"):
        console.print(
            "[bold yellow]WARNING: Web panel is exposed publicly without authentication. "
            "Set the TURKICDOCGEN_WEB_TOKEN environment variable to enforce Bearer token authentication, "
            "or bind to 127.0.0.1 for local use.[/bold yellow]"
        )
    console.print(
        f"[bold blue]TurkicDocGen Web Panel[/bold blue] http://{host}:{port}{suffix}"
    )
    uvicorn.run(
        "turkicdocgen.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command("web-index")
def web_index(
    input_dir: Annotated[Path, typer.Option("--input")],
) -> None:
    """Build the lightweight web manifest index for an output run."""
    from turkicdocgen.web.routers.utils import _build_manifest_index

    index_path = _build_manifest_index(input_dir)
    if index_path is None:
        console.print(f"[red]Web index failed[/red]: {input_dir}")
        raise typer.Exit(1)
    console.print(f"[green]Web index ready[/green]: {index_path}")


def _run_core(args: list[str]) -> None:
    result = run_rust_core(args, require=False)
    if result is None:
        console.print("[yellow]Rust core unavailable[/yellow]")
        raise typer.Exit(1)
    if result.stdout.strip():
        console.print(result.stdout.strip())
    if result.stderr.strip():
        console.print(result.stderr.strip())
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command("core-summary")
def core_summary(
    manifest: Annotated[Path, typer.Option("--manifest")],
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    """Run Rust dataset-summary if available."""
    _run_core(["dataset-summary", "--manifest", str(manifest), "--out", str(out)])


@app.command("core-hash-files")
def core_hash_files(
    manifest: Annotated[Path, typer.Option("--manifest")],
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    """Run Rust file hashing if available."""
    _run_core(["hash-files", "--manifest", str(manifest), "--out", str(out)])


@app.command("core-dedup-text")
def core_dedup_text(
    manifest: Annotated[Path, typer.Option("--manifest")],
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    """Run Rust exact text deduplication if available."""
    _run_core(["dedup-text", "--manifest", str(manifest), "--out", str(out)])


@app.command("core-dedup-text-minhash")
def core_dedup_text_minhash(
    manifest: Annotated[Path, typer.Option("--manifest")],
    out: Annotated[Path, typer.Option("--out")],
    threshold: Annotated[float | None, typer.Option("--threshold")] = None,
) -> None:
    """Run Rust minhash text deduplication if available."""
    args = ["dedup-text-minhash", "--manifest", str(manifest), "--out", str(out)]
    if threshold is not None:
        args.extend(["--threshold", str(threshold)])
    _run_core(args)


@app.command("core-image-ahash")
def core_image_ahash(
    image: Annotated[Path, typer.Option("--image")],
) -> None:
    """Run Rust image ahash if available."""
    _run_core(["image-ahash", "--image", str(image)])


@app.command("core-leakage-check")
def core_leakage_check(
    left: Annotated[Path, typer.Option("--left")],
    right: Annotated[Path, typer.Option("--right")],
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    """Run Rust leakage check if available."""
    _run_core(
        ["leakage-check", "--left", str(left), "--right", str(right), "--out", str(out)]
    )


@app.command("core-split-manifest")
def core_split_manifest(
    manifest: Annotated[Path, typer.Option("--manifest")],
    train_out: Annotated[Path, typer.Option("--train-out")],
    validation_out: Annotated[Path, typer.Option("--validation-out")],
    split_ratio: Annotated[float | None, typer.Option("--split-ratio")] = None,
) -> None:
    """Run Rust manifest split if available."""
    args = [
        "split-manifest",
        "--manifest",
        str(manifest),
        "--train-out",
        str(train_out),
        "--validation-out",
        str(validation_out),
    ]
    if split_ratio is not None:
        args.extend(["--split-ratio", str(split_ratio)])
    _run_core(args)


@app.command("core-special-char-stats")
def core_special_char_stats(
    manifest: Annotated[Path, typer.Option("--manifest")],
) -> None:
    """Run Rust special character statistics if available."""
    _run_core(["special-char-stats", "--manifest", str(manifest)])


if __name__ == "__main__":
    app()
