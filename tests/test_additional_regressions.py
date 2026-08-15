from __future__ import annotations

import inspect
import json
import re
import typing
from pathlib import Path

from typer.testing import CliRunner

from turkicdocgen.cli import app, generate
from turkicdocgen.dataset import GenerationOptions
from turkicdocgen.export import export_page
from turkicdocgen.hf.release import validate_hf_release
from turkicdocgen.page_planning.planner import PlannerOverrides
from turkicdocgen.web.models import Job, JobConfig

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_cli_output(text: str) -> str:
    return ANSI_RE.sub("", text)


def test_cli_typer_honest_signatures() -> None:
    # Direct inspect.signature check
    sig = inspect.signature(generate)
    assert "out" in sig.parameters
    assert "profile" in sig.parameters
    assert sig.parameters["profile"].default == "visual_300"
    assert sig.parameters["count"].default is None
    assert getattr(generate, "__signature__", None) is None

    # CLI runners
    runner = CliRunner()
    # Help output check
    res_gen = runner.invoke(app, ["generate", "--help"])
    assert res_gen.exit_code == 0
    generate_help = _plain_cli_output(res_gen.stdout)
    assert re.search(r"-+out\b", generate_help)
    assert re.search(r"-+profile\b", generate_help)

    res_pipe = runner.invoke(app, ["pipeline", "--help"])
    assert res_pipe.exit_code == 0
    assert re.search(r"-+out\b", _plain_cli_output(res_pipe.stdout))


def test_malformed_provenance_json(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    provenance_file = release_dir / "provenance.json"

    # Create dummy files to satisfy structural check
    for f in (
        "README.md",
        "CITATION.cff",
        "family_index.json",
        "dataset_info.json",
        "checksums.sha256",
    ):
        (release_dir / f).write_text("{}", encoding="utf-8")

    # 1. provenance.json file missing
    errors = validate_hf_release(release_dir)
    assert any("missing provenance.json" in err for err in errors)

    # 2. Malformed JSON
    provenance_file.write_text("{invalid json", encoding="utf-8")
    errors = validate_hf_release(release_dir)
    assert any(
        "failed to read provenance.json: Expecting property" in err for err in errors
    )

    # 3. Invalid structure (keys missing)
    provenance_file.write_text(json.dumps({"git_commit": "mock"}), encoding="utf-8")
    errors = validate_hf_release(release_dir)
    assert any("invalid provenance.json structure" in err for err in errors)


def test_runtime_type_introspection() -> None:
    # 1. GenerationOptions
    hints = typing.get_type_hints(GenerationOptions)
    assert hints["out"] is Path
    assert hints["profile"] is str

    # 2. export_page
    hints_export = typing.get_type_hints(export_page)
    assert hints_export["out_dir"] is Path
    assert hints_export["plan"].__name__ == "PagePlan"
    assert hints_export["qa"].__name__ == "QAReport"

    # 3. validate_hf_release
    hints_release = typing.get_type_hints(validate_hf_release)
    assert hints_release["release_dir"] is Path


def test_job_config_deep_immutability() -> None:
    config = JobConfig(
        profile="visual_300",
        out_dir="outputs/visual_300",
        count=8,
        seed=42,
        languages=["kk", "ky"],
        layouts=["book_page_single_column"],
        effects=["clean"],
    )
    assert isinstance(config.languages, tuple)
    assert isinstance(config.layouts, tuple)
    assert isinstance(config.effects, tuple)

    job = Job(config=config)
    job.languages.append("ru_kk")
    expected_lang_count = 2
    assert len(config.languages) == expected_lang_count
    assert "ru_kk" not in config.languages


def test_dataclass_adapters() -> None:
    opts = GenerationOptions(profile="visual_300", out=Path("out"))
    assert opts.profile == "visual_300"
    assert opts.out == Path("out")
    assert opts.seed is None

    planner_ovr = PlannerOverrides(language="kk")
    assert planner_ovr.language == "kk"
    assert planner_ovr.layout is None
