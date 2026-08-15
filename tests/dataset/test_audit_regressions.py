from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from turkicdocgen.config_validation import (
    validate_dataset_profile_config,
    validate_effects_config,
)
from turkicdocgen.dataset import generate_dataset
from turkicdocgen.hf.release import export_hf_release, validate_hf_release
from turkicdocgen.qa import validate_page_plan
from turkicdocgen.safety import safe_prepare_output_dir
from turkicdocgen.schema import PagePlan, TextStyle, Zone


def _hash_tree(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return out


def test_same_seed_produces_identical_outputs(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset("visual_300", first, seed=42, count=4, force=True)
    generate_dataset("visual_300", second, seed=42, count=4, force=True)

    assert (first / "manifest.jsonl").read_text(encoding="utf-8") == (
        second / "manifest.jsonl"
    ).read_text(encoding="utf-8")
    assert _hash_tree(first / "images") == _hash_tree(second / "images")


def test_safe_prepare_output_refuses_source_roots() -> None:
    with pytest.raises(ValueError, match="protected directory"):
        safe_prepare_output_dir(Path("src") / "bad", force=True)
    with pytest.raises(ValueError, match="repository root"):
        safe_prepare_output_dir(Path("."), force=True)


def test_bad_config_ranges_fail_with_clear_errors() -> None:
    with pytest.raises(ValueError, match=re.escape("profiles.bad.count")):
        validate_dataset_profile_config(
            {
                "profiles": {"bad": {"count": 0}},
                "languages": {"kk": 1.0},
                "layouts": {"book_page_single_column": 1.0},
                "quality": {"clean": 1.0},
            }
        )
    with pytest.raises(ValueError, match=re.escape("parameters.rotation_degrees")):
        validate_effects_config(
            {
                "profiles": {"clean": {"effects": []}},
                "parameters": {"rotation_degrees": [1, -1]},
            }
        )


def test_invalid_bbox_length_is_rejected() -> None:
    plan = PagePlan(
        page_id="bad",
        width=100,
        height=100,
        layout_id="simple_form_page",
        language_mix="kk",
        quality_profile="clean",
        zones=[
            Zone(
                zone_id="z1",
                zone_type="form",
                bbox=(1, 2, 3),  # type: ignore[arg-type]
                polygon=[],
                text="A: B",
                language="kk",
                reading_order=1,
                style=TextStyle("DejaVuSans", 20),
            )
        ],
    )
    qa = validate_page_plan(plan)
    assert not qa.ok
    assert any(issue.code == "invalid_zone_box" for issue in qa.issues)


def test_release_validation_catches_alignment_errors(tmp_path: Path) -> None:
    run = tmp_path / "run"
    release = tmp_path / "release"
    generate_dataset("visual_300", run, seed=7, count=3, force=True)
    export_hf_release(run, release, hf_card=True)

    # Delete samples.parquet to trigger a validation error
    (release / "indexes" / "samples.parquet").unlink()
    errors = validate_hf_release(release)
    assert any(
        "missing index file" in error or "does not exist" in error for error in errors
    )
