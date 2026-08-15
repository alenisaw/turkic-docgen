from __future__ import annotations

from turkicdocgen.web.models import Job, JobConfig
from turkicdocgen.web.routers.api import StartJobRequest


def test_job_config_is_deeply_immutable_and_job_copies_lists() -> None:
    languages = ["kk", "ky"]
    layouts = ["book_page_single_column"]
    effects = ["clean"]
    config = JobConfig(
        profile="visual_300",
        out_dir="outputs/run",
        count=2,
        seed=42,
        languages=languages,
        layouts=layouts,
        effects=effects,
    )

    languages.append("ru")
    layouts.append("official")
    effects.append("noisy")

    assert config.languages == ("kk", "ky")
    assert config.layouts == ("book_page_single_column",)
    assert config.effects == ("clean",)

    job = Job(config)
    job.languages.append("uz")
    job.layouts.append("special")
    job.effects.append("shadow")

    assert config.languages == ("kk", "ky")
    assert config.layouts == ("book_page_single_column",)
    assert config.effects == ("clean",)
    assert job.languages == ["kk", "ky", "uz"]
    assert job.layouts == ["book_page_single_column", "special"]
    assert job.effects == ["clean", "shadow"]


def test_api_request_normalizes_lists_before_job_config() -> None:
    req = StartJobRequest(
        profile="visual_300",
        out_dir="outputs/run",
        count=1,
        seed=99,
        languages=["kk"],
        layouts=["book_page_single_column"],
        effects=["clean"],
    )

    config = req.normalized_job_config()
    assert isinstance(config.languages, tuple)
    assert isinstance(config.layouts, tuple)
    assert isinstance(config.effects, tuple)
    assert config.languages == ("kk",)
    assert config.layouts == ("book_page_single_column",)
    assert config.effects == ("clean",)
    assert req.languages == ["kk"]
    assert req.layouts == ["book_page_single_column"]
    assert req.effects == ["clean"]
