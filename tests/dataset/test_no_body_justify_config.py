import importlib.resources
from pathlib import Path

import yaml


def test_render_profile_disables_justify():
    cfg = yaml.safe_load(
        Path(
            str(
                importlib.resources.files("turkicdocgen")
                / "configs"
                / "render_profile.yaml"
            )
        ).read_text(encoding="utf-8")
    )
    assert cfg["alignment"]["body"] == "left"
    assert cfg["alignment"]["allow_justify"] is False
