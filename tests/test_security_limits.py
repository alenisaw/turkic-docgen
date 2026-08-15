from __future__ import annotations

import pytest

from turkicdocgen.safety import redact_sensitive, validate_structure_limits


def test_structure_limits_reject_deep_payload() -> None:
    payload: dict[str, object] = {}
    cursor = payload
    for _ in range(40):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    with pytest.raises(ValueError, match="maximum depth"):
        validate_structure_limits(payload, max_depth=16)


def test_sensitive_values_are_redacted() -> None:
    message = "Authorization: Bearer-secret-token api_key=hf_abcdefghijklmnop"
    redacted = redact_sensitive(message)
    assert "hf_abcdefghijklmnop" not in redacted
    assert "Bearer-secret-token" not in redacted
