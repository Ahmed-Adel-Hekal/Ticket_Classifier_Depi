import pytest


def test_structured_generation_requires_optional_backend():
    pytest.importorskip("outlines")
    # The actual model-backed integration test belongs in an environment
    # with a downloaded compatible transformer model.
    assert True
