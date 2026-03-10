"""
Shared pytest fixtures for all test modules.

Provides:
- mock_db_config: patches source.core.db.config module globals
- mock_httpx: patches httpx.post/put at module level
- mock_sleep: patches time.sleep to no-op
- sample_task_params: minimal valid task parameter dict
- sample_phase_config: 2-phase config with 2 LoRAs
"""

from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_db_config():
    """Patch source.core.db.config module globals with MagicMock values.

    All DB submodules import ``from . import config as _cfg`` and access
    ``_cfg.SUPABASE_CLIENT``, so patching the module-level attributes is
    the correct target.
    """
    import source.core.db.config as _cfg

    originals = {
        "SUPABASE_CLIENT": _cfg.SUPABASE_CLIENT,
        "SUPABASE_URL": _cfg.SUPABASE_URL,
        "SUPABASE_ACCESS_TOKEN": _cfg.SUPABASE_ACCESS_TOKEN,
        "SUPABASE_EDGE_COMPLETE_TASK_URL": _cfg.SUPABASE_EDGE_COMPLETE_TASK_URL,
        "SUPABASE_EDGE_CREATE_TASK_URL": _cfg.SUPABASE_EDGE_CREATE_TASK_URL,
        "SUPABASE_EDGE_CLAIM_TASK_URL": _cfg.SUPABASE_EDGE_CLAIM_TASK_URL,
    }

    mock_client = MagicMock()
    _cfg.SUPABASE_CLIENT = mock_client
    _cfg.SUPABASE_URL = "https://test.supabase.co"
    _cfg.SUPABASE_ACCESS_TOKEN = "test-access-token"
    _cfg.SUPABASE_EDGE_COMPLETE_TASK_URL = "https://test.supabase.co/functions/v1/complete_task"
    _cfg.SUPABASE_EDGE_CREATE_TASK_URL = "https://test.supabase.co/functions/v1/create-task"
    _cfg.SUPABASE_EDGE_CLAIM_TASK_URL = "https://test.supabase.co/functions/v1/claim-next-task"

    yield {
        "client": mock_client,
        "url": _cfg.SUPABASE_URL,
        "token": _cfg.SUPABASE_ACCESS_TOKEN,
    }

    # Restore originals
    for attr, val in originals.items():
        setattr(_cfg, attr, val)


@pytest.fixture
def mock_httpx():
    """Patch ``httpx.post`` and ``httpx.put`` with configurable mock responses."""
    mock_post = MagicMock()
    mock_put = MagicMock()

    # Default: 200 with empty JSON body
    for mock in (mock_post, mock_put):
        mock.return_value = MagicMock(
            status_code=200,
            text="{}",
            json=MagicMock(return_value={}),
        )

    with patch("httpx.post", mock_post), patch("httpx.put", mock_put):
        yield {"post": mock_post, "put": mock_put}


@pytest.fixture
def mock_sleep():
    """Patch ``time.sleep`` to no-op for retry tests."""
    with patch("time.sleep") as mock:
        yield mock


@pytest.fixture
def sample_task_params():
    """Minimal valid task parameter dict."""
    return {
        "prompt": "a beautiful landscape",
        "resolution": "896x496",
        "video_length": 81,
        "model_name": "vace_14B_cocktail_2_2",
        "seed": 42,
    }


@pytest.fixture
def sample_phase_config():
    """2-phase config with 2 LoRAs."""
    return {
        "num_phases": 2,
        "steps_per_phase": [3, 3],
        "phases": [
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://hf.co/high_noise.safetensors", "multiplier": 0.9},
                    {"url": "https://hf.co/style.safetensors", "multiplier": 0.5},
                ],
            },
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://hf.co/high_noise.safetensors", "multiplier": 0.0},
                    {"url": "https://hf.co/style.safetensors", "multiplier": 0.8},
                ],
            },
        ],
    }
