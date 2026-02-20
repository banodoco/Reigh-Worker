"""Tests for source/core/platform_utils.py."""

import os
from unittest.mock import patch

from source.core.platform_utils import setup_headless_environment, suppress_alsa_errors


class TestSuppressAlsaErrors:
    def test_no_crash_on_non_linux(self):
        """suppress_alsa_errors should silently no-op on non-Linux systems."""
        suppress_alsa_errors()


class TestSetupHeadlessEnvironment:
    def test_sets_expected_env_vars(self):
        """setup_headless_environment should set required env vars."""
        # Clear any pre-existing values to test setdefault behavior
        env_keys = [
            "PYTHONWARNINGS",
            "XDG_RUNTIME_DIR",
            "SDL_AUDIODRIVER",
            "PYGAME_HIDE_SUPPORT_PROMPT",
        ]
        saved = {k: os.environ.get(k) for k in env_keys}

        try:
            for k in env_keys:
                os.environ.pop(k, None)

            setup_headless_environment()

            assert os.environ["PYTHONWARNINGS"] == "ignore::FutureWarning"
            assert os.environ["XDG_RUNTIME_DIR"] == "/tmp/runtime-root"
            assert os.environ["SDL_AUDIODRIVER"] == "dummy"
            assert os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] == "1"
        finally:
            # Restore
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_does_not_overwrite_existing(self):
        """setdefault should not overwrite pre-existing values."""
        saved = os.environ.get("SDL_AUDIODRIVER")
        try:
            os.environ["SDL_AUDIODRIVER"] = "pulse"
            setup_headless_environment()
            assert os.environ["SDL_AUDIODRIVER"] == "pulse"
        finally:
            if saved is None:
                os.environ.pop("SDL_AUDIODRIVER", None)
            else:
                os.environ["SDL_AUDIODRIVER"] = saved
