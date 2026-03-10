"""Tests for source/core/db/config.py validate_config."""

from unittest import mock

import pytest

from source.core.db import config as db_config


class TestValidateConfig:
    """Tests for validate_config."""

    def test_all_set_no_errors(self):
        with mock.patch.multiple(
            db_config,
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_KEY="secret",
            SUPABASE_CLIENT=mock.MagicMock(),
            SUPABASE_ACCESS_TOKEN="token",
        ):
            errors = db_config.validate_config()
            assert errors == []

    def test_missing_url(self):
        with mock.patch.multiple(
            db_config,
            SUPABASE_URL=None,
            SUPABASE_SERVICE_KEY="secret",
            SUPABASE_CLIENT=mock.MagicMock(),
            SUPABASE_ACCESS_TOKEN="token",
        ):
            errors = db_config.validate_config()
            assert any("SUPABASE_URL" in e for e in errors)

    def test_invalid_url(self):
        with mock.patch.multiple(
            db_config,
            SUPABASE_URL="not-a-url",
            SUPABASE_SERVICE_KEY="secret",
            SUPABASE_CLIENT=mock.MagicMock(),
            SUPABASE_ACCESS_TOKEN="token",
        ):
            errors = db_config.validate_config()
            assert any("does not look like a URL" in e for e in errors)

    def test_missing_service_key(self):
        with mock.patch.multiple(
            db_config,
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_KEY=None,
            SUPABASE_CLIENT=mock.MagicMock(),
            SUPABASE_ACCESS_TOKEN="token",
        ):
            errors = db_config.validate_config()
            assert any("SUPABASE_SERVICE_KEY" in e for e in errors)

    def test_missing_client(self):
        with mock.patch.multiple(
            db_config,
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_KEY="secret",
            SUPABASE_CLIENT=None,
            SUPABASE_ACCESS_TOKEN="token",
        ):
            errors = db_config.validate_config()
            assert any("SUPABASE_CLIENT" in e for e in errors)

    def test_missing_access_token(self):
        with mock.patch.multiple(
            db_config,
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_KEY="secret",
            SUPABASE_CLIENT=mock.MagicMock(),
            SUPABASE_ACCESS_TOKEN=None,
        ):
            errors = db_config.validate_config()
            assert any("SUPABASE_ACCESS_TOKEN" in e for e in errors)

    def test_multiple_errors_reported(self):
        with mock.patch.multiple(
            db_config,
            SUPABASE_URL=None,
            SUPABASE_SERVICE_KEY=None,
            SUPABASE_CLIENT=None,
            SUPABASE_ACCESS_TOKEN=None,
        ):
            errors = db_config.validate_config()
            assert len(errors) == 4
