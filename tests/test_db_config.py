"""Tests for source/core/db/config.py.

Focuses on behavioral contracts: status constants are used in equality
comparisons and DB filters throughout the codebase, _log_thumbnail routes
to the correct logger level, and __all__ re-exports everything the facade
modules need.
"""

from unittest.mock import MagicMock, patch

import source.core.db.config as _cfg
from source.core.db.config import (
    STATUS_QUEUED,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETE,
    STATUS_FAILED,
    EDGE_FAIL_PREFIX,
    RETRYABLE_STATUS_CODES,
    _log_thumbnail,
)


class TestStatusConstants:
    def test_statuses_are_distinct(self):
        """All four statuses must be unique — DB queries use equality checks."""
        statuses = {STATUS_QUEUED, STATUS_IN_PROGRESS, STATUS_COMPLETE, STATUS_FAILED}
        assert len(statuses) == 4

    def test_statuses_are_strings(self):
        """Statuses are compared via .eq() in Supabase queries, must be str."""
        for s in (STATUS_QUEUED, STATUS_IN_PROGRESS, STATUS_COMPLETE, STATUS_FAILED):
            assert isinstance(s, str)
            assert len(s) > 0


class TestEdgeFailPrefix:
    def test_can_detect_edge_errors(self):
        """edge_helpers.py builds messages like f'{EDGE_FAIL_PREFIX}:fn:TYPE] ...'
        and consumers use startswith() to detect them."""
        error_msg = f"{EDGE_FAIL_PREFIX}:complete_task:HTTP_500] 500 Internal Server Error"
        assert error_msg.startswith(EDGE_FAIL_PREFIX)

    def test_non_edge_errors_dont_match(self):
        assert not "Some other error".startswith(EDGE_FAIL_PREFIX)


class TestRetryableStatusCodes:
    def test_retries_server_errors(self):
        """edge_helpers.py checks `status_code in RETRYABLE_STATUS_CODES`."""
        for code in (500, 502, 503, 504):
            assert code in RETRYABLE_STATUS_CODES

    def test_no_retry_on_client_errors(self):
        for code in (400, 401, 403, 404, 409, 422):
            assert code not in RETRYABLE_STATUS_CODES


class TestLogThumbnail:
    def test_routes_info_to_info(self):
        mock_logger = MagicMock()
        with patch.object(_cfg, 'headless_logger', mock_logger):
            _log_thumbnail("uploaded", level="info", task_id="t1")
        mock_logger.info.assert_called_once()
        assert "[THUMBNAIL]" in mock_logger.info.call_args[0][0]

    def test_routes_warning_to_warning(self):
        mock_logger = MagicMock()
        with patch.object(_cfg, 'headless_logger', mock_logger):
            _log_thumbnail("failed", level="warning", task_id="t1")
        mock_logger.warning.assert_called_once()

    def test_routes_debug_by_default(self):
        mock_logger = MagicMock()
        with patch.object(_cfg, 'headless_logger', mock_logger):
            _log_thumbnail("checking file")
        mock_logger.debug.assert_called_once()

    def test_noop_when_logger_is_none(self):
        with patch.object(_cfg, 'headless_logger', None):
            _log_thumbnail("no crash")  # Should not raise

    def test_prepends_thumbnail_tag(self):
        mock_logger = MagicMock()
        with patch.object(_cfg, 'headless_logger', mock_logger):
            _log_thumbnail("some message", level="info")
        msg = mock_logger.info.call_args[0][0]
        assert msg == "[THUMBNAIL] some message"


class TestAllExports:
    def test_all_contains_status_constants(self):
        """Facades use `from config import *` — statuses must be in __all__."""
        for name in ("STATUS_QUEUED", "STATUS_IN_PROGRESS", "STATUS_COMPLETE", "STATUS_FAILED"):
            assert name in _cfg.__all__

    def test_all_contains_connection_vars(self):
        for name in ("SUPABASE_CLIENT", "SUPABASE_URL", "SUPABASE_ACCESS_TOKEN"):
            assert name in _cfg.__all__

    def test_all_contains_edge_helpers(self):
        for name in ("EDGE_FAIL_PREFIX", "RETRYABLE_STATUS_CODES"):
            assert name in _cfg.__all__
