"""Tests for source/core/db/edge_helpers.py."""

import json
import base64

import httpx
import pytest
from unittest.mock import MagicMock

from source.core.db.edge_helpers import (
    _call_edge_function_with_retry,
    _get_user_id_from_jwt,
    _is_jwt_token,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ok_response(body=None, status=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = json.dumps(body or {})
    resp.json.return_value = body or {}
    return resp


def _err_response(status, body="error"):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = body
    resp.json.return_value = {}
    return resp


HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer tok"}
URL = "https://edge.test/fn"


# ── _call_edge_function_with_retry ───────────────────────────────────────────

class TestCallEdgeFunctionWithRetry:
    def test_success_first_attempt(self, mock_httpx, mock_sleep):
        mock_httpx["post"].return_value = _ok_response({"ok": True})
        resp, err = _call_edge_function_with_retry(URL, {"x": 1}, HEADERS, "test-fn")
        assert err is None
        assert resp.status_code == 200
        mock_httpx["post"].assert_called_once()

    def test_success_201(self, mock_httpx, mock_sleep):
        mock_httpx["post"].return_value = _ok_response(status=201)
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn")
        assert err is None
        assert resp.status_code == 201

    def test_retry_on_502(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            _err_response(502),
            _ok_response(),
        ]
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=3)
        assert err is None
        assert mock_httpx["post"].call_count == 2

    def test_retry_on_503(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            _err_response(503),
            _ok_response(),
        ]
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=2)
        assert err is None
        assert mock_httpx["post"].call_count == 2

    def test_retry_on_504(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            _err_response(504),
            _ok_response(),
        ]
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=2)
        assert err is None
        assert mock_httpx["post"].call_count == 2

    def test_retry_on_500(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            _err_response(500),
            _ok_response(),
        ]
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=2)
        assert err is None
        assert mock_httpx["post"].call_count == 2

    def test_retry_on_timeout(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            httpx.TimeoutException("timed out"),
            _ok_response(),
        ]
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=2)
        assert err is None
        assert resp.status_code == 200

    def test_retry_on_network_error(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            httpx.ConnectError("refused"),
            _ok_response(),
        ]
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=2)
        assert err is None

    def test_no_retry_on_400(self, mock_httpx, mock_sleep):
        mock_httpx["post"].return_value = _err_response(400, "bad request")
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=3)
        assert err is not None
        assert "HTTP_400" in err
        assert mock_httpx["post"].call_count == 1

    def test_no_retry_on_401(self, mock_httpx, mock_sleep):
        mock_httpx["post"].return_value = _err_response(401, "unauthorized")
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=3)
        assert err is not None
        assert "HTTP_401" in err
        assert mock_httpx["post"].call_count == 1

    def test_404_with_fallback_url(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            _err_response(404, "not found"),
            _ok_response({"fallback": True}),
        ]
        resp, err = _call_edge_function_with_retry(
            URL, {}, HEADERS, "test-fn",
            fallback_url="https://edge.test/fn-fallback",
        )
        assert err is None
        assert resp.status_code == 200
        # Verify fallback URL was actually used for the second call
        assert mock_httpx["post"].call_count == 2
        second_call_url = mock_httpx["post"].call_args_list[1][0][0]
        assert second_call_url == "https://edge.test/fn-fallback"

    def test_404_with_retryable_pattern(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            _err_response(404, "Task not found"),
            _ok_response(),
        ]
        resp, err = _call_edge_function_with_retry(
            URL, {}, HEADERS, "test-fn",
            max_retries=2,
            retry_on_404_patterns=["Task not found"],
        )
        assert err is None

    def test_max_retries_exhausted(self, mock_httpx, mock_sleep):
        mock_httpx["post"].return_value = _err_response(502, "bad gateway")
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=3)
        assert err is not None
        assert "5XX_TRANSIENT" in err
        assert mock_httpx["post"].call_count == 3

    def test_timeout_exhausted(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = httpx.TimeoutException("timeout")
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=2)
        assert resp is None
        assert err is not None
        assert "TIMEOUT" in err

    def test_network_error_exhausted(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = httpx.ConnectError("refused")
        resp, err = _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", max_retries=2)
        assert resp is None
        assert "NETWORK" in err

    def test_put_method(self, mock_httpx, mock_sleep):
        mock_httpx["put"].return_value = _ok_response()
        resp, err = _call_edge_function_with_retry(
            URL, b"file-data", HEADERS, "upload", method="PUT",
        )
        assert err is None
        mock_httpx["put"].assert_called_once()

    def test_put_with_file_path(self, mock_httpx, mock_sleep, tmp_path):
        f = tmp_path / "upload.bin"
        f.write_bytes(b"binary-data")
        mock_httpx["put"].return_value = _ok_response()
        resp, err = _call_edge_function_with_retry(
            URL, str(f), HEADERS, "upload", method="PUT",
        )
        assert err is None

    def test_timeout_increases_per_attempt(self, mock_httpx, mock_sleep):
        mock_httpx["post"].side_effect = [
            _err_response(502),
            _err_response(502),
            _ok_response(),
        ]
        _call_edge_function_with_retry(URL, {}, HEADERS, "test-fn", timeout=10, max_retries=3)
        # Check timeout values passed to httpx.post
        calls = mock_httpx["post"].call_args_list
        assert calls[0].kwargs.get("timeout") == 10       # attempt 0
        assert calls[1].kwargs.get("timeout") == 25       # attempt 1: 10 + 15
        assert calls[2].kwargs.get("timeout") == 40       # attempt 2: 10 + 30

    def test_error_message_format(self, mock_httpx, mock_sleep):
        mock_httpx["post"].return_value = _err_response(400, "bad input")
        _, err = _call_edge_function_with_retry(
            URL, {}, HEADERS, "my-fn", context_id="task-123",
        )
        assert "[EDGE_FAIL" in err
        assert "my-fn" in err
        assert "task-123" in err


# ── _get_user_id_from_jwt ────────────────────────────────────────────────────

class TestGetUserIdFromJwt:
    def _make_jwt(self, payload: dict) -> str:
        header = base64.b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
        body = base64.b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        return f"{header}.{body}.fakesig"

    def test_valid_jwt(self):
        jwt = self._make_jwt({"sub": "user-abc-123"})
        assert _get_user_id_from_jwt(jwt) == "user-abc-123"

    def test_empty_string(self):
        assert _get_user_id_from_jwt("") is None

    def test_invalid_format(self):
        assert _get_user_id_from_jwt("not.a.valid-base64!") is None

    def test_no_sub_claim(self):
        jwt = self._make_jwt({"iss": "test"})
        assert _get_user_id_from_jwt(jwt) is None


# ── _is_jwt_token ────────────────────────────────────────────────────────────

class TestIsJwtToken:
    def test_valid_format(self):
        assert _is_jwt_token("header.payload.signature") is True

    def test_two_parts(self):
        assert _is_jwt_token("header.payload") is False

    def test_empty(self):
        assert _is_jwt_token("") is False

    def test_no_dots(self):
        assert _is_jwt_token("nodots") is False
