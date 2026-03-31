"""Security tests for Kronos WebUI — path traversal, auth, error sanitization, input validation."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Flask test-client fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Create a Flask test client with a known API key."""
    os.environ["KRONOS_API_KEY"] = "test-secret-key-12345"
    # Ensure DATA_DIR is predictable
    os.environ["KRONOS_DATA_DIR"] = str(
        Path(__file__).resolve().parent.parent / "data"
    )
    # Must import *after* setting env vars so the module picks them up
    import importlib
    import webui.app as app_module
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c, app_module


@pytest.fixture()
def api_client(client):
    """Return (test_client, app_module, valid_api_key)."""
    test_client, app_module = client
    return test_client, app_module, "test-secret-key-12345"


# ===================================================================
# 1. Path traversal prevention in load_data_file()
# ===================================================================

class TestPathTraversal:
    """Verify load_data_file() rejects paths outside DATA_DIR."""

    def test_dotdot_slash_etc_passwd(self, api_client):
        _, app_module, _ = api_client
        df, err = app_module.load_data_file("../../../etc/passwd")
        assert df is None
        assert err == "Access denied"

    def test_absolute_path_outside_data_dir(self, api_client):
        _, app_module, _ = api_client
        df, err = app_module.load_data_file("/etc/shadow")
        assert df is None
        assert err == "Access denied"

    def test_absolute_root(self, api_client):
        _, app_module, _ = api_client
        df, err = app_module.load_data_file("/")
        assert df is None
        # Either "Access denied" (resolved outside DATA_DIR) or "File not found"
        assert err in ("Access denied", "File not found")

    def test_null_byte_in_path(self, api_client):
        _, app_module, _ = api_client
        # Null bytes should be rejected or cause failure
        df, err = app_module.load_data_file("data.csv\x00../../etc/passwd")
        # Path with null bytes should not succeed
        assert df is None

    def test_symlink_escape(self, api_client, tmp_path):
        """Create a symlink pointing outside DATA_DIR and verify rejection."""
        _, app_module, _ = api_client
        link = tmp_path / "evil_link.csv"
        target = Path("/etc/passwd")
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Cannot create symlinks in this environment")
        df, err = app_module.load_data_file(str(link))
        assert df is None

    def test_nested_traversal(self, api_client):
        _, app_module, _ = api_client
        df, err = app_module.load_data_file("foo/../../../../../../etc/hosts")
        assert df is None
        assert err == "Access denied"

    def test_valid_but_nonexistent_file(self, api_client):
        _, app_module, _ = api_client
        data_dir = app_module.DATA_DIR
        safe_path = str(data_dir / "nonexistent_safe_file.csv")
        df, err = app_module.load_data_file(safe_path)
        assert df is None
        assert err == "File not found"


# ===================================================================
# 2. API key authentication (@require_api_key)
# ===================================================================

class TestApiKeyAuth:
    """Verify @require_api_key enforces authentication on protected endpoints."""

    def test_missing_api_key_returns_401(self, api_client):
        client, _, _ = api_client
        resp = client.post("/api/load-data", json={"file_path": "/tmp/x"})
        assert resp.status_code == 401
        data = resp.get_json()
        assert "error" in data

    def test_wrong_api_key_returns_401(self, api_client):
        client, _, _ = api_client
        resp = client.post(
            "/api/load-data",
            json={"file_path": "/tmp/x"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_valid_api_key_passes(self, api_client):
        client, _, key = api_client
        # /api/load-data with valid key should get past auth (may 400 for bad data, but NOT 401)
        resp = client.post(
            "/api/load-data",
            json={"file_path": "/nonexistent"},
            headers={"X-API-Key": key},
        )
        assert resp.status_code != 401

    def test_empty_api_key_returns_401(self, api_client):
        client, _, _ = api_client
        resp = client.post(
            "/api/load-data",
            json={"file_path": "/tmp/x"},
            headers={"X-API-Key": ""},
        )
        assert resp.status_code == 401

    def test_compare_digest_used(self, api_client):
        """Verify secrets.compare_digest is used for timing-attack resistance."""
        _, app_module, _ = api_client
        import inspect
        source = inspect.getsource(app_module.require_api_key)
        assert "compare_digest" in source

    def test_predict_endpoint_requires_auth(self, api_client):
        client, _, _ = api_client
        resp = client.post("/api/predict", json={})
        assert resp.status_code == 401

    def test_load_model_requires_auth(self, api_client):
        client, _, _ = api_client
        resp = client.post("/api/load-model", json={})
        assert resp.status_code == 401

    def test_public_endpoints_no_auth(self, api_client):
        """Endpoints without @require_api_key should work without key."""
        client, _, _ = api_client
        # /api/data-files has no decorator
        resp = client.get("/api/data-files")
        assert resp.status_code == 200
        # /api/available-models has no decorator
        resp = client.get("/api/available-models")
        assert resp.status_code == 200
        # /api/model-status has no decorator
        resp = client.get("/api/model-status")
        assert resp.status_code == 200


# ===================================================================
# 3. Error message sanitization
# ===================================================================

class TestErrorSanitization:
    """Verify 500 responses do not leak internal details or stack traces."""

    def test_predict_error_no_stack_trace(self, api_client):
        client, _, key = api_client
        resp = client.post(
            "/api/predict",
            json={"file_path": "/nonexistent"},
            headers={"X-API-Key": key},
        )
        data = resp.get_json()
        # Should be generic message, no Python traceback
        if resp.status_code == 500:
            msg = json.dumps(data)
            assert "Traceback" not in msg
            assert "Exception" not in msg

    def test_load_data_error_generic(self, api_client):
        """Force an internal error and check the 500 response is generic."""
        client, app_module, key = api_client
        # Patch load_data_file to raise, which triggers the outer except → generic 500
        with patch("webui.app.load_data_file", side_effect=RuntimeError("secret-internal-boom")):
            resp = client.post(
                "/api/load-data",
                json={"file_path": "/some/path"},
                headers={"X-API-Key": key},
            )
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data
        msg = json.dumps(data)
        assert "RuntimeError" not in msg
        assert "secret-internal-boom" not in msg

    def test_predict_error_no_internal_details(self, api_client):
        client, app_module, key = api_client
        # Patch load_data_file to raise deep inside /api/predict handler
        with patch("webui.app.load_data_file", side_effect=RuntimeError("secret-details")):
            resp = client.post(
                "/api/predict",
                json={"file_path": "/some/path"},
                headers={"X-API-Key": key},
            )
        assert resp.status_code == 500
        data = resp.get_json()
        msg = json.dumps(data)
        assert "secret-details" not in msg

    def test_load_model_error_generic(self, api_client):
        """Model loading failures return generic messages."""
        client, app_module, key = api_client
        # MODEL_AVAILABLE is False in test env (no torch/HF), so this returns 400 not 500
        resp = client.post(
            "/api/load-model",
            json={"model_key": "kronos-small"},
            headers={"X-API-Key": key},
        )
        data = resp.get_json()
        msg = json.dumps(data)
        assert "Traceback" not in msg


# ===================================================================
# 4. Input validation
# ===================================================================

class TestInputValidation:
    """Verify malformed or missing input returns 400 with clear errors."""

    def test_load_data_missing_file_path(self, api_client):
        client, _, key = api_client
        resp = client.post(
            "/api/load-data",
            json={},
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_load_data_empty_file_path(self, api_client):
        client, _, key = api_client
        resp = client.post(
            "/api/load-data",
            json={"file_path": ""},
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 400

    def test_load_data_null_file_path(self, api_client):
        client, _, key = api_client
        resp = client.post(
            "/api/load-data",
            json={"file_path": None},
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 400

    def test_predict_missing_file_path(self, api_client):
        client, _, key = api_client
        resp = client.post(
            "/api/predict",
            json={},
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_load_model_invalid_model_key(self, api_client):
        client, _, key = api_client
        resp = client.post(
            "/api/load-model",
            json={"model_key": "nonexistent-model"},
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 400

    def test_malformed_json_returns_400_or_error(self, api_client):
        client, _, key = api_client
        resp = client.post(
            "/api/load-data",
            data="not-json{{{",
            content_type="application/json",
            headers={"X-API-Key": key},
        )
        # Flask returns 400 on bad JSON or the handler may 500
        assert resp.status_code in (400, 500)

    def test_predict_invalid_lookback_type(self, api_client):
        """Non-numeric lookback should cause a controlled error."""
        client, _, key = api_client
        resp = client.post(
            "/api/predict",
            json={"file_path": "/tmp/x", "lookback": "not-a-number"},
            headers={"X-API-Key": key},
        )
        # Should not be 200 — either 400 or 500
        assert resp.status_code != 200

    def test_predict_invalid_pred_len_type(self, api_client):
        client, _, key = api_client
        resp = client.post(
            "/api/predict",
            json={"file_path": "/tmp/x", "pred_len": "abc"},
            headers={"X-API-Key": key},
        )
        assert resp.status_code != 200

    def test_unsupported_file_format(self, api_client, tmp_path):
        """load_data_file should reject non-CSV/feather files inside DATA_DIR."""
        _, app_module, _ = api_client
        # Create a .txt file inside DATA_DIR
        bad_file = app_module.DATA_DIR / "test_unsupported.txt"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("hello")
        try:
            df, err = app_module.load_data_file(str(bad_file))
            assert df is None
            assert "Unsupported" in err
        finally:
            bad_file.unlink(missing_ok=True)
