"""
Tests for source/models/wgp/lora_setup.py — setup_loras_for_model.

Covers:
  - Happy path: LoRAs discovered and state updated
  - No LoRAs found: empty state
  - Exception path: state defaults set on error
  - Many LoRAs: log truncation

Run with: python -m pytest tests/test_lora_setup.py -v
"""

import sys
from unittest.mock import patch, MagicMock

# Patch target: the logger as it lives in the lora_setup module's namespace
_LOGGER = "source.models.wgp.lora_setup.model_logger"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _make_orchestrator():
    """Create a mock orchestrator with a state dict."""
    orch = MagicMock()
    orch.state = {}
    return orch


class TestSetupLorasSuccess:
    """setup_loras_for_model should populate orchestrator.state on success."""

    @patch(_LOGGER)
    def test_loras_discovered(self, mock_logger):
        from source.models.wgp.lora_setup import setup_loras_for_model

        fake_wgp = MagicMock()
        fake_wgp.get_lora_dir.return_value = "/fake/lora/dir"
        fake_wgp.setup_loras.return_value = (
            ["/path/lora1.safetensors", "/path/lora2.safetensors"],  # loras
            ["lora1", "lora2"],     # loras_names
            {"preset1": {}},        # loras_presets
            [],                     # default_loras_choices
            "",                     # default_loras_multis_str
            "",                     # default_lora_preset_prompt
            "",                     # default_lora_preset
        )

        orch = _make_orchestrator()

        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            setup_loras_for_model(orch, "t2v")

        assert orch.state["loras"] == ["/path/lora1.safetensors", "/path/lora2.safetensors"]
        assert orch.state["loras_names"] == ["lora1", "lora2"]
        assert orch.state["loras_presets"] == {"preset1": {}}
        mock_logger.debug.assert_called_once()
        assert "Discovered 2 LoRAs" in mock_logger.debug.call_args[0][0]

    @patch(_LOGGER)
    def test_no_loras_found(self, mock_logger):
        from source.models.wgp.lora_setup import setup_loras_for_model

        fake_wgp = MagicMock()
        fake_wgp.get_lora_dir.return_value = "/empty/dir"
        fake_wgp.setup_loras.return_value = ([], [], {}, [], "", "", "")

        orch = _make_orchestrator()

        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            setup_loras_for_model(orch, "i2v")

        assert orch.state["loras"] == []
        assert orch.state["loras_names"] == []
        assert orch.state["loras_presets"] == {}
        mock_logger.debug.assert_called_once()
        assert "No LoRAs found" in mock_logger.debug.call_args[0][0]


class TestSetupLorasError:
    """setup_loras_for_model should set empty defaults on error."""

    @patch(_LOGGER)
    def test_runtime_error_sets_defaults(self, mock_logger):
        from source.models.wgp.lora_setup import setup_loras_for_model

        fake_wgp = MagicMock()
        fake_wgp.setup_loras.side_effect = RuntimeError("CUDA OOM")
        fake_wgp.get_lora_dir.return_value = "/dir"

        orch = _make_orchestrator()

        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            setup_loras_for_model(orch, "vace_14B")

        assert orch.state["loras"] == []
        assert orch.state["loras_names"] == []
        assert orch.state["loras_presets"] == {}
        mock_logger.warning.assert_called_once()
        assert "LoRA discovery failed" in mock_logger.warning.call_args[0][0]

    @patch(_LOGGER)
    def test_os_error_sets_defaults(self, mock_logger):
        from source.models.wgp.lora_setup import setup_loras_for_model

        fake_wgp = MagicMock()
        fake_wgp.get_lora_dir.side_effect = OSError("permission denied")

        orch = _make_orchestrator()

        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            setup_loras_for_model(orch, "t2v")

        assert orch.state["loras"] == []
        mock_logger.warning.assert_called_once()

    @patch(_LOGGER)
    def test_value_error_sets_defaults(self, mock_logger):
        from source.models.wgp.lora_setup import setup_loras_for_model

        fake_wgp = MagicMock()
        fake_wgp.get_lora_dir.return_value = "/dir"
        fake_wgp.setup_loras.side_effect = ValueError("bad model type")

        orch = _make_orchestrator()

        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            setup_loras_for_model(orch, "bad")

        assert orch.state["loras"] == []
        assert orch.state["loras_names"] == []
        assert orch.state["loras_presets"] == {}


class TestSetupLorasMany:
    """Edge cases: many LoRAs (>3 should be truncated in log)."""

    @patch(_LOGGER)
    def test_more_than_three_loras_truncated_in_log(self, mock_logger):
        from source.models.wgp.lora_setup import setup_loras_for_model

        fake_wgp = MagicMock()
        fake_wgp.get_lora_dir.return_value = "/dir"
        lora_paths = [f"/dir/lora{i}.safetensors" for i in range(5)]
        fake_wgp.setup_loras.return_value = (
            lora_paths,
            [f"lora{i}" for i in range(5)],
            {},
            [],
            "",
            "",
            "",
        )

        orch = _make_orchestrator()

        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            setup_loras_for_model(orch, "t2v")

        msg = mock_logger.debug.call_args[0][0]
        assert "Discovered 5 LoRAs" in msg
        assert "..." in msg  # truncation indicator

    @patch(_LOGGER)
    def test_exactly_three_loras_no_truncation(self, mock_logger):
        from source.models.wgp.lora_setup import setup_loras_for_model

        fake_wgp = MagicMock()
        fake_wgp.get_lora_dir.return_value = "/dir"
        lora_paths = [f"/dir/lora{i}.safetensors" for i in range(3)]
        fake_wgp.setup_loras.return_value = (
            lora_paths,
            [f"lora{i}" for i in range(3)],
            {},
            [],
            "",
            "",
            "",
        )

        orch = _make_orchestrator()

        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            setup_loras_for_model(orch, "t2v")

        msg = mock_logger.debug.call_args[0][0]
        assert "Discovered 3 LoRAs" in msg
        assert "..." not in msg
