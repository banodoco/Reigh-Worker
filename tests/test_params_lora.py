"""Tests for source/core/params/lora.py."""

import pytest

from source.core.params.lora import LoRAEntry, LoRAConfig, LoRAStatus


# ── LoRAEntry ───────────────────────────────────────────────────────────────

class TestLoRAEntry:
    def test_mark_downloaded(self):
        entry = LoRAEntry(url="https://hf.co/lora.safetensors", status=LoRAStatus.PENDING)
        entry.mark_downloaded("/local/path/lora.safetensors")
        assert entry.local_path == "/local/path/lora.safetensors"
        assert entry.filename == "lora.safetensors"
        assert entry.status == LoRAStatus.DOWNLOADED

    def test_get_effective_path_prefers_local(self):
        entry = LoRAEntry(filename="f.safetensors", local_path="/abs/f.safetensors")
        assert entry.get_effective_path() == "/abs/f.safetensors"

    def test_get_effective_path_fallback_to_filename(self):
        entry = LoRAEntry(filename="f.safetensors", local_path=None)
        assert entry.get_effective_path() == "f.safetensors"

    def test_get_effective_path_none(self):
        entry = LoRAEntry()
        assert entry.get_effective_path() is None

    def test_is_phase_config_multiplier_true(self):
        entry = LoRAEntry(multiplier="0.9;0.5")
        assert entry.is_phase_config_multiplier() is True

    def test_is_phase_config_multiplier_false_float(self):
        entry = LoRAEntry(multiplier=1.0)
        assert entry.is_phase_config_multiplier() is False

    def test_is_phase_config_multiplier_false_string(self):
        entry = LoRAEntry(multiplier="0.9")
        assert entry.is_phase_config_multiplier() is False


# ── LoRAConfig.from_params ──────────────────────────────────────────────────

class TestLoRAConfigFromParams:
    def test_empty(self):
        config = LoRAConfig.from_params({})
        assert len(config.entries) == 0

    def test_local_filenames(self):
        config = LoRAConfig.from_params({"activated_loras": ["a.safetensors", "b.safetensors"]})
        assert len(config.entries) == 2
        assert all(e.status == LoRAStatus.LOCAL for e in config.entries)

    def test_urls(self):
        config = LoRAConfig.from_params({"activated_loras": ["https://hf.co/lora.safetensors"]})
        assert len(config.entries) == 1
        assert config.entries[0].status == LoRAStatus.PENDING
        assert config.entries[0].url == "https://hf.co/lora.safetensors"

    def test_mixed_local_and_url(self):
        config = LoRAConfig.from_params({
            "activated_loras": ["local.safetensors", "https://hf.co/remote.safetensors"],
            "loras_multipliers": "0.8 1.2",
        })
        assert len(config.entries) == 2
        assert config.entries[0].status == LoRAStatus.LOCAL
        assert config.entries[1].status == LoRAStatus.PENDING
        assert config.entries[0].multiplier == 0.8
        assert config.entries[1].multiplier == 1.2

    def test_multiplier_space_separated(self):
        config = LoRAConfig.from_params({
            "activated_loras": ["a.safetensors", "b.safetensors"],
            "loras_multipliers": "0.5 1.5",
        })
        assert config.entries[0].multiplier == 0.5
        assert config.entries[1].multiplier == 1.5

    def test_multiplier_comma_separated(self):
        config = LoRAConfig.from_params({
            "activated_loras": ["a.safetensors"],
            "loras_multipliers": "0.7",
        })
        assert config.entries[0].multiplier == 0.7

    def test_multiplier_phase_config(self):
        config = LoRAConfig.from_params({
            "activated_loras": ["a.safetensors", "b.safetensors"],
            "loras_multipliers": "0.9;0.5 1.1;1.1",
        })
        assert config.entries[0].multiplier == "0.9;0.5"
        assert config.entries[1].multiplier == "1.1;1.1"

    def test_additional_loras_merge(self):
        config = LoRAConfig.from_params({
            "activated_loras": ["existing.safetensors"],
            "additional_loras": {"https://hf.co/new.safetensors": 0.8},
        })
        assert len(config.entries) == 2
        filenames = [e.filename for e in config.entries]
        assert "existing.safetensors" in filenames
        assert "new.safetensors" in filenames

    def test_deduplication(self):
        url = "https://hf.co/lora.safetensors"
        config = LoRAConfig.from_params({
            "activated_loras": [url],
            "loras_multipliers": "0.5;0.8",
            "additional_loras": {url: 1.0},
        })
        assert len(config.entries) == 1
        # activated_loras entry wins — keeps phase-config multiplier
        assert config.entries[0].multiplier == "0.5;0.8"


# ── LoRAConfig.from_phase_config ────────────────────────────────────────────

class TestLoRAConfigFromPhaseConfig:
    def test_two_phase(self, sample_phase_config):
        config = LoRAConfig.from_phase_config(sample_phase_config)
        assert len(config.entries) == 2
        for entry in config.entries:
            vals = entry.multiplier.split(";")
            assert len(vals) == 2

    def test_three_phase(self):
        pc = {
            "phases": [
                {"loras": [{"url": "https://hf.co/a.safetensors", "multiplier": 1.0}]},
                {"loras": [{"url": "https://hf.co/a.safetensors", "multiplier": 0.5}]},
                {"loras": [{"url": "https://hf.co/a.safetensors", "multiplier": 0.0}]},
            ]
        }
        config = LoRAConfig.from_phase_config(pc)
        assert len(config.entries) == 1
        assert config.entries[0].multiplier == "1.0;0.5;0.0"

    def test_empty_phases(self):
        config = LoRAConfig.from_phase_config({"phases": []})
        assert len(config.entries) == 0


# ── LoRAConfig.from_segment_loras ───────────────────────────────────────────

class TestLoRAConfigFromSegmentLoras:
    def test_local_paths(self):
        segment_loras = [{"path": "style.safetensors", "strength": 0.8}]
        config = LoRAConfig.from_segment_loras(segment_loras)
        assert len(config.entries) == 1
        assert config.entries[0].multiplier == 0.8
        assert config.entries[0].status == LoRAStatus.LOCAL

    def test_urls(self):
        segment_loras = [{"path": "https://hf.co/lora.safetensors", "strength": 1.0}]
        config = LoRAConfig.from_segment_loras(segment_loras)
        assert config.entries[0].status == LoRAStatus.PENDING

    def test_empty_list(self):
        config = LoRAConfig.from_segment_loras([])
        assert len(config.entries) == 0

    def test_empty_path_skipped(self):
        config = LoRAConfig.from_segment_loras([{"path": "", "strength": 1.0}])
        assert len(config.entries) == 0


# ── LoRAConfig.merge ────────────────────────────────────────────────────────

class TestLoRAConfigMerge:
    def test_no_overlap(self):
        a = LoRAConfig(entries=[LoRAEntry(filename="a.safetensors")])
        b = LoRAConfig(entries=[LoRAEntry(filename="b.safetensors")])
        merged = a.merge(b)
        assert len(merged.entries) == 2

    def test_duplicate_replacement(self):
        a = LoRAConfig(entries=[LoRAEntry(filename="same.safetensors", multiplier=0.5)])
        b = LoRAConfig(entries=[LoRAEntry(filename="same.safetensors", multiplier=1.0)])
        merged = a.merge(b)
        assert len(merged.entries) == 1
        assert merged.entries[0].multiplier == 1.0

    def test_phase_config_takes_precedence(self):
        a = LoRAConfig(entries=[LoRAEntry(filename="lora.safetensors", multiplier=1.0)])
        b = LoRAConfig(entries=[LoRAEntry(filename="lora.safetensors", multiplier="0.9;0.5")])
        merged = a.merge(b)
        assert merged.entries[0].multiplier == "0.9;0.5"


# ── LoRAConfig.to_wgp_format ───────────────────────────────────────────────

class TestLoRAConfigToWgpFormat:
    def test_excludes_pending(self):
        config = LoRAConfig(entries=[
            LoRAEntry(url="https://hf.co/a.safetensors", filename="a.safetensors", status=LoRAStatus.PENDING),
        ])
        wgp = config.to_wgp_format()
        assert wgp["activated_loras"] == []
        assert wgp["loras_multipliers"] == ""

    def test_includes_downloaded(self):
        entry = LoRAEntry(filename="a.safetensors", local_path="/dl/a.safetensors", status=LoRAStatus.DOWNLOADED, multiplier=0.8)
        config = LoRAConfig(entries=[entry])
        wgp = config.to_wgp_format()
        assert wgp["activated_loras"] == ["/dl/a.safetensors"]
        assert wgp["loras_multipliers"] == "0.8"

    def test_includes_local(self):
        config = LoRAConfig(entries=[
            LoRAEntry(filename="local.safetensors", status=LoRAStatus.LOCAL, multiplier=1.0),
        ])
        wgp = config.to_wgp_format()
        assert wgp["activated_loras"] == ["local.safetensors"]

    def test_empty(self):
        config = LoRAConfig()
        wgp = config.to_wgp_format()
        assert wgp == {"activated_loras": [], "loras_multipliers": ""}

    def test_multiplier_formatting_space_separated(self):
        config = LoRAConfig(entries=[
            LoRAEntry(filename="a.safetensors", status=LoRAStatus.LOCAL, multiplier="0.9;0.5"),
            LoRAEntry(filename="b.safetensors", status=LoRAStatus.LOCAL, multiplier="1.1;1.1"),
        ])
        wgp = config.to_wgp_format()
        assert wgp["loras_multipliers"] == "0.9;0.5 1.1;1.1"


# ── LoRAConfig.validate ────────────────────────────────────────────────────

class TestLoRAConfigValidate:
    def test_valid_config(self):
        config = LoRAConfig(entries=[
            LoRAEntry(filename="lora.safetensors", status=LoRAStatus.LOCAL),
        ])
        assert config.validate() == []

    def test_pending_without_url(self):
        config = LoRAConfig(entries=[
            LoRAEntry(filename="lora.safetensors", status=LoRAStatus.PENDING, url=None),
        ])
        errors = config.validate()
        assert len(errors) == 1
        assert "no URL" in errors[0]
