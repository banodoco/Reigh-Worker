"""Tests for source/models/lora/lora_paths.py."""

from pathlib import Path

from source.models.lora.lora_paths import get_lora_search_dirs, get_lora_dir_for_model


class TestGetLoraSearchDirs:
    def test_basic_dirs(self):
        wan = Path("/fake/Wan2GP")
        dirs = get_lora_search_dirs(wan)
        assert wan / "loras" in dirs
        assert wan / "loras" / "wan" in dirs
        assert wan / "loras_i2v" in dirs
        assert wan / "loras_hunyuan" in dirs
        assert wan / "loras_flux" in dirs
        assert wan / "loras_qwen" in dirs
        assert wan / "loras_ltxv" in dirs
        assert wan / "loras_kandinsky5" in dirs

    def test_without_repo_root(self):
        wan = Path("/fake/Wan2GP")
        dirs = get_lora_search_dirs(wan)
        # Should not include repo root dirs
        assert all("fake/loras" not in str(d) or "Wan2GP" in str(d) for d in dirs)

    def test_with_repo_root(self):
        wan = Path("/project/Wan2GP")
        root = Path("/project")
        dirs = get_lora_search_dirs(wan, repo_root=root)
        assert root / "loras" in dirs
        assert root / "loras" / "wan" in dirs


class TestGetLoraDirForModel:
    def test_wan_models(self):
        wan = Path("/w")
        assert get_lora_dir_for_model("wan", wan) == wan / "loras" / "wan"
        assert get_lora_dir_for_model("vace_14B", wan) == wan / "loras" / "wan"
        assert get_lora_dir_for_model("WAN_22", wan) == wan / "loras" / "wan"

    def test_hunyuan_models(self):
        wan = Path("/w")
        assert get_lora_dir_for_model("hunyuan", wan) == wan / "loras_hunyuan"
        assert get_lora_dir_for_model("hunyuan_i2v", wan) == wan / "loras_hunyuan_i2v"
        assert get_lora_dir_for_model("hunyuan_1_5", wan) == wan / "loras_hunyuan" / "1.5"
        assert get_lora_dir_for_model("hunyuan_1.5", wan) == wan / "loras_hunyuan" / "1.5"

    def test_flux_models(self):
        wan = Path("/w")
        assert get_lora_dir_for_model("flux", wan) == wan / "loras_flux"
        assert get_lora_dir_for_model("FLUX_schnell", wan) == wan / "loras_flux"

    def test_qwen_models(self):
        wan = Path("/w")
        assert get_lora_dir_for_model("qwen_image_edit", wan) == wan / "loras_qwen"

    def test_ltxv_models(self):
        wan = Path("/w")
        assert get_lora_dir_for_model("ltxv_13B", wan) == wan / "loras_ltxv"

    def test_kandinsky_models(self):
        wan = Path("/w")
        assert get_lora_dir_for_model("kandinsky5", wan) == wan / "loras_kandinsky5"

    def test_empty_string_returns_default(self):
        wan = Path("/w")
        assert get_lora_dir_for_model("", wan) == wan / "loras"

    def test_none_returns_default(self):
        wan = Path("/w")
        assert get_lora_dir_for_model(None, wan) == wan / "loras"

    def test_unknown_model_returns_default(self):
        wan = Path("/w")
        assert get_lora_dir_for_model("some_new_model", wan) == wan / "loras"
