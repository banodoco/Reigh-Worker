"""Tests for source/core/params/task.py."""

from source.core.params.generation import GenerationConfig
from source.core.params.vace import VACEConfig
from source.core.params.phase import PhaseConfig
from source.core.params.task import TaskConfig


class TestTaskConfigFromParams:
    def test_integrates_all_groups(self):
        cfg = TaskConfig.from_params({
            "prompt": "a landscape",
            "resolution": "960x544",
            "video_guide": "/guide.mp4",
            "activated_loras": ["style.safetensors"],
        }, task_id="t1", task_type="vace", model="wan_14B")
        assert cfg.generation.prompt == "a landscape"
        assert cfg.vace.guide_path == "/guide.mp4"
        assert len(cfg.lora.entries) == 1
        assert cfg.task_id == "t1"
        assert cfg.model == "wan_14B"

    def test_model_from_params_fallback(self):
        """model_name in params is used when context.model is not given."""
        cfg = TaskConfig.from_params({"model_name": "vace_14B"})
        assert cfg.model == "vace_14B"

    def test_context_model_takes_precedence(self):
        cfg = TaskConfig.from_params({"model_name": "fallback"}, model="primary")
        assert cfg.model == "primary"

    def test_handled_keys_excluded_from_extra(self):
        """Known param keys should be consumed, not leak into extra_params."""
        cfg = TaskConfig.from_params({
            "prompt": "test",
            "resolution": "960x544",
            "seed": 42,
            "video_guide": "/path",
            "activated_loras": [],
            "custom": "kept",
        })
        assert "prompt" not in cfg.extra_params
        assert "resolution" not in cfg.extra_params
        assert "seed" not in cfg.extra_params
        assert "video_guide" not in cfg.extra_params
        assert cfg.extra_params["custom"] == "kept"


class TestTaskConfigFromDbTask:
    def test_flattens_orchestrator_details(self):
        db_params = {
            "orchestrator_details": {"resolution": "1920x1080", "steps": 20},
            "prompt": "db test",
        }
        cfg = TaskConfig.from_db_task(db_params, task_id="t1")
        assert cfg.generation.prompt == "db test"
        assert cfg.generation.resolution == "1920x1080"
        assert cfg.generation.num_inference_steps == 20


class TestTaskConfigFromSegmentParams:
    def test_precedence_ordering(self):
        """individual_params > segment_params > orchestrator_payload."""
        cfg = TaskConfig.from_segment_params(
            segment_params={"prompt": "segment", "resolution": "960x544"},
            orchestrator_payload={"prompt": "orch", "resolution": "640x360", "seed": 42},
            individual_params={"prompt": "individual"},
            task_id="t1",
        )
        assert cfg.generation.prompt == "individual"   # highest wins
        assert cfg.generation.resolution == "960x544"   # segment wins over orch
        assert cfg.generation.seed == 42                 # only in orch, propagates

    def test_none_individual_params(self):
        """individual_params=None should not crash."""
        cfg = TaskConfig.from_segment_params(
            segment_params={"prompt": "seg"},
            orchestrator_payload={"resolution": "640x360"},
        )
        assert cfg.generation.prompt == "seg"
        assert cfg.generation.resolution == "640x360"


class TestTaskConfigToWgpFormat:
    def test_combines_all_groups(self):
        cfg = TaskConfig(
            generation=GenerationConfig(prompt="test", seed=42),
            vace=VACEConfig(guide_path="/guide.mp4"),
            model="wan_14B",
        )
        wgp = cfg.to_wgp_format()
        assert wgp["prompt"] == "test"
        assert wgp["seed"] == 42
        assert wgp["video_guide"] == "/guide.mp4"
        assert wgp["model"] == "wan_14B"
        assert wgp["activated_loras"] == []  # LoRA defaults

    def test_extra_params_passthrough(self):
        cfg = TaskConfig(
            generation=GenerationConfig(prompt="test"),
            extra_params={"custom": "value"},
        )
        assert cfg.to_wgp_format()["custom"] == "value"

    def test_no_model_omitted(self):
        cfg = TaskConfig(generation=GenerationConfig(prompt="test"))
        assert "model" not in cfg.to_wgp_format()


class TestTaskConfigValidate:
    def test_valid_config(self):
        cfg = TaskConfig(generation=GenerationConfig(video_length=81))
        assert cfg.validate() == []

    def test_propagates_generation_errors(self):
        cfg = TaskConfig(generation=GenerationConfig(video_length=80))
        errors = cfg.validate()
        assert any("4N+1" in e for e in errors)

    def test_propagates_phase_errors(self):
        cfg = TaskConfig(
            phase=PhaseConfig(raw_config={"phases": [1]}, parsed_output={})
        )
        errors = cfg.validate()
        assert any("failed to parse" in e.lower() for e in errors)


class TestTaskConfigLogSummary:
    def test_calls_log_func_with_summary(self):
        messages = []
        cfg = TaskConfig(
            generation=GenerationConfig(
                prompt="test", resolution="960x544",
                video_length=81, num_inference_steps=6,
            ),
            task_id="t1", task_type="vace", model="wan_14B",
        )
        cfg.log_summary(log_func=messages.append)
        combined = "\n".join(messages)
        assert "t1" in combined
        assert "wan_14B" in combined
        assert "960x544" in combined
