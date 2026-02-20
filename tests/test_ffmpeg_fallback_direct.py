from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_ffmpeg_fallback_direct_builds_xfade_and_succeeds(monkeypatch, tmp_path):
    import source.task_handlers.travel.ffmpeg_fallback as ff

    class _Cap:
        def __init__(self, *_args, **_kwargs):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 5:
                return 30.0
            if prop == 7:
                return 120.0
            return 0.0

        def release(self):
            pass

    fake_cv2 = SimpleNamespace(VideoCapture=_Cap, CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    captured_cmd: list[str] = []
    output_path = tmp_path / "out.mp4"

    def _run(cmd, **_kwargs):
        captured_cmd.extend(cmd)
        output_path.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setitem(sys.modules, "subprocess", SimpleNamespace(run=_run, TimeoutExpired=TimeoutError, SubprocessError=RuntimeError))
    ok = ff.attempt_ffmpeg_crossfade_fallback(
        segment_video_paths=[str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
        overlaps=[30],
        output_path=output_path,
        task_id="t1",
    )
    assert ok is True
    full_cmd = " ".join(captured_cmd)
    assert "xfade=transition=fade:duration=1.000:offset=3.000" in full_cmd


def test_ffmpeg_fallback_direct_timeout_returns_false(monkeypatch, tmp_path):
    import source.task_handlers.travel.ffmpeg_fallback as ff

    class _Cap:
        def __init__(self, *_args, **_kwargs):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 5:
                return 30.0
            if prop == 7:
                return 120.0
            return 0.0

        def release(self):
            pass

    class _Subprocess:
        TimeoutExpired = RuntimeError
        SubprocessError = RuntimeError

        @staticmethod
        def run(*_args, **_kwargs):
            raise RuntimeError("timeout")

    fake_cv2 = SimpleNamespace(VideoCapture=_Cap, CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "subprocess", _Subprocess)

    ok = ff.attempt_ffmpeg_crossfade_fallback(
        segment_video_paths=[str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
        overlaps=[30],
        output_path=Path(tmp_path / "out.mp4"),
        task_id="t2",
    )
    assert ok is False


def test_ffmpeg_fallback_direct_not_enough_videos_returns_false(tmp_path):
    import source.task_handlers.travel.ffmpeg_fallback as ff

    ok = ff.attempt_ffmpeg_crossfade_fallback(
        segment_video_paths=[str(tmp_path / "only.mp4")],
        overlaps=[],
        output_path=Path(tmp_path / "out.mp4"),
        task_id="t3",
    )
    assert ok is False


def test_ffmpeg_fallback_direct_invalid_fps_returns_false(monkeypatch, tmp_path):
    import source.task_handlers.travel.ffmpeg_fallback as ff

    class _Cap:
        def __init__(self, *_args, **_kwargs):
            pass

        def isOpened(self):
            return True

        def get(self, _prop):
            return 0.0

        def release(self):
            pass

    fake_cv2 = SimpleNamespace(VideoCapture=_Cap, CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "subprocess", SimpleNamespace(run=lambda *_a, **_k: None, TimeoutExpired=TimeoutError, SubprocessError=RuntimeError))

    ok = ff.attempt_ffmpeg_crossfade_fallback(
        segment_video_paths=[str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
        overlaps=[10],
        output_path=Path(tmp_path / "out.mp4"),
        task_id="t4",
    )
    assert ok is False


def test_ffmpeg_fallback_direct_real_integration(tmp_path):
    import cv2
    import numpy as np
    import source.task_handlers.travel.ffmpeg_fallback as ff

    def _write_video(path: Path, color: tuple[int, int, int]):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (64, 64))
        for _ in range(24):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :] = color
            writer.write(frame)
        writer.release()

    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    out = tmp_path / "out.mp4"
    _write_video(a, (0, 0, 255))
    _write_video(b, (0, 255, 0))

    ok = ff.attempt_ffmpeg_crossfade_fallback(
        segment_video_paths=[str(a), str(b)],
        overlaps=[8],
        output_path=out,
        task_id="it1",
    )
    assert ok is True
    assert out.exists()
    assert out.stat().st_size > 0
