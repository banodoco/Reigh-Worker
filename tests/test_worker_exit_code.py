from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

import pytest

from source.runtime import supervisor


def _write_fake_host(source: Path, *, exit_code: int = 0, sleep_seconds: float = 0.05) -> None:
    module = source / "astrid" / "core" / "execution" / "generic_host.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import json, os, signal, sys, time\n"
        "from pathlib import Path\n"
        "args = sys.argv\n"
        "ready = Path(args[args.index('--ready-file') + 1])\n"
        "support = Path(args[args.index('--support-root') + 1])\n"
        "def stop(signum, _frame):\n"
        "    (support / 'host-signal.json').write_text(json.dumps({'signal': signum, 'pid': os.getpid()}), encoding='utf-8')\n"
        "    raise SystemExit(128 + signum)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        "ready.parent.mkdir(parents=True, exist_ok=True)\n"
        "ready.write_text(json.dumps({'status': 'ready', 'pid': os.getpid()}), encoding='utf-8')\n"
        f"time.sleep({sleep_seconds})\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )


def _config(tmp_path: Path, *, exit_code: int = 0, sleep_seconds: float = 0.05) -> supervisor.HostLaunchConfig:
    source = tmp_path / "Astrid"
    (source / "astrid" / "packs").mkdir(parents=True)
    support = tmp_path / "support"
    support.mkdir()
    credential = support / "credential"
    credential.write_text("worker-token", encoding="utf-8")
    credential.chmod(0o600)
    manifest = support / "boot-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    _write_fake_host(source, exit_code=exit_code, sleep_seconds=sleep_seconds)
    return supervisor.HostLaunchConfig(
        host_python=Path(sys.executable).resolve(),
        source_checkout=source.resolve(),
        pack_root=(source / "astrid" / "packs").resolve(),
        runtime_endpoint="http://runtime.test",
        credential_file=credential.resolve(),
        support_root=support.resolve(),
        runtime_instance_id="runtime-instance",
        ready_file=(support / "host-ready.json").resolve(),
        state_file=(support / "worker-state.json").resolve(),
        boot_manifest_path=manifest.resolve(),
        boot_manifest_hash="sha256:test",
    )


def test_build_command_is_fixed_to_generic_pack_host(tmp_path: Path) -> None:
    config = _config(tmp_path)
    argv = config.argv()
    assert argv[:4] == [str(Path(sys.executable).resolve()), "-m", supervisor.GENERIC_HOST_MODULE, "run"]
    assert "--executor-id" in argv
    assert supervisor.GENERIC_HOST_EXECUTOR_ID in argv
    assert "REIGH_BACKEND" not in argv
    assert "WORKER_BACKEND" not in argv


def test_host_config_rejects_relative_or_missing_interpreter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    env = {
        "ASTRID_HOST_PYTHON": "python3",
        "ASTRID_HOST_SOURCE_CHECKOUT": str(config.source_checkout),
        "ASTRID_HOST_PACK_ROOT": str(config.pack_root),
        "ASTRID_HOST_RUNTIME_ENDPOINT": config.runtime_endpoint,
        "ASTRID_HOST_CREDENTIAL_FILE": str(config.credential_file),
        "ASTRID_HOST_SUPPORT_ROOT": str(config.support_root),
        "ASTRID_HOST_RUNTIME_INSTANCE_ID": config.runtime_instance_id,
        "ASTRID_HOST_READY_FILE": str(config.ready_file),
        "ASTRID_HOST_STATE_FILE": str(config.state_file),
        "ASTRID_HOST_BOOT_MANIFEST_PATH": str(config.boot_manifest_path),
        "ASTRID_HOST_BOOT_MANIFEST_HASH": config.boot_manifest_hash,
    }
    with pytest.raises(supervisor.LauncherConfigurationError, match="absolute"):
        supervisor.HostLaunchConfig.from_environment(env)


def test_host_environment_is_an_explicit_allowlist(tmp_path: Path) -> None:
    config = _config(tmp_path)
    child = supervisor._host_environment(
        {"PATH": "/bin", "HOME": "/tmp", "SECRET": "must-not-pass", "PYTHONPATH": "/wrong"},
        config,
    )
    assert child["PATH"] == "/bin"
    assert child["PYTHONPATH"] == str(config.source_checkout)
    assert "SECRET" not in child
    assert "HOME" not in child


def test_atomic_state_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    supervisor._atomic_write_json(path, {"status": "starting"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "starting"


def test_host_exit_code_is_returned_unchanged_and_state_is_final(tmp_path: Path) -> None:
    config = _config(tmp_path, exit_code=17)
    result = supervisor.launch_generic_pack_host(config, environ={"PATH": "/bin"})
    assert result == 17
    state = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert state["status"] == "exited"
    assert state["returncode"] == 17
    assert state["ready"] is True
    assert stat.S_IMODE(config.state_file.stat().st_mode) == 0o600


def test_worker_forwards_signal_to_owned_host_group(tmp_path: Path) -> None:
    config = _config(tmp_path, sleep_seconds=30)
    env = os.environ.copy()
    env.update(
        {
            "ASTRID_HOST_PYTHON": str(config.host_python),
            "ASTRID_HOST_SOURCE_CHECKOUT": str(config.source_checkout),
            "ASTRID_HOST_PACK_ROOT": str(config.pack_root),
            "ASTRID_HOST_RUNTIME_ENDPOINT": config.runtime_endpoint,
            "ASTRID_HOST_CREDENTIAL_FILE": str(config.credential_file),
            "ASTRID_HOST_SUPPORT_ROOT": str(config.support_root),
            "ASTRID_HOST_RUNTIME_INSTANCE_ID": config.runtime_instance_id,
            "ASTRID_HOST_READY_FILE": str(config.ready_file),
            "ASTRID_HOST_STATE_FILE": str(config.state_file),
            "ASTRID_HOST_BOOT_MANIFEST_PATH": str(config.boot_manifest_path),
            "ASTRID_HOST_BOOT_MANIFEST_HASH": config.boot_manifest_hash,
            "WORKER_LAUNCHER_TEST_SECRET": "must-not-pass",
        }
    )
    runner = "from source.runtime.supervisor import main; raise SystemExit(main())"
    process = subprocess.Popen([sys.executable, "-c", runner], env=env)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not config.ready_file.exists():
            time.sleep(0.02)
        assert config.ready_file.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        host_signal = json.loads((config.support_root / "host-signal.json").read_text(encoding="utf-8"))
        assert host_signal["signal"] == signal.SIGTERM
        state = json.loads(config.state_file.read_text(encoding="utf-8"))
        assert state["signals"] == ["SIGTERM"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("failing_write", [1, 2])
def test_post_launch_publication_failure_cleans_up_and_restores_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_write: int
) -> None:
    config = _config(tmp_path, sleep_seconds=30)
    real_atomic_write = supervisor._atomic_write_json
    writes = 0

    def _fail_selected_write(path: Path, value: dict[str, object]) -> None:
        nonlocal writes
        writes += 1
        if writes == failing_write:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not config.ready_file.exists():
                time.sleep(0.01)
            assert config.ready_file.exists()
            raise OSError("simulated publication failure")
        real_atomic_write(path, value)

    monkeypatch.setattr(supervisor, "_atomic_write_json", _fail_selected_write)
    original_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    marker = lambda _signum, _frame: None
    signal.signal(signal.SIGINT, marker)
    signal.signal(signal.SIGTERM, marker)
    try:
        result = supervisor.launch_generic_pack_host(config, environ={"PATH": "/bin"})
        assert result == 78
        assert writes == failing_write
        host_signal = json.loads((config.support_root / "host-signal.json").read_text(encoding="utf-8"))
        assert host_signal["signal"] == signal.SIGTERM
        assert signal.getsignal(signal.SIGINT) is marker
        assert signal.getsignal(signal.SIGTERM) is marker
    finally:
        signal.signal(signal.SIGINT, original_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, original_handlers[signal.SIGTERM])
