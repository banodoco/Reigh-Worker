"""Launch and supervise the one Astrid GenericPackHost owned by a Worker.

The Worker is deliberately not a task executor. It only validates the
operator-supplied host profile, starts one external GenericPackHost, publishes
neutral process state, forwards termination signals, and returns the host's
exit status unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Mapping, Sequence


GENERIC_HOST_MODULE = "astrid.core.execution.generic_host"
GENERIC_HOST_EXECUTOR_ID = "astrid-pack-host"

# Ambient process settings only. Host bindings that affect identity are
# validated and passed as argv values rather than inherited from the process.
HOST_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
        "TMPDIR",
        "TEMP",
        "TMP",
        "XDG_RUNTIME_DIR",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    }
)


class LauncherConfigurationError(ValueError):
    """A trusted host binding is missing or unsafe."""


def _required_env(name: str, environ: Mapping[str, str]) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise LauncherConfigurationError(f"{name} is required")
    return value


def _resolved_path(
    name: str,
    environ: Mapping[str, str],
    *,
    directory: bool = False,
    executable: bool = False,
) -> Path:
    raw = _required_env(name, environ)
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise LauncherConfigurationError(f"{name} must be an absolute path")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LauncherConfigurationError(f"{name} is unavailable") from exc
    if directory:
        if not resolved.is_dir():
            raise LauncherConfigurationError(f"{name} must name a directory")
    elif not resolved.is_file() or (executable and not os.access(resolved, os.X_OK)):
        raise LauncherConfigurationError(f"{name} must name a file")
    return resolved


def _absolute_value(name: str, environ: Mapping[str, str]) -> str:
    value = _required_env(name, environ)
    path = Path(value)
    if not path.is_absolute():
        raise LauncherConfigurationError(f"{name} must be an absolute path")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise LauncherConfigurationError(f"{name} parent directory is unavailable") from exc
    if not parent.is_dir():
        raise LauncherConfigurationError(f"{name} parent must be a directory")
    return str(parent / path.name)


@dataclass(frozen=True)
class HostLaunchConfig:
    """The complete trusted binding for one GenericPackHost."""

    host_python: Path
    source_checkout: Path
    pack_root: Path
    runtime_endpoint: str
    credential_file: Path
    support_root: Path
    runtime_instance_id: str
    ready_file: Path
    state_file: Path
    boot_manifest_path: Path
    boot_manifest_hash: str
    capability_matrix: Path | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "HostLaunchConfig":
        env = os.environ if environ is None else environ
        source_checkout = _resolved_path("ASTRID_HOST_SOURCE_CHECKOUT", env, directory=True)
        pack_root = _resolved_path("ASTRID_HOST_PACK_ROOT", env, directory=True)
        if not pack_root.is_relative_to(source_checkout):
            raise LauncherConfigurationError("ASTRID_HOST_PACK_ROOT must be inside ASTRID_HOST_SOURCE_CHECKOUT")
        support_root = _resolved_path("ASTRID_HOST_SUPPORT_ROOT", env, directory=True)
        credential_file = _resolved_path("ASTRID_HOST_CREDENTIAL_FILE", env)
        boot_manifest_path = _resolved_path("ASTRID_HOST_BOOT_MANIFEST_PATH", env)
        capability_matrix = None
        if env.get("ASTRID_HOST_CAPABILITY_MATRIX", "").strip():
            capability_matrix = _resolved_path("ASTRID_HOST_CAPABILITY_MATRIX", env)
        return cls(
            host_python=_resolved_path("ASTRID_HOST_PYTHON", env, executable=True),
            source_checkout=source_checkout,
            pack_root=pack_root,
            runtime_endpoint=_required_env("ASTRID_HOST_RUNTIME_ENDPOINT", env).rstrip("/"),
            credential_file=credential_file,
            support_root=support_root,
            runtime_instance_id=_required_env("ASTRID_HOST_RUNTIME_INSTANCE_ID", env),
            ready_file=Path(_absolute_value("ASTRID_HOST_READY_FILE", env)),
            state_file=Path(_absolute_value("ASTRID_HOST_STATE_FILE", env)),
            boot_manifest_path=boot_manifest_path,
            boot_manifest_hash=_required_env("ASTRID_HOST_BOOT_MANIFEST_HASH", env),
            capability_matrix=capability_matrix,
        )

    def argv(self) -> list[str]:
        args = [
            str(self.host_python),
            "-m",
            GENERIC_HOST_MODULE,
            "run",
            "--pack-root",
            str(self.pack_root),
            "--runtime-endpoint",
            self.runtime_endpoint,
            "--credential-file",
            str(self.credential_file),
            "--executor-id",
            GENERIC_HOST_EXECUTOR_ID,
            "--ready-file",
            str(self.ready_file),
            "--support-root",
            str(self.support_root),
            "--source-checkout",
            str(self.source_checkout),
            "--runtime-instance-id",
            self.runtime_instance_id,
            "--register",
            "--boot-manifest-path",
            str(self.boot_manifest_path),
            "--boot-manifest-hash",
            self.boot_manifest_hash,
        ]
        if self.capability_matrix is not None:
            args.extend(("--capability-matrix", str(self.capability_matrix)))
        return args


def _host_environment(environ: Mapping[str, str], config: HostLaunchConfig) -> dict[str, str]:
    child_env = {key: value for key, value in environ.items() if key in HOST_ENV_ALLOWLIST}
    # Ambient PYTHONPATH could select another checkout. The configured source
    # checkout is the sole code root admitted to this host.
    child_env["PYTHONPATH"] = str(config.source_checkout)
    child_env["PYTHONUNBUFFERED"] = "1"
    return child_env


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(dict(value), sort_keys=True, indent=2), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_returncode(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 + (-returncode)


def _signal_owned_group(process: subprocess.Popen[bytes], signum: int) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _ready_file_is_owned(path: Path, process: subprocess.Popen[bytes]) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("status") == "ready"
        and str(value.get("pid")) == str(process.pid)
    )


def launch_generic_pack_host(
    config: HostLaunchConfig,
    *,
    environ: Mapping[str, str] | None = None,
    ready_timeout_seconds: float = 20.0,
) -> int:
    """Start exactly one host and return its exit status.

    The process is a fresh session leader, so signal forwarding and cleanup are
    limited to the group created for this launch. A host that never publishes a
    matching ready record is terminated and the Worker fails closed.
    """

    env = os.environ if environ is None else environ
    argv = config.argv()
    child_env = _host_environment(env, config)
    config.ready_file.unlink(missing_ok=True)
    received: list[str] = []
    child: subprocess.Popen[bytes] | None = None

    def _forward_signal(signum: int, _frame: object) -> None:
        try:
            received.append(signal.Signals(signum).name)
        except ValueError:
            received.append(f"SIG{signum}")
        if child is not None:
            _signal_owned_group(child, signum)

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)
    try:
        child = subprocess.Popen(
            argv,
            cwd=str(config.source_checkout),
            env=child_env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        if os.name != "nt":
            try:
                own_group = os.getpgid(child.pid) == child.pid
            except OSError:
                own_group = False
            if not own_group:
                _signal_owned_group(child, signal.SIGTERM)
                child.wait(timeout=3)
                raise LauncherConfigurationError(
                    "GenericPackHost did not become its own process-group leader"
                )
    except BaseException:
        signal.signal(signal.SIGINT, previous_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous_handlers[signal.SIGTERM])
        raise
    state = {
        "status": "starting",
        "pid": child.pid,
        "pgid": child.pid,
        "argv": argv,
        "interpreter": str(config.host_python),
        "ready_file": str(config.ready_file),
        "state_file": str(config.state_file),
        "env_keys": sorted(child_env),
        "allowed_env": sorted(HOST_ENV_ALLOWLIST | {"PYTHONPATH"}),
    }
    _atomic_write_json(config.state_file, state)

    deadline = time.monotonic() + ready_timeout_seconds
    ready = False
    while time.monotonic() < deadline:
        if _ready_file_is_owned(config.ready_file, child):
            ready = True
            break
        if child.poll() is not None:
            break
        time.sleep(0.05)

    if not ready:
        _signal_owned_group(child, signal.SIGTERM)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _signal_owned_group(child, signal.SIGKILL)
            child.wait(timeout=3)
        returncode = _normalize_returncode(child.returncode)
        _atomic_write_json(
            config.state_file,
            {**state, "status": "failed", "returncode": returncode, "ready": False, "signals": received},
        )
        signal.signal(signal.SIGINT, previous_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous_handlers[signal.SIGTERM])
        return returncode or 1

    _atomic_write_json(config.state_file, {**state, "status": "ready", "ready": True})
    try:
        returncode = _normalize_returncode(child.wait())
    finally:
        signal.signal(signal.SIGINT, previous_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous_handlers[signal.SIGTERM])
    _atomic_write_json(
        config.state_file,
        {
            **state,
            "status": "exited",
            "ready": True,
            "returncode": returncode,
            "signals": received,
        },
    )
    return returncode


def main(argv: Sequence[str] | None = None) -> int:
    # Task/route arguments cannot select a host interpreter, source path,
    # engine, template, or backend.
    del argv
    try:
        config = HostLaunchConfig.from_environment()
        return launch_generic_pack_host(config)
    except LauncherConfigurationError as exc:
        print(f"Worker launcher configuration error: {exc}", file=sys.stderr)
        return 78


__all__ = [
    "GENERIC_HOST_EXECUTOR_ID",
    "GENERIC_HOST_MODULE",
    "HOST_ENV_ALLOWLIST",
    "HostLaunchConfig",
    "LauncherConfigurationError",
    "launch_generic_pack_host",
    "main",
]
