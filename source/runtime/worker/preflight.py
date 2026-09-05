"""Worker preflight checks and readiness metadata publishing."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import socket
import subprocess
import sys
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from source.core.log import headless_logger
from source.runtime.vibecomfy_profile import VerifiedFacts, empty_verified_facts, validate_verified_facts


PREFLIGHT_STATUS_PENDING = "pending"
PREFLIGHT_STATUS_RUNNING = "running"
PREFLIGHT_STATUS_PASSED = "passed"
PREFLIGHT_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class WorkerPreflightResult:
    status: str
    checks: list[PreflightCheck]
    started_at: float
    completed_at: float
    phase: str | None = None
    verified_facts: VerifiedFacts = field(default_factory=empty_verified_facts)

    @property
    def ok(self) -> bool:
        return self.status == PREFLIGHT_STATUS_PASSED

    @property
    def failed_checks(self) -> list[str]:
        return [check.name for check in self.checks if check.required and not check.ok]

    @property
    def facts_complete(self) -> bool:
        return set(self.verified_facts.exact) == {
            "interpreter",
            "runtime_lock",
            "engine_lock",
            "model_digest",
            "custom_node_digest",
            "driver",
            "root",
            "port",
        } and set(self.verified_facts.minimum) == {"vram_bytes", "scratch_bytes"}

    @property
    def ready_for_tasks(self) -> bool:
        return self.ok and self.facts_complete

    @property
    def readiness(self) -> str:
        return "ready" if self.ready_for_tasks else "not_ready"

    @property
    def readiness_reason(self) -> str | None:
        failed = self.failed_checks
        return failed[0] if failed else (None if self.facts_complete else "verified_facts_incomplete")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "preflight_status": self.status,
            "preflight_ok": self.ok,
            "preflight_failed_checks": self.failed_checks,
            "preflight_checks": [asdict(check) for check in self.checks],
            "preflight_started_at": self.started_at,
            "preflight_completed_at": self.completed_at,
            "preflight_phase": self.phase or self.status,
            "readiness": self.readiness,
            "readiness_reason": self.readiness_reason,
            "verified_facts": self.verified_facts.to_dict(),
            "verified_facts_digest": self.verified_facts.digest,
            "verified_facts_complete": self.facts_complete,
        }


@dataclass(frozen=True)
class FactProbeOverrides:
    """Optional test seams; production probes always inspect the host."""

    gpu: Callable[[], Mapping[str, Any]] | None = None
    port: Callable[[str, int], Mapping[str, Any]] | None = None


def collect_verified_facts(
    *,
    repo_root: Path,
    main_output_dir: Path,
    probes: FactProbeOverrides | None = None,
) -> tuple[VerifiedFacts, list[PreflightCheck]]:
    """Collect only independently verifiable, HC-02-shaped host facts.

    Every configured identity is hashed from local bytes.  Missing configuration
    is an error, not an invitation to infer a default, download a dependency, or
    select an engine.  The optional probe object exists solely for deterministic
    readiness tests and is never serialized or accepted from task data.
    """

    checks: list[PreflightCheck] = []
    exact: dict[str, str | int] = {}
    minimum: dict[str, int] = {}
    probes = probes or FactProbeOverrides()

    interpreter_identities: list[str] = []
    interpreter_names = ["REIGH_INTERPRETER"]
    for env_name in ("REIGH_ASTRID_INTERPRETER", "REIGH_ENGINE_INTERPRETER"):
        if os.environ.get(env_name):
            interpreter_names.append(env_name)
    for env_name in interpreter_names:
        configured = os.environ.get(env_name) or sys.executable
        identity, detail = _probe_interpreter(Path(configured))
        ok = identity is not None
        checks.append(PreflightCheck(f"fact:{env_name.lower()}", ok, detail))
        if identity is not None:
            interpreter_identities.append(identity)
    if interpreter_identities and len(interpreter_identities) == len(interpreter_names):
        exact["interpreter"] = json.dumps(interpreter_identities, separators=(",", ":"))

    runtime_lock = _configured_path("REIGH_RUNTIME_LOCK_PATH", repo_root / "uv.lock")
    runtime_digest, detail = _digest_file(runtime_lock)
    checks.append(PreflightCheck("fact:runtime_lock", runtime_digest is not None, detail))
    if runtime_digest is not None:
        exact["runtime_lock"] = runtime_digest

    engine_lock = _configured_path("REIGH_ENGINE_LOCK_PATH")
    engine_digest, detail = _digest_file(engine_lock)
    checks.append(PreflightCheck("fact:engine_lock", engine_digest is not None, detail))
    if engine_digest is not None:
        exact["engine_lock"] = engine_digest

    model_root = _configured_path("REIGH_MODEL_ROOT")
    model_manifest = _configured_path("REIGH_MODEL_MANIFEST_PATH")
    model_digest, detail = _digest_manifest(model_root, model_manifest, label="model")
    checks.append(PreflightCheck("fact:model_digest", model_digest is not None, detail))
    if model_digest is not None:
        exact["model_digest"] = model_digest

    custom_root = _configured_path("REIGH_CUSTOM_NODE_ROOT")
    custom_manifest = _configured_path(
        "REIGH_CUSTOM_NODE_MANIFEST_PATH",
        _configured_path("REIGH_CUSTOM_NODE_LOCK_PATH"),
    )
    if os.environ.get("REIGH_CUSTOM_NODE_LOCK_PATH") and not os.environ.get("REIGH_CUSTOM_NODE_MANIFEST_PATH"):
        custom_digest, detail = _digest_file(custom_manifest)
    else:
        custom_digest, detail = _digest_manifest(custom_root, custom_manifest, label="custom-node")
    checks.append(PreflightCheck("fact:custom_node_digest", custom_digest is not None, detail))
    if custom_digest is not None:
        exact["custom_node_digest"] = custom_digest

    try:
        gpu = dict((probes.gpu or _probe_gpu)())
        driver = _gpu_driver_identity(gpu)
        vram_bytes = gpu.get("vram_bytes")
        if not driver or isinstance(vram_bytes, bool) or not isinstance(vram_bytes, int) or vram_bytes < 1:
            raise ValueError("GPU identity and positive VRAM capacity are required")
        exact["driver"] = driver
        minimum["vram_bytes"] = vram_bytes
        checks.append(PreflightCheck("fact:gpu_driver", True, driver))
        checks.append(PreflightCheck("fact:vram_bytes", True, str(vram_bytes)))
    except (OSError, RuntimeError, TypeError, ValueError, ImportError) as exc:
        checks.append(PreflightCheck("fact:gpu_driver", False, str(exc)))
        checks.append(PreflightCheck("fact:vram_bytes", False, str(exc)))

    scratch_root = _configured_path("REIGH_SCRATCH_ROOT", main_output_dir)
    scratch_identity, detail = _strict_root_identity(scratch_root)
    if scratch_identity is None:
        checks.append(PreflightCheck("fact:scratch_root", False, detail))
        checks.append(PreflightCheck("fact:scratch_bytes", False, detail))
    else:
        try:
            free_bytes = shutil.disk_usage(scratch_root).free
        except OSError as exc:
            free_bytes = None
            detail = f"{scratch_root}: {exc}"
        ok = isinstance(free_bytes, int) and free_bytes > 0
        checks.append(PreflightCheck("fact:scratch_root", True, scratch_identity))
        checks.append(PreflightCheck("fact:scratch_bytes", ok, str(free_bytes) if free_bytes is not None else detail))
        if ok:
            minimum["scratch_bytes"] = free_bytes

    roots: dict[str, str] = {}
    for name, path in (
        ("source", repo_root),
        ("model", model_root),
        ("custom_node", custom_root),
        ("scratch", scratch_root),
        ("cas", _configured_path("REIGH_CAS_ROOT", main_output_dir)),
    ):
        identity, detail = _strict_root_identity(path)
        checks.append(PreflightCheck(f"fact:root:{name}", identity is not None, detail))
        if identity is not None:
            roots[name] = identity
    if len(roots) == 5:
        exact["root"] = _canonical_digest(roots)

    host = os.environ.get("REIGH_RUNTIME_HOST", "127.0.0.1").strip()
    raw_port = os.environ.get("REIGH_RUNTIME_PORT", "8765")
    try:
        port = int(raw_port)
        if port < 0 or port > 65535:
            raise ValueError("port is outside 0..65535")
        port_info = dict((probes.port or _probe_port)(host, port))
        if not port_info.get("ok"):
            raise OSError(str(port_info.get("detail") or "port is not free or owned"))
        exact["port"] = port
        checks.append(PreflightCheck("fact:port", True, str(port_info.get("detail") or f"{host}:{port}")))
    except (OSError, TypeError, ValueError) as exc:
        checks.append(PreflightCheck("fact:port", False, str(exc)))

    try:
        facts = validate_verified_facts({"exact": exact, "minimum": minimum})
    except ValueError as exc:
        checks.append(PreflightCheck("fact_schema", False, str(exc)))
        facts = empty_verified_facts()
    return facts, checks


def run_worker_preflight(
    *,
    repo_root: Path,
    wan2gp_path: Path,
    main_output_dir: Path,
    backend: str,
    probes: FactProbeOverrides | None = None,
) -> WorkerPreflightResult:
    started_at = time.time()
    checks: list[PreflightCheck] = []

    _append_path_check(checks, "repo_root", repo_root, expected_type="dir")
    wan2gp_required = _wan2gp_preflight_required(backend)
    _append_path_check(checks, "wan2gp_path", wan2gp_path, expected_type="dir", required=wan2gp_required)
    _append_path_check(
        checks,
        "wan2gp_submodule_marker",
        wan2gp_path / ".git",
        expected_type="any",
        required=wan2gp_required,
    )
    _append_path_check(checks, "wgp_entrypoint", wan2gp_path / "wgp.py", expected_type="file", required=wan2gp_required)
    _append_import_check(checks, "torch")
    _append_import_check(checks, "dotenv")
    _append_import_check(checks, "fastapi")

    vibecomfy_required = _vibecomfy_preflight_required(backend)
    _append_vibecomfy_check(checks, repo_root=repo_root, required=vibecomfy_required)
    _append_vibecomfy_attention_check(checks, required=vibecomfy_required)

    _append_path_check(
        checks,
        "task_dispatch_manifest",
        repo_root / "source" / "task_handlers" / "tasks" / "dispatch_manifest.py",
        expected_type="file",
    )
    _append_path_check(
        checks,
        "lora_module_manifest",
        repo_root / "source" / "models" / "lora" / "module_manifest.py",
        expected_type="file",
    )
    _append_path_check(checks, "wan2gp_models_dir", wan2gp_path / "models", expected_type="dir", required=wan2gp_required)
    _append_path_check(checks, "wan2gp_plugins_dir", wan2gp_path / "plugins", expected_type="dir", required=wan2gp_required)
    _append_writable_dir_check(checks, "main_output_dir", main_output_dir)
    _append_writable_dir_check(checks, "uv_cache_dir", Path(os.environ.get("UV_CACHE_DIR", str(repo_root / ".uv-cache"))))

    verified_facts, fact_checks = collect_verified_facts(
        repo_root=repo_root,
        main_output_dir=main_output_dir,
        probes=probes,
    )
    checks.extend(fact_checks)

    status = PREFLIGHT_STATUS_PASSED if all(check.ok or not check.required for check in checks) else PREFLIGHT_STATUS_FAILED
    result = WorkerPreflightResult(
        status=status,
        checks=checks,
        started_at=started_at,
        completed_at=time.time(),
        verified_facts=verified_facts,
    )
    headless_logger.essential(
        f"[PREFLIGHT] status={result.status} backend={backend} failed={','.join(result.failed_checks) or 'none'}"
    )
    return result


def _configured_path(env_name: str, default: Path | None = None) -> Path | None:
    raw = os.environ.get(env_name)
    if raw is not None and raw.strip():
        return Path(raw).expanduser()
    return default


def _digest_file(path: Path | None) -> tuple[str | None, str]:
    if path is None:
        return None, "not configured"
    try:
        if not path.is_file() or path.is_symlink():
            return None, f"{path}: regular file required"
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    except OSError as exc:
        return None, f"{path}: {exc}"
    return f"sha256:{digest}", str(path.resolve())


def _digest_manifest(
    root: Path | None,
    manifest: Path | None,
    *,
    label: str,
) -> tuple[str | None, str]:
    if root is None or manifest is None:
        return None, f"{label} root and manifest are required"
    root_identity, root_detail = _strict_root_identity(root)
    if root_identity is None:
        return None, root_detail
    root = root.expanduser().absolute()
    try:
        if not manifest.is_file() or manifest.is_symlink():
            return None, f"{manifest}: regular manifest file required"
        document = json.loads(manifest.read_text(encoding="utf-8"))
        entries = document.get("files") if isinstance(document, dict) else None
        if not isinstance(entries, list) or not entries:
            return None, f"{manifest}: non-empty files list required"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{manifest}: {exc}"

    observed: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return None, f"{manifest}: each files entry needs a relative path"
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            return None, f"{manifest}: path escapes root: {relative}"
        expected = entry.get("sha256")
        if not isinstance(expected, str) or not expected:
            return None, f"{manifest}: each {label} file needs sha256"
        candidate = root / relative
        try:
            if candidate.resolve(strict=True) != candidate.absolute():
                return None, f"{candidate}: symlinked model/node bytes are not verifiable"
        except OSError as exc:
            return None, f"{candidate}: {exc}"
        actual, detail = _digest_file(candidate)
        if actual is None:
            return None, detail
        expected_digest = expected if expected.startswith("sha256:") else f"sha256:{expected}"
        if actual != expected_digest:
            return None, f"{candidate}: digest mismatch"
        observed.append({"path": relative.as_posix(), "sha256": actual})

    # Model/node identity is content-addressed and therefore portable across
    # mounts.  The absolute roots are represented separately by HC-03's root
    # fact and must not contaminate the byte digest.
    canonical = {"files": observed}
    return _canonical_digest(canonical), str(manifest.resolve())


def _strict_root_identity(path: Path | None) -> tuple[str | None, str]:
    if path is None:
        return None, "root is not configured"
    try:
        absolute = path.expanduser().absolute()
        if not absolute.is_dir():
            return None, f"{absolute}: directory required"
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            if current.is_symlink():
                return None, f"{absolute}: symlink root is not verifiable"
        resolved = absolute.resolve(strict=True)
        if resolved != absolute:
            return None, f"{absolute}: root resolves elsewhere"
        return _canonical_digest({"path": str(absolute)}), str(absolute)
    except OSError as exc:
        return None, f"{path}: {exc}"


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _probe_interpreter(path: Path) -> tuple[str | None, str]:
    try:
        if not path.is_absolute():
            return None, f"{path}: absolute executable required"
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            return None, f"{resolved}: executable required"
        completed = subprocess.run(
            [str(resolved), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = completed.stdout.strip()
        if not version or len(version.split(".")) != 3:
            return None, f"{resolved}: interpreter version could not be verified"
        executable_digest, detail = _digest_file(resolved)
        if executable_digest is None:
            return None, detail
        identity = json.dumps(
            {"path": str(resolved), "version": version, "sha256": executable_digest},
            sort_keys=True,
            separators=(",", ":"),
        )
        return identity, str(resolved)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return None, f"{path}: {exc}"


def _probe_gpu() -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"torch unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")
    index = int(os.environ.get("REIGH_CUDA_DEVICE_INDEX", "0"))
    properties = torch.cuda.get_device_properties(index)
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    fields = [field.strip() for field in query.stdout.splitlines()[0].split(",")]
    if len(fields) != 3 or not all(fields):
        raise RuntimeError("nvidia-smi did not return GPU UUID, name, and driver")
    return {
        "uuid": fields[0],
        "name": fields[1],
        "driver": fields[2],
        "cuda": str(getattr(torch.version, "cuda", "unknown") or "unknown"),
        "vram_bytes": int(properties.total_memory),
    }


def _gpu_driver_identity(gpu: Mapping[str, Any]) -> str:
    values = {
        "cuda": gpu.get("cuda"),
        "driver": gpu.get("driver"),
        "name": gpu.get("name"),
        "uuid": gpu.get("uuid"),
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ValueError("GPU UUID, name, driver, and CUDA identities are required")
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _probe_port(host: str, port: int) -> Mapping[str, Any]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            return {"ok": False, "detail": f"{host}:{port}: {exc}"}
    return {"ok": True, "detail": f"{host}:{port} is free"}


def _wan2gp_preflight_required(backend: str) -> bool:
    return str(backend).strip().lower() != "vibecomfy"


def result_from_failure(name: str, detail: str, *, started_at: float | None = None) -> WorkerPreflightResult:
    started = started_at or time.time()
    return WorkerPreflightResult(
        status=PREFLIGHT_STATUS_FAILED,
        checks=[PreflightCheck(name=name, ok=False, detail=detail, required=True)],
        started_at=started,
        completed_at=time.time(),
        verified_facts=empty_verified_facts(),
    )


def finalize_preflight_result(
    base: WorkerPreflightResult,
    *,
    extra_checks: list[PreflightCheck],
) -> WorkerPreflightResult:
    checks = [*base.checks, *extra_checks]
    status = PREFLIGHT_STATUS_PASSED if all(check.ok or not check.required for check in checks) else PREFLIGHT_STATUS_FAILED
    return WorkerPreflightResult(
        status=status,
        checks=checks,
        started_at=base.started_at,
        completed_at=time.time(),
        phase=status,
        verified_facts=base.verified_facts,
    )


def publish_preflight_metadata(
    *,
    supabase_client: Any,
    worker_id: str,
    result: WorkerPreflightResult,
    ready_for_tasks: bool,
) -> bool:
    metadata_update = {
        **result.to_metadata(),
        "ready_for_tasks": bool(ready_for_tasks and result.ready_for_tasks),
    }
    write_preflight_state(worker_id, metadata_update)
    try:
        current_result = (
            supabase_client.table("workers")
            .select("metadata")
            .eq("id", worker_id)
            .limit(1)
            .execute()
        )
        rows = getattr(current_result, "data", None) or []
        metadata = dict(rows[0].get("metadata") or {}) if rows else {}
        metadata.update(metadata_update)
        (
            supabase_client.table("workers")
            .update({"metadata": metadata})
            .eq("id", worker_id)
            .execute()
        )
        return True
    except Exception as exc:
        headless_logger.warning(f"[PREFLIGHT] Failed to publish worker metadata for {worker_id}: {exc}")
        return False


def write_preflight_state(worker_id: str, metadata: dict[str, Any]) -> Path:
    path = preflight_state_path(worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return path


def read_preflight_state(worker_id: str | None) -> dict[str, Any] | None:
    if not worker_id:
        return None
    path = preflight_state_path(worker_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def preflight_state_path(worker_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in worker_id)
    return Path(os.environ.get("REIGH_PREFLIGHT_STATE_DIR", "/tmp")) / f"reigh_worker_preflight_{safe}.json"


def _append_path_check(
    checks: list[PreflightCheck],
    name: str,
    path: Path,
    *,
    expected_type: str,
    required: bool = True,
) -> None:
    if expected_type == "dir":
        ok = path.is_dir()
    elif expected_type == "file":
        ok = path.is_file()
    else:
        ok = path.exists()
    checks.append(PreflightCheck(name=name, ok=ok, detail=str(path), required=required))


def _append_import_check(checks: list[PreflightCheck], module_name: str, *, required: bool = True) -> None:
    spec = importlib.util.find_spec(module_name)
    checks.append(
        PreflightCheck(
            name=f"import:{module_name}",
            ok=spec is not None,
            detail=getattr(spec, "origin", None) or "not found",
            required=required,
        )
    )


def _vibecomfy_preflight_required(backend: str) -> bool:
    configured = os.environ.get("REIGH_PREFLIGHT_REQUIRE_VIBECOMFY")
    if configured is not None:
        return configured.strip().lower() not in {"0", "false", "no"}
    return str(backend).strip().lower() == "vibecomfy"


def _vibecomfy_attention_profile() -> str:
    raw = (
        os.environ.get("REIGH_VIBECOMFY_ATTENTION_PROFILE")
        or os.environ.get("VIBECOMFY_ATTENTION_PROFILE")
        or os.environ.get("REIGH_WORKER_PROFILE")
        or ""
    )
    value = raw.strip().lower()
    if value in {"", "default", "portable", "sdpa"}:
        return "portable"
    if value in {"optimized", "sage", "sageattn", "sageattention"}:
        return "sage"
    return value


def _append_vibecomfy_attention_check(checks: list[PreflightCheck], *, required: bool) -> None:
    profile = _vibecomfy_attention_profile()
    if profile == "portable":
        checks.append(
            PreflightCheck(
                name="vibecomfy_attention_profile",
                ok=True,
                detail="portable/sdpa",
                required=False,
            )
        )
        return
    if profile != "sage":
        checks.append(
            PreflightCheck(
                name="vibecomfy_attention_profile",
                ok=False,
                detail=f"unsupported profile: {profile}",
                required=required,
            )
        )
        return
    spec = importlib.util.find_spec("sageattention")
    checks.append(
        PreflightCheck(
            name="import:sageattention",
            ok=spec is not None,
            detail=getattr(spec, "origin", None) or "not found",
            required=required,
        )
    )


def _append_vibecomfy_check(checks: list[PreflightCheck], *, repo_root: Path, required: bool) -> None:
    spec = importlib.util.find_spec("vibecomfy")
    candidates = [
        Path(os.environ["VIBECOMFY_PATH"]) if os.environ.get("VIBECOMFY_PATH") else None,
        repo_root.parent / "vibecomfy",
        Path("/workspace/vibecomfy"),
    ]
    existing = [path for path in candidates if path and path.exists()]
    checks.append(
        PreflightCheck(
            name="vibecomfy_available",
            ok=spec is not None or bool(existing),
            detail=getattr(spec, "origin", None) or ",".join(str(path) for path in existing) or "not found",
            required=required,
        )
    )
    if existing:
        vibecomfy_root = existing[0]
        _append_path_check(
            checks,
            "vibecomfy_template_index",
            vibecomfy_root / "template_index.json",
            expected_type="file",
            required=required,
        )
        _append_path_check(
            checks,
            "vibecomfy_custom_nodes_manifest",
            vibecomfy_root / "workflow_corpus" / "manifests" / "coverage.json",
            expected_type="file",
            required=required,
        )


def _append_writable_dir_check(checks: list[PreflightCheck], name: str, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".reigh-preflight-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append(PreflightCheck(name=name, ok=True, detail=str(path), required=True))
    except OSError as exc:
        checks.append(PreflightCheck(name=name, ok=False, detail=f"{path}: {exc}", required=True))
