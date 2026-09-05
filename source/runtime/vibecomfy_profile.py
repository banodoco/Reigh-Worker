"""Numeric VibeComfy memory-profile protocol helpers.

This module is intentionally worker-local.  The worker runs on Python 3.10 in
some environments, while VibeComfy owns the concrete profile mapping in its own
package.  Keep this boundary to numeric CLI protocol data only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

PROCESS_DEFAULT_PROFILE = -1
VALID_MEMORY_PROFILES = frozenset((1, 2, 3, 4, 5))

# This is the wire shape owned by Runtime HC-02.  Keep the Worker-side
# representation deliberately boring: no backend, route, template, or engine
# policy can be smuggled into verified facts.
EXACT_FACT_KEYS = (
    "interpreter",
    "runtime_lock",
    "engine_lock",
    "model_digest",
    "custom_node_digest",
    "driver",
    "root",
    "port",
)
MINIMUM_FACT_KEYS = ("vram_bytes", "scratch_bytes")


@dataclass(frozen=True)
class VerifiedFacts:
    """The secret-free, engine-neutral HC-02 fact payload."""

    exact: Mapping[str, str | int]
    minimum: Mapping[str, int]

    def to_dict(self) -> dict[str, dict[str, str | int]]:
        return {
            "exact": dict(sorted(self.exact.items())),
            "minimum": dict(sorted(self.minimum.items())),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def empty_verified_facts() -> VerifiedFacts:
    return VerifiedFacts(exact={}, minimum={})


def validate_verified_facts(facts: VerifiedFacts | Mapping[str, Mapping[str, Any]]) -> VerifiedFacts:
    """Validate and normalize the exact HC-02 shape without adding policy."""

    if isinstance(facts, VerifiedFacts):
        payload = facts.to_dict()
    elif isinstance(facts, Mapping):
        payload = dict(facts)
    else:
        raise ValueError("verified facts must be an object")

    unknown_sections = set(payload) - {"exact", "minimum"}
    if unknown_sections:
        raise ValueError(f"unsupported verified fact sections: {sorted(unknown_sections)}")

    raw_exact = payload.get("exact") or {}
    raw_minimum = payload.get("minimum") or {}
    if not isinstance(raw_exact, Mapping) or not isinstance(raw_minimum, Mapping):
        raise ValueError("verified fact sections must be objects")
    exact = dict(raw_exact)
    minimum = dict(raw_minimum)
    unknown_exact = set(exact) - set(EXACT_FACT_KEYS)
    unknown_minimum = set(minimum) - set(MINIMUM_FACT_KEYS)
    if unknown_exact or unknown_minimum:
        unknown = sorted(unknown_exact | unknown_minimum)
        raise ValueError(f"unsupported verified facts: {unknown}")

    normalized_exact: dict[str, str | int] = {}
    for key, value in exact.items():
        if key == "port":
            if isinstance(value, bool) or not isinstance(value, (str, int)) or not value or (isinstance(value, int) and value < 0):
                raise ValueError("verified fact port must be a non-negative integer or non-empty string")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"verified fact {key} must be a non-empty string")
        normalized_exact[key] = value

    normalized_minimum: dict[str, int] = {}
    for key, value in minimum.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"verified fact {key} must be a non-negative integer")
        normalized_minimum[key] = value

    return VerifiedFacts(exact=normalized_exact, minimum=normalized_minimum)


@dataclass(frozen=True)
class VibeComfyProfileProtocol:
    """One-run VibeComfy profile protocol payload."""

    memory_profile: int

    def to_cli_args(self) -> list[str]:
        return ["--memory-profile", str(self.memory_profile)]

    def to_dict(self) -> dict[str, int]:
        return {"memory_profile": self.memory_profile}


def _validate_numeric_profile(value: Any, *, field_name: str, allow_default: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer memory profile")
    if allow_default and value == PROCESS_DEFAULT_PROFILE:
        return value
    if value not in VALID_MEMORY_PROFILES:
        allowed = "1, 2, 3, 4, 5"
        if allow_default:
            allowed = f"{PROCESS_DEFAULT_PROFILE}, {allowed}"
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return value


def resolve_memory_profile(
    *,
    process_default: int | None,
    override_profile: int | None = PROCESS_DEFAULT_PROFILE,
) -> int | None:
    """Resolve worker process default plus one-task override to a numeric profile.

    ``None`` means no VibeComfy profile flag should be emitted.  ``-1`` is only
    accepted for ``override_profile`` and means "use the process default".
    """

    resolved_default = None
    if process_default is not None:
        resolved_default = _validate_numeric_profile(
            process_default,
            field_name="process_default",
            allow_default=False,
        )

    if override_profile is None or override_profile == PROCESS_DEFAULT_PROFILE:
        return resolved_default

    return _validate_numeric_profile(
        override_profile,
        field_name="override_profile",
        allow_default=False,
    )


def build_profile_protocol(
    *,
    process_default: int | None,
    override_profile: int | None = PROCESS_DEFAULT_PROFILE,
) -> VibeComfyProfileProtocol | None:
    """Return one-run VibeComfy CLI protocol data for the selected profile."""

    resolved = resolve_memory_profile(
        process_default=process_default,
        override_profile=override_profile,
    )
    if resolved is None:
        return None
    return VibeComfyProfileProtocol(memory_profile=resolved)


def build_memory_profile_cli_args(
    *,
    process_default: int | None,
    override_profile: int | None = PROCESS_DEFAULT_PROFILE,
) -> list[str]:
    """Return ``vibecomfy run`` CLI args for the selected memory profile."""

    protocol = build_profile_protocol(
        process_default=process_default,
        override_profile=override_profile,
    )
    if protocol is None:
        return []
    return protocol.to_cli_args()
