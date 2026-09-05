"""Neutral Worker entrypoint for the sole external Astrid pack host."""

from __future__ import annotations

from source.runtime.supervisor import launch_generic_pack_host, main as launch_main


def main() -> int:
    """Launch one GenericPackHost; Runtime remains the task authority."""
    return launch_main()


__all__ = ["launch_generic_pack_host", "main"]
