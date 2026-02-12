"""Backward-compatible import location. Actual implementation in source/media/structure/.

Required by vendored Wan2GP code (any2video.py) which uses sys.path to import directly.
"""
from source.media.structure import *  # noqa: F401,F403
