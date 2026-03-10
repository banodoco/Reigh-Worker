from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_formatter_task_direct():
    from debug.formatters import Formatter
    from debug.models import TaskInfo

    info = TaskInfo(task_id="t1", state={"status": "Queued", "task_type": "demo"}, logs=[])
    out = Formatter.format_task(info)
    assert "TASK: t1" in out
