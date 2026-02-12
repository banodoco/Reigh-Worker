"""Database operations - re-export facade for backward compatibility.

Worker.py sets global config on this module (e.g. db_ops.SUPABASE_URL = ...).
We intercept those writes and propagate them to source.core.db.config so the
sub-modules see the updated values.
"""
import sys as _sys

import source.core.db.config as _config  # noqa: E402
from source.core.db.config import *  # noqa: F401,F403,E402
from source.core.db.edge_helpers import *  # noqa: F401,F403,E402
from source.core.db.task_claim import *  # noqa: F401,F403,E402
from source.core.db.task_completion import *  # noqa: F401,F403,E402
from source.core.db.task_polling import *  # noqa: F401,F403,E402
from source.core.db.task_dependencies import *  # noqa: F401,F403,E402

# ---------------------------------------------------------------------------
# Mutable config attributes that worker.py sets at runtime via
#   db_ops.SUPABASE_URL = ...
# We must propagate these writes into source.core.db.config so every sub-module
# (which reads from config at call-time) picks them up.
# ---------------------------------------------------------------------------
_CONFIG_ATTRS = frozenset({
    "PG_TABLE_NAME",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_VIDEO_BUCKET",
    "SUPABASE_CLIENT",
    "SUPABASE_EDGE_COMPLETE_TASK_URL",
    "SUPABASE_ACCESS_TOKEN",
    "SUPABASE_EDGE_CREATE_TASK_URL",
    "SUPABASE_EDGE_CLAIM_TASK_URL",
    "STATUS_QUEUED",
    "STATUS_IN_PROGRESS",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "BYTES_PER_MB",
    "debug_mode",
    "EDGE_FAIL_PREFIX",
    "RETRYABLE_STATUS_CODES",
    # DB_TYPE is set by worker.py but not used inside the package; still propagate
    "DB_TYPE",
})

_this_module = _sys.modules[__name__]


class _ConfigProxy:
    """Module wrapper that intercepts attribute writes for config globals."""

    def __init__(self, module):
        self.__dict__["_module"] = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __setattr__(self, name, value):
        # Always set on the facade module itself
        setattr(self._module, name, value)
        # Also propagate to the config module so sub-modules see the change
        if name in _CONFIG_ATTRS:
            setattr(_config, name, value)

    def __dir__(self):
        return dir(self._module)

    # Ensure pickling / representation still work
    def __repr__(self):
        return repr(self._module)


_sys.modules[__name__] = _ConfigProxy(_this_module)
