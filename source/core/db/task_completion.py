"""
Task insertion into the database via Edge Functions.
"""
import os
import time

from postgrest.exceptions import APIError

from source.core.log import headless_logger

__all__ = [
    "add_task_to_db",
]

from . import config as _cfg
from .config import (
    STATUS_QUEUED,
    STATUS_IN_PROGRESS)
from .edge_helpers import _call_edge_function_with_retry


def add_task_to_db(task_payload: dict, task_type_str: str, dependant_on: str | list[str] | None = None, db_path: str | None = None) -> str:
    """
    Adds a new task to the Supabase database via Edge Function.

    Args:
        task_payload: Task parameters dictionary
        task_type_str: Type of task being created
        dependant_on: Optional dependency - single task ID string or list of task IDs.
                      When list is provided, task will only run when ALL dependencies complete.
        db_path: Ignored (kept for API compatibility)

    Returns:
        str: The database row ID (UUID) assigned to the task
    """
    # Generate a new UUID for the database row ID
    import uuid
    actual_db_row_id = str(uuid.uuid4())

    # Sanitize payload and get project_id
    params_for_db = task_payload.copy()
    params_for_db.pop("task_type", None)  # Ensure task_type is not duplicated in params
    project_id = task_payload.get("project_id", "default_project_id")

    # Build Edge URL
    edge_url = (
        _cfg.SUPABASE_EDGE_CREATE_TASK_URL
    ) or os.getenv("SUPABASE_EDGE_CREATE_TASK_URL") or (
        f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/create-task" if _cfg.SUPABASE_URL else None
    )

    if not edge_url:
        raise ValueError("Edge Function URL for create-task is not configured")

    headers = {"Content-Type": "application/json"}
    if _cfg.SUPABASE_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

    # Normalize dependant_on to list format for consistency
    # Edge Function expects: null, or JSON array of UUIDs
    dependant_on_list: list[str] | None = None
    if dependant_on:
        if isinstance(dependant_on, str):
            dependant_on_list = [dependant_on]
        else:
            dependant_on_list = list(dependant_on)

    # Defensive check: if dependencies are specified, ensure they all exist before enqueueing
    if dependant_on_list:
        max_retries = 3
        retry_delay = 0.5

        for dep_id in dependant_on_list:
            for attempt in range(max_retries):
                try:
                    if _cfg.SUPABASE_CLIENT:
                        resp_exist = _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME).select("id").eq("id", dep_id).single().execute()
                        if not getattr(resp_exist, "data", None):
                            headless_logger.debug(f"[ERROR][DEBUG_DEPENDENCY_CHAIN] dependant_on not found: {dep_id}. Refusing to create task of type {task_type_str} with broken dependency.")
                            raise RuntimeError(f"dependant_on {dep_id} not found")
                        # Successfully verified this dependency
                        break
                except (APIError, RuntimeError, ValueError, OSError) as e_depchk:
                    error_str = str(e_depchk)
                    # Check if it's a "0 rows" error (race condition) and we have retries left
                    if "0 rows" in error_str and attempt < max_retries - 1:
                        headless_logger.debug(f"[RETRY][DEBUG_DEPENDENCY_CHAIN] Dependency {dep_id} not visible yet (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Final attempt failed or different error - log warning but continue
                        headless_logger.debug(f"[WARN][DEBUG_DEPENDENCY_CHAIN] Could not verify dependant_on {dep_id} existence prior to enqueue: {e_depchk}")

    payload_edge = {
        "task_id": actual_db_row_id,
        "params": params_for_db,
        "task_type": task_type_str,
        "project_id": project_id,
        "dependant_on": dependant_on_list,  # Always list or None for Edge Function
    }

    headless_logger.debug(f"Supabase Edge call >>> POST {edge_url} payload={str(payload_edge)[:120]}\u2026")

    # Use standardized retry helper (handles 5xx, timeout, and network errors)
    resp, edge_error = _call_edge_function_with_retry(
        edge_url=edge_url,
        payload=payload_edge,
        headers=headers,
        function_name="create-task",
        context_id=f"{task_type_str}:{actual_db_row_id[:8]}",
        timeout=45,  # Base timeout
        max_retries=3)

    # Handle failure cases
    if edge_error:
        raise RuntimeError(f"Edge Function create-task failed: {edge_error}")
    if resp is None:
        raise RuntimeError(f"Edge Function create-task returned no response for {actual_db_row_id}")

    if resp.status_code == 200:
        headless_logger.essential(f"Task {actual_db_row_id} (Type: {task_type_str}) queued via Edge Function.", task_id=actual_db_row_id)

        # Verify task visibility and status to catch race conditions
        if _cfg.SUPABASE_CLIENT:
            max_verify_retries = 10  # Increased from 5 to handle longer replication delays
            verify_start_time = time.time()

            for attempt in range(max_verify_retries):
                try:
                    # Check status, created_at, and project_id for comprehensive debugging
                    verify_resp = _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME).select("status, created_at, project_id, task_type").eq("id", actual_db_row_id).single().execute()
                    if verify_resp.data:
                        status = verify_resp.data.get("status")
                        created_at = verify_resp.data.get("created_at")
                        db_project_id = verify_resp.data.get("project_id")
                        db_task_type = verify_resp.data.get("task_type")
                        elapsed = time.time() - verify_start_time

                        headless_logger.debug(f"[VERIFY] Task {actual_db_row_id} found after {elapsed:.2f}s: status={status}, project_id={db_project_id}, task_type={db_task_type}, created_at={created_at}")

                        if status == STATUS_QUEUED:
                            headless_logger.essential(f"[VERIFY] Task {actual_db_row_id} verified visible and Queued (took {elapsed:.2f}s)", task_id=actual_db_row_id)
                            break
                        elif status == STATUS_IN_PROGRESS:
                            headless_logger.warning(f"[VERIFY] Task {actual_db_row_id} already In Progress (claimed in {elapsed:.2f}s - unusually fast)", task_id=actual_db_row_id)
                            break
                        else:
                            headless_logger.warning(f"[VERIFY] Task {actual_db_row_id} has unexpected status '{status}' after {elapsed:.2f}s", task_id=actual_db_row_id)
                            break
                    else:
                        headless_logger.debug(f"[VERIFY] Attempt {attempt+1}/{max_verify_retries}: Task {actual_db_row_id} not visible yet (no data returned)")
                except (APIError, RuntimeError, ValueError, OSError) as e_ver:
                    error_str = str(e_ver)
                    if "0 rows" in error_str:
                        headless_logger.debug(f"[VERIFY] Attempt {attempt+1}/{max_verify_retries}: Task {actual_db_row_id} not visible yet (0 rows)")
                    else:
                        headless_logger.debug(f"[VERIFY] Attempt {attempt+1}/{max_verify_retries}: Error verifying task {actual_db_row_id}: {e_ver}")

                if attempt < max_verify_retries - 1:
                    time.sleep(1.0)
            else:
                # Verification failed after all retries
                elapsed = time.time() - verify_start_time
                headless_logger.warning(f"Task {actual_db_row_id} creation confirmed by Edge Function but NOT VISIBLE in DB after {max_verify_retries} attempts ({elapsed:.2f}s)", task_id=actual_db_row_id)
                headless_logger.warning(f"This is likely due to database replication lag. The task IS CREATED and will be processed when visible.", task_id=actual_db_row_id)
                headless_logger.debug(f"[WARN] Task details: type={task_type_str}, project_id={project_id}, dependant_on={dependant_on}")

        return actual_db_row_id
    else:
        error_msg = f"Edge Function create-task failed: {resp.status_code} - {resp.text}"
        headless_logger.error(error_msg, task_id=actual_db_row_id)
        raise RuntimeError(error_msg)
