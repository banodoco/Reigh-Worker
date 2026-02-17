"""
Task polling, output queries, and parameter retrieval.
"""
import os
import time
from pathlib import Path

import httpx
from postgrest.exceptions import APIError

from source.core.log import headless_logger

__all__ = [
    "poll_task_status",
    "get_task_output_location_from_db",
    "get_task_params",
    "get_abs_path_from_db_path",
]

from . import config as _cfg
from .config import (
    STATUS_QUEUED,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETE,
    STATUS_FAILED)
from .edge_helpers import _call_edge_function_with_retry

def poll_task_status(task_id: str, poll_interval_seconds: int = 10, timeout_seconds: int = 1800, db_path: str | None = None) -> str | None:
    """
    Polls Supabase for task completion and returns the output_location.

    Args:
        task_id: Task ID to poll
        poll_interval_seconds: Seconds between polls
        timeout_seconds: Maximum time to wait
        db_path: Ignored (kept for API compatibility)

    Returns:
        Output location string if successful, None otherwise
    """
    headless_logger.essential(f"Polling for completion of task {task_id} (timeout: {timeout_seconds}s)...", task_id=task_id)
    start_time = time.time()
    last_status_print_time = 0

    while True:
        current_time = time.time()
        if current_time - start_time > timeout_seconds:
            headless_logger.error(f"Timeout polling for task {task_id} after {timeout_seconds} seconds.", task_id=task_id)
            return None

        status = None
        output_location = None

        if not _cfg.SUPABASE_CLIENT:
            headless_logger.error("Supabase client not initialized. Cannot poll status.", task_id=task_id)
            time.sleep(poll_interval_seconds)
            continue
        try:
            # Direct table query for polling status
            resp = _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME).select("status, output_location").eq("id", task_id).single().execute()
            if resp.data:
                status = resp.data.get("status")
                output_location = resp.data.get("output_location")
        except (APIError, httpx.HTTPError, OSError, ValueError, KeyError) as e:
            headless_logger.error(f"Supabase error while polling task {task_id}: {e}. Retrying...", task_id=task_id)

        if status:
            if current_time - last_status_print_time > poll_interval_seconds * 2:
                headless_logger.essential(f"Task {task_id}: Status = {status} (Output: {output_location if output_location else 'N/A'})", task_id=task_id)
                last_status_print_time = current_time

            if status == STATUS_COMPLETE:
                if output_location:
                    headless_logger.essential(f"Task {task_id} completed successfully. Output: {output_location}", task_id=task_id)
                    return output_location
                else:
                    headless_logger.error(f"Task {task_id} is COMPLETE but output_location is missing. Assuming failure.", task_id=task_id)
                    return None
            elif status == STATUS_FAILED:
                headless_logger.error(f"Task {task_id} failed. Error details: {output_location}", task_id=task_id)
                return None
            elif status not in [STATUS_QUEUED, STATUS_IN_PROGRESS]:
                headless_logger.warning(f"Task {task_id} has unknown status '{status}'. Treating as error.", task_id=task_id)
                return None
        else:
            if current_time - last_status_print_time > poll_interval_seconds * 2:
                headless_logger.essential(f"Task {task_id}: Not found in DB yet or status pending...", task_id=task_id)
                last_status_print_time = current_time

        time.sleep(poll_interval_seconds)

# Helper to query DB for a specific task's output (needed by segment handler)
def get_task_output_location_from_db(task_id_to_find: str) -> str | None:
    """
    Fetches a task's output location via the get-task-output Edge Function.

    This uses an edge function instead of direct DB query to work with
    workers that only have anon key access (RLS would block direct queries).

    Args:
        task_id_to_find: Task ID to look up

    Returns:
        Output location string if task is complete, None otherwise
    """
    headless_logger.debug(f"Fetching output location for task: {task_id_to_find}")

    # Build edge function URL
    edge_url = (
        os.getenv("SUPABASE_EDGE_GET_TASK_OUTPUT_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/get-task-output" if _cfg.SUPABASE_URL else None)
    )

    if not edge_url:
        headless_logger.error(f"No edge function URL available for get-task-output", task_id=task_id_to_find)
        return None

    if not _cfg.SUPABASE_ACCESS_TOKEN:
        headless_logger.error(f"No auth configuration available for get-task-output", task_id=task_id_to_find)
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"
    }

    payload = {"task_id": task_id_to_find}

    try:
        resp, edge_error = _call_edge_function_with_retry(
            edge_url=edge_url,
            payload=payload,
            headers=headers,
            function_name="get-task-output",
            context_id=task_id_to_find,
            timeout=30,
            max_retries=3,
            method="POST",
            retry_on_404_patterns=["not found", "Task not found"],  # Handle race conditions
        )

        if edge_error:
            headless_logger.error(f"get-task-output failed for {task_id_to_find}: {edge_error}", task_id=task_id_to_find)
            return None

        if resp and resp.status_code == 200:
            data = resp.json()
            status = data.get("status")
            output_location = data.get("output_location")

            if status == STATUS_COMPLETE and output_location:
                headless_logger.debug(f"Task {task_id_to_find} output fetched successfully")
                return output_location
            else:
                headless_logger.debug(f"Task {task_id_to_find} not complete or no output. Status: {status}")
                return None
        elif resp and resp.status_code == 404:
            headless_logger.debug(f"Task {task_id_to_find} not found")
            return None
        else:
            status_code = resp.status_code if resp else "no response"
            headless_logger.error(f"get-task-output unexpected response for {task_id_to_find}: {status_code}", task_id=task_id_to_find)
            return None

    except (httpx.HTTPError, OSError, ValueError) as e:
        headless_logger.error(f"get-task-output exception for {task_id_to_find}: {e}", task_id=task_id_to_find, exc_info=True)
        return None

def get_task_params(task_id: str) -> str | None:
    """Gets the raw params JSON string for a given task ID via edge function."""
    headless_logger.debug(f"Fetching params for task: {task_id}")

    # Build edge function URL
    edge_url = (
        os.getenv("SUPABASE_EDGE_GET_TASK_OUTPUT_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/get-task-output" if _cfg.SUPABASE_URL else None)
    )

    if not edge_url or not _cfg.SUPABASE_ACCESS_TOKEN:
        # Fallback to direct query if edge function not available
        if _cfg.SUPABASE_CLIENT:
            try:
                resp = _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME).select("params").eq("id", task_id).single().execute()
                if resp.data:
                    return resp.data.get("params")
            except (APIError, RuntimeError, ValueError, OSError) as e:
                headless_logger.debug(f"Error getting task params for {task_id}: {e}")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"
    }

    payload = {"task_id": task_id}

    try:
        resp, edge_error = _call_edge_function_with_retry(
            edge_url=edge_url,
            payload=payload,
            headers=headers,
            function_name="get-task-output",
            context_id=task_id,
            timeout=30,
            max_retries=3,
            method="POST",
            retry_on_404_patterns=["not found", "Task not found"],  # Handle race conditions
        )

        if edge_error:
            headless_logger.debug(f"get-task-output (params) failed for {task_id}: {edge_error}")
            return None

        if resp and resp.status_code == 200:
            data = resp.json()
            return data.get("params")
        return None

    except (httpx.HTTPError, OSError, ValueError) as e:
        headless_logger.debug(f"Error getting task params for {task_id}: {e}")
        return None

def get_abs_path_from_db_path(db_path: str) -> Path | None:
    """
    Helper to resolve a path from the DB to a usable absolute path.
    Assumes paths from Supabase are already absolute or valid URLs.
    """
    if not db_path:
        return None

    # Path from DB is assumed to be absolute (Supabase) or a URL
    resolved_path = Path(db_path).resolve()

    if resolved_path and resolved_path.exists():
        return resolved_path
    else:
        headless_logger.debug(f"Warning: Resolved path '{resolved_path}' from DB path '{db_path}' does not exist.")
        return None
