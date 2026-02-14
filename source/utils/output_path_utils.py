"""Output path preparation, filename sanitization, and upload helpers."""

import os
import time
from pathlib import Path

from source.core.log import headless_logger
from source.core.constants import BYTES_PER_KB as BYTES_PER_KIBIBYTE

__all__ = [
    "prepare_output_path",
    "sanitize_filename_for_storage",
    "prepare_output_path_with_upload",
    "upload_and_get_final_output_location",
    "upload_intermediate_file_to_storage",
    "wait_for_file_stable",
]

# --- SM_RESTRUCTURE: Function moved from worker.py ---
def _get_task_type_directory(task_type: str) -> str:
    """
    Map task types to their output subdirectories.

    Single-level directory structure: outputs/{task_type}/{files}
    Examples:
    - vace -> outputs/vace/
    - travel_orchestrator -> outputs/travel_orchestrator/
    - extract_frame -> outputs/extract_frame/

    Args:
        task_type: The task type string (e.g., 'vace', 't2v', 'travel_orchestrator')

    Returns:
        Task type as directory name (e.g., 'vace', 'travel_orchestrator')
    """
    # Simply return the task type as the directory name
    # This creates outputs/{task_type}/ structure
    return task_type if task_type else 'misc'

def prepare_output_path(
    task_id: str,
    filename: str,
    main_output_dir_base: Path,
    task_type: str | None = None,  # NEW PARAMETER
    *,
    custom_output_dir: str | Path | None = None
) -> tuple[Path, str]:
    """
    Prepares the output path for a task's artifact.

    If `custom_output_dir` is provided, it's used as the base. Otherwise,
    the output is placed in a subdirectory based on task_type
    or directly in `main_output_dir_base` (backwards compatibility).

    Args:
        task_id: Unique task identifier
        filename: Output filename
        main_output_dir_base: Base output directory (from worker --main-output-dir)
        task_type: Optional task type for subdirectory organization
        custom_output_dir: Optional custom output directory (overrides all)

    Returns:
        Tuple of (Path object for file location, string path for database)
    """
    # Decide base directory for the file
    if custom_output_dir:
        output_dir_for_task = Path(custom_output_dir)
        headless_logger.debug(f"Task {task_id}: Using custom output directory: {output_dir_for_task}", task_id=task_id)
    else:
        # Create task-type-specific subdirectory structure
        if task_type:
            type_subdir = _get_task_type_directory(task_type)
            output_dir_for_task = main_output_dir_base / type_subdir
            headless_logger.debug(
                f"Task {task_id}: Using task-type subdirectory: {output_dir_for_task} "
                f"(task_type='{task_type}' -> '{type_subdir}')",
                task_id=task_id
            )
        else:
            # Backwards compatibility: No task_type provided, use root directory
            output_dir_for_task = main_output_dir_base
            headless_logger.debug(
                f"Task {task_id}: No task_type provided, using root output directory: {output_dir_for_task} "
                f"(backwards compatibility)",
                task_id=task_id
            )

        # To avoid name collisions we prefix the filename with the task_id
        # Skip prefixing for files with UUID patterns (they guarantee uniqueness)
        import re
        # Match UUID pattern: _HHMMSS_uuid6.ext
        uuid_pattern = r'_\d{6}_[a-f0-9]{6}\.(mp4|png|jpg|jpeg)$'
        has_uuid_pattern = re.search(uuid_pattern, filename, re.IGNORECASE)

        if not filename.startswith(task_id) and not has_uuid_pattern:
            filename = f"{task_id}_{filename}"

    output_dir_for_task.mkdir(parents=True, exist_ok=True)

    final_save_path = output_dir_for_task / filename

    # Handle filename conflicts by adding _1, _2, etc.
    if final_save_path.exists():
        stem = final_save_path.stem
        suffix = final_save_path.suffix
        counter = 1
        while final_save_path.exists():
            new_filename = f"{stem}_{counter}{suffix}"
            final_save_path = output_dir_for_task / new_filename
            counter += 1
        headless_logger.debug(f"Task {task_id}: Filename conflict resolved - using {final_save_path.name}", task_id=task_id)

    # Build DB path string - use relative path to current working directory
    try:
        db_output_location = str(final_save_path.relative_to(Path.cwd()))
    except ValueError:
        db_output_location = str(final_save_path.resolve())

    headless_logger.debug(f"Task {task_id}: final_save_path='{final_save_path}', db_output_location='{db_output_location}'", task_id=task_id)

    return final_save_path, db_output_location

def sanitize_filename_for_storage(filename: str) -> str:
    """
    Sanitizes a filename to be safe for storage systems like Supabase Storage.

    Removes characters that are invalid for S3/Supabase storage keys:
    - Control characters (0x00-0x1F, 0x7F-0x9F)
    - Special characters: etc.
    - Path separators and other problematic characters

    Args:
        filename: Original filename that may contain unsafe characters

    Returns:
        Sanitized filename safe for storage systems
    """
    import re

    # Define characters to remove (includes the special character causing the issue)
    # This is based on S3/Supabase storage key restrictions and common filesystem issues
    unsafe_chars = r'[\u00a7\u00ae\u00a9\u2122@\u00b7\u00ba\u00bd\u00be\u00bf\u00a1~\x00-\x1F\x7F-\x9F<>:"/\\|?*,]'

    # Replace unsafe characters with empty string
    sanitized = re.sub(unsafe_chars, '', filename)

    # Replace multiple consecutive spaces/dots with single ones
    sanitized = re.sub(r'[ \.]{2,}', ' ', sanitized)

    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')

    # Ensure we have a non-empty filename
    if not sanitized:
        sanitized = "sanitized_file"

    return sanitized

def prepare_output_path_with_upload(
    task_id: str,
    filename: str,
    main_output_dir_base: Path,
    task_type: str | None = None,  # NEW PARAMETER
    *,
    custom_output_dir: str | Path | None = None
) -> tuple[Path, str]:
    """
    Prepares the output path for a task's artifact and handles Supabase upload if configured.

    Args:
        task_id: Unique task identifier
        filename: Output filename
        main_output_dir_base: Base output directory (from worker --main-output-dir)
        task_type: Optional task type for subdirectory organization
        custom_output_dir: Optional custom output directory (overrides all)

    Returns:
        tuple[Path, str]: (local_file_path, db_output_location)
        - local_file_path: Where to save the file locally (for generation)
        - db_output_location: What to store in the database (local path or Supabase URL)
    """
    # Sanitize filename BEFORE any processing to prevent storage upload issues
    original_filename = filename
    sanitized_filename = sanitize_filename_for_storage(filename)

    if original_filename != sanitized_filename:
        headless_logger.debug(f"Task {task_id}: Sanitized filename '{original_filename}' -> '{sanitized_filename}'", task_id=task_id)

    # First, get the local path where we'll save the file (using sanitized filename)
    # Forward task_type parameter
    local_save_path, initial_db_location = prepare_output_path(
        task_id, sanitized_filename, main_output_dir_base,
        task_type=task_type,  # Forward task_type
        custom_output_dir=custom_output_dir
    )

    # Return the local path for now - we'll handle Supabase upload after file is created
    return local_save_path, initial_db_location

def upload_and_get_final_output_location(
    local_file_path: Path,
    initial_db_location: str) -> str:
    """
    Returns the local file path. Upload is now handled by the edge function.

    Args:
        local_file_path: Path to the local file
        initial_db_location: The initial DB location (local path)

    Returns:
        str: Local file path (upload now handled by edge function)
    """
    # Edge function will handle the upload, so we just return the local path
    headless_logger.debug(f"File ready for edge function upload: {local_file_path}")
    return str(local_file_path.resolve())

def upload_intermediate_file_to_storage(
    local_file_path: Path,
    task_id: str,
    filename: str) -> str | None:
    """
    Upload an intermediate file to Supabase storage for cross-worker access.

    This is used when orchestrators create intermediate files (like reversed videos)
    that need to be accessible by child tasks running on different workers.

    Includes retry logic for transient failures (502/503/504, timeouts, network errors).

    Args:
        local_file_path: Path to the local file to upload
        task_id: Task ID for organizing uploads
        filename: Filename to use in storage

    Returns:
        Public URL of the uploaded file, or None on failure
    """
    import httpx
    import mimetypes

    # Retry configuration
    RETRYABLE_STATUS_CODES = {502, 503, 504}
    MAX_RETRIES = 3

    # Check if Supabase is configured (read URL from canonical config set by worker.py)
    from source.core.db import config as _db_config
    SUPABASE_URL = _db_config.SUPABASE_URL
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        headless_logger.warning(f"[UPLOAD_INTERMEDIATE] Supabase not configured, cannot upload", task_id=task_id)
        return None

    if not local_file_path.exists():
        headless_logger.warning(f"[UPLOAD_INTERMEDIATE] File not found: {local_file_path}", task_id=task_id)
        return None

    try:
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
            "Content-Type": "application/json"
        }

        # Step 1: Get signed upload URL - WITH RETRY
        generate_url_edge = f"{SUPABASE_URL.rstrip('/')}/functions/v1/generate-upload-url"
        content_type = mimetypes.guess_type(str(local_file_path))[0] or 'application/octet-stream'

        upload_url_resp = None
        for attempt in range(MAX_RETRIES):
            try:
                upload_url_resp = httpx.post(
                    generate_url_edge,
                    headers=headers,
                    json={
                        "task_id": task_id,
                        "filename": filename,
                        "content_type": content_type
                    },
                    timeout=30 + (attempt * 15)
                )

                if upload_url_resp.status_code == 200:
                    break
                elif upload_url_resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    headless_logger.warning(f"[UPLOAD_INTERMEDIATE] generate-upload-url got {upload_url_resp.status_code}, retrying in {wait_time}s...", task_id=task_id)
                    time.sleep(wait_time)
                    continue
                else:
                    break  # Non-retryable error

            except (httpx.HTTPError, OSError, ValueError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    headless_logger.warning(f"[UPLOAD_INTERMEDIATE] generate-upload-url error, retrying in {wait_time}s: {e}", task_id=task_id)
                    time.sleep(wait_time)
                    continue
                else:
                    headless_logger.error(f"[UPLOAD_INTERMEDIATE] [EDGE_FAIL:generate-upload-url:NETWORK] {e}", task_id=task_id)
                    return None

        if not upload_url_resp or upload_url_resp.status_code != 200:
            error_text = upload_url_resp.text[:200] if upload_url_resp else "No response"
            error_code = upload_url_resp.status_code if upload_url_resp else "N/A"
            headless_logger.error(f"[UPLOAD_INTERMEDIATE] [EDGE_FAIL:generate-upload-url:HTTP_{error_code}] {error_text}", task_id=task_id)
            return None

        upload_data = upload_url_resp.json()
        upload_url = upload_data.get("upload_url")
        storage_path = upload_data.get("storage_path")

        if not upload_url:
            headless_logger.error(f"[UPLOAD_INTERMEDIATE] No upload_url in response", task_id=task_id)
            return None

        # Step 2: Upload file via signed URL - WITH RETRY
        # IMPORTANT: do NOT read entire file into memory (can be large).
        file_size_mb = local_file_path.stat().st_size / BYTES_PER_KIBIBYTE / BYTES_PER_KIBIBYTE
        headless_logger.debug(f"[UPLOAD_INTERMEDIATE] Uploading {local_file_path.name} ({file_size_mb:.1f} MB)", task_id=task_id)

        put_resp = None
        for attempt in range(MAX_RETRIES):
            try:
                with open(local_file_path, 'rb') as f:
                    put_resp = httpx.put(
                        upload_url,
                        headers={"Content-Type": content_type},
                        content=f,
                        timeout=300 + (attempt * 60)  # 5 min base, +1 min per retry
                    )

                if put_resp.status_code in [200, 201]:
                    break
                elif put_resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    headless_logger.warning(f"[UPLOAD_INTERMEDIATE] storage-upload got {put_resp.status_code}, retrying in {wait_time}s...", task_id=task_id)
                    time.sleep(wait_time)
                    continue
                else:
                    break  # Non-retryable error

            except (httpx.HTTPError, OSError, ValueError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    headless_logger.warning(f"[UPLOAD_INTERMEDIATE] storage-upload error, retrying in {wait_time}s: {e}", task_id=task_id)
                    time.sleep(wait_time)
                    continue
                else:
                    headless_logger.error(f"[UPLOAD_INTERMEDIATE] [EDGE_FAIL:storage-upload:NETWORK] {e}", task_id=task_id)
                    return None

        if not put_resp or put_resp.status_code not in [200, 201]:
            error_text = put_resp.text[:200] if put_resp else "No response"
            error_code = put_resp.status_code if put_resp else "N/A"
            headless_logger.error(f"[UPLOAD_INTERMEDIATE] [EDGE_FAIL:storage-upload:HTTP_{error_code}] {error_text}", task_id=task_id)
            return None

        # Construct public URL
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/image_uploads/{storage_path}"
        headless_logger.debug(f"[UPLOAD_INTERMEDIATE] Uploaded to: {public_url}", task_id=task_id)

        return public_url

    except (httpx.HTTPError, OSError, ValueError) as e:
        headless_logger.error(f"[UPLOAD_INTERMEDIATE] Exception: {e}", task_id=task_id, exc_info=True)
        return None

def wait_for_file_stable(path: Path | str, checks: int = 3, interval: float = 1.0) -> bool:
    """Return True when the file size stays constant for a few consecutive checks.
    Useful to make sure long-running encoders have finished writing before we
    copy/move the file.
    """
    p = Path(path)
    if not p.exists():
        return False
    last_size = p.stat().st_size
    stable_count = 0
    for _ in range(checks):
        time.sleep(interval)
        new_size = p.stat().st_size
        if new_size == last_size and new_size > 0:
            stable_count += 1
            if stable_count >= checks - 1:
                return True
        else:
            stable_count = 0
            last_size = new_size
    return False
