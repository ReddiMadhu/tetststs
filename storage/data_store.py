"""
storage/data_store.py — Unified data access layer with local/azure fallback.

Controls which backend is used for pipeline row data (NOT session metadata):

    STORAGE_MODE=local  (default)
        → Stores DataFrames in a module-level Python dict.
        → Identical to the previous behaviour; zero config required.

    STORAGE_MODE=azure
        → Stores DataFrames in DuckDB on the Azure File Share mount.
        → Uploads original files to Azure Blob Storage.
        → Requires: DUCKDB_PATH, AZURE_STORAGE_CONNECTION_STRING,
                    AZURE_BLOB_CONTAINER env vars.

Public API (same regardless of mode):
    save_stage(upload_id, stage, df)
    load_stage(upload_id, stage) -> Optional[DataFrame]
    load_chunk(upload_id, stage, offset, limit) -> Optional[DataFrame]
    append_stage(upload_id, stage, df)
    get_row_count(upload_id, stage) -> int
    delete_session_data(upload_id)
    upload_raw_file(upload_id, content, filename)   # no-op in local mode
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger("data_store")

STORAGE_MODE: str = os.getenv("STORAGE_MODE", "local").lower()

logger.info(f"data_store initialised — STORAGE_MODE={STORAGE_MODE!r}")

# ── LOCAL backend ──────────────────────────────────────────────────────────────
# Keyed as:  _local_store[upload_id][stage] = pd.DataFrame

_local_store: dict = {}


def _local_save(upload_id: str, stage: str, df: pd.DataFrame) -> None:
    _local_store.setdefault(upload_id, {})[stage] = df.reset_index(drop=True)


def _local_load(upload_id: str, stage: str) -> Optional[pd.DataFrame]:
    return _local_store.get(upload_id, {}).get(stage)


def _local_load_chunk(upload_id: str, stage: str,
                      offset: int, limit: int) -> Optional[pd.DataFrame]:
    df = _local_load(upload_id, stage)
    if df is None:
        return None
    return df.iloc[offset: offset + limit].reset_index(drop=True)


def _local_append(upload_id: str, stage: str, df: pd.DataFrame) -> None:
    existing = _local_load(upload_id, stage)
    if existing is None:
        _local_save(upload_id, stage, df)
    else:
        merged = pd.concat([existing, df], ignore_index=True)
        _local_store[upload_id][stage] = merged


def _local_delete(upload_id: str) -> None:
    _local_store.pop(upload_id, None)


def _local_row_count(upload_id: str, stage: str) -> int:
    df = _local_load(upload_id, stage)
    return len(df) if df is not None else 0


# ── AZURE / DuckDB backend ─────────────────────────────────────────────────────

def _duck_table(upload_id: str, stage: str) -> str:
    """Return a safe DuckDB table name for (upload_id, stage)."""
    return f"s_{upload_id.replace('-', '_')}_{stage}"


def _duck_connect():
    """Return a DuckDB connection to the persistent database file."""
    import duckdb
    db_path = os.getenv("DUCKDB_PATH", "/mnt/azurefiles/pipeline.duckdb")
    return duckdb.connect(db_path)


def _duck_save(upload_id: str, stage: str, df: pd.DataFrame) -> None:
    table = _duck_table(upload_id, stage)
    with _duck_connect() as con:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    logger.debug(f"DuckDB: saved {len(df)} rows → {table}")


def _duck_load(upload_id: str, stage: str) -> Optional[pd.DataFrame]:
    table = _duck_table(upload_id, stage)
    try:
        with _duck_connect() as con:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if table not in tables:
                return None
            return con.execute(f"SELECT * FROM {table}").df()
    except Exception as exc:
        logger.error(f"DuckDB load failed ({table}): {exc}")
        return None


def _duck_load_chunk(upload_id: str, stage: str,
                     offset: int, limit: int) -> Optional[pd.DataFrame]:
    table = _duck_table(upload_id, stage)
    try:
        with _duck_connect() as con:
            return con.execute(
                f"SELECT * FROM {table} LIMIT {limit} OFFSET {offset}"
            ).df()
    except Exception as exc:
        logger.error(f"DuckDB chunk load failed ({table}): {exc}")
        return None


def _duck_append(upload_id: str, stage: str, df: pd.DataFrame) -> None:
    table = _duck_table(upload_id, stage)
    try:
        with _duck_connect() as con:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if table not in tables:
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
            else:
                con.execute(f"INSERT INTO {table} SELECT * FROM df")
    except Exception as exc:
        logger.error(f"DuckDB append failed ({table}): {exc}")


def _duck_row_count(upload_id: str, stage: str) -> int:
    table = _duck_table(upload_id, stage)
    try:
        with _duck_connect() as con:
            result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return result[0] if result else 0
    except Exception:
        return 0


def _duck_delete(upload_id: str) -> None:
    prefix = f"s_{upload_id.replace('-', '_')}_"
    try:
        with _duck_connect() as con:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            for t in tables:
                if t.startswith(prefix):
                    con.execute(f"DROP TABLE IF EXISTS {t}")
        logger.info(f"DuckDB: dropped all tables for session {upload_id}")
    except Exception as exc:
        logger.error(f"DuckDB delete failed for {upload_id}: {exc}")


def _blob_upload(upload_id: str, content: bytes, filename: str) -> None:
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        container = os.getenv("AZURE_BLOB_CONTAINER", "catpipeline")
        if not conn_str:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING not set; blob upload skipped.")
            return
        client = BlobServiceClient.from_connection_string(conn_str)
        blob   = client.get_blob_client(container, f"{upload_id}/original_{filename}")
        blob.upload_blob(content, overwrite=True)
        logger.info(f"Blob: uploaded {filename} for session {upload_id}")
    except Exception as exc:
        logger.error(f"Blob upload failed for {upload_id}/{filename}: {exc}")


def _blob_delete(upload_id: str) -> None:
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        container = os.getenv("AZURE_BLOB_CONTAINER", "catpipeline")
        if not conn_str:
            return
        client = BlobServiceClient.from_connection_string(conn_str)
        cc     = client.get_container_client(container)
        blobs  = list(cc.list_blobs(name_starts_with=f"{upload_id}/"))
        for b in blobs:
            cc.delete_blob(b.name)
        logger.info(f"Blob: deleted {len(blobs)} object(s) for session {upload_id}")
    except Exception as exc:
        logger.error(f"Blob delete failed for {upload_id}: {exc}")


# ── Public API ─────────────────────────────────────────────────────────────────

def save_stage(upload_id: str, stage: str, df: pd.DataFrame) -> None:
    """Persist a full processed DataFrame for the given pipeline stage."""
    if STORAGE_MODE == "azure":
        _duck_save(upload_id, stage, df)
    else:
        _local_save(upload_id, stage, df)


def load_stage(upload_id: str, stage: str) -> Optional[pd.DataFrame]:
    """Load the complete DataFrame for a pipeline stage."""
    if STORAGE_MODE == "azure":
        return _duck_load(upload_id, stage)
    return _local_load(upload_id, stage)


def load_chunk(upload_id: str, stage: str,
               offset: int, limit: int) -> Optional[pd.DataFrame]:
    """Load a slice of rows — used during chunked processing to keep RAM low."""
    if STORAGE_MODE == "azure":
        return _duck_load_chunk(upload_id, stage, offset, limit)
    return _local_load_chunk(upload_id, stage, offset, limit)


def append_stage(upload_id: str, stage: str, df: pd.DataFrame) -> None:
    """Append rows to an existing stage (used during chunked writes)."""
    if STORAGE_MODE == "azure":
        _duck_append(upload_id, stage, df)
    else:
        _local_append(upload_id, stage, df)


def get_row_count(upload_id: str, stage: str) -> int:
    """Return the total number of rows in a stage."""
    if STORAGE_MODE == "azure":
        return _duck_row_count(upload_id, stage)
    df = _local_load(upload_id, stage)
    return len(df) if df is not None else 0


def delete_session_data(upload_id: str) -> None:
    """Remove all stored data for a session (called by TTL cleanup)."""
    if STORAGE_MODE == "azure":
        _duck_delete(upload_id)
        _blob_delete(upload_id)
    else:
        _local_delete(upload_id)


def upload_raw_file(upload_id: str, content: bytes, filename: str) -> None:
    """Store the original uploaded file bytes (azure mode → Blob; local → no-op)."""
    if STORAGE_MODE == "azure":
        _blob_upload(upload_id, content, filename)
