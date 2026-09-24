"""Append-and-dedup helper
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd


def append_dedup(new_rows: pd.DataFrame, store_path: Path, dedup_cols: list[str],
                  verbose: bool = True) -> pd.DataFrame:
    """Appends new_rows to store_path (creating it if absent), drops
    duplicates on dedup_cols (keep="last" -- a rerun's fresher value wins
    over what was already on disk), and re-writes the combined CSV.
    Returns the combined DataFrame either way (including when new_rows is
    empty, in which case it's just whatever was already on disk, or an
    empty frame if there was nothing at all).

    dedup_cols are cast to str on BOTH the existing and new rows before
    comparing -- see module docstring for the exact dtype-mismatch bug
    this guards against (a numeric-looking key like NSE's seq_id reloads
    from CSV as int64 but arrives as str on a fresh fetch, and pandas
    treats `1` and `"1"` as different values in an object-dtype column).

    verbose=True (default) prints a one-line status summary; set False
    when the caller does its own reporting off the returned DataFrame
    (e.g. ingest_trendlyne_breadth.py's per-source row/date counts).

    Raises ValueError if any of dedup_cols is missing from the data --
    silently deduping on a subset of the key would merge rows that the
    full key says are distinct.

    CORRUPTED-STORE HANDLING: an EMPTY store file (a run killed before it
    wrote anything) is treated as absent. Any other unreadable store (a
    malformed line, bad encoding) still holds history, so it is moved
    aside to `<name>.corrupt-<timestamp>` before the store is rebuilt from
    this run's rows -- never silently overwritten. Writes go to a temp file
    in the same directory and are swapped in with os.replace(), so a run
    killed mid-write cannot truncate the existing store."""
    if new_rows.empty:
        if verbose:
            print(f"  No new rows for {store_path.name} this run.")
        if not store_path.exists():
            return new_rows
        try:
            return _read_store(store_path, dedup_cols)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            print(f"  WARNING: {store_path.name} is corrupted/unreadable ({e}) -- treating as empty.")
            return new_rows

    store_path.parent.mkdir(parents=True, exist_ok=True)
    combined = new_rows
    if store_path.exists():
        try:
            existing = _read_store(store_path, dedup_cols)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        except pd.errors.EmptyDataError as e:
            print(f"  WARNING: {store_path.name} is corrupted/unreadable ({e}) -- rebuilding from today's rows only.")
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            backup = _quarantine(store_path)
            print(f"  WARNING: {store_path.name} is corrupted/unreadable ({e}) -- moved to {backup.name}, "
                  "rebuilding from today's rows only.")

    missing = [c for c in dedup_cols if c not in combined.columns]
    if missing:
        raise ValueError(f"dedup_cols {missing} not found in data columns {list(combined.columns)}")

    before = len(combined)
    combined = combined.assign(**{c: combined[c].map(_key_to_str).astype(object) for c in dedup_cols})
    combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    combined = combined.sort_values(dedup_cols).reset_index(drop=True)
    _write_atomic(combined, store_path)

    if verbose:
        print(f"  Wrote {store_path} -- {len(new_rows)} row(s) this run, "
              f"{before - len(combined)} duplicate(s) dropped, {len(combined)} total row(s) now stored.")
    return combined


def _read_store(store_path: Path, dedup_cols: list[str]) -> pd.DataFrame:
    """Reads key columns as str so a numeric-looking key isn't re-inferred
    as int64 -- or float64, once the column holds a missing value, which
    would turn "1" into "1.0"."""
    return pd.read_csv(store_path, dtype={c: str for c in dedup_cols})


def _key_to_str(value):
    """str() a key value, rendering integral floats without ".0" (a key
    column that picked up a NaN upstream arrives as float) and leaving
    missing values missing."""
    if pd.isna(value):
        return value
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _quarantine(store_path: Path) -> Path:
    backup = store_path.with_name(f"{store_path.name}.corrupt-{datetime.now():%Y%m%dT%H%M%S%f}")
    os.replace(store_path, backup)
    return backup


def _write_atomic(df: pd.DataFrame, store_path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=store_path.parent, prefix=f".{store_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            df.to_csv(f, index=False)
        os.replace(tmp, store_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
