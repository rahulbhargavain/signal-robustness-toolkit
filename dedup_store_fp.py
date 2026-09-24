"""
dedup_store_fp.py -- Functional-programming rewrite of dedup_store.py.

DROP-IN COMPATIBLE: append_dedup() has the same signature, same return
value, same printed messages, same corrupted-store handling (empty store
treated as absent; any other unreadable store moved aside before a
rebuild; atomic writes) and same ValueError on a missing dedup col.

WHAT CHANGED, AND WHY THIS ONE IS DIFFERENT FROM THE OTHER THREE:

dedup_store.py is fundamentally I/O -- it reads a file, maybe writes a
file, maybe prints. There's no way to make disk access itself "pure",
and pretending otherwise (e.g. wrapping every read in a monad) would
just be ceremony around code any caller still has to call for its side
effects. So the FP move here isn't "eliminate the impurity" -- it's the
more modest and more useful one: **shrink the impure part to the
absolute minimum, and put everything else in pure functions**.

Concretely, four pure functions now do all of the actual decision-making
and data transformation, and never touch disk or print:
  - `_cast_dedup_cols_to_str`: DataFrame -> DataFrame (via `.assign`, no
    `df[c] = ...` in-place mutation of the input)
  - `_combine`: (existing_df | None, new_rows_df) -> DataFrame
  - `_dedupe_and_sort`: DataFrame -> (deduped_df, n_dropped)
  - `_corrupted_store_message`: (path, reason, fallback_desc) -> str
    (the two slightly-different warning strings the original built
    inline with an f-string in two different except blocks are now one
    pure formatter called from both call sites)

`_read_store` and `_write_store` are the only two functions that touch
disk, and `append_dedup` itself is now a thin orchestrator: read (if
needed) -> combine -> dedupe -> write (if needed) -> report. Every
branch that used to interleave try/except, string formatting, and the
actual pandas logic is now testable as a pure function with an
in-memory DataFrame -- no tmp_path/disk fixture required at all.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _corrupted_store_message(store_path: Path, reason: BaseException, fallback_desc: str) -> str:
    """Pure: builds the one warning string format used in both places
    the original inlined it (empty-new-rows path says 'treating as
    empty'; combine path says 'rebuilding from today's rows only')."""
    return f"  WARNING: {store_path.name} is corrupted/unreadable ({reason}) -- {fallback_desc}."


def _read_store(store_path: Path, dedup_cols: list[str]) -> tuple[Optional[pd.DataFrame], Optional[BaseException]]:
    """The only disk READ. Returns (df_or_None, exception_or_None).
    df is None both when the store doesn't exist yet and when it's
    corrupted/unreadable; the exception is populated only in the latter
    case, letting the caller decide how to phrase the warning. Key columns
    are read as str so a numeric-looking key isn't re-inferred as int64 --
    or float64 once the column holds a missing value ("1" -> "1.0")."""
    if not store_path.exists():
        return None, None
    try:
        return pd.read_csv(store_path, dtype={c: str for c in dedup_cols}), None
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
        return None, e


def _quarantine(store_path: Path) -> Path:
    """Disk MOVE: sets an unreadable (but non-empty) store aside so the
    rebuild never overwrites history."""
    backup = store_path.with_name(f"{store_path.name}.corrupt-{datetime.now():%Y%m%dT%H%M%S%f}")
    os.replace(store_path, backup)
    return backup


def _write_store(df: pd.DataFrame, store_path: Path) -> None:
    """The only disk WRITE (besides mkdir): temp file in the same
    directory + os.replace(), so a run killed mid-write can't truncate the
    existing store."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=store_path.parent, prefix=f".{store_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            df.to_csv(f, index=False)
        os.replace(tmp, store_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _key_to_str(value):
    """Pure: str() a key value, rendering integral floats without ".0" and
    leaving missing values missing."""
    if pd.isna(value):
        return value
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _cast_dedup_cols_to_str(df: pd.DataFrame, dedup_cols: list[str]) -> pd.DataFrame:
    """Pure: returns a NEW frame with dedup_cols normalized to str, guarding
    against the int64/float64-on-reload vs. str-on-fresh-fetch mismatch."""
    return df.assign(**{c: df[c].map(_key_to_str).astype(object) for c in dedup_cols})


def _check_dedup_cols(df: pd.DataFrame, dedup_cols: list[str]) -> None:
    """Deduping on a subset of the key would merge rows the full key says
    are distinct -- refuse instead."""
    missing = [c for c in dedup_cols if c not in df.columns]
    if missing:
        raise ValueError(f"dedup_cols {missing} not found in data columns {list(df.columns)}")


def _dedupe_and_sort(df: pd.DataFrame, dedup_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Pure: validate -> cast -> drop_duplicates(keep='last') -> sort.
    Returns the result plus how many rows were dropped."""
    _check_dedup_cols(df, dedup_cols)
    deduped = _cast_dedup_cols_to_str(df, dedup_cols).drop_duplicates(subset=dedup_cols, keep="last")
    return deduped.sort_values(dedup_cols).reset_index(drop=True), len(df) - len(deduped)


def _combine(existing: Optional[pd.DataFrame], new_rows: pd.DataFrame) -> pd.DataFrame:
    """Pure: existing store (or None) + new rows -> one combined frame."""
    if existing is None:
        return new_rows
    return pd.concat([existing, new_rows], ignore_index=True)


def append_dedup(new_rows: pd.DataFrame, store_path: Path, dedup_cols: list[str],
                  verbose: bool = True) -> pd.DataFrame:
    """Appends new_rows to store_path (creating it if absent), drops
    duplicates on dedup_cols (keep="last"), re-writes the combined CSV,
    and returns the combined DataFrame. Same behavior as the original,
    including: ValueError on a missing dedup col, an empty store treated as
    absent, any other unreadable store moved aside to
    `<name>.corrupt-<timestamp>` before rebuilding, and atomic writes."""
    if new_rows.empty:
        if verbose:
            print(f"  No new rows for {store_path.name} this run.")
        if not store_path.exists():
            return new_rows
        existing, error = _read_store(store_path, dedup_cols)
        if error is not None:
            print(_corrupted_store_message(store_path, error, "treating as empty"))
            return new_rows
        return existing

    existing, error = _read_store(store_path, dedup_cols)
    if isinstance(error, pd.errors.EmptyDataError):
        print(_corrupted_store_message(store_path, error, "rebuilding from today's rows only"))
    elif error is not None:
        backup = _quarantine(store_path)
        print(_corrupted_store_message(store_path, error,
                                       f"moved to {backup.name}, rebuilding from today's rows only"))

    combined, n_dropped = _dedupe_and_sort(_combine(existing, new_rows), dedup_cols)
    _write_store(combined, store_path)

    if verbose:
        print(f"  Wrote {store_path} -- {len(new_rows)} row(s) this run, "
              f"{n_dropped} duplicate(s) dropped, {len(combined)} total row(s) now stored.")
    return combined
