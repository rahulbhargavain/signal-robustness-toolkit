"""Append-and-dedup helper
"""

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

    CORRUPTED-STORE HANDLING (added 2026-08-20, found via boundary-
    condition audit): both read_csv calls below used to be unguarded --
    a store file truncated by a run killed mid-write (the same failure
    mode already guarded against for every other disk cache/store) raised 
    an uncaught pandas.errors.EmptyDataError, crashing every caller. Falls back
    to treating the store as absent -- new_rows becomes the whole store
    again, same as a fresh start -- rather than losing the run entirely."""
    if new_rows.empty:
        if verbose:
            print(f"  No new rows for {store_path.name} this run.")
        if not store_path.exists():
            return new_rows
        try:
            return pd.read_csv(store_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            print(f"  WARNING: {store_path.name} is corrupted/unreadable ({e}) -- treating as empty.")
            return new_rows

    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        try:
            existing = pd.read_csv(store_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            print(f"  WARNING: {store_path.name} is corrupted/unreadable ({e}) -- rebuilding from today's rows only.")
            combined = new_rows
    else:
        combined = new_rows

    before = len(combined)
    present_dedup_cols = [c for c in dedup_cols if c in combined.columns]
    for c in present_dedup_cols:
        combined[c] = combined[c].astype(str)
    combined = combined.drop_duplicates(subset=present_dedup_cols, keep="last")
    if present_dedup_cols:
        combined = combined.sort_values(present_dedup_cols).reset_index(drop=True)
    combined.to_csv(store_path, index=False)

    if verbose:
        print(f"  Wrote {store_path} -- {len(new_rows)} row(s) this run, "
              f"{before - len(combined)} duplicate(s) dropped, {len(combined)} total row(s) now stored.")
    return combined
