"""Tests for dedup_store.py's append_dedup()"""
import pandas as pd

from dedup_store import append_dedup


def test_creates_new_store_when_absent(tmp_path):
    store_path = tmp_path / "store.csv"
    rows = pd.DataFrame([{"id": "1", "value": "a"}])
    result = append_dedup(rows, store_path, dedup_cols=["id"])
    assert store_path.exists()
    assert len(result) == 1


def test_dedupes_numeric_looking_key_across_a_real_csv_round_trip(tmp_path):
    """The regression case: a numeric-looking id must not silently
    duplicate on a second run due to an int64-vs-str dtype mismatch
    between the reloaded store and a freshly-arriving row."""
    store_path = tmp_path / "store.csv"
    rows = pd.DataFrame([{"id": "1234567", "symbol": "AAPL"}])
    append_dedup(rows, store_path, dedup_cols=["id"])
    result = append_dedup(rows, store_path, dedup_cols=["id"])  # same row again
    assert len(result) == 1
    assert len(pd.read_csv(store_path)) == 1


def test_keep_last_lets_a_rerun_overwrite_stale_fields(tmp_path):
    store_path = tmp_path / "store.csv"
    append_dedup(pd.DataFrame([{"id": "1", "status": "pending"}]), store_path, dedup_cols=["id"])
    result = append_dedup(pd.DataFrame([{"id": "1", "status": "confirmed"}]), store_path, dedup_cols=["id"])
    assert len(result) == 1
    assert result.iloc[0]["status"] == "confirmed"


def test_empty_new_rows_returns_existing_store_unchanged(tmp_path):
    store_path = tmp_path / "store.csv"
    append_dedup(pd.DataFrame([{"id": "1", "value": "a"}]), store_path, dedup_cols=["id"])
    result = append_dedup(pd.DataFrame(), store_path, dedup_cols=["id"])
    assert len(result) == 1


def test_empty_new_rows_no_existing_store_returns_empty(tmp_path):
    store_path = tmp_path / "nonexistent.csv"
    result = append_dedup(pd.DataFrame(), store_path, dedup_cols=["id"])
    assert result.empty
    assert not store_path.exists()


def test_output_sorted_by_dedup_cols(tmp_path):
    store_path = tmp_path / "store.csv"
    rows = pd.DataFrame([{"date": "2026-08-19", "id": "3"}, {"date": "2026-08-17", "id": "1"}])
    result = append_dedup(rows, store_path, dedup_cols=["date", "id"])
    assert list(result["date"]) == ["2026-08-17", "2026-08-19"]


def test_multi_column_dedup_key_requires_all_columns_to_match(tmp_path):
    store_path = tmp_path / "store.csv"
    append_dedup(pd.DataFrame([{"date": "2026-08-19", "symbol": "AAPL", "value": 1}]),
                 store_path, dedup_cols=["date", "symbol"])
    # same date, different symbol -- must NOT be treated as a duplicate
    result = append_dedup(pd.DataFrame([{"date": "2026-08-19", "symbol": "MSFT", "value": 2}]),
                           store_path, dedup_cols=["date", "symbol"])
    assert len(result) == 2


def test_verbose_false_suppresses_status_print(tmp_path, capsys):
    store_path = tmp_path / "store.csv"
    append_dedup(pd.DataFrame([{"id": "1"}]), store_path, dedup_cols=["id"], verbose=False)
    assert capsys.readouterr().out == ""


def test_verbose_true_prints_status(tmp_path, capsys):
    store_path = tmp_path / "store.csv"
    append_dedup(pd.DataFrame([{"id": "1"}]), store_path, dedup_cols=["id"], verbose=True)
    assert "Wrote" in capsys.readouterr().out


def test_creates_parent_directory_if_missing(tmp_path):
    store_path = tmp_path / "nested" / "dir" / "store.csv"
    append_dedup(pd.DataFrame([{"id": "1"}]), store_path, dedup_cols=["id"])
    assert store_path.exists()


def test_corrupted_existing_store_falls_through_rebuilds_not_crash(tmp_path, capsys):
    """REAL BUG FIXED 2026-08-20 (found via boundary-condition audit): a
    store file truncated by a run killed mid-write (a real failure mode,
    same as every other disk cache/store in this repo) raised an
    uncaught pandas.errors.EmptyDataError instead of falling through --
    this is a SHARED module across 3+ ingest scripts, so one bad write on
    any of them would have broken all future runs of that ingest until
    manually fixed."""
    store_path = tmp_path / "store.csv"
    store_path.write_text("")  # empty file -- EmptyDataError on read_csv
    result = append_dedup(pd.DataFrame([{"id": "1", "value": "a"}]), store_path, dedup_cols=["id"])
    assert len(result) == 1
    assert "corrupted/unreadable" in capsys.readouterr().out


def test_corrupted_existing_store_empty_new_rows_falls_through_not_crash(tmp_path, capsys):
    store_path = tmp_path / "store.csv"
    store_path.write_text("")
    result = append_dedup(pd.DataFrame(), store_path, dedup_cols=["id"])
    assert result.empty
    assert "corrupted/unreadable" in capsys.readouterr().out


def test_malformed_csv_existing_store_falls_through_not_crash(tmp_path):
    """A genuinely malformed (not just empty) CSV -- inconsistent field
    counts -- must also fall through, not just the empty-file case."""
    store_path = tmp_path / "store.csv"
    store_path.write_text("a,b,c\n1,2,3\n4,5\n6,7,8,9,10\n")
    result = append_dedup(pd.DataFrame([{"id": "1", "value": "a"}]), store_path, dedup_cols=["id"])
    assert len(result) == 1
