
import sys
import re
import datetime
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BINS_CONFIG = [
    {"temporal_bin": "2019-Q1", "min": "2019-01", "max": "2019-04", "total_population":  2440},
    {"temporal_bin": "2019-Q2", "min": "2019-05", "max": "2019-08", "total_population":  1895},
    {"temporal_bin": "2019-Q3", "min": "2019-09", "max": "2019-12", "total_population":  1043},
    {"temporal_bin": "2020-Q1", "min": "2020-01", "max": "2020-04", "total_population":  2500},
    {"temporal_bin": "2020-Q2", "min": "2020-05", "max": "2020-08", "total_population":  1401},
    {"temporal_bin": "2020-Q3", "min": "2020-09", "max": "2020-12", "total_population":  2198},
    {"temporal_bin": "2021-Q1", "min": "2021-01", "max": "2021-04", "total_population":  4128},
    {"temporal_bin": "2021-Q2", "min": "2021-05", "max": "2021-08", "total_population":  3494},
    {"temporal_bin": "2021-Q3", "min": "2021-09", "max": "2021-12", "total_population":  5541},
    {"temporal_bin": "2022-Q1", "min": "2022-01", "max": "2022-04", "total_population": 18864},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_unix(val) -> "int | pd.NA":
    """Normalize a date field to a Unix timestamp (int).

    Accepts:
    - Unix timestamp as string/int/float  (e.g. "1632023078", 1632023078.0)
    - Human-readable datetime             (e.g. "2021-09-19 14:37:58")
    - N/A, empty, NaN                     → pd.NA

    FIX: str(float) produces "1632023078.0" which breaks the digit-only
    regex. We now strip the fractional part before matching.
    """
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if s in ("N/A", "", "nan"):
        return pd.NA

    # Strip trailing ".0" that float→str conversion introduces
    s_int = re.sub(r"\.0+$", "", s)

    if re.match(r"^\d{9,11}$", s_int):
        return int(s_int)

    # Try human-readable datetime
    try:
        dt = datetime.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    except ValueError:
        return pd.NA


def unix_to_ym(val) -> "str | None":
    """Convert a Unix timestamp to 'YYYY-MM' string, or None if invalid."""
    if pd.isna(val):
        return None
    try:
        dt = datetime.datetime.fromtimestamp(int(val), tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m")
    except (ValueError, OSError):
        return None


def month_diff(ym1: str, ym2: str) -> float:
    """Return (ym2 - ym1) in months. Both must be 'YYYY-MM' strings."""
    y1, m1 = int(ym1[:4]), int(ym1[5:7])
    y2, m2 = int(ym2[:4]), int(ym2[5:7])
    return (y2 - y1) * 12 + (m2 - m1)


def assign_bin(ym: "str | None") -> "str | None":
    """Return the bin name for a 'YYYY-MM' string, or None if out of range."""
    if not ym:
        return None
    for b in BINS_CONFIG:
        if b["min"] <= ym <= b["max"]:
            return b["temporal_bin"]
    return None


def safe_int(val) -> int:
    """Convert to int safely — handles object/NA results from empty Series.sum()."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    print(f"[load]  {len(df):,} rows read from {path}")

    # Deduplicate on hash
    n_before = len(df)
    df = df.drop_duplicates(subset="hash", keep="first")
    print(f"[dedup] removed {n_before - len(df):,} duplicate rows → {len(df):,} unique hashes")

    # Normalize date columns to unix int (Int64 = nullable integer)
    for col in ("first_seen_itw", "first_submission"):
        df[col] = pd.array(df[col].apply(to_unix), dtype="Int64")
        print(f"[dates] {col}: {df[col].notna().sum():,} non-null values")

    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_bins(df: pd.DataFrame) -> pd.DataFrame:
    # Derive YYYY-MM strings used for bin matching
    df = df.copy()
    df["ts_ym"]  = df["timestamp"].astype(str).str[:7]          # already YYYY-MM
    df["fs_ym"]  = df["first_seen_itw"].apply(unix_to_ym)
    df["sub_ym"] = df["first_submission"].apply(unix_to_ym)
    df["has_error"] = (
        df["error"].notna()
        & ~df["error"].astype(str).isin(["", "nan", "<NA>"])
    )
    df["bin"] = df["ts_ym"].apply(assign_bin)

    rows = []
    for b in BINS_CONFIG:
        bname, bmin, bmax = b["temporal_bin"], b["min"], b["max"]
        total_population  = b["total_population"]

        subset  = df[df["bin"] == bname]
        n_total = len(subset)
        n_fs    = subset["fs_ym"].notna().sum()
        n_sub   = subset["sub_ym"].notna().sum()
        n_err   = subset["has_error"].sum()

        has_fs  = subset[subset["fs_ym"].notna()]
        has_sub = subset[subset["sub_ym"].notna()]

        diffs_fs  = has_fs.apply( lambda r: month_diff(r["ts_ym"], r["fs_ym"]),  axis=1)
        diffs_sub = has_sub.apply(lambda r: month_diff(r["ts_ym"], r["sub_ym"]), axis=1)

        # How many first_seen / first_submission actually fall inside this bin
        n_fs_in_bin  = int(subset["fs_ym"].apply( lambda y: isinstance(y, str) and bmin <= y <= bmax).sum())
        n_sub_in_bin = int(subset["sub_ym"].apply(lambda y: isinstance(y, str) and bmin <= y <= bmax).sum())

        def fmt(n, pct):
            if pct is np.nan or pct != pct:
                return f"{n} (n/a)"
            return f"{n} ({pct}%)"

        rows.append({
            "bin": bname,
            # first_seen_itw
            "n_first_seen (% of total)": fmt(int(n_fs), round(n_fs / n_total * 100, 2) if n_total else np.nan),
            "n_first_seen_in_bin (% of first_seen)": fmt(n_fs_in_bin,
                                                         round(n_fs_in_bin / n_fs * 100, 2) if n_fs else np.nan),
            "fs_avg_abs_diff_months": round(diffs_fs.abs().mean(), 2) if len(diffs_fs) else np.nan,
            "fs_n_before_timestamp (% of first_seen)": fmt(safe_int((diffs_fs < 0).sum()),
                                                           round((diffs_fs < 0).sum() / n_fs * 100,
                                                                 2) if n_fs else np.nan),
            "fs_n_same_month (% of first_seen)": fmt(safe_int((diffs_fs == 0).sum()),
                                                     round((diffs_fs == 0).sum() / n_fs * 100, 2) if n_fs else np.nan),
            "fs_n_after_timestamp (% of first_seen)": fmt(safe_int((diffs_fs > 0).sum()),
                                                          round((diffs_fs > 0).sum() / n_fs * 100,
                                                                2) if n_fs else np.nan),
            # first_submission
            "n_first_submission (% of total)": fmt(int(n_sub), round(n_sub / n_total * 100, 2) if n_total else np.nan),
            "n_first_submission_in_bin (% of first_sub)": fmt(n_sub_in_bin, round(n_sub_in_bin / n_sub * 100,
                                                                                  2) if n_sub else np.nan),
            "sub_avg_abs_diff_months": round(diffs_sub.abs().mean(), 2) if len(diffs_sub) else np.nan,
            "sub_n_before_timestamp (% of first_sub)": fmt(safe_int((diffs_sub < 0).sum()),
                                                           round((diffs_sub < 0).sum() / n_sub * 100,
                                                                 2) if n_sub else np.nan),
            "sub_n_same_month (% of first_sub)": fmt(safe_int((diffs_sub == 0).sum()),
                                                     round((diffs_sub == 0).sum() / n_sub * 100,
                                                           2) if n_sub else np.nan),
            "sub_n_after_timestamp (% of first_sub)": fmt(safe_int((diffs_sub > 0).sum()),
                                                          round((diffs_sub > 0).sum() / n_sub * 100,
                                                                2) if n_sub else np.nan),
            # errors
            "n_errors (% of total)": fmt(int(n_err), round(n_err / n_total * 100, 2) if n_total else np.nan),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "vt_results.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".csv", "_bin_analysis.csv")

    df     = load_and_clean(input_path)
    result = analyze_bins(df)

    print("\n" + result.to_string(index=False))
    result.to_csv(output_path, index=False)
    print(f"\n[out]   saved to {output_path}")


if __name__ == "__main__":
    main()