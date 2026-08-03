---
name: tabular-load
description: Load tabular data from CSV, Parquet, Excel, JSON, or SQL into a typed pandas DataFrame. Handles delimiters, encodings, dtype inference, datetime parsing, and memory reduction. Produces the canonical DataFrame all other tabular skills operate on.
---

# Purpose

Get data into memory correctly, once, with sane dtypes.

Every other tabular skill assumes the DataFrame produced here.

Invoked by `tabular` as step 2, usually right after `tabular-inspect` has looked at
a sample.

---

# Supported sources

| Source | Reader |
| --- | --- |
| `.csv`, `.tsv`, `.txt` | `pd.read_csv` |
| `.parquet` | `pd.read_parquet` (pyarrow) |
| `.xlsx`, `.xls` | `pd.read_excel` |
| `.json`, `.jsonl` | `pd.read_json` |
| `.feather` | `pd.read_feather` |
| SQL | `pd.read_sql` with a user-supplied connection |

Prefer Parquet when the user has a choice. It preserves dtypes and loads faster.

---

# Procedure

1. **Peek first.** For files over 100 MB, read `nrows=5000` and profile before the
   full load. Never blindly load a multi-gigabyte CSV.
2. **Sniff the delimiter** with `csv.Sniffer` when the extension is ambiguous.
3. **Detect encoding.** Try `utf-8`, then `utf-8-sig`, then `latin-1`. Report which
   one worked.
4. **Parse dates explicitly.** Do not rely on `infer_datetime_format` across the
   whole frame; pass an explicit `parse_dates` list from the profile.
5. **Downcast** numerics and convert low-cardinality object columns to `category`.
6. **Normalize column names** — strip whitespace, keep original names in a mapping so
   output can be reported in the user's vocabulary.
7. **Report** shape, dtypes, memory before and after, and any parse warnings.

---

# Missing value sentinels

Real datasets encode missing as more than `NaN`. Pass these to `na_values`:

```python
NA_SENTINELS = [
    "", " ", "NA", "N/A", "n/a", "na", "null", "NULL", "None", "none",
    "-", "--", "?", "unknown", "UNKNOWN", "missing", "NaN", "nan",
]
```

Numeric sentinels like `-999`, `-1`, `9999` are **not** safe to convert
automatically. Flag them and ask.

---

# Reference implementation

```python
import csv
import pandas as pd

NA_SENTINELS = ["", " ", "NA", "N/A", "n/a", "null", "NULL", "None",
                "-", "--", "?", "unknown", "missing", "NaN", "nan"]


def sniff_delimiter(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="replace") as fh:
        sample = fh.read(64_000)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load(path: str, parse_dates=None, nrows=None) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    elif p.endswith(".feather"):
        df = pd.read_feather(path)
    elif p.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, na_values=NA_SENTINELS, nrows=nrows)
    elif p.endswith((".json", ".jsonl")):
        df = pd.read_json(path, lines=p.endswith(".jsonl"))
    else:
        enc_used = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                sep = sniff_delimiter(path, enc)
                df = pd.read_csv(
                    path, sep=sep, encoding=enc, na_values=NA_SENTINELS,
                    keep_default_na=True, low_memory=False, nrows=nrows,
                )
                enc_used = enc
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if enc_used is None:
            raise ValueError(f"Could not parse {path} with common encodings.")

    df.columns = [str(c).strip() for c in df.columns]
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def reduce_memory(df: pd.DataFrame, cat_threshold: float = 0.5) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif s.dtype == object and s.nunique(dropna=True) / max(len(s), 1) < cat_threshold:
            df[col] = s.astype("category")
    return df
```

---

# Large data

Above roughly 5 million rows, or when the file exceeds available RAM:

- Load only needed columns via `usecols`.
- Use Parquet with column pruning.
- Sample for exploration, then fit on the full set only if the model supports it.
- Tell the user you sampled and state the sample size. Never sample silently.

---

# Output

Return the DataFrame plus a load report:

- source path and format
- encoding and delimiter used
- shape
- dtypes after conversion
- memory before and after reduction
- columns coerced to datetime
- rows dropped, if any, and why

---

# Do not

Do not fill missing values here. `NaN` is information; downstream skills decide.

Do not encode categoricals here. CatBoost and the baseline want different things.

Do not shuffle or sort. `tabular-split` owns row ordering.
