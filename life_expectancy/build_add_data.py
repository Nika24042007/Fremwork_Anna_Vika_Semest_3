import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("add_data")
OUTPUT_FILE = OUTPUT_DIR / "Life_expectancy_new.csv"
YEAR_MIN, YEAR_MAX = 2016, 2022

EXPLICIT_MAP = {
    "GGHE-D": "Total expenditure",
    "BMI_less_18": "thinness 1-19 years",
    "BMI_more_25": "BMI",
    "HepB": "Hepatitis B",
    "Percent_expenditure": "percentage expenditure",
    "Thinness_children": "thinness 5-9 years",
    "Under-five_mortality": "under-five deaths",
    "Under_five_mortality": "under-five deaths",
}

COUNTRY_ALIASES = {
    "Türkiye": "Turkey",
    "Viet Nam": "Vietnam",
    "Naoero": "Nauru",
    "Cabo Verde": "Cape Verde",
    "Republic of Korea": "South Korea",
    "Democratic People's Republic of Korea": "North Korea",
    "Russian Federation": "Russia",
    "Iran (Islamic Republic of)": "Iran",
    "Republic of Moldova": "Moldova",
    "United Republic of Tanzania": "Tanzania",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Netherlands (Kingdom of the)": "Netherlands",
    "occupied Palestinian territory, including east Jerusalem": "Palestine",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United States of America": "United States",
    "Lao People's Democratic Republic": "Laos",
    "Syrian Arab Republic": "Syria",
    "Micronesia (Federated States of)": "Micronesia",
}

DIM_PREFERRED = ("All types", "Both sexes", "All ages")


def _clean(text):
    """Lowercase a string and strip spaces, underscores and hyphens."""
    return re.sub(r"[\s_\-]", "", str(text).lower())


def normalize_country(name):
    """Normalize a country name by applying aliases and stripping parentheticals."""
    if not isinstance(name, str):
        return name
    name = name.strip()
    if name in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[name]
    return re.sub(r"\s*\([^)]*\)", "", name).strip()


def match_column(prefix, columns):
    """Return the exact original column name matching a file prefix, or None."""
    hint = EXPLICIT_MAP.get(prefix, prefix)
    hint_clean = _clean(hint)
    for col in columns:
        if _clean(col) == hint_clean:
            return col
    partial = [
        c for c in columns
        if _clean(c).startswith(hint_clean) or hint_clean.startswith(_clean(c))
    ]
    if partial:
        return min(partial, key=lambda c: abs(len(_clean(c)) - len(hint_clean)))
    return None


def read_any(path):
    """Read a CSV trying several (encoding, separator) combinations."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(
                    path, sep=sep, encoding=enc,
                    engine="python", on_bad_lines="skip",
                )
            except UnicodeDecodeError:
                break
            except Exception:
                continue
            if "Period" in df.columns and (
                "Location" in df.columns or "SpatialDim" in df.columns
            ):
                return df
    return None


def extract_values(df):
    """Return a numeric Series for the file's value column, or None."""
    for col in ("FactValueNumeric", "FactValueNumericHigh", "FactValueNumericLow"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                return vals
    if "Value" in df.columns:
        vals = df["Value"].astype(str).str.extract(r"^\s*(-?\d+(?:\.\d+)?)")[0]
        vals = pd.to_numeric(vals, errors="coerce")
        if vals.notna().any():
            return vals
    return None


def filter_dim(df):
    """Keep only the aggregate breakdown (All types / Both sexes / All ages)."""
    if "Dim1" not in df.columns:
        return df
    for val in DIM_PREFERRED:
        if (df["Dim1"] == val).any():
            return df[df["Dim1"] == val].copy()
    return df


def process_additional_file(path, target_col):
    """Convert one WHO GHO file into a (Country, Year, target_col) frame."""
    df = read_any(path)
    if df is None or df.empty:
        print(f"[warn] cannot read {path.name}")
        return None

    if "Location" not in df.columns and "SpatialDim" in df.columns:
        df = df.rename(columns={"SpatialDim": "Location"})
    if "Location type" in df.columns:
        df = df[df["Location type"] == "Country"]

    values = extract_values(df)
    if values is None:
        print(f"[warn] no numeric values in {path.name}")
        return None
    df = df.assign(_val=values)

    df["Period"] = pd.to_numeric(df["Period"], errors="coerce")
    df = df[df["Period"].between(YEAR_MIN, YEAR_MAX)].copy()
    if df.empty:
        print(f"[warn] no rows in {YEAR_MIN}-{YEAR_MAX} in {path.name}")
        return None

    df = filter_dim(df)
    df = df[["Location", "Period", "_val"]].rename(
        columns={"Location": "Country", "Period": "Year", "_val": target_col}
    )
    df["Country"] = df["Country"].map(normalize_country)
    df = df.dropna(subset=[target_col])
    if df.empty:
        return None
    return df.groupby(["Country", "Year"], as_index=False)[target_col].first()


def merge_piece(merged, piece, target_col):
    """Merge a piece into merged, filling gaps if the target column already exists."""
    if target_col in merged.columns:
        tmp = piece.rename(columns={target_col: target_col + "__new"})
        merged = merged.merge(tmp, on=["Country", "Year"], how="outer")
        merged[target_col] = merged[target_col].fillna(merged[target_col + "__new"])
        return merged.drop(columns=[target_col + "__new"])
    return merged.merge(piece, on=["Country", "Year"], how="outer")


def find_files():
    """Return (original dataset path, sorted list of additional file paths)."""
    all_csv = list(DATA_DIR.glob("*.csv"))
    add_files = sorted(f for f in all_csv if f.stem.endswith("_add_data"))
    orig_files = [f for f in all_csv if not f.stem.endswith("_add_data")]
    if not orig_files:
        raise FileNotFoundError("Original dataset not found in data/")
    return orig_files[0], add_files


def build_new_dataset():
    """Assemble the new life-expectancy dataset from all additional files."""
    orig_path, add_paths = find_files()
    original = pd.read_csv(orig_path)
    target_columns = [c for c in original.columns if c != "Status"]
    original_countries = set(original["Country"].map(normalize_country).dropna())

    merged = None
    for path in add_paths:
        prefix = path.stem[: -len("_add_data")]
        target_col = match_column(prefix, target_columns)
        if target_col is None:
            print(f"[warn] no column match for {path.name} (prefix={prefix!r})")
            continue
        piece = process_additional_file(path, target_col)
        if piece is None or piece.empty:
            continue
        merged = piece if merged is None else merge_piece(merged, piece, target_col)

    if merged is None:
        merged = pd.DataFrame(columns=["Country", "Year"])

    merged = merged[merged["Country"].isin(original_countries)].copy()

    for col in target_columns:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = (
        merged[target_columns]
        .sort_values(["Country", "Year"])
        .reset_index(drop=True)
    )
    return merged


def main():
    """Build and save the new dataset to add_data/Life_expectancy_new.csv."""
    df = build_new_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")
    print("\nNon-null counts per column:")
    print(df.notna().sum().to_string())


if __name__ == "__main__":
    main()