import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = Path("data")
OUTPUT_DIR = Path("add_data")
OUTPUT_FILE = OUTPUT_DIR / "Life_expectancy_new.csv"
YEAR_MIN, YEAR_MAX = 2016, 2022

EXPLICIT_MAP = {
    "GGHE-D": "Total expenditure",
    "HepB": "Hepatitis B",
    "Percent_expenditure": "percentage expenditure",
    "Thinness_children": "thinness 5-9 years",
    "Under-five_mortality": "under-five deaths",
    "Under_five_mortality": "under-five deaths",
}

BMI_CONSTRAINT_FILES = {
    "BMI_less_18": "p_under",
    "BMI_more_25": "p_over25",
    "BMI_more_30": "p_ob30",
}

EXPECTED_INDICATORS = {
    "Infant_deaths": {"MDG_0000000001"},
    "Under-five_mortality": {"MDG_0000000007"},
    "Measles": {"MDG_0000000026"},
    "Thinness_children": {"NCD_0000000033", "NCD_0000000032"},
}

SUM_DIM1 = {"Population"}

COUNTRY_ALIASES = {
    "Türkiye": "Turkey", "Viet Nam": "Vietnam", "Naoero": "Nauru",
    "Cabo Verde": "Cape Verde", "Republic of Korea": "South Korea",
    "Democratic People's Republic of Korea": "North Korea",
    "Russian Federation": "Russia", "Iran (Islamic Republic of)": "Iran",
    "Republic of Moldova": "Moldova", "United Republic of Tanzania": "Tanzania",
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

DIM1_PREFERRED = ("All types", "Both sexes", "Both", "Total", "All ages")
DIM2_PREFERRED = ("At birth", "0 years", "0-4 years")
DEFAULT_BMI_SIGMA = 5.0
BMI_MU_MIN, BMI_MU_MAX = 12.0, 40.0
BMI_SIGMA_MIN, BMI_SIGMA_MAX = 1.5, 15.0
PROBIT_EPS = 1e-4


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
    partial = [c for c in columns
               if _clean(c).startswith(hint_clean) or hint_clean.startswith(_clean(c))]
    if partial:
        return min(partial, key=lambda c: abs(len(_clean(c)) - len(hint_clean)))
    return None


def read_any(path):
    """Read a CSV trying several (encoding, separator) combinations."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc,
                                 engine="python", on_bad_lines="skip")
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


def report_indicator(df, path, expected_codes):
    """Print IndicatorCode/Indicator and warn if code is not in expected set."""
    if "IndicatorCode" not in df.columns:
        return
    code = df["IndicatorCode"].dropna().iloc[0] if df["IndicatorCode"].notna().any() else None
    text = df["Indicator"].dropna().iloc[0] if "Indicator" in df.columns and df["Indicator"].notna().any() else ""
    print(f"  [{path.stem}] code={code} : {text[:90]}")
    if expected_codes and code and code not in expected_codes:
        print(f"  [warn] {path.name}: expected one of {expected_codes}, got {code} "
              f"— semantic mismatch with original column is likely")


def collapse_sex_dim(df, prefer_sum):
    """Collapse Dim1 to a single aggregate row per (Location, Period).

    Order of preference:
      1. Any value in DIM1_PREFERRED → keep those rows.
      2. If both Male and Female present → sum (for counts) or mean (for rates).
      3. Otherwise keep whatever is there and use first per group.
    """
    if "Dim1" not in df.columns:
        return df
    for val in DIM1_PREFERRED:
        if (df["Dim1"] == val).any():
            return df[df["Dim1"] == val].copy()

    sexes = set(df["Dim1"].dropna().unique())
    if {"Male", "Female"}.issubset(sexes):
        df = df[df["Dim1"].isin(["Male", "Female"])].copy()
        agg = "sum" if prefer_sum else "mean"
        print(f"    [info] no 'Both sexes' — {agg} Male+Female")
        df = (df.groupby(["Location", "Period"], as_index=False)
                .agg({"_val": agg}))
        return df

    return df


def filter_dim2(df):
    """Keep only the reference age group if Dim2 has multiple values."""
    if "Dim2" not in df.columns or df["Dim2"].nunique(dropna=True) <= 1:
        return df
    for val in DIM2_PREFERRED:
        if (df["Dim2"] == val).any():
            return df[df["Dim2"] == val].copy()
    return df


def process_additional_file(path, target_col, expected_codes=None):
    """Convert one WHO GHO file into a (Country, Year, target_col) frame."""
    df = read_any(path)
    if df is None or df.empty:
        print(f"[warn] cannot read {path.name}")
        return None
    if "Location" not in df.columns and "SpatialDim" in df.columns:
        df = df.rename(columns={"SpatialDim": "Location"})
    if "Location type" in df.columns:
        df = df[df["Location type"] == "Country"]

    report_indicator(df, path, expected_codes)

    values = extract_values(df)
    if values is None:
        print(f"[warn] no numeric values in {path.name}")
        return None
    df = df.assign(_val=values)

    df["Period"] = pd.to_numeric(df["Period"], errors="coerce")
    df = df[df["Period"].between(YEAR_MIN, YEAR_MAX)].copy()
    if df.empty:
        return None

    df = filter_dim2(df)
    df = collapse_sex_dim(df, prefer_sum=target_col in SUM_DIM1)

    if "Dim1" in df.columns and "_val" in df.columns:
        df = df[["Location", "Period", "_val"]]
    df = df.rename(columns={"Location": "Country", "Period": "Year", "_val": target_col})
    df["Country"] = df["Country"].map(normalize_country)
    df = df.dropna(subset=[target_col])
    if df.empty:
        return None
    return df.groupby(["Country", "Year"], as_index=False)[target_col].first()


def estimate_mean_bmi(p_under, p_over25, p_ob30):
    """Estimate mean adult BMI from three prevalence percentages (probit fit)."""
    points = []
    for value, x, is_under in ((p_under, 18.5, True),
                               (p_over25, 25.0, False),
                               (p_ob30, 30.0, False)):
        if pd.isna(value) or not (0 < value < 100):
            continue
        F = value / 100.0 if is_under else 1.0 - value / 100.0
        points.append((x, F))
    if not points:
        return np.nan

    xs = np.array([x for x, _ in points])
    Fs = np.clip([F for _, F in points], PROBIT_EPS, 1.0 - PROBIT_EPS)
    zs = norm.ppf(Fs)

    if len(points) >= 2:
        A = np.vstack([xs, np.ones(len(xs))]).T
        try:
            coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan
        a, b = coef
        if a <= 0:
            return np.nan
        sigma = 1.0 / a
        mu = -b / a
        if not (BMI_SIGMA_MIN <= sigma <= BMI_SIGMA_MAX):
            return np.nan
    else:
        x, z = xs[0], zs[0]
        mu = x - DEFAULT_BMI_SIGMA * z

    return float(mu) if BMI_MU_MIN <= mu <= BMI_MU_MAX else np.nan


def load_bmi_constraints(add_paths):
    """Load the three BMI prevalence files and merge them on (Country, Year)."""
    merged = None
    for prefix, new_col in BMI_CONSTRAINT_FILES.items():
        path = next((p for p in add_paths if p.stem == f"{prefix}_add_data"), None)
        if path is None:
            continue
        piece = process_additional_file(path, new_col)
        if piece is None or piece.empty:
            continue
        merged = piece if merged is None else merged.merge(
            piece, on=["Country", "Year"], how="outer")
    return merged


def build_bmi_column(add_paths, original_countries):
    """Estimate mean BMI for every (Country, Year) covered by the three files."""
    constraints = load_bmi_constraints(add_paths)
    if constraints is None or constraints.empty:
        return pd.DataFrame(columns=["Country", "Year", "BMI"])
    constraints = constraints[constraints["Country"].isin(original_countries)].copy()
    constraints["BMI"] = constraints.apply(
        lambda r: estimate_mean_bmi(r.get("p_under", np.nan),
                                     r.get("p_over25", np.nan),
                                     r.get("p_ob30", np.nan)),
        axis=1)
    return constraints[["Country", "Year", "BMI"]].dropna(subset=["BMI"])


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

    bmi_col = next((c for c in target_columns if _clean(c) == "bmi"), None)
    bmi_input_files = {f"{p}_add_data" for p in BMI_CONSTRAINT_FILES}

    merged = None
    print("Processing additional files:")
    for path in add_paths:
        if path.stem in bmi_input_files:
            continue
        prefix = path.stem[: -len("_add_data")]
        target_col = match_column(prefix, target_columns)
        if target_col is None:
            print(f"[warn] no column match for {path.name} (prefix={prefix!r})")
            continue
        piece = process_additional_file(path, target_col,
                                        EXPECTED_INDICATORS.get(prefix))
        if piece is None or piece.empty:
            continue
        merged = piece if merged is None else merge_piece(merged, piece, target_col)

    print("\nBMI prevalence files:")
    bmi_piece = build_bmi_column(add_paths, original_countries)
    if bmi_col is not None and not bmi_piece.empty:
        bmi_piece = bmi_piece.rename(columns={"BMI": bmi_col})
        merged = bmi_piece if merged is None else merge_piece(merged, bmi_piece, bmi_col)

    if merged is None:
        merged = pd.DataFrame(columns=["Country", "Year"])
    merged = merged[merged["Country"].isin(original_countries)].copy()

    for col in target_columns:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = (merged[target_columns]
              .sort_values(["Country", "Year"])
              .reset_index(drop=True))
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