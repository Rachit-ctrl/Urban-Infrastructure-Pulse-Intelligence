"""Create ZIP-area complaint composition features for Urban Infrastructure Pulse Intelligence.

Input:  data/processed/area_features_by_zip.csv
Output: data/processed/area_composition_features.csv

Run from project root:
    python -m src.features.create_composition_features

Preserves the existing downstream schema: all original area features,
share_<category> for each count_<category>, and category_share_sum.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "area_features_by_zip.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "area_composition_features.csv"
EXPECTED_COMPLAINT_CATEGORIES = 15
TOLERANCE = 1e-6


def load_data():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Area-level input not found: {INPUT_FILE}\n"
            "First run: python -m src.features.aggregate_area_features"
        )
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    if df.empty:
        raise ValueError("Area-level dataset is empty.")
    print(f"Loaded {len(df):,} ZIP/borough areas, {len(df.columns)} columns.")
    return df



def identify_category_columns(df):

    category_columns = [
        col for col in df.columns
        if col.startswith("count_")
    ]

    if not category_columns:
        raise ValueError(
            "No category count columns found."
        )

    # Verify counts match the number of complaints
    count_totals = df[category_columns].sum(axis=1)

    if not count_totals.eq(df["total_complaints"]).all():
        raise ValueError(
            "Category counts do not match total complaints."
        )

    print(
        "Available ZIP-level complaint categories:",
        len(category_columns)
    )

    print(
        "Note: Bridge Condition is excluded from ZIP-level "
        "aggregation because all 840 records lack ZIP codes."
    )

    return category_columns



def create_composition_features(df):
    df = df.copy()
    count_cols = identify_category_columns(df)
    required = ["area_id", "total_complaints"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    if df["area_id"].isna().any() or df["area_id"].duplicated().any():
        raise ValueError("area_id must be unique and nonmissing.")

    totals = pd.to_numeric(df["total_complaints"], errors="coerce")
    counts = df[count_cols].apply(pd.to_numeric, errors="coerce")
    if totals.isna().any() or (totals <= 0).any():
        bad = df.loc[totals.isna() | (totals <= 0), "area_id"].head(5).tolist()
        raise ValueError(f"Every area needs total_complaints > 0. Invalid areas: {bad}")
    if counts.isna().any().any() or (counts < 0).any().any():
        raise ValueError("Category counts must be nonmissing and nonnegative.")

    # A difference here means at least one complaint was not represented
    # in a count_* category, so percentages would be misleading.
    count_totals = counts.sum(axis=1)
    mismatch = ~np.isclose(count_totals, totals, rtol=0, atol=TOLERANCE)
    if mismatch.any():
        examples = df.loc[mismatch, ["area_id", "total_complaints"]].head(5)
        raise ValueError(
            f"Category counts do not add up to total_complaints for "
            f"{int(mismatch.sum()):,} areas. First examples:\n{examples.to_string(index=False)}"
        )

    shares = counts.div(totals, axis=0)
    share_cols = [c.replace("count_", "share_", 1) for c in count_cols]
    shares.columns = share_cols
    # Idempotent if the function is rerun with an already enriched dataframe.
    df = df.drop(columns=[c for c in share_cols + ["category_share_sum"] if c in df.columns])
    df[share_cols] = shares
    df["category_share_sum"] = shares.sum(axis=1)
    print(f"Created {len(share_cols)} complaint-share features.")
    return df


def validate_composition(df):
    share_cols = [c for c in df.columns if c.startswith("share_")]
    values = df[share_cols].to_numpy(dtype=float)
    valid = (
        np.isfinite(values).all()
        and np.all(values >= -TOLERANCE)
        and np.all(values <= 1 + TOLERANCE)
        and np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=TOLERANCE)
    )
    if not valid:
        raise ValueError("Invalid category shares: each area must sum to 1, with values in [0, 1].")
    print("Composition validation: PASSED")
    print(f"Share range: {values.min():.6f} to {values.max():.6f}")
    print(f"Complaint total represented: {int(df['total_complaints'].sum()):,}")


def main():
    df = load_data()
    enriched = create_composition_features(df)
    validate_composition(enriched)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_FILE, index=False)
    print("\nCOMPOSITION FEATURE ENGINEERING COMPLETE")
    print("Saved:", OUTPUT_FILE)
    print("Output shape:", enriched.shape)
    print(enriched[["area_id", "total_complaints", "category_share_sum"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
