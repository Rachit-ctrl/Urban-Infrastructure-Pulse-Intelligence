
from pathlib import Path

import pandas as pd


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "area_features_by_zip.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "area_composition_features.csv"
)


# ==================================================
# LOAD AREA-LEVEL DATA
# ==================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Area-level dataset is empty.")

    print("Area-level dataset loaded.")
    print("Shape:", df.shape)

    return df


# ==================================================
# IDENTIFY CATEGORY COUNT COLUMNS
# ==================================================

def identify_category_columns(df):

    category_columns = [
        col for col in df.columns
        if col.startswith("count_")
    ]

    if not category_columns:
        raise ValueError(
            "No category count columns found."
        )

    return category_columns


# ==================================================
# CREATE CATEGORY COMPOSITION FEATURES
# ==================================================

def create_composition_features(df):

    category_columns = identify_category_columns(df)

    total_complaints = df["total_complaints"]

    for column in category_columns:

        feature_name = (
            column.replace("count_", "share_", 1)
        )

        df[feature_name] = (
            df[column]
            / total_complaints
        )

    # Validate category proportions.
    share_columns = [
        col for col in df.columns
        if col.startswith("share_")
    ]

    df["category_share_sum"] = (
        df[share_columns].sum(axis=1)
    )

    print("\nComposition features created.")
    print("Category count features:", len(category_columns))
    print("Category share features:", len(share_columns))

    return df


# ==================================================
# VALIDATE COMPOSITION
# ==================================================

def validate_composition(df):

    share_columns = [
        col for col in df.columns
        if col.startswith("share_")
    ]

    share_sums = df[share_columns].sum(axis=1)

    invalid_rows = (
        (share_sums - 1).abs() > 0.0001
    )

    print("\nComposition validation")
    print("----------------------")

    print("Invalid composition rows:", invalid_rows.sum())

    print(
        "Minimum category share:",
        df[share_columns].min().min()
    )

    print(
        "Maximum category share:",
        df[share_columns].max().max()
    )

    if invalid_rows.any():
        raise ValueError(
            "Category shares do not sum to 1. "
            "Investigate the input data."
        )


# ==================================================
# MAIN
# ==================================================

def main():

    df = load_data()

    df = create_composition_features(df)

    validate_composition(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nComposition dataset saved successfully.")
    print("Output:", OUTPUT_FILE)

    print("\nDataset shape:", df.shape)

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()

