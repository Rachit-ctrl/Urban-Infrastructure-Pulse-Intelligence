
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nyc311_infrastructure_100k_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "area_features_by_zip.csv"


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    return df


# --------------------------------------------------
# 3. Prepare aggregation columns
# --------------------------------------------------

def prepare_data(df):

    required_columns = [
        "unique_key",
        "borough",
        "incident_zip",
        "complaint_type",
        "latitude",
        "longitude",
        "resolution_hours",
        "is_closed_status",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Remove duplicate complaint IDs, if any.
    df = df.drop_duplicates(
        subset=["unique_key"]
    ).copy()

    # Standardize geographical identifiers.
    df["borough"] = (
        df["borough"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["incident_zip"] = (
        pd.to_numeric(
            df["incident_zip"],
            errors="coerce"
        )
    )

    # Keep records with valid area identifiers.
    df = df.dropna(
        subset=["borough", "incident_zip"]
    ).copy()

    df = df[
        df["borough"].ne("UNSPECIFIED")
    ]

    df["incident_zip"] = (
        df["incident_zip"]
        .astype(int)
        .astype(str)
    )

    # Convert numerical fields safely.
    df["resolution_hours"] = pd.to_numeric(
        df["resolution_hours"],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Exclude negative resolution durations.
    df.loc[
        df["resolution_hours"] < 0,
        "resolution_hours"
    ] = np.nan

    return df


# --------------------------------------------------
# 4. Aggregate area-level features
# --------------------------------------------------

def create_area_features(df):

    group_columns = [
        "borough",
        "incident_zip"
    ]

    grouped = df.groupby(
        group_columns,
        observed=True
    )

    area_features = grouped.agg(

        total_complaints=(
            "unique_key",
            "count"
        ),

        category_diversity=(
            "complaint_type",
            "nunique"
        ),

        median_resolution_hours=(
            "resolution_hours",
            "median"
        ),

        p90_resolution_hours=(
            "resolution_hours",
            lambda x: x.quantile(0.90)
        ),

        resolution_available_rate=(
            "resolution_hours",
            lambda x: x.notna().mean()
        ),

        unresolved_rate=(
            "is_closed_status",
            lambda x: (~x.fillna(False).astype(bool)).mean()
        ),

        latitude=(
            "latitude",
            "mean"
        ),

        longitude=(
            "longitude",
            "mean"
        )

    ).reset_index()

    # Create complaint counts for each category.
    category_counts = pd.crosstab(
        index=[
            df["borough"],
            df["incident_zip"]
        ],
        columns=df["complaint_type"]
    )

    category_counts.columns = [
        f"count_{str(col).lower().replace(' ', '_').replace('/', '_')}"
        for col in category_counts.columns
    ]

    category_counts = category_counts.reset_index()

    # Merge general and category-specific features.
    area_features = area_features.merge(
        category_counts,
        on=group_columns,
        how="left"
    )

    area_features = area_features.fillna({
        col: 0
        for col in category_counts.columns
        if col not in group_columns
    })

    area_features.insert(
        0,
        "area_id",
        area_features["borough"].astype(str)
        + "_"
        + area_features["incident_zip"].astype(str)
    )

    return area_features


# --------------------------------------------------
# 5. Save and validate
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    df = prepare_data(df)

    area_features = create_area_features(df)

    area_features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nArea-level feature dataset created.")

    print("Output:", OUTPUT_FILE)

    print("\nFinal shape:")
    print(area_features.shape)

    print("\nFirst five rows:")
    print(area_features.head())

    print("\nMissing values:")
    print(
        area_features.isnull().sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTotal complaints represented:")
    print(
        area_features["total_complaints"].sum()
    )

    print("\nNumber of geographical areas:")
    print(len(area_features))


if __name__ == "__main__":
    main()

