import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import OUTPUT_DIR, PROCESSED_DATA_DIR


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "area_modeling_features_with_persistence.csv"
)

REPORT_DIR = OUTPUT_DIR / "tables" / "feature_quality"

MISSING_THRESHOLD = 0.20
NEAR_CONSTANT_THRESHOLD = 0.95
CORRELATION_THRESHOLD = 0.90
SKEWNESS_THRESHOLD = 2.0


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    logger.info("Loading modeling dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Modeling dataset is empty.")

    logger.info("Dataset loaded: %s", df.shape)

    return df


# --------------------------------------------------
# BASIC DATA QUALITY
# --------------------------------------------------

def check_basic_quality(df):

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_area_ids": (
            int(df["area_id"].duplicated().sum())
            if "area_id" in df.columns else None
        ),
        "duplicate_columns": int(
            df.columns.duplicated().sum()
        ),
        "total_missing_values": int(
            df.isna().sum().sum()
        )
    }

    return pd.DataFrame(
        report.items(),
        columns=["metric", "value"]
    )


# --------------------------------------------------
# MISSING VALUE ANALYSIS
# --------------------------------------------------

def check_missing_values(df):

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (
            df.isna().mean().values * 100
        )
    })

    report["status"] = np.where(
        report["missing_percentage"] > MISSING_THRESHOLD * 100,
        "HIGH_MISSING",
        np.where(
            report["missing_count"] > 0,
            "HAS_MISSING",
            "COMPLETE"
        )
    )

    return report.sort_values(
        "missing_percentage",
        ascending=False
    )


# --------------------------------------------------
# NUMERICAL FEATURE ANALYSIS
# --------------------------------------------------

def check_numerical_features(df):

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    records = []

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if series.empty:
            continue

        unique_values = series.nunique()

        value_counts = series.value_counts(
            normalize=True
        )

        dominant_percentage = (
            value_counts.iloc[0]
            if not value_counts.empty else 1.0
        )

        skewness = (
            series.skew()
            if unique_values > 2 else 0
        )

        records.append({

            "feature": column,

            "unique_values": unique_values,

            "minimum": series.min(),

            "maximum": series.max(),

            "mean": series.mean(),

            "std": series.std(),

            "skewness": skewness,

            "dominant_value_percentage": (
                dominant_percentage * 100
            ),

            "constant": unique_values <= 1,

            "near_constant": (
                dominant_percentage
                >= NEAR_CONSTANT_THRESHOLD
            ),

            "highly_skewed": (
                abs(skewness) > SKEWNESS_THRESHOLD
            )
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------

def check_correlations(df):

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    correlation_matrix = numeric_df.corr(
        method="pearson"
    )

    records = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            feature_1 = columns[i]
            feature_2 = columns[j]

            correlation = correlation_matrix.loc[
                feature_1, feature_2
            ]

            if pd.notna(correlation):

                if abs(correlation) >= CORRELATION_THRESHOLD:

                    records.append({

                        "feature_1": feature_1,

                        "feature_2": feature_2,

                        "correlation": correlation,

                        "absolute_correlation": abs(
                            correlation
                        )
                    })

    return pd.DataFrame(records)


# --------------------------------------------------
# CLUSTERING EXCLUSION REVIEW
# --------------------------------------------------

def identify_excluded_features(df):

    excluded = {}

    for column in df.columns:

        reason = None

        if column in [
            "area_id",
            "borough",
            "incident_zip"
        ]:
            reason = "Geographic identifier"

        elif column in [
            "latitude",
            "longitude",
            "mean_latitude",
            "mean_longitude"
        ]:
            reason = "Geographic coordinates; assess separately"

        elif column in [
            "category_share_sum"
        ]:
            reason = "Validation column, not a clustering feature"

        elif not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            reason = "Non-numeric column"

        if reason:

            excluded[column] = reason

    return pd.DataFrame(
        [
            {
                "feature": column,
                "reason": reason
            }
            for column, reason in excluded.items()
        ]
    )


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    # 1. Basic quality
    basic_report = check_basic_quality(df)

    # 2. Missing values
    missing_report = check_missing_values(df)

    # 3. Numerical features
    numerical_report = check_numerical_features(df)

    # 4. Correlations
    correlation_report = check_correlations(df)

    # 5. Excluded features
    excluded_report = identify_excluded_features(df)

    # Save reports
    basic_report.to_csv(
        REPORT_DIR / "basic_quality_report.csv",
        index=False
    )

    missing_report.to_csv(
        REPORT_DIR / "missing_values_report.csv",
        index=False
    )

    numerical_report.to_csv(
        REPORT_DIR / "numerical_feature_report.csv",
        index=False
    )

    correlation_report.to_csv(
        REPORT_DIR / "high_correlation_report.csv",
        index=False
    )

    excluded_report.to_csv(
        REPORT_DIR / "excluded_features_review.csv",
        index=False
    )

    # Summary
    print("\nFEATURE QUALITY ASSESSMENT")
    print("=" * 50)

    print("\nBasic Quality:")
    print(basic_report.to_string(index=False))

    print("\nMissing Values:")
    print(
        "Columns with missing values:",
        int((missing_report["missing_count"] > 0).sum())
    )

    print("\nNumerical Feature Analysis:")
    print(
        "Total numerical features:",
        len(numerical_report)
    )

    print(
        "Constant features:",
        int(numerical_report["constant"].sum())
    )

    print(
        "Near-constant features:",
        int(numerical_report["near_constant"].sum())
    )

    print(
        "Highly skewed features:",
        int(numerical_report["highly_skewed"].sum())
    )

    print("\nCorrelation Analysis:")
    print(
        "Highly correlated feature pairs:",
        len(correlation_report)
    )

    if not correlation_report.empty:
        print(
            correlation_report.sort_values(
                "absolute_correlation",
                ascending=False
            ).head(15).to_string(index=False)
        )

    print("\nExcluded Feature Review:")
    print(excluded_report.to_string(index=False))

    print("\nReports saved to:")
    print(REPORT_DIR)

    logger.info("Feature quality assessment completed.")


if __name__ == "__main__":
    main()