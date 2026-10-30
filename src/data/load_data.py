import logging

import pandas as pd

from config.config import (
    MASTER_DATA_FILE,
    EXPECTED_RECORDS,
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

def load_master_data(
    file_path=MASTER_DATA_FILE,
    low_memory=False
):

    file_path = str(file_path)

    try:

        logger.info("Loading master dataset: %s", file_path)

        df = pd.read_csv(
            file_path,
            low_memory=low_memory
        )

        if df.empty:
            raise ValueError(
                "The dataset is empty."
            )

        logger.info(
            "Dataset loaded: %s rows, %s columns",
            len(df),
            len(df.columns)
        )

        return df

    except FileNotFoundError:

        logger.exception(
            "Master dataset not found: %s",
            file_path
        )

        raise

    except Exception:

        logger.exception(
            "Error while loading the master dataset."
        )

        raise


# ============================================================
# BASIC DATASET VALIDATION
# ============================================================

def validate_dataset(df):

    required_columns = [
        "unique_key",
        "created_date",
        "complaint_type",
        "borough",
        "latitude",
        "longitude",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    validation_results = {

        "total_records": len(df),

        "total_columns": len(df.columns),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "duplicate_complaint_ids": int(
            df["unique_key"].duplicated().sum()
        ),

        "missing_complaint_ids": int(
            df["unique_key"].isna().sum()
        ),

        "missing_creation_dates": int(
            df["created_date"].isna().sum()
        ),

        "distinct_categories": int(
            df["complaint_type"].nunique()
        ),

        "distinct_boroughs": int(
            df["borough"].nunique()
        ),

    }

    return validation_results


# ============================================================
# LOAD AND VALIDATE
# ============================================================

def load_and_validate():

    df = load_master_data()

    results = validate_dataset(df)

    if len(df) != EXPECTED_RECORDS:

        logger.warning(
            "Expected %s records but found %s.",
            EXPECTED_RECORDS,
            len(df)
        )

    return df, results