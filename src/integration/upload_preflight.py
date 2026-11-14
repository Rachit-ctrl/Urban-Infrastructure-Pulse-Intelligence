"""Safe in-memory upload validation for NYC 311 batches.

Uses the existing bulk complaint contract.
This module does not run models or modify files.
"""

from io import BytesIO

import pandas as pd

from src.integration.bulk_complaint_contract import (
    REQUIRED_COLUMNS,
    read_batch_csv,
    validate_batch
)


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 5000


def _find_validation_frame(value, expected_rows):
    """Find row-level results in the contract output."""

    if isinstance(value, pd.DataFrame):
        if len(value) == expected_rows:
            return value.copy()
        return None

    if isinstance(value, dict):
        for item in value.values():
            found = _find_validation_frame(
                item, expected_rows
            )
            if found is not None:
                return found

    if isinstance(value, (tuple, list)):

        if (
            len(value) == expected_rows
            and all(
                isinstance(item, dict)
                for item in value
            )
        ):
            return pd.DataFrame(value)

        for item in value:
            found = _find_validation_frame(
                item, expected_rows
            )
            if found is not None:
                return found

    return None


def run_upload_preflight(
    csv_bytes,
    supported_categories,
    max_bytes=MAX_UPLOAD_BYTES,
    max_rows=MAX_UPLOAD_ROWS
):
    """Read and validate CSV, without processing it."""

    if not isinstance(
        csv_bytes, (bytes, bytearray)
    ):
        raise TypeError(
            "Uploaded CSV must be provided as bytes."
        )

    if len(csv_bytes) == 0:
        raise ValueError("Uploaded CSV is empty.")

    if len(csv_bytes) > max_bytes:
        raise ValueError(
            "Uploaded CSV exceeds the size limit."
        )

    frame = read_batch_csv(
        BytesIO(csv_bytes)
    )

    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            "CSV reader did not return a DataFrame."
        )

    if frame.columns.duplicated().any():
        raise ValueError(
            "Duplicate CSV column names."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    if frame.empty:
        raise ValueError(
            "CSV has no complaint rows."
        )

    if len(frame) > max_rows:
        raise ValueError(
            f"Maximum {max_rows} complaints per upload."
        )

    # The established project contract is
    # the sole source of row validation.
    validation = validate_batch(
        frame.copy(deep=True),
        supported_categories
    )

    results = _find_validation_frame(
        validation,
        len(frame)
    )

    if results is None:
        raise ValueError(
            "Could not identify one validation result "
            "per uploaded row. Review the bulk "
            "contract return structure."
        )

    status_column = next(
        (
            column for column in [
                "validation_status",
                "status"
            ]
            if column in results.columns
        ),
        None
    )

    if status_column is not None:
        status_counts = {
            str(key): int(value)
            for key, value in results[
                status_column
            ].astype(str).value_counts().items()
        }
    else:
        status_counts = {}

    return {
        "input_records": frame,
        "validation_results": results,
        "row_count": int(len(frame)),
        "status_column": status_column,
        "status_counts": status_counts,
        "model_processing_performed": False,
        "historical_models_modified": False
    }
