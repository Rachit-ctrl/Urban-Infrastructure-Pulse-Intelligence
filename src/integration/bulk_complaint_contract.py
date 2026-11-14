"""Validation contract for bulk infrastructure complaints.

This module checks input syntax, required fields,
categories, dates, IDs and coordinate ranges.

It does not assign NYC boroughs, DBSCAN clusters,
K-Means profiles or operational priorities.
"""

from datetime import date, datetime
import math
import re
import pandas as pd

INPUT_COLUMNS = (
    "complaint_id",
    "created_date",
    "latitude",
    "longitude",
    "complaint_category",
    "incident_zip",
)

REQUIRED_COLUMNS = INPUT_COLUMNS[:-1]


def _text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def _valid_date(value):
    # Date-only inputs: YYYY-MM-DD
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", value
    ):
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            return False

    # Timestamp inputs require an explicit
    # UTC offset or Z suffix.
    pattern = (
        r"\d{4}-\d{2}-\d{2}"
        r"T\d{2}:\d{2}"
        r"(?::\d{2}(?:\.\d+)?)?"
        r"(?:Z|[+-]\d{2}:\d{2})"
    )

    if not re.fullmatch(pattern, value):
        return False

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        return parsed.utcoffset() is not None
    except ValueError:
        return False


def _coordinate(value, low, high):
    if isinstance(value, bool):
        return None

    text = _text(value)

    if not text:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    if not low <= number <= high:
        return None

    return number


def read_batch_csv(path_or_file):
    """Read all fields as text to preserve ZIP zeros."""
    return pd.read_csv(
        path_or_file,
        dtype="string",
        keep_default_na=False
    )


def validate_batch(frame, supported_categories):
    """Return one result for every submitted row.

    Invalid records are retained with error reasons.
    Valid records remain eligible for later
    geographic validation and historical analysis.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if frame.columns.duplicated().any():
        raise ValueError(
            "Duplicate CSV column names are not allowed."
        )

    missing = [
        name
        for name in REQUIRED_COLUMNS
        if name not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if frame.empty:
        raise ValueError(
            "The complaint CSV contains no records."
        )

    allowed = {
        str(value).strip().upper()
        for value in supported_categories
    }

    if not allowed:
        raise ValueError(
            "Supported complaint categories are empty."
        )

    work = frame.copy().reset_index(drop=True)

    if "incident_zip" not in work.columns:
        work["incident_zip"] = ""

    ids = work["complaint_id"].map(_text)

    duplicate_ids = (
        ids.ne("")
        & ids.duplicated(keep=False)
    )

    records = []

    for index, row in work.iterrows():

        errors = []

        complaint_id = _text(
            row["complaint_id"]
        )

        created_date = _text(
            row["created_date"]
        )

        category = _text(
            row["complaint_category"]
        ).upper()

        incident_zip = _text(
            row["incident_zip"]
        )

        latitude = _coordinate(
            row["latitude"], -90, 90
        )

        longitude = _coordinate(
            row["longitude"], -180, 180
        )

        if not complaint_id:
            errors.append("missing_complaint_id")

        if bool(duplicate_ids.iloc[index]):
            errors.append("duplicate_complaint_id")

        if not _valid_date(created_date):
            errors.append("invalid_created_date")

        if latitude is None:
            errors.append("invalid_latitude")

        if longitude is None:
            errors.append("invalid_longitude")

        if category not in allowed:
            errors.append(
                "unsupported_complaint_category"
            )

        if incident_zip and not re.fullmatch(
            r"[0-9]{5}", incident_zip
        ):
            errors.append("invalid_incident_zip")

        status = (
            "ready_for_geographic_validation"
            if not errors
            else "invalid_input"
        )

        records.append({
            "source_row": index + 2,
            "complaint_id": complaint_id,
            "created_date": created_date,
            "latitude": latitude,
            "longitude": longitude,
            "complaint_category": category,
            "incident_zip": incident_zip or None,
            "validation_status": status,
            "validation_errors": "; ".join(errors)
        })

    result = pd.DataFrame.from_records(records)

    # Pandas may convert optional missing ZIPs
    # into NaN. The backend expects Python None.
    result["incident_zip"] = (
        result["incident_zip"]
        .astype(object)
        .where(result["incident_zip"].notna(), None)
    )

    return result
