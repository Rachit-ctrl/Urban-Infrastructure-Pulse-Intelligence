"""Select one NYC-local reporting day from an incoming batch.

Naive timestamps are interpreted as New York local
time. Explicit timestamp offsets are converted to
New York time.

Ambiguous/nonexistent local DST times are rejected.
All original records remain auditable.
"""

from datetime import date, datetime
import pandas as pd


NYC_TIMEZONE = "America/New_York"


def _parse_nyc_day(value):
    """Return (NYC day, parsing status)."""

    if pd.isna(value):
        return None, "invalid_or_missing_date"

    text = str(value).strip()

    if not text:
        return None, "invalid_or_missing_date"

    parsed = None

    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError:
        pass

    if parsed is None:

        for fmt in [
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %H:%M:%S"
        ]:
            try:
                parsed = datetime.strptime(
                    text, fmt
                )
                break
            except ValueError:
                continue

    if parsed is None:
        return None, "invalid_or_missing_date"

    try:
        timestamp = pd.Timestamp(parsed)

        if timestamp.tzinfo is None:

            timestamp = timestamp.tz_localize(
                NYC_TIMEZONE,
                ambiguous="NaT",
                nonexistent="NaT"
            )

            parse_status = "assumed_nyc_local"

        else:

            timestamp = timestamp.tz_convert(
                NYC_TIMEZONE
            )

            parse_status = "converted_timezone"

        if pd.isna(timestamp):
            return None, "invalid_or_missing_date"

        return (
            timestamp.date().isoformat(),
            parse_status
        )

    except (ValueError, TypeError, OverflowError):
        return None, "invalid_or_missing_date"


def select_daily_complaints(
    records,
    reporting_day
):
    """Select an explicit NYC reporting day.

    Returns selected records plus a row-level audit.
    Does not mutate the input DataFrame.
    """

    if not isinstance(records, pd.DataFrame):
        raise TypeError("Expected a DataFrame.")

    required = {
        "source_row",
        "complaint_id",
        "created_date"
    }

    missing = required - set(records.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if records["source_row"].duplicated().any():
        raise ValueError(
            "Duplicate source_row values."
        )

    if not isinstance(reporting_day, str):
        raise ValueError(
            "reporting_day must be YYYY-MM-DD."
        )

    try:
        target = date.fromisoformat(
            reporting_day
        ).isoformat()
    except ValueError as exc:
        raise ValueError(
            "Invalid reporting day."
        ) from exc

    if target != reporting_day:
        raise ValueError(
            "Use YYYY-MM-DD reporting day."
        )

    work = records.copy().reset_index(drop=True)

    parsed = [
        _parse_nyc_day(value)
        for value in work["created_date"]
    ]

    work["nyc_reporting_day"] = [
        item[0] for item in parsed
    ]

    work["date_parse_status"] = [
        item[1] for item in parsed
    ]

    work["daily_window_status"] = (
        "outside_selected_day"
    )

    invalid = work["nyc_reporting_day"].isna()

    work.loc[
        invalid,
        "daily_window_status"
    ] = "invalid_or_missing_date"

    selected = work[
        "nyc_reporting_day"
    ].eq(target).fillna(False)

    work.loc[
        selected,
        "daily_window_status"
    ] = "selected_day"

    # Select from the original records, preserving
    # original model input columns and row order.
    selected_records = (
        records.reset_index(drop=True)
        .loc[selected.to_numpy()]
        .copy()
    )

    audit_columns = [
        "source_row",
        "complaint_id",
        "created_date",
        "nyc_reporting_day",
        "date_parse_status",
        "daily_window_status"
    ]

    audit = work[audit_columns].copy()

    stats = {
        "reporting_day": target,
        "timezone": NYC_TIMEZONE,
        "input_records": int(len(work)),
        "selected_records": int(selected.sum()),
        "outside_selected_day": int(
            work["daily_window_status"].eq(
                "outside_selected_day"
            ).sum()
        ),
        "invalid_or_missing_date": int(
            invalid.sum()
        ),
        "naive_local_assumptions": int(
            work["date_parse_status"].eq(
                "assumed_nyc_local"
            ).sum()
        ),
        "timezone_conversions": int(
            work["date_parse_status"].eq(
                "converted_timezone"
            ).sum()
        )
    }

    assert (
        stats["selected_records"]
        + stats["outside_selected_day"]
        + stats["invalid_or_missing_date"]
    ) == stats["input_records"]

    return {
        "selected_records": selected_records,
        "audit": audit,
        "stats": stats
    }
