"""Exploratory historical-versus-incoming complaint screen.

Uses weekday-matched daily count references.
Screening candidates are not confirmed anomalies,
statistical significance claims, or repair priorities.

Historical area coverage must be explicitly supplied.
"""

import re
import numpy as np
import pandas as pd

from src.integration.daily_batch_window import (
    select_daily_complaints
)


VALID_BOROUGHS = {
    "BRONX",
    "BROOKLYN",
    "MANHATTAN",
    "QUEENS",
    "STATEN ISLAND"
}


def _window_evidence(
    daily_counts,
    area_id,
    category,
    dates,
    min_historical_reports,
    min_active_days,
    min_years
):
    labels = dates.strftime("%Y-%m-%d")

    values = np.array([
        int(daily_counts.get(
            (area_id, category, day),
            0
        ))
        for day in labels
    ], dtype=np.int64)

    active = values > 0

    years = len(set(
        dates.year[active].tolist()
    ))

    total = int(values.sum())
    active_days = int(active.sum())

    supported = (
        total >= min_historical_reports
        and active_days >= min_active_days
        and years >= min_years
    )

    return {
        "calendar_days": len(values),
        "reports": total,
        "active_days": active_days,
        "years_with_reports": years,
        "supported": bool(supported),
        "mean": (
            float(values.mean())
            if len(values) else 0.0
        ),
        "p95": (
            float(np.quantile(values, 0.95))
            if len(values) else 0.0
        ),
        "maximum": (
            int(values.max())
            if len(values) else 0
        )
    }


def screen_daily_anomalies(
    records,
    supported_categories,
    historical_daily_counts,
    historical_calendar,
    historical_area_ids,
    reporting_day,
    min_incoming_reports=8,
    min_historical_reports=8,
    min_active_days=5,
    min_years=2
):
    """Return exploratory anomaly evidence and row audit.

    Only processed, ZIP-consistent complaints
    belonging to the selected NYC reporting day
    enter the area/category count comparison.
    """

    required = {
        "source_row",
        "complaint_id",
        "created_date",
        "processing_status",
        "zip_validation_status",
        "borough",
        "incident_zip",
        "complaint_category"
    }

    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a DataFrame")

    if records.columns.duplicated().any():
        raise ValueError("Duplicate column names")

    missing = required - set(records.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if min_incoming_reports < 1:
        raise ValueError(
            "min_incoming_reports must be positive"
        )

    categories = {
        str(x).strip().upper()
        for x in supported_categories
    }

    known_areas = {
        str(x).strip().upper()
        for x in historical_area_ids
    }

    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        str(reporting_day)
    ):
        raise ValueError("Use YYYY-MM-DD")

    daily = select_daily_complaints(
        records,
        reporting_day
    )

    selected = daily[
        "selected_records"
    ].copy().reset_index(drop=True)

    borough = (
        selected["borough"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    zip_code = (
        selected["incident_zip"]
        .astype(str)
        .str.strip()
    )

    category = (
        selected["complaint_category"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    processed = selected[
        "processing_status"
    ].astype(str).eq("processed")

    zip_consistent = selected[
        "zip_validation_status"
    ].astype(str).eq("zcta_consistent")

    valid_borough = borough.isin(
        VALID_BOROUGHS
    )

    valid_zip = zip_code.str.fullmatch(
        r"\d{5}",
        na=False
    )

    supported = category.isin(
        categories
    )

    # Order of conditions gives a single
    # auditable reason per excluded record.
    reasons = np.select(
        [
            ~processed,
            ~zip_consistent,
            ~valid_borough,
            ~valid_zip,
            ~supported
        ],
        [
            "not_processed",
            "zip_not_verified",
            "invalid_borough",
            "invalid_zip",
            "unsupported_category"
        ],
        default="eligible"
    )

    selected["anomaly_input_status"] = reasons

    selected["area_id"] = (
        borough + "_" + zip_code
    )

    selected["complaint_category"] = category

    eligible = selected.loc[
        selected["anomaly_input_status"].eq(
            "eligible"
        )
    ].copy()

    incoming_counts = (
        eligible.groupby(
            ["area_id", "complaint_category"]
        )
        .size()
        .rename("incoming_reports")
        .reset_index()
    )

    # Do not include historical reference days
    # that occur on or after the reporting day.
    dates = pd.DatetimeIndex(
        pd.to_datetime(
            list(historical_calendar)
        )
    ).normalize().unique().sort_values()

    if dates.tz is not None:
        raise ValueError(
            "Historical calendar must use local dates"
        )

    target = pd.Timestamp(reporting_day)

    all_weekdays = dates[
        (dates < target)
        & (dates.weekday == target.weekday())
    ]

    # Three-month comparison window centered
    # on the reporting month, wrapping years.
    seasonal_months = {
        ((target.month - 2) % 12) + 1,
        target.month,
        (target.month % 12) + 1
    }

    seasonal = all_weekdays[
        np.isin(
            all_weekdays.month,
            list(seasonal_months)
        )
    ]

    rows = []

    for item in incoming_counts.itertuples(
        index=False
    ):

        area = str(item.area_id)
        complaint_category = str(
            item.complaint_category
        )

        count = int(item.incoming_reports)

        seasonal_ev = _window_evidence(
            historical_daily_counts,
            area,
            complaint_category,
            seasonal,
            min_historical_reports,
            min_active_days,
            min_years
        )

        broad_ev = _window_evidence(
            historical_daily_counts,
            area,
            complaint_category,
            all_weekdays,
            min_historical_reports,
            min_active_days,
            min_years
        )

        area_known = area in known_areas

        # Raw candidate flags are retained solely
        # for audit and threshold comparison.
        seasonal_raw_5 = (
            area_known
            and seasonal_ev["supported"]
            and count >= 5
            and count > seasonal_ev["maximum"]
        )

        broad_raw_5 = (
            area_known
            and broad_ev["supported"]
            and count >= 5
            and count > broad_ev["maximum"]
        )

        seasonal_above_max = (
            count > seasonal_ev["maximum"]
        )

        broad_above_max = (
            count > broad_ev["maximum"]
        )

        if not area_known:

            review_status = (
                "no_historical_area_coverage"
            )

        elif not seasonal_ev["supported"]:

            review_status = (
                "insufficient_historical_support"
            )

        # PHASE 23.3 — STRONG HISTORICAL EXCEEDANCE
        #
        # Require at least 3 extra reports AND
        # at least 50% above each historical maximum.
        #
        # A historical maximum of zero does not
        # require a percentage calculation.
        # Existing history-support checks still apply.

        elif (
            count >= min_incoming_reports
            and broad_ev["supported"]
            and seasonal_above_max
            and broad_above_max
            and (count - seasonal_ev["maximum"]) >= 3
            and (
                seasonal_ev["maximum"] == 0
                or count >= 1.5 * seasonal_ev["maximum"]
            )
            and (count - broad_ev["maximum"]) >= 3
            and (
                broad_ev["maximum"] == 0
                or count >= 1.5 * broad_ev["maximum"]
            )
        ):

            review_status = (
                "cross_window_consistent_candidate"
            )

        # A count may exceed historical maxima
        # without meeting the stronger threshold.
        # Do not classify it as an anomaly candidate.

        elif (
            count >= min_incoming_reports
            and broad_ev["supported"]
            and seasonal_above_max
            and broad_above_max
        ):

            review_status = (
                "historical_max_exceeded_below_strength_threshold"
            )

        elif (
            count >= min_incoming_reports
            and seasonal_above_max
        ):

            review_status = (
                "reference_window_sensitive_review"
            )

        elif seasonal_above_max:

            review_status = (
                "above_seasonal_max_below_report_floor"
            )

        elif count > seasonal_ev["p95"]:

            review_status = "above_seasonal_p95"

        else:

            review_status = (
                "within_matched_historical_range"
            )

        rows.append({
            "reporting_day": reporting_day,
            "area_id": area,
            "complaint_category":
                complaint_category,
            "incoming_reports": count,
            "historical_area_covered": area_known,
            "seasonal_days":
                seasonal_ev["calendar_days"],
            "seasonal_historical_reports":
                seasonal_ev["reports"],
            "seasonal_active_days":
                seasonal_ev["active_days"],
            "seasonal_years_with_reports":
                seasonal_ev["years_with_reports"],
            "seasonal_support":
                seasonal_ev["supported"],
            "seasonal_daily_mean":
                seasonal_ev["mean"],
            "seasonal_daily_p95":
                seasonal_ev["p95"],
            "seasonal_daily_max":
                seasonal_ev["maximum"],
            "all_weekday_days":
                broad_ev["calendar_days"],
            "all_weekday_support":
                broad_ev["supported"],
            "all_weekday_daily_max":
                broad_ev["maximum"],
            "seasonal_candidate_floor_5":
                bool(seasonal_raw_5),
            "all_weekday_candidate_floor_5":
                bool(broad_raw_5),
            "anomaly_review_status":
                review_status
        })

    comparisons = pd.DataFrame(rows)

    # Ensure a stable schema when input is empty.
    if comparisons.empty:
        comparisons = pd.DataFrame(columns=[
            "reporting_day",
            "area_id",
            "complaint_category",
            "incoming_reports",
            "historical_area_covered",
            "seasonal_days",
            "seasonal_historical_reports",
            "seasonal_active_days",
            "seasonal_years_with_reports",
            "seasonal_support",
            "seasonal_daily_mean",
            "seasonal_daily_p95",
            "seasonal_daily_max",
            "all_weekday_days",
            "all_weekday_support",
            "all_weekday_daily_max",
            "seasonal_candidate_floor_5",
            "all_weekday_candidate_floor_5",
            "anomaly_review_status"
        ])

    quality_audit = selected[[
        "source_row",
        "anomaly_input_status"
    ]]

    eligible_review = (
        eligible[[
            "source_row",
            "area_id",
            "complaint_category"
        ]]
        .merge(
            comparisons[[
                "area_id",
                "complaint_category",
                "anomaly_review_status"
            ]],
            on=[
                "area_id",
                "complaint_category"
            ],
            how="left",
            validate="many_to_one"
        )[[
            "source_row",
            "anomaly_review_status"
        ]]
    )

    record_audit = (
        daily["audit"]
        .merge(
            quality_audit,
            on="source_row",
            how="left",
            validate="one_to_one"
        )
        .merge(
            eligible_review,
            on="source_row",
            how="left",
            validate="one_to_one"
        )
    )

    record_audit[
        "anomaly_input_status"
    ] = record_audit[
        "anomaly_input_status"
    ].fillna("not_in_reporting_day")

    record_audit[
        "anomaly_review_status"
    ] = record_audit[
        "anomaly_review_status"
    ].fillna("not_analyzed")

    candidates = comparisons.loc[
        comparisons["anomaly_review_status"].eq(
            "cross_window_consistent_candidate"
        )
    ].copy()

    stats = {
        "input_records": int(len(records)),
        "selected_records": int(len(selected)),
        "date_exclusions": int(
            len(records) - len(selected)
        ),
        "eligible_records": int(len(eligible)),
        "quality_exclusions": int(
            len(selected) - len(eligible)
        ),
        "active_area_category_combinations":
            int(len(comparisons)),
        "candidate_combinations":
            int(len(candidates)),
        "candidate_complaints": (
            int(candidates[
                "incoming_reports"
            ].sum())
            if not candidates.empty else 0
        ),
        "seasonal_reference_days":
            int(len(seasonal)),
        "all_weekday_reference_days":
            int(len(all_weekdays))
    }

    if (
        stats["date_exclusions"]
        + stats["quality_exclusions"]
        + stats["eligible_records"]
        != stats["input_records"]
    ):
        raise AssertionError(
            "Input coverage did not reconcile"
        )

    if (
        int(comparisons[
            "incoming_reports"
        ].sum()) != len(eligible)
    ):
        raise AssertionError(
            "Incoming comparison counts disagree"
        )

    if len(record_audit) != len(records):
        raise AssertionError(
            "Record audit lost source records"
        )

    return {
        "comparisons": comparisons,
        "candidates": candidates,
        "record_audit": record_audit,
        "stats": stats,
        "parameters": {
            "reporting_day": reporting_day,
            "min_incoming_reports":
                min_incoming_reports,
            "min_historical_reports":
                min_historical_reports,
            "min_active_days": min_active_days,
            "min_years": min_years,
            "seasonal_months":
                sorted(seasonal_months),
            "screening_only": True
        }
    }
