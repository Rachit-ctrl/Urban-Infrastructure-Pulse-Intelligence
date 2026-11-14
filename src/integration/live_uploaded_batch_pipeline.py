"""Process uploaded complaints using existing models.

Historical models and source files remain read-only.
Uploaded results remain in memory until exported.
"""

from pathlib import Path

import pandas as pd

from src.integration.upload_preflight import (
    run_upload_preflight
)

from src.integration.bulk_complaint_processor import (
    process_complaint_batch
)

from src.integration.daily_batch_window import (
    select_daily_complaints
)

from src.integration.anomaly_snapshot_integration import (
    load_anomaly_snapshot,
    analyze_batch_with_anomaly_context
)

from src.integration.smart_assistant_investigation_briefs import (
    build_investigation_briefs
)

from src.integration.dashboard_intelligence_feed import (
    build_dashboard_feed
)

from src.ui.service_factory import get_gui_adapter


def load_upload_baseline(project_root):

    root = Path(project_root).resolve()

    return load_anomaly_snapshot(
        root / "outputs/tables/anomaly"
        / "historical_daily_counts_2023_2025_v1.csv.gz",

        root / "outputs/tables/anomaly"
        / "historical_daily_counts_2023_2025_v1_manifest.json"
    )


def build_verified_day_feed(
    processed_records,
    baseline,
    reporting_day,
    synthetic_data=True
):
    """Analyze only geographically verified selected-day rows."""

    if not isinstance(
        processed_records,
        pd.DataFrame
    ):
        raise TypeError(
            "Backend records must be a DataFrame."
        )

    required = {
        "source_row",
        "complaint_id",
        "created_date",
        "borough",
        "incident_zip",
        "complaint_category",
        "latitude",
        "longitude",
        "processing_status",
        "zip_validation_status"
    }

    missing = required - set(
        processed_records.columns
    )

    if missing:
        raise ValueError(
            f"Backend missing columns: {sorted(missing)}"
        )

    if processed_records[
        "source_row"
    ].duplicated().any():
        raise ValueError(
            "Duplicate backend source_row values."
        )

    quality_ok = (
        processed_records[
            "processing_status"
        ].astype(str).eq("processed")
        &
        processed_records[
            "zip_validation_status"
        ].astype(str).eq("zcta_consistent")
    )

    verified = processed_records.loc[
        quality_ok
    ].copy()

    window = select_daily_complaints(
        verified,
        reporting_day
    )

    selected = window[
        "selected_records"
    ].copy()

    details = {
        "backend_records":
            int(len(processed_records)),
        "geographically_verified":
            int(len(verified)),
        "excluded_by_backend_quality":
            int((~quality_ok).sum()),
        "excluded_by_reporting_day":
            int(len(verified) - len(selected)),
        "selected_for_new_models":
            int(len(selected))
    }

    if selected.empty:
        return {
            "feed": None,
            "briefs": None,
            "integrated": None,
            "selection_stats": details,
            "status":
                "no_eligible_reporting_day_records"
        }

    integrated = analyze_batch_with_anomaly_context(
        selected,
        baseline,
        reporting_day
    )

    briefs = build_investigation_briefs(
        integrated,
        synthetic_data=synthetic_data
    )

    feed = build_dashboard_feed(
        integrated,
        briefs
    )

    feed["metadata"]["input_origin"] = (
        "uploaded_csv"
    )

    feed["metadata"]["data_verified_in_field"] = (
        False
    )

    feed["metadata"]["upload_note"] = (
        "Uploaded reports were geographically "
        "screened. No infrastructure fault "
        "has been independently verified."
    )

    if feed["summary"][
        "input_complaints"
    ] != len(selected):
        raise AssertionError(
            "Dashboard record count mismatch."
        )

    if len(
        briefs["record_audit"]
    ) != len(selected):
        raise AssertionError(
            "Investigation audit lost records."
        )

    return {
        "feed": feed,
        "briefs": briefs,
        "integrated": integrated,
        "selection_stats": details,
        "status": "analyzed"
    }


def process_uploaded_csv_bytes(
    csv_bytes,
    reporting_day,
    project_root,
    synthetic_data=True,
    adapter=None,
    baseline=None
):
    """Run validation, backend and three-model analysis.

    The backend audit retains all uploaded rows.
    The dashboard contains only verified rows
    belonging to the selected reporting day.
    """

    if baseline is None:
        baseline = load_upload_baseline(
            project_root
        )

    categories = baseline[
        "supported_categories"
    ]

    preflight = run_upload_preflight(
        csv_bytes,
        categories
    )

    original = preflight[
        "input_records"
    ].copy(deep=True)

    if adapter is None:
        adapter = get_gui_adapter()

    backend = process_complaint_batch(
        original,
        categories,
        adapter,
        use_cache=True
    )

    if (
        not isinstance(backend, dict)
        or "records" not in backend
    ):
        raise ValueError(
            "Unexpected bulk backend response."
        )

    rows = backend["records"]

    if isinstance(rows, pd.DataFrame):
        rows = rows.copy(deep=True)

    elif isinstance(rows, (list, tuple)):
        rows = pd.DataFrame(list(rows))

    else:
        raise TypeError(
            "Unexpected backend records container."
        )

    if len(rows) != len(original):
        raise ValueError(
            "Backend dropped uploaded rows."
        )

    analysis = build_verified_day_feed(
        rows,
        baseline,
        reporting_day,
        synthetic_data=synthetic_data
    )

    return {
        **analysis,
        "uploaded_rows": int(len(original)),
        "backend_records": rows,
        "backend_stats": backend.get(
            "stats", {}
        ),
        "preflight_status_counts":
            preflight["status_counts"]
    }
