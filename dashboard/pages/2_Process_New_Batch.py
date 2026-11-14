"""Process uploaded complaints with existing backend."""

from datetime import date
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integration.live_uploaded_batch_pipeline import (
    process_uploaded_csv_bytes,
    load_upload_baseline
)

from src.ui.service_factory import (
    get_gui_adapter
)

from dashboard.urban_pulse_dashboard import (
    validate_feed,
    filter_records,
    build_filtered_view
)


st.set_page_config(
    page_title="Process New Batch | Urban Pulse",
    page_icon="🏙️",
    layout="wide"
)

st.title("Process New Complaint Batch")

st.caption(
    "Upload CSV → Geographic backend → "
    "K-Means + DBSCAN + Anomaly screening"
)

st.warning(
    "Reporting evidence only. Results do not "
    "confirm infrastructure defects or "
    "determine repair priority."
)


@st.cache_resource(
    show_spinner="Loading historical services..."
)
def existing_adapter():
    return get_gui_adapter()


@st.cache_resource(
    show_spinner="Loading historical daily baseline..."
)
def existing_baseline():
    return load_upload_baseline(ROOT)


uploaded_file = st.file_uploader(
    "Choose complaint CSV",
    type=["csv"],
    key="batch_csv"
)

reporting_day = st.date_input(
    "NYC reporting day",
    value=date.today(),
    key="batch_day"
)

synthetic = st.checkbox(
    "Synthetic / test data",
    value=True,
    help=(
        "Leave selected for demonstrations. "
        "Uploaded real-world reports are not "
        "independently verified."
    )
)

st.caption(
    "Maximum upload: 10 MB and 5,000 rows. "
    "Historical models remain read-only."
)

upload_key = None

if uploaded_file is not None:

    payload = uploaded_file.getvalue()

    upload_key = (
        hashlib.sha256(payload).hexdigest(),
        reporting_day.isoformat(),
        bool(synthetic)
    )

if st.button(
    "Process Uploaded Complaints",
    type="primary",
    disabled=uploaded_file is None
):

    try:

        with st.spinner(
            "Validating, geographically verifying "
            "and analyzing complaints..."
        ):

            result = process_uploaded_csv_bytes(
                payload,
                reporting_day.isoformat(),
                ROOT,
                synthetic_data=synthetic,
                adapter=existing_adapter(),
                baseline=existing_baseline()
            )

        st.session_state[
            "urban_pulse_live_result"
        ] = result

        st.session_state[
            "urban_pulse_live_key"
        ] = upload_key

    except Exception as exc:

        st.session_state.pop(
            "urban_pulse_live_result",
            None
        )

        st.session_state.pop(
            "urban_pulse_live_key",
            None
        )

        st.error(
            f"Batch processing failed: "
            f"{type(exc).__name__}: {exc}"
        )


result = st.session_state.get(
    "urban_pulse_live_result"
)

if (
    result is None
    or st.session_state.get(
        "urban_pulse_live_key"
    ) != upload_key
):

    st.info(
        "Upload a CSV and press "
        "Process Uploaded Complaints."
    )

    st.stop()


st.success(
    "Backend processing completed."
)

st.subheader("Processing audit")

metrics = result["selection_stats"]

columns = st.columns(4)

columns[0].metric(
    "Uploaded",
    result["uploaded_rows"]
)

columns[1].metric(
    "Geographically verified",
    metrics["geographically_verified"]
)

columns[2].metric(
    "Selected reporting day",
    metrics["selected_for_new_models"]
)

columns[3].metric(
    "Excluded",
    (
        metrics["excluded_by_backend_quality"]
        + metrics["excluded_by_reporting_day"]
    )
)

st.caption(
    "All uploaded rows remain available in the "
    "backend audit. Only geographically verified "
    "reports from the selected NYC reporting day "
    "enter the new-batch models."
)

with st.expander(
    "Backend audit — including rejected records"
):

    st.dataframe(
        result["backend_records"],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download backend audit CSV",
        result["backend_records"]
        .to_csv(index=False)
        .encode("utf-8"),
        "uploaded_batch_backend_audit.csv",
        "text/csv"
    )


feed = result["feed"]

if feed is None:

    st.info(
        "No geographically verified complaints "
        "matched the selected reporting day. "
        "No model conclusions were generated."
    )

    st.stop()


try:
    validate_feed(feed)

except (ValueError, KeyError, TypeError) as exc:

    st.error(
        f"Dashboard feed validation failed: {exc}"
    )

    st.stop()


st.subheader("Verified-day intelligence")

filters = feed["filters"]

f1, f2, f3 = st.columns(3)

borough = f1.selectbox(
    "Borough",
    ["All"] + filters["boroughs"]
)

category = f2.selectbox(
    "Category",
    ["All"] + filters["categories"]
)

signal = f3.selectbox(
    "Signal",
    [
        "All",
        "Both signals",
        "Anomaly only",
        "Spatial only",
        "Neither"
    ]
)

records = filter_records(
    feed["records"],
    borough=borough,
    category=category,
    signal=signal
)

view = build_filtered_view(
    feed,
    records
)

summary = view["summary"]

a, b, c, d = st.columns(4)

a.metric(
    "Filtered reports",
    summary["selected"]
)

b.metric(
    "Anomaly-associated",
    summary["anomaly_reports"]
)

c.metric(
    "Spatial-associated",
    summary["spatial_reports"]
)

d.metric(
    "Distinct with either",
    summary["distinct_either"]
)

st.caption(
    "Anomaly and spatial totals overlap. "
    "The explanations and area cards refer "
    "to the complete processed reporting-day batch."
)

tabs = st.tabs([
    "Area patterns",
    "Anomaly reviews",
    "Spatial map",
    "Records"
])

with tabs[0]:

    areas = pd.DataFrame(
        feed["area_cards"]
    )

    selected_areas = {
        row["area_id"]
        for row in records
    }

    if not areas.empty:

        st.dataframe(
            areas.loc[
                areas["area_id"].isin(
                    selected_areas
                )
            ],
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info(
            "No area reporting patterns available."
        )


with tabs[1]:

    if not view["anomaly_cards"]:
        st.info(
            "No anomaly candidates match the filters."
        )

    for card in view["anomaly_cards"]:

        with st.expander(
            f"{card['area_id']} — "
            f"{card['complaint_category']}"
        ):

            st.write(
                card["explanation"]
            )

            st.caption(
                card["suggested_review"]
            )


with tabs[2]:

    points = []

    for feature in view["map_features"]:

        longitude, latitude = (
            feature["geometry"]["coordinates"]
        )

        points.append({
            "latitude": latitude,
            "longitude": longitude,
            "candidate_id": feature["id"]
        })

    if points:

        st.map(
            pd.DataFrame(points)[[
                "latitude",
                "longitude"
            ]]
        )

    else:

        st.info(
            "No spatial candidate locations "
            "match the current filters."
        )

    for card in view["spatial_cards"]:

        with st.expander(
            f"{card['candidate_id']} — "
            f"{card['complaint_category']}"
        ):

            st.write(
                card["explanation"]
            )

            st.caption(
                card["suggested_review"]
            )


with tabs[3]:

    if records:

        st.dataframe(
            pd.DataFrame(records),
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info(
            "No complaint records match the filters."
        )


feed_bytes = json.dumps(
    feed,
    indent=2,
    ensure_ascii=False,
    allow_nan=False
).encode("utf-8")

st.download_button(
    "Download dashboard feed JSON",
    feed_bytes,
    "uploaded_batch_dashboard_feed.json",
    "application/json"
)

st.caption(
    "Uploaded input and generated outputs "
    "remain in this Streamlit session unless "
    "explicitly downloaded. Historical models "
    "are not modified."
)
