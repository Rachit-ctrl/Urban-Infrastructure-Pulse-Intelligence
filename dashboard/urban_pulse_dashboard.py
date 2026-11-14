"""Read-only Urban Infrastructure Pulse dashboard.

Launch from the project root:
    python -m streamlit run dashboard/urban_pulse_dashboard.py

This prototype displays saved synthetic demonstration
results. It does not retrain or modify historical models.
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEED_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tests"
    / "phase_19_2e_dashboard_feed"
    / "dashboard_feed.json"
)


def load_feed(path=FEED_PATH):
    with Path(path).open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def signal_type(row):
    anomaly = bool(
        row.get("anomaly_candidate", False)
    )
    spatial = bool(
        row.get("spatial_candidate", False)
    )

    if anomaly and spatial:
        return "Both signals"

    if anomaly:
        return "Anomaly only"

    if spatial:
        return "Spatial only"

    return "Neither"


def filter_records(
    records,
    borough="All",
    category="All",
    pattern="All",
    signal="All"
):
    selected = []

    for record in records:

        area_id = str(
            record.get("area_id") or ""
        )

        current_borough = (
            area_id.split("_", 1)[0]
        )

        if (
            borough != "All"
            and current_borough != borough
        ):
            continue

        if (
            category != "All"
            and record.get(
                "complaint_category"
            ) != category
        ):
            continue

        if (
            pattern != "All"
            and record.get(
                "new_kmeans_pattern"
            ) != pattern
        ):
            continue

        if (
            signal != "All"
            and signal_type(record) != signal
        ):
            continue

        selected.append(record)

    return selected


def build_filtered_view(feed, records):
    """Link selected reports to saved evidence cards."""

    anomaly_keys = {
        (
            str(item["area_id"]),
            str(item["complaint_category"])
        )
        for item in records
        if item["anomaly_candidate"]
    }

    spatial_ids = {
        str(item["new_dbscan_candidate_id"])
        for item in records
        if (
            item["spatial_candidate"]
            and item.get(
                "new_dbscan_candidate_id"
            ) is not None
        )
    }

    anomaly_cards = [
        item
        for item in feed["anomaly_cards"]
        if (
            str(item["area_id"]),
            str(item["complaint_category"])
        ) in anomaly_keys
    ]

    spatial_cards = [
        item
        for item in feed["spatial_cards"]
        if str(item["candidate_id"]) in spatial_ids
    ]

    map_features = [
        feature
        for feature in feed[
            "map_geojson"
        ]["features"]
        if str(feature["id"]) in spatial_ids
    ]

    both = sum(
        signal_type(item) == "Both signals"
        for item in records
    )

    anomaly_only = sum(
        signal_type(item) == "Anomaly only"
        for item in records
    )

    spatial_only = sum(
        signal_type(item) == "Spatial only"
        for item in records
    )

    neither = sum(
        signal_type(item) == "Neither"
        for item in records
    )

    summary = {
        "selected": len(records),
        "both": both,
        "anomaly_only": anomaly_only,
        "spatial_only": spatial_only,
        "neither": neither,
        "anomaly_reports":
            both + anomaly_only,
        "spatial_reports":
            both + spatial_only,
        "distinct_either":
            both + anomaly_only + spatial_only,
        "anomaly_groups":
            len(anomaly_cards),
        "spatial_groups":
            len(spatial_cards),
        "areas": len({
            str(item["area_id"])
            for item in records
        })
    }

    if sum([
        both,
        anomaly_only,
        spatial_only,
        neither
    ]) != len(records):
        raise ValueError(
            "Selected complaint counts disagree."
        )

    return {
        "records": records,
        "anomaly_cards": anomaly_cards,
        "spatial_cards": spatial_cards,
        "map_features": map_features,
        "summary": summary
    }


def validate_feed(feed):
    """Reject inconsistent dashboard source data."""

    required = {
        "metadata",
        "summary",
        "filters",
        "area_cards",
        "anomaly_cards",
        "spatial_cards",
        "records",
        "map_geojson"
    }

    if not required.issubset(feed):
        raise ValueError(
            "Dashboard feed is missing required sections."
        )

    records = feed["records"]
    summary = feed["summary"]

    if len(records) != summary[
        "input_complaints"
    ]:
        raise ValueError(
            "Input record count mismatch."
        )

    identities = [
        str(item["source_row"])
        for item in records
    ]

    if len(identities) != len(
        set(identities)
    ):
        raise ValueError(
            "Duplicate complaint source rows."
        )

    view = build_filtered_view(
        feed,
        records
    )

    counts = view["summary"]

    expected = {
        "anomaly_candidate_complaints":
            counts["anomaly_reports"],
        "spatial_candidate_complaints":
            counts["spatial_reports"],
        "complaints_in_both_signals":
            counts["both"],
        "complaints_anomaly_only":
            counts["anomaly_only"],
        "complaints_spatial_only":
            counts["spatial_only"],
        "complaints_in_neither_signal":
            counts["neither"],
        "distinct_complaints_in_either_signal":
            counts["distinct_either"]
    }

    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(
                f"Dashboard count mismatch: {key}"
            )

    if counts["anomaly_groups"] != summary[
        "anomaly_candidate_groups"
    ]:
        raise ValueError(
            "Anomaly group count mismatch."
        )

    if counts["spatial_groups"] != summary[
        "spatial_candidate_groups"
    ]:
        raise ValueError(
            "Spatial group count mismatch."
        )

    return view


def main():
    import streamlit as st

    st.set_page_config(
        page_title="Urban Infrastructure Pulse",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Urban Infrastructure Pulse")
    st.caption(
        "Smart infrastructure reporting intelligence"
    )

    if not FEED_PATH.is_file():
        st.error(
            "The validated dashboard feed was not found."
        )
        st.code(str(FEED_PATH))
        st.stop()

    try:
        feed = load_feed()
        validate_feed(feed)
    except (OSError, ValueError, KeyError) as exc:
        st.error(
            f"Invalid dashboard feed: {exc}"
        )
        st.stop()

    if feed["metadata"].get(
        "synthetic_data", False
    ):
        st.warning(
            "Synthetic demonstration data — "
            "not live NYC 311 incidents or "
            "verified infrastructure problems."
        )

    st.caption(
        "Reporting day: "
        + str(feed["metadata"]["reporting_day"])
        + " | Historical comparison: 2023–2025"
    )

    # ----------------------------------------------
    # Interactive filters
    # ----------------------------------------------

    st.sidebar.header("Explore complaints")

    borough = st.sidebar.selectbox(
        "Borough",
        ["All"] + feed["filters"]["boroughs"]
    )

    category = st.sidebar.selectbox(
        "Complaint category",
        ["All"] + feed["filters"]["categories"]
    )

    pattern = st.sidebar.selectbox(
        "K-Means reporting pattern",
        ["All"] + feed["filters"][
            "kmeans_patterns"
        ]
    )

    signal = st.sidebar.selectbox(
        "Investigation signal",
        [
            "All",
            "Both signals",
            "Anomaly only",
            "Spatial only",
            "Neither"
        ]
    )

    st.sidebar.caption(
        "Filters change the displayed report subset. "
        "Historical maxima and investigation "
        "explanations refer to the complete "
        "original reporting-day batch."
    )

    records = filter_records(
        feed["records"],
        borough=borough,
        category=category,
        pattern=pattern,
        signal=signal
    )

    view = build_filtered_view(
        feed, records
    )

    stats = view["summary"]

    # ----------------------------------------------
    # Overview metrics
    # ----------------------------------------------

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Selected reports",
        stats["selected"]
    )

    m2.metric(
        "Anomaly-associated",
        stats["anomaly_reports"]
    )

    m3.metric(
        "Spatial-associated",
        stats["spatial_reports"]
    )

    m4.metric(
        "Distinct with either",
        stats["distinct_either"]
    )

    m5.metric(
        "Areas represented",
        stats["areas"]
    )

    st.caption(
        "Anomaly and spatial totals overlap. "
        "'Distinct with either' counts each "
        "complaint once."
    )

    tabs = st.tabs([
        "Overview",
        "K-Means Areas",
        "Anomaly Reviews",
        "DBSCAN Map",
        "Complaint Records"
    ])

    # ----------------------------------------------
    # Tab 1: Overview
    # ----------------------------------------------

    with tabs[0]:

        st.subheader("Reporting overview")

        summary_frame = pd.DataFrame([
            {
                "Signal classification":
                    "Both signals",
                "Complaints": stats["both"]
            },
            {
                "Signal classification":
                    "Anomaly only",
                "Complaints":
                    stats["anomaly_only"]
            },
            {
                "Signal classification":
                    "Spatial only",
                "Complaints":
                    stats["spatial_only"]
            },
            {
                "Signal classification":
                    "Neither",
                "Complaints":
                    stats["neither"]
            }
        ])

        left, right = st.columns(2)

        with left:
            st.markdown(
                "**Distinct complaint accounting**"
            )
            st.dataframe(
                summary_frame,
                hide_index=True,
                use_container_width=True
            )

        with right:
            st.markdown(
                "**Complaints by category**"
            )

            if records:
                frame = pd.DataFrame(records)
                st.bar_chart(
                    frame[
                        "complaint_category"
                    ].value_counts()
                )
            else:
                st.info(
                    "No complaints match these filters."
                )

        if records:
            st.markdown(
                "**Complaints by borough–ZIP area**"
            )

            frame = pd.DataFrame(records)

            st.bar_chart(
                frame["area_id"]
                .value_counts()
                .sort_index()
            )

    # ----------------------------------------------
    # Tab 2: K-Means area patterns
    # ----------------------------------------------

    with tabs[1]:

        st.subheader(
            "K-Means area reporting patterns"
        )

        st.caption(
            "Patterns describe relative complaint "
            "category mixtures, not infrastructure "
            "condition or repair urgency."
        )

        if records:
            frame = pd.DataFrame(records)

            area_table = []

            for area_id, group in frame.groupby(
                "area_id", sort=True
            ):

                most_common = (
                    group["complaint_category"]
                    .value_counts()
                    .index[0]
                )

                patterns = sorted({
                    str(value)
                    for value in group[
                        "new_kmeans_pattern"
                    ].dropna()
                })

                area_table.append({
                    "Area": area_id,
                    "Selected reports":
                        len(group),
                    "K-Means pattern":
                        " | ".join(patterns),
                    "Most reported category":
                        most_common,
                    "Anomaly-associated":
                        int(group[
                            "anomaly_candidate"
                        ].sum()),
                    "Spatial-associated":
                        int(group[
                            "spatial_candidate"
                        ].sum())
                })

            st.dataframe(
                pd.DataFrame(area_table),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info(
                "No area patterns match the filters."
            )

    # ----------------------------------------------
    # Tab 3: Anomaly screening evidence
    # ----------------------------------------------

    with tabs[2]:

        st.subheader(
            "Historical reporting-increase reviews"
        )

        st.caption(
            "These are screening candidates, "
            "not statistically confirmed anomalies."
        )

        if not view["anomaly_cards"]:
            st.info(
                "No anomaly candidate groups match "
                "the selected filters."
            )

        for card in view["anomaly_cards"]:

            matched = sum(
                record["anomaly_candidate"]
                and record["area_id"]
                    == card["area_id"]
                and record["complaint_category"]
                    == card["complaint_category"]
                for record in records
            )

            heading = (
                f"{card['area_id']} — "
                f"{card['complaint_category']} "
                f"({matched} selected reports)"
            )

            with st.expander(heading):

                st.write(
                    card["explanation"]
                )

                st.markdown(
                    "**Historical comparison**"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Full-batch reports",
                    card["incoming_complaints"]
                )

                c2.metric(
                    "Seasonal historical max",
                    card[
                        "seasonal_historical_max"
                    ]
                )

                c3.metric(
                    "All-Monday historical max",
                    card[
                        "all_weekday_historical_max"
                    ]
                )

                st.markdown(
                    "**Suggested verification**"
                )

                st.write(
                    card["suggested_review"]
                )

    # ----------------------------------------------
    # Tab 4: DBSCAN spatial concentrations
    # ----------------------------------------------

    with tabs[3]:

        st.subheader(
            "New-batch spatial concentrations"
        )

        st.caption(
            "Pins represent submitted complaint "
            "coordinates. They are not verified "
            "infrastructure fault locations."
        )

        if not view["map_features"]:
            st.info(
                "No spatial concentrations match "
                "the current filters."
            )

        else:
            points = []

            for feature in view[
                "map_features"
            ]:

                longitude, latitude = (
                    feature["geometry"][
                        "coordinates"
                    ]
                )

                points.append({
                    "latitude": latitude,
                    "longitude": longitude,
                    "candidate_id": feature["id"]
                })

            points_frame = pd.DataFrame(
                points
            )

            st.map(
                points_frame[[
                    "latitude",
                    "longitude"
                ]]
            )

            st.caption(
                "One representative point per "
                "spatial candidate; not every "
                "individual complaint location."
            )

        for card in view["spatial_cards"]:

            matched = sum(
                record["spatial_candidate"]
                and record[
                    "new_dbscan_candidate_id"
                ] == card["candidate_id"]
                for record in records
            )

            with st.expander(
                f"{card['candidate_id']} — "
                f"{card['complaint_category']} "
                f"({matched} selected reports)"
            ):

                st.write(
                    card["explanation"]
                )

                st.write(
                    "**Full-batch spatial reports:**",
                    card["incoming_complaints"]
                )

                st.write(
                    "**Distinct submitted locations:**",
                    card[
                        "distinct_submitted_locations"
                    ]
                )

                st.write(
                    "**Suggested verification:** "
                    + card["suggested_review"]
                )

    # ----------------------------------------------
    # Tab 5: Record-level evidence
    # ----------------------------------------------

    with tabs[4]:

        st.subheader(
            "Complaint-level evidence"
        )

        st.caption(
            "This table retains both new-batch "
            "signals and historical comparison "
            "statuses for traceability."
        )

        if not records:
            st.info(
                "No complaint records match the filters."
            )

        else:
            frame = pd.DataFrame(
                records
            )

            st.dataframe(
                frame,
                hide_index=True,
                use_container_width=True
            )

            st.download_button(
                "Export filtered complaints as CSV",
                data=frame.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "urban_pulse_filtered_complaints.csv"
                ),
                mime="text/csv"
            )

    st.divider()

    st.caption(
        "Urban Infrastructure Pulse Intelligence "
        "| Synthetic-data pilot | "
        "Human verification required before "
        "operational action."
    )


if __name__ == "__main__":
    main()
