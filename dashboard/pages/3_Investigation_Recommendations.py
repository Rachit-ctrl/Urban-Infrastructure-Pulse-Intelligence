"""Human-readable investigation recommendations page."""

from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integration.investigation_review_queue import (
    build_review_queue
)


DEMO_FEED = (
    ROOT
    / "outputs/tests/phase_19_2e_dashboard_feed"
    / "dashboard_feed.json"
)


st.set_page_config(
    page_title="Investigation Recommendations",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Where Should We Investigate First?")

st.caption(
    "Urban Infrastructure Pulse — "
    "explainable investigation recommendations"
)

live = st.session_state.get(
    "urban_pulse_live_result"
)

if (
    isinstance(live, dict)
    and isinstance(live.get("feed"), dict)
):

    feed = live["feed"]

    st.info(
        "Showing your most recently processed "
        "uploaded batch from this browser session."
    )

else:

    if not DEMO_FEED.is_file():

        st.error(
            "No completed upload or demonstration "
            "feed is available."
        )

        st.stop()

    with DEMO_FEED.open(
        encoding="utf-8"
    ) as file:

        feed = json.load(file)

    st.info(
        "Showing the saved 500-complaint "
        "synthetic demonstration. Process a "
        "new batch to view uploaded results."
    )


try:

    result = build_review_queue(
        feed
    )

except (ValueError, KeyError, TypeError) as exc:

    st.error(
        f"Could not build recommendations: {exc}"
    )

    st.stop()


if result["metadata"]["synthetic_data"]:

    st.warning(
        "Synthetic demonstration data. "
        "These are test recommendations, "
        "not confirmed NYC incidents."
    )

else:

    st.warning(
        "Uploaded complaint data is unverified "
        "in the field. Screening results are "
        "not confirmed faults or repair priorities."
    )


st.markdown(
    "**What this page answers:** "
    "Which places have noteworthy reporting "
    "evidence, why should they be reviewed, "
    "and what should investigators check?"
)

st.caption(
    "Review order is not a measure of danger. "
    "It considers historical reporting increases "
    "and new spatial concentrations; emergency "
    "severity requires independent assessment."
)

summary = result["summary"]

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Analyzed reports",
    summary["reports"]
)

m2.metric(
    "Review first groups",
    summary["review_first_groups"]
)

m3.metric(
    "Spatial-only reviews",
    summary["spatial_only_groups"]
)

m4.metric(
    "Distinct reports flagged",
    summary["distinct_with_signals"]
)


st.divider()

# Filters only change displayed cards;
# rankings remain based on the complete batch.

queue = result["queue"]

boroughs = sorted({
    str(area).rsplit("_", 1)[0]
    for item in queue
    for area in item["area_ids"]
})

categories = sorted({
    item["category"]
    for item in queue
})

f1, f2 = st.columns(2)

borough = f1.selectbox(
    "Filter borough",
    ["All"] + boroughs,
    key="recommendation_borough"
)

category = f2.selectbox(
    "Filter complaint category",
    ["All"] + categories,
    key="recommendation_category"
)


def is_visible(item):

    if (
        category != "All"
        and item["category"] != category
    ):
        return False

    if borough != "All":

        if not any(
            area.startswith(
                borough + "_"
            )
            for area in item["area_ids"]
        ):
            return False

    return True


visible = [
    item
    for item in queue
    if is_visible(item)
]


def show_recommendation(item):

    with st.container(border=True):

        st.markdown(
            f"### {item['review_order']}. "
            f"{item['location']}"
        )

        st.markdown(
            f"**Complaint type:** {item['category']}"
        )

        st.caption(
            item["review_label"]
        )

        if item["kmeans_patterns"]:

            st.write(
                "**Area reporting pattern:**",
                ", ".join(
                    item["kmeans_patterns"]
                )
            )

        st.markdown("**Why investigate here?**")

        st.write(
            item["why"]
        )

        st.markdown("**Suggested action**")

        st.write(
            item["suggested_action"]
        )

        st.caption(
            "Evidence limitation: "
            + item["evidence_note"]
        )


tabs = st.tabs([
    "Investigate first",
    "Other review findings",
    "Spatial map",
    "Area overview"
])


with tabs[0]:

    st.subheader(
        "Places to review first"
    )

    st.write(
        "These areas show both an elevated "
        "historical reporting-count signal "
        "and new spatial concentration evidence."
    )

    first_items = [
        item for item in visible
        if item["tier"] == 1
    ]

    if not first_items:

        st.info(
            "No two-signal reviews match "
            "the selected filters."
        )

    else:

        for item in first_items[:3]:

            show_recommendation(item)

        remaining = first_items[3:]

        if remaining:

            with st.expander(
                f"View {len(remaining)} "
                "more investigation recommendations"
            ):

                for item in remaining:

                    show_recommendation(item)


with tabs[1]:

    st.subheader(
        "Other findings worth reviewing"
    )

    st.write(
        "These findings are based on a "
        "historical reporting increase alone "
        "or a spatial concentration without "
        "a linked historical-count anomaly."
    )

    other_items = [
        item for item in visible
        if item["tier"] != 1
    ]

    if not other_items:

        st.info(
            "No additional findings match "
            "the selected filters."
        )

    for item in other_items:

        show_recommendation(item)


with tabs[2]:

    st.subheader(
        "New spatial concentration locations"
    )

    st.caption(
        "Map pins are representative submitted "
        "complaint coordinates, not verified "
        "fault sites or intersections."
    )

    matching_spatial_ids = {
        str(item["candidate_id"])
        for item in result["spatial_cards"]

        if (
            category == "All"
            or item["complaint_category"]
                == category
        )

        and (
            borough == "All"
            or any(
                str(area).startswith(
                    borough + "_"
                )
                for area in item["area_ids"]
            )
        )
    }

    points = []

    for feature in result[
        "map_geojson"
    ]["features"]:

        if str(feature["id"]) not in (
            matching_spatial_ids
        ):
            continue

        lon, lat = (
            feature["geometry"][
                "coordinates"
            ]
        )

        points.append({
            "latitude": lat,
            "longitude": lon,
            "candidate_id": feature["id"]
        })

    if points:

        st.map(
            pd.DataFrame(points)[[
                "latitude",
                "longitude"
            ]]
        )

        st.dataframe(
            pd.DataFrame(points),
            hide_index=True,
            use_container_width=True
        )

    else:

        st.info(
            "No candidate map locations "
            "match these filters."
        )


with tabs[3]:

    st.subheader(
        "Area reporting patterns"
    )

    st.write(
        "The following is a simplified "
        "K-Means and model-evidence overview."
    )

    frame = pd.DataFrame(
        result["area_table"]
    )

    if borough != "All":

        frame = frame.loc[
            frame["area_id"].str.startswith(
                borough + "_"
            )
        ]

    if category != "All":

        frame = frame.loc[
            frame["dominant_category"]
            == category
        ]

    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True
    )


st.divider()

st.subheader("Export investigation recommendations")

export_rows = [
    {
        "review_order": item["review_order"],
        "review_label": item["review_label"],
        "location": item["location"],
        "complaint_category": item["category"],
        "incoming_reports": item["incoming_reports"],
        "why_investigate": item["why"],
        "suggested_action": item["suggested_action"],
        "evidence_note": item["evidence_note"]
    }
    for item in visible
]

if export_rows:

    export_csv = pd.DataFrame(
        export_rows
    ).to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Investigation Recommendations CSV",
        data=export_csv,
        file_name=(
            "urban_pulse_investigation_"
            "recommendations.csv"
        ),
        mime="text/csv"
    )


st.caption(
    "Suggestions are for human verification. "
    "Do not dispatch repairs or claim confirmed "
    "faults based solely on these screens."
)
