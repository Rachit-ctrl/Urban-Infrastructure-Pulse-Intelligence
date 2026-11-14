from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.recommendations.hotspot_recommender import (
    HotspotRecommendationEngine
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HOTSPOT_PATH = (
    PROJECT_ROOT / "outputs/tables/dbscan"
    / "dbscan_primary_f_gui_hotspots.csv"
)

RECOMMENDATION_DIR = (
    PROJECT_ROOT / "outputs/tables/dbscan"
)

BOROUGH_COLORS = {
    "BRONX": [239, 158, 72, 195],
    "BROOKLYN": [49, 185, 164, 195],
    "MANHATTAN": [239, 92, 104, 195],
    "QUEENS": [166, 126, 228, 195],
    "STATEN ISLAND": [72, 158, 231, 195],
}


@st.cache_data(show_spinner=False)
def load_hotspot_map_data(path):

    data = pd.read_csv(path)

    required = {
        "dbscan_cluster_id",
        "primary_borough",
        "center_latitude",
        "center_longitude",
        "total_complaints",
        "dominant_category",
        "dominant_category_pct",
        "recurring_reporting_pct",
        "complaints_2023",
        "complaints_2024",
        "complaints_2025",
        "active_months",
        "yoy_change_pct",
        "trend_interpretation_caution",
        "cross_borough",
        "model_version",
        "data_start_year",
        "data_end_year",
    }

    if not required.issubset(data.columns):
        raise ValueError(
            "Required DBSCAN map columns are missing: "
            + str(sorted(required - set(data.columns)))
        )

    if len(data) != 2846:
        raise ValueError("Unexpected DBSCAN hotspot count.")

    if data["dbscan_cluster_id"].duplicated().any():
        raise ValueError("Duplicate hotspot IDs.")

    coordinates = data[
        ["center_latitude", "center_longitude"]
    ].to_numpy(dtype=float)

    if not np.isfinite(coordinates).all():
        raise ValueError("Invalid hotspot coordinates.")

    if (
        (data["center_latitude"].abs() > 90).any()
        or (data["center_longitude"].abs() > 180).any()
    ):
        raise ValueError("Coordinates outside valid ranges.")

    if (data["total_complaints"] <= 0).any():
        raise ValueError("Invalid hotspot complaint counts.")

    if int(data["total_complaints"].sum()) != 317936:
        raise ValueError(
            "Historical clustered complaint total differs."
        )

    if set(data["model_version"]) != {
        "dbscan_f_75m_min40_v1"
    }:
        raise ValueError("Unexpected DBSCAN model version.")

    if (
        not data["data_start_year"].eq(2023).all()
        or not data["data_end_year"].eq(2025).all()
    ):
        raise ValueError("Unexpected historical period.")

    return data.sort_values(
        ["total_complaints", "dbscan_cluster_id"],
        ascending=[False, True]
    ).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_hotspot_recommender(data_directory):

    return HotspotRecommendationEngine(
        Path(data_directory)
    )


def render_hotspot_explorer():

    st.divider()
    st.header("Interactive NYC DBSCAN Hotspot Explorer")

    st.caption(
        "Explore 2,846 historical spatial reporting "
        "clusters derived from NYC 311 complaints "
        "during 2023–2025."
    )

    hotspots = load_hotspot_map_data(
        str(HOTSPOT_PATH)
    )

    # ----------------------------------------------
    # 2. Borough, category and volume filters
    # ----------------------------------------------

    borough_options = sorted(
        hotspots["primary_borough"]
        .dropna()
        .unique()
        .tolist()
    )

    category_options = sorted(
        hotspots["dominant_category"]
        .dropna()
        .unique()
        .tolist()
    )

    st.markdown("#### Map Filters")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        chosen_boroughs = st.multiselect(
            "Primary borough",
            options=borough_options,
            default=borough_options,
            key="hotspot_map_boroughs"
        )

    with filter_col2:
        chosen_category = st.selectbox(
            "Dominant complaint category",
            options=["All categories"] + category_options,
            key="hotspot_map_category"
        )

    max_complaints = int(
        hotspots["total_complaints"].max()
    )

    min_complaints = st.slider(
        "Minimum historical complaints per hotspot",
        min_value=0,
        max_value=max_complaints,
        value=0,
        step=25,
        key="hotspot_map_min_complaints"
    )

    cross_only = st.checkbox(
        "Show only cross-borough hotspots",
        value=False,
        key="hotspot_map_cross_only"
    )

    # ----------------------------------------------
    # 3. Apply filters
    # ----------------------------------------------

    filtered = hotspots.loc[
        hotspots["primary_borough"].isin(
            chosen_boroughs
        )
        & (
            hotspots["total_complaints"]
            >= min_complaints
        )
    ].copy()

    if chosen_category != "All categories":
        filtered = filtered.loc[
            filtered["dominant_category"]
            == chosen_category
        ].copy()

    if cross_only:
        filtered = filtered.loc[
            filtered["cross_borough"] == True
        ].copy()

    # ----------------------------------------------
    # 4. Filtered map metrics
    # ----------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Visible Hotspots",
        f"{len(filtered):,}"
    )

    metric2.metric(
        "Complaints in Visible Hotspots",
        f"{int(filtered['total_complaints'].sum()):,}"
    )

    metric3.metric(
        "Boroughs Represented",
        str(filtered["primary_borough"].nunique())
    )

    if filtered.empty:
        st.info(
            "No historical hotspots match these filters. "
            "Adjust the filters to display locations."
        )
        return

    # ----------------------------------------------
    # 5. Prepare map markers
    # ----------------------------------------------

    map_data = filtered.copy()

    # Visual scaling only, not DBSCAN cluster extent.
    map_data["marker_radius_m"] = (
        25
        + 3 * np.sqrt(
            map_data["total_complaints"]
        )
    ).clip(upper=130)

    map_data["marker_color"] = (
        map_data["primary_borough"]
        .apply(
            lambda name: BOROUGH_COLORS.get(
                name, [160, 166, 175, 195]
            )
        )
    )

    map_data["popup_complaints"] = (
        map_data["total_complaints"]
        .map(lambda x: f"{int(x):,}")
    )

    map_data["popup_recurrence"] = (
        map_data["recurring_reporting_pct"]
        .map(lambda x: f"{float(x):.1f}%")
    )

    # ----------------------------------------------
    # 6. Hotspot selector
    # ----------------------------------------------

    cluster_ids = (
        map_data["dbscan_cluster_id"]
        .astype(int)
        .tolist()
    )

    labels = {
        int(row.dbscan_cluster_id): (
            f"Cluster {int(row.dbscan_cluster_id)}"
            f" | {row.primary_borough}"
            f" | {int(row.total_complaints):,} reports"
        )
        for row in map_data.itertuples(index=False)
    }

    # Prefer a complaint's associated hotspot
    # when that cluster passes the current filters.
    recent_complaint = st.session_state.get(
        "historical_complaint_response"
    ) or {}

    recent_cluster = (
        recent_complaint.get("dbscan") or {}
    ).get("cluster_id")

    preferred_cluster = (
        recent_cluster
        if recent_cluster in cluster_ids
        else (
            1421
            if 1421 in cluster_ids
            else cluster_ids[0]
        )
    )

    selected_cluster = st.selectbox(
        "Select a hotspot for detailed inspection",
        options=cluster_ids,
        index=cluster_ids.index(preferred_cluster),
        format_func=lambda cluster_id: labels[cluster_id],
        key="hotspot_map_selected_cluster"
    )

    selected_row = map_data.loc[
        map_data["dbscan_cluster_id"]
        == selected_cluster
    ].iloc[0]

    # ----------------------------------------------
    # 7. Create interactive PyDeck layers
    # ----------------------------------------------

    hotspot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        id="historical_dbscan_hotspots",
        pickable=True,
        auto_highlight=True,
        stroked=True,
        filled=True,
        opacity=0.75,
        radius_units="meters",
        radius_min_pixels=3,
        radius_max_pixels=17,
        line_width_min_pixels=1,
        get_position=(
            "[center_longitude, center_latitude]"
        ),
        get_radius="marker_radius_m",
        get_fill_color="marker_color",
        get_line_color=[18, 25, 38, 200]
    )

    # Highlight the currently selected centroid.
    selected_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data.loc[
            map_data["dbscan_cluster_id"]
            == selected_cluster
        ],
        id="selected_hotspot_highlight",
        pickable=False,
        stroked=True,
        filled=False,
        radius_units="meters",
        radius_min_pixels=10,
        radius_max_pixels=26,
        line_width_min_pixels=3,
        get_position=(
            "[center_longitude, center_latitude]"
        ),
        get_radius=180,
        get_line_color=[255, 223, 79, 255]
    )

    viewport = pdk.ViewState(
        latitude=float(selected_row["center_latitude"]),
        longitude=float(selected_row["center_longitude"]),
        zoom=(
            10.2
            if len(map_data) > 500
            else 11.6
        ),
        pitch=0,
        bearing=0
    )

    tooltip = {
        "html": (
            "<b>Historical Hotspot {dbscan_cluster_id}</b>"
            "<br/>Borough: {primary_borough}"
            "<br/>Complaints: {popup_complaints}"
            "<br/>Dominant category: {dominant_category}"
            "<br/>Recurring reporting: {popup_recurrence}"
        ),
        "style": {
            "backgroundColor": "#172435",
            "color": "#ffffff",
            "fontSize": "12px"
        }
    }

    deck = pdk.Deck(
        layers=[hotspot_layer, selected_layer],
        initial_view_state=viewport,
        tooltip=tooltip
    )

    st.pydeck_chart(
        deck,
        use_container_width=True
    )

    st.caption(
        "Each marker represents a historical DBSCAN "
        "cluster centroid, not its spatial footprint. "
        "Marker size represents historical reporting "
        "volume; colors distinguish primary boroughs. "
        "Hover over markers for details, or use the "
        "selector above to inspect a cluster."
    )

    # ----------------------------------------------
    # 8. Selected hotspot intelligence
    # ----------------------------------------------

    st.markdown("#### Selected Historical Hotspot")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Cluster",
        str(selected_cluster)
    )

    col2.metric(
        "Historical Complaints",
        f"{int(selected_row['total_complaints']):,}"
    )

    col3.metric(
        "Recurring Reporting",
        f"{float(selected_row['recurring_reporting_pct']):.1f}%"
    )

    col4.metric(
        "Active Months",
        str(int(selected_row["active_months"]))
    )

    st.write(
        "**Primary borough:**",
        selected_row["primary_borough"]
    )

    st.write(
        "**Dominant complaint category:**",
        selected_row["dominant_category"]
    )

    st.write(
        "**Dominant category share:**",
        f"{float(selected_row['dominant_category_pct']):.2f}%"
    )

    # ----------------------------------------------
    # 9. Historical yearly reporting
    # ----------------------------------------------

    yearly = pd.DataFrame({
        "Year": ["2023", "2024", "2025"],
        "Complaint Reports": [
            int(selected_row["complaints_2023"]),
            int(selected_row["complaints_2024"]),
            int(selected_row["complaints_2025"])
        ]
    }).set_index("Year")

    st.markdown("#### Reporting by Year")

    st.bar_chart(yearly)

    if bool(selected_row["trend_interpretation_caution"]):
        st.warning(
            "Interpret year-over-year changes with "
            "caution because the historical baseline "
            "is small or otherwise unsuitable."
        )

    if pd.isna(selected_row["yoy_change_pct"]):
        st.caption(
            "Year-over-year percentage change "
            "is undefined for this hotspot."
        )

    # ----------------------------------------------
    # === PHASE 15.5.4: REFINED MAP RECOMMENDATIONS ===

    from src.ui.recommendation_panel import (
        render_recommendation_panel
    )

    recommender = load_hotspot_recommender(
        str(RECOMMENDATION_DIR)
    )

    report = recommender.get_hotspot_report(
        int(selected_cluster)
    )

    if report["status"] != "established_cluster":
        st.error("Selected hotspot report unavailable.")
        return

    render_recommendation_panel(
        report["actions_by_tier"],
        key_prefix=f"hotspot_{selected_cluster}"
    )

    # 11. Historical interpretation safeguards
    # ----------------------------------------------

    st.caption(
        "Historical complaint recurrence indicates "
        "repeated reporting, not necessarily repeated "
        "physical failure. Closed complaint status "
        "does not prove repair. These are suggested "
        "investigations, not verified defects or "
        "physical infrastructure risk estimates."
    )

    # ----------------------------------------------
    # 12. Optional filtered data table
    # ----------------------------------------------

    with st.expander("View Filtered Hotspot Data"):

        visible_columns = [
            "dbscan_cluster_id",
            "primary_borough",
            "dominant_category",
            "total_complaints",
            "recurring_reporting_pct",
            "complaints_2023",
            "complaints_2024",
            "complaints_2025"
        ]

        st.dataframe(
            filtered[visible_columns],
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Export Filtered Hotspots (CSV)",
            data=filtered[visible_columns].to_csv(
                index=False
            ),
            file_name="filtered_dbscan_hotspots.csv",
            mime="text/csv"
        )
