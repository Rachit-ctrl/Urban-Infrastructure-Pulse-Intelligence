from pathlib import Path
import json
import pandas as pd
import streamlit as st

from src.integration.kmeans_area_lookup import (
    KMeansAreaLookup
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROFILE_PATH = (
    PROJECT_ROOT / "outputs/tables/kmeans"
    / "kmeans_full_data_k3_descriptive_profiles.csv"
)

BOROUGHS = {
    "BRONX",
    "BROOKLYN",
    "MANHATTAN",
    "QUEENS",
    "STATEN ISLAND"
}


@st.cache_data(show_spinner=False)
def load_kmeans_area_profiles(path):

    data = pd.read_csv(
        path,
        dtype={"area_id": "string"}
    )

    required = {
        "area_id",
        "k2_cluster",
        "k3_cluster",
        "reporting_profile"
    }

    if (
        len(data) != 238
        or not required.issubset(data.columns)
    ):
        raise ValueError(
            "Historical K-Means profile schema changed."
        )

    if data[list(required)].isna().any().any():
        raise ValueError(
            "Missing historical K-Means classifications."
        )

    data["area_id"] = (
        data["area_id"].str.strip().str.upper()
    )

    data["reporting_profile"] = (
        data["reporting_profile"]
        .astype(str)
        .str.strip()
    )

    data["k3_cluster"] = pd.to_numeric(
        data["k3_cluster"],
        errors="raise"
    ).astype(int)

    data["k2_cluster"] = pd.to_numeric(
        data["k2_cluster"],
        errors="raise"
    ).astype(int)

    if data["area_id"].duplicated().any():
        raise ValueError("Duplicate area IDs.")

    if data["reporting_profile"].eq("").any():
        raise ValueError("Empty reporting profile.")

    if set(data["k3_cluster"]) != {0, 1, 2}:
        raise ValueError("Unexpected K-Means labels.")

    if (
        data.groupby("k3_cluster")[
            "reporting_profile"
        ].nunique().max() != 1
    ):
        raise ValueError(
            "Inconsistent profile labels."
        )

    # Preserve original historical area IDs.
    parts = data["area_id"].str.rsplit(
        "_", n=1
    )

    data["borough"] = parts.str[0]

    data["historical_zip_token"] = parts.str[-1]

    data["standard_zip_format"] = (
        data["historical_zip_token"]
        .str.fullmatch(r"[0-9]{5}")
        .fillna(False)
    )

    if set(data["borough"]) != BOROUGHS:
        raise ValueError("Unexpected borough labels.")

    nonstandard = data.loc[
        ~data["standard_zip_format"],
        "area_id"
    ].tolist()

    if nonstandard != ["MANHATTAN_83"]:
        raise ValueError(
            "Unexpected nonstandard historical IDs."
        )

    return data.sort_values(
        ["borough", "area_id"]
    ).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def get_kmeans_reference_lookup(path):
    return KMeansAreaLookup(path)


def render_kmeans_dashboard():

    st.divider()

    st.header(
        "K-Means Historical Area Reporting Dashboard"
    )

    st.caption(
        "PCA90 K-Means K=3 classifications "
        "for 238 historical borough-ZIP reporting "
        "areas, 2023–2025."
    )

    profiles = load_kmeans_area_profiles(
        str(PROFILE_PATH)
    )

    lookup = get_kmeans_reference_lookup(
        str(PROFILE_PATH)
    )

    # ----------------------------------------------
    # 2. Dashboard summary
    # ----------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Historical Areas",
        f"{len(profiles):,}"
    )

    col2.metric(
        "Reporting Profiles",
        profiles["k3_cluster"].nunique()
    )

    col3.metric(
        "NYC Boroughs",
        profiles["borough"].nunique()
    )

    # ----------------------------------------------
    # 3. K-Means cluster distribution
    # ----------------------------------------------

    st.markdown(
        "#### Historical Reporting Pattern Distribution"
    )

    summary = (
        profiles.groupby(
            ["k3_cluster", "reporting_profile"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "area_count"})
        .sort_values("k3_cluster")
    )

    summary["area_share_pct"] = (
        100 * summary["area_count"] / len(profiles)
    ).round(2)

    st.bar_chart(
        summary.set_index("k3_cluster")[
            ["area_count"]
        ]
    )

    st.dataframe(
        summary.rename(columns={
            "k3_cluster": "K=3 Cluster",
            "reporting_profile": "Reporting Profile",
            "area_count": "Number of Areas",
            "area_share_pct": "Share of Areas (%)"
        }),
        hide_index=True,
        use_container_width=True
    )

    # ----------------------------------------------
    # 4. Borough comparison
    # ----------------------------------------------

    st.markdown(
        "#### Borough-Level Reporting Pattern Comparison"
    )

    borough_distribution = (
        profiles.assign(
            cluster_label=profiles[
                "k3_cluster"
            ].map(lambda x: f"Cluster {x}")
        )
        .groupby(["borough", "cluster_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=sorted(BOROUGHS),
            fill_value=0
        )
        .reindex(
            columns=[
                "Cluster 0",
                "Cluster 1",
                "Cluster 2"
            ],
            fill_value=0
        )
    )

    st.bar_chart(borough_distribution)

    st.caption(
        "These charts count historical reporting "
        "areas, not infrastructure defects or "
        "individual complaint predictions."
    )

    # ----------------------------------------------
    # 5. Interactive area filters
    # ----------------------------------------------

    st.markdown(
        "#### Explore Historical Borough-ZIP Areas"
    )

    filter_col1, filter_col2 = st.columns(2)

    borough_options = sorted(BOROUGHS)

    with filter_col1:

        selected_boroughs = st.multiselect(
            "Filter Boroughs",
            options=borough_options,
            default=borough_options,
            key="kmeans_dashboard_borough_filter"
        )

    with filter_col2:

        selected_clusters = st.multiselect(
            "Filter K-Means Clusters",
            options=[0, 1, 2],
            default=[0, 1, 2],
            key="kmeans_dashboard_cluster_filter"
        )

    include_nonstandard = st.checkbox(
        "Include nonstandard historical area "
        "MANHATTAN_83",
        value=True,
        key="kmeans_dashboard_include_nonstandard"
    )

    filtered = profiles.loc[
        profiles["borough"].isin(selected_boroughs)
        & profiles["k3_cluster"].isin(
            selected_clusters
        )
    ].copy()

    if not include_nonstandard:

        filtered = filtered.loc[
            filtered["standard_zip_format"]
        ].copy()

    st.caption(
        f"Showing {len(filtered)} of "
        f"{len(profiles)} historical areas."
    )

    if filtered.empty:

        st.info(
            "No historical areas match these filters."
        )

        return

    # ----------------------------------------------
    # 6. Check connection to last complaint lookup
    # ----------------------------------------------

    recent = st.session_state.get(
        "historical_complaint_response"
    ) or {}

    recent_kmeans = recent.get("kmeans") or {}

    linked_area = (
        recent_kmeans.get("area_id")
        if recent_kmeans.get("display_allowed") is True
        else None
    )

    if recent_kmeans.get("status") == (
        "area_held_for_zip_review"
    ):

        st.warning(
            "The latest complaint's K-Means profile "
            "was withheld for ZIP geographic review. "
            "Browsing historical areas below does not "
            "override that review."
        )

    elif linked_area:

        support = recent_kmeans.get(
            "geographic_support",
            "not_established"
        )

        st.info(
            f"Latest complaint area lookup: "
            f"{linked_area}. "
            f"Geographic support: "
            f"{support.replace('_', ' ')}."
        )

    # ----------------------------------------------
    # 7. Area selection
    # ----------------------------------------------

    area_options = filtered["area_id"].tolist()

    default_index = (
        area_options.index(linked_area)
        if linked_area in area_options
        else 0
    )

    selected_id = st.selectbox(
        "Inspect Historical Area",
        options=area_options,
        index=default_index,
        key="kmeans_dashboard_selected_area"
    )

    selected = filtered.loc[
        filtered["area_id"] == selected_id
    ].iloc[0]

    # Use the previously verified historical
    # area lookup, not a new K-Means prediction.
    area_record = lookup.lookup_by_area_id(
        selected_id
    )

    if area_record["k3_cluster"] != int(
        selected["k3_cluster"]
    ):
        raise ValueError(
            "Historical lookup classification mismatch."
        )

    detail_col1, detail_col2 = st.columns(2)

    detail_col1.metric(
        "K-Means K=3 Cluster",
        str(area_record["k3_cluster"])
    )

    detail_col2.metric(
        "Borough",
        str(selected["borough"])
    )

    st.write(
        "**Historical Area ID:**",
        selected_id
    )

    st.write(
        "**Reporting Profile:**",
        area_record["reporting_profile"]
    )

    # ----------------------------------------------
    # 8. Geographic interpretation safeguards
    # ----------------------------------------------

    if not bool(selected["standard_zip_format"]):

        st.warning(
            "This record uses the historical token "
            "83, not a valid five-digit postal ZIP. "
            "It is preserved for historical completeness."
        )

    elif linked_area != selected_id:

        st.caption(
            "This manually selected area is a "
            "historical catalog record. It has not "
            "been assigned to the latest complaint "
            "coordinates."
        )

    elif recent_kmeans.get(
        "geographic_support"
    ) == "supplied_zip_unverified":

        st.warning(
            "The latest complaint's supplied ZIP "
            "has no ZCTA geographic confirmation."
        )

    elif linked_area == selected_id:

        st.caption(
            "The latest complaint's historical "
            "area lookup matches this record. "
            "ZCTA matching does not constitute "
            "USPS postal verification."
        )

    with st.expander(
        "Historical Profile Interpretation"
    ):

        st.write(
            "The K-Means classification describes "
            "aggregated historical area reporting "
            "patterns, not physical infrastructure "
            "condition, damage, or risk."
        )

        st.write(
            "No new area features are calculated "
            "and the PCA/K-Means model is not retrained."
        )

        for warning in area_record.get(
            "warnings", []
        ):
            st.caption(str(warning))

    # ----------------------------------------------
    # 9. Historical area table and CSV export
    # ----------------------------------------------

    display_columns = [
        "area_id",
        "borough",
        "historical_zip_token",
        "standard_zip_format",
        "k3_cluster",
        "reporting_profile",
        "k2_cluster"
    ]

    with st.expander(
        "View and Export Historical Area Data"
    ):

        st.dataframe(
            filtered[display_columns],
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Export Filtered K-Means Areas (CSV)",
            data=filtered[
                display_columns
            ].to_csv(index=False),
            file_name="historical_kmeans_areas_filtered.csv",
            mime="text/csv",
            key="kmeans_dashboard_export_csv"
        )

    st.download_button(
        "Export Selected Area (JSON)",
        data=json.dumps(
            area_record,
            indent=2,
            allow_nan=False
        ),
        file_name="historical_kmeans_selected_area.json",
        mime="application/json",
        key="kmeans_dashboard_export_area"
    )
