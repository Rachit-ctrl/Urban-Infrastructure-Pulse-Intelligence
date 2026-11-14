import streamlit as st

from src.ui.service_factory import (
    get_gui_adapter,
    verify_artifacts
)


st.set_page_config(
    page_title="Urban Infrastructure Pulse Intelligence",
    page_icon="🏙️",
    layout="wide"
)

st.title("Urban Infrastructure Pulse Intelligence")

st.caption(
    "NYC 311 Infrastructure Reporting Intelligence "
    "| Historical period: 2023–2025"
)

st.markdown("---")


@st.cache_resource(show_spinner=False)
def load_backend():
    return get_gui_adapter()


try:
    with st.spinner("Loading historical infrastructure models..."):
        verify_artifacts()
        adapter = load_backend()

except Exception as exc:
    st.error("Application backend could not be loaded.")
    st.exception(exc)
    st.stop()


st.success("Historical intelligence backend connected successfully.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Historical Complaints",
        "685,284"
    )

with col2:
    st.metric(
        "DBSCAN Hotspots",
        "2,846"
    )

with col3:
    st.metric(
        "K-Means Areas",
        "238"
    )

st.sidebar.header("System Information")
st.sidebar.write("Historical period: 2023–2025")
st.sidebar.write("Spatial model: DBSCAN F")
st.sidebar.write("Area model: PCA90 K-Means K=3")
st.sidebar.write("Geographic reference: NYC ZCTA")
st.sidebar.caption("Historical analysis, not live monitoring.")

# === PHASE 15.5.6: DASHBOARD NAVIGATION ===

st.sidebar.divider()

st.sidebar.subheader("Dashboard Navigation")

page = st.sidebar.radio(
    "Select a section",
    options=[
        "Overview",
        "Complaint Lookup",
        "Hotspot Explorer",
        "K-Means Areas"
    ],
    index=0,
    key="urban_pulse_dashboard_navigation"
)

st.sidebar.divider()

st.sidebar.caption(
    "NYC 311 historical reporting intelligence. "
    "Data period: 2023–2025. "
    "This is not real-time infrastructure monitoring."
)

if page == "Overview":

    st.info(
        "Use the sidebar navigation to explore "
        "complaint intelligence, DBSCAN reporting "
        "hotspots and K-Means area patterns."
    )

    st.subheader("Application Modules")

    st.markdown(
        """
        **Complaint Intelligence**

        Associate complaint coordinates with established
        historical DBSCAN reporting hotspots.

        **Investigation Recommendations**

        Explore evidence-based infrastructure
        investigation suggestions.

        **Area Reporting Patterns**

        Review historical borough-ZIP K-Means
        reporting profiles with ZCTA consistency checks.

        **Geographic Intelligence**

        Explore historical hotspot distributions
        and reporting patterns.
        """
    )

    st.info(
        "Interactive complaint lookup is available below. "
        "Hotspot mapping and area dashboards follow next."
    )

    with st.expander("Models and interpretation safeguards"):
        st.write(
            "DBSCAN: historical spatial reporting clusters."
        )
        st.write(
            "K-Means: aggregated borough-ZIP "
            "historical reporting patterns."
        )
        st.write(
            "Recommendations: suggested investigations, "
            "not confirmed infrastructure defects."
        )
        st.write(
            "ZCTA matching does not establish "
            "USPS postal-address verification."
        )
        st.write(
            "Closed complaint status does not "
            "prove physical infrastructure repair."
        )
        st.write(
            "This application does not predict "
            "infrastructure damage or physical risk."
        )

elif page == "Complaint Lookup":

    # === PHASE 15.5.2: COMPLAINT LOOKUP ===

    import json
    import pandas as pd

    st.divider()
    st.header("New Complaint Intelligence Lookup")

    st.caption(
        "Explore historical reporting patterns for a "
        "submitted location. No complaint is submitted "
        "to NYC 311 or added to the training dataset."
    )

    # Load supported categories from the verified taxonomy.
    taxonomy_path = verify_artifacts()["Action taxonomy"]

    categories = sorted(
        pd.read_csv(taxonomy_path)["complaint_category"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )

    if len(categories) != 15:
        st.error("The infrastructure category reference is invalid.")
        st.stop()

    # --------------------------------------------------
    # Complaint input form
    # --------------------------------------------------

    with st.form("historical_complaint_form"):

        input_col1, input_col2, input_col3 = st.columns(3)

        with input_col1:
            latitude_text = st.text_input(
                "Latitude",
                placeholder="e.g. 40.7500"
            )

        with input_col2:
            longitude_text = st.text_input(
                "Longitude",
                placeholder="e.g. -73.9900"
            )

        with input_col3:
            zip_text = st.text_input(
                "Incident ZIP (optional)",
                placeholder="Five-digit ZIP"
            )

        category = st.selectbox(
            "Infrastructure Complaint Category",
            options=categories
        )

        submitted = st.form_submit_button(
            "Analyze Historical Reporting",
            type="primary"
        )

    # --------------------------------------------------
    # Validate input and request GUI-ready response
    # --------------------------------------------------

    if submitted:

        st.session_state.pop(
            "historical_complaint_response",
            None
        )

        try:
            lat = float(latitude_text.strip())
            lon = float(longitude_text.strip())

            incident_zip = zip_text.strip() or None

            with st.spinner(
                "Retrieving historical infrastructure insights..."
            ):

                result = adapter.evaluate(
                    lat,
                    lon,
                    category,
                    incident_zip
                )

            st.session_state[
                "historical_complaint_response"
            ] = result

        except (ValueError, TypeError) as exc:

            st.error(
                "Please check the complaint input: "
                + str(exc)
            )

        except Exception as exc:

            st.error(
                "The historical intelligence lookup failed."
            )

            st.exception(exc)

    # --------------------------------------------------
    # Display retained response after Streamlit reruns
    # --------------------------------------------------

    if "historical_complaint_response" in st.session_state:

        result = st.session_state[
            "historical_complaint_response"
        ]

        geography = result["geography"]
        zip_validation = result["zip_validation"]
        dbscan = result["dbscan"]
        kmeans = result["kmeans"]
        recommendations = result["recommendations"]

        st.divider()
        st.subheader("Historical Intelligence Results")

        st.caption(
            "Analysis period: 2023–2025. "
            "This is not a real-time infrastructure "
            "condition or damage assessment."
        )

        # --------------------------------------------------
        # 1. Geographic validation
        # --------------------------------------------------

        st.markdown("#### Geographic Validation")

        geo_col1, geo_col2, geo_col3 = st.columns(3)

        with geo_col1:
            st.metric(
                "Coordinate Borough",
                geography.get("borough") or "Unresolved"
            )

        with geo_col2:
            st.metric(
                "Borough Validation",
                geography["status"].replace("_", " ").title()
            )

        with geo_col3:
            st.metric(
                "ZIP Geographic Evidence",
                zip_validation["status"].replace("_", " ").title()
            )

        if result["status"] == "geographic_review_required":
            st.warning(
                "Geographic review is required. "
                "Automatic historical model association "
                "has been withheld. This does not mean "
                "the complaint is unimportant."
            )

        # --------------------------------------------------
        # 2. DBSCAN reporting hotspot
        # --------------------------------------------------

        st.markdown("#### DBSCAN Spatial Reporting Insight")

        spatial_status = dbscan["status"]

        if spatial_status == "established_cluster":

            st.success(
                "An established historical reporting "
                "hotspot was found."
            )

            st.metric(
                "Historical DBSCAN Cluster",
                str(dbscan["cluster_id"])
            )

            context = dbscan.get("category_context")

            if context and context.get("status") == (
                "historically_reported"
            ):

                st.write(
                    "**Matching category:**",
                    context["category"]
                )

                st.metric(
                    "Historical Category Complaints",
                    f"{context['category_complaints']:,}"
                )

                if context.get("suggested_investigation"):
                    st.markdown(
                        "**Suggested investigation:**"
                    )

                    st.write(
                        context["suggested_investigation"]
                    )

            elif context:
                st.info(
                    "This category has no matching "
                    "historical evidence within the "
                    "associated hotspot."
                )

            if dbscan.get("hotspot"):
                with st.expander(
                    "View historical hotspot details"
                ):
                    st.json(dbscan["hotspot"])

        elif spatial_status == "ambiguous_association":

            st.warning(
                "More than one historical DBSCAN hotspot "
                "is associated with these coordinates. "
                "No single cluster was automatically selected."
            )

            candidates = dbscan.get(
                "candidate_hotspots", []
            )

            if candidates:
                st.dataframe(
                    pd.DataFrame(candidates),
                    hide_index=True
                )

        elif spatial_status == "outside_existing_clusters":

            st.info(
                "No established historical DBSCAN hotspot "
                "was associated with these coordinates. "
                "This does not imply low infrastructure risk."
            )

        else:

            st.info(
                "DBSCAN lookup was not performed because "
                "geographic validation requires review."
            )

        # --------------------------------------------------
        # 3. K-Means historical area reporting pattern
        # --------------------------------------------------

        st.markdown("#### K-Means Area Reporting Pattern")

        if kmeans["display_allowed"]:

            st.metric(
                "Historical K-Means Cluster",
                str(kmeans["cluster_id"])
            )

            st.write(
                "**Historical area:**",
                kmeans["area_id"]
            )

            st.write(
                "**Reporting profile:**",
                kmeans["reporting_profile"]
            )

            if kmeans["geographic_support"] == (
                "mapped_zcta_consistent"
            ):

                st.success(
                    "Supplied ZIP is geographically "
                    "consistent with the mapped ZCTA."
                )

            elif kmeans["geographic_support"] == (
                "supplied_zip_unverified"
            ):

                st.warning(
                    "Historical area profile is based on "
                    "a supplied ZIP without ZCTA confirmation."
                )

            st.caption(
                "K-Means describes historical reporting "
                "patterns for an aggregated borough-ZIP area. "
                "ZCTA matching is not USPS postal verification."
            )

        elif kmeans["status"] == "area_held_for_zip_review":

            st.warning(
                "K-Means reporting profile withheld "
                "pending ZIP geographic review."
            )

            if kmeans.get("candidate_area_id"):
                st.write(
                    "Area requiring review:",
                    kmeans["candidate_area_id"]
                )

        else:

            st.info(
                "No K-Means historical area profile "
                "is available for this lookup."
            )

            st.caption(
                "Lookup status: " + str(kmeans["status"])
            )

        # --------------------------------------------------
        # === PHASE 15.5.4: REFINED COMPLAINT RECOMMENDATIONS ===

        from src.ui.recommendation_panel import (
            render_recommendation_panel
        )

        render_recommendation_panel(
            recommendations,
            key_prefix="complaint_lookup",
            preferred_category=result[
                "complaint_input"
            ]["complaint_category"],
            category_context=dbscan.get("category_context")
        )

        # 5. Interpretation warnings
        # --------------------------------------------------

        warnings = result.get("warnings", [])

        with st.expander(
            f"Interpretation Warnings ({len(warnings)})"
        ):

            for warning in warnings:
                st.write("- " + str(warning))

        # --------------------------------------------------
        # 6. Download structured response
        # --------------------------------------------------

        st.download_button(
            label="Export Intelligence Response (JSON)",
            data=json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            ),
            file_name="historical_complaint_insight.json",
            mime="application/json"
        )

elif page == "Hotspot Explorer":

    # === PHASE 15.5.3B: HOTSPOT EXPLORER ===

    from src.ui.hotspot_explorer import (
        render_hotspot_explorer
    )

    render_hotspot_explorer()

elif page == "K-Means Areas":

    # === PHASE 15.5.5: K-MEANS DASHBOARD ===

    from src.ui.kmeans_dashboard import (
        render_kmeans_dashboard
    )

    render_kmeans_dashboard()
