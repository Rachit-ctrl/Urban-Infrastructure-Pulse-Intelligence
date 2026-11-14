import streamlit as st


TIERS = [
    ("featured", "Featured"),
    ("supporting", "Supporting"),
    ("limited_evidence", "Limited Evidence"),
]


def prepare_recommendations(
    recommendations,
    preferred_category=None
):
    if not isinstance(recommendations, dict):
        raise TypeError("Expected recommendation dictionary.")

    actions = {}

    for key, title in TIERS:
        records = recommendations.get(
            key,
            recommendations.get(title, [])
        )
        actions[key] = list(records or [])

    categories = sorted({
        str(action["complaint_category"]).strip().upper()
        for records in actions.values()
        for action in records
    })

    preferred = (
        str(preferred_category).strip().upper()
        if preferred_category is not None
        else None
    )

    return {
        "actions": actions,
        "categories": categories,
        "preferred_category": preferred,
        "counts": {
            key: len(actions[key])
            for key, _ in TIERS
        },
        "total": sum(
            len(records) for records in actions.values()
        )
    }


def render_recommendation_panel(
    recommendations,
    key_prefix,
    preferred_category=None,
    category_context=None
):
    prepared = prepare_recommendations(
        recommendations,
        preferred_category
    )

    st.markdown(
        "#### Historical Evidence-Based Investigations"
    )

    st.caption(
        "Suggested investigations based on historical "
        "complaint reporting, not confirmed defects "
        "or infrastructure risk predictions."
    )

    if prepared["total"] == 0:
        st.info(
            "No hotspot-specific investigation "
            "recommendations are available."
        )
        return

    # Evidence tier counts
    columns = st.columns(3)

    for col, (key, title) in zip(columns, TIERS):
        col.metric(
            title,
            prepared["counts"][key]
        )

    st.caption(
        "Evidence tiers describe reporting support, "
        "not physical severity or repair urgency."
    )

    # Category filtering
    options = (
        ["All categories"] + prepared["categories"]
    )

    preferred = prepared["preferred_category"]

    if preferred and preferred not in options:
        options.append(preferred)

    default_index = (
        options.index(preferred)
        if preferred in options
        else 0
    )

    chosen_category = st.selectbox(
        "Filter investigations by category",
        options=options,
        index=default_index,
        key=f"recommendation_filter_{key_prefix}"
    )

    filtered = {}

    for key, title in TIERS:
        records = prepared["actions"][key]

        if chosen_category != "All categories":
            records = [
                action for action in records
                if str(
                    action.get("complaint_category", "")
                ).strip().upper() == chosen_category
            ]

        filtered[key] = records

    visible_count = sum(
        len(records) for records in filtered.values()
    )

    st.caption(
        f"Showing {visible_count} of "
        f"{prepared['total']} available investigations."
    )

    # Category-specific context
    if (
        category_context
        and chosen_category == preferred
    ):
        status = category_context.get("status")

        if status == "historically_reported":
            reports = int(
                category_context.get(
                    "category_complaints", 0
                )
            )
            st.info(
                f"{reports:,} historical reports "
                "for this category within the hotspot."
            )

        elif status == "no_historical_category_match":
            st.info(
                "No historical reports for this category "
                "within the associated hotspot."
            )

    if visible_count == 0:
        st.info(
            "No investigations match this category. "
            "This does not indicate low risk."
        )
        return

    # Investigation entries
    for key, title in TIERS:
        records = filtered[key]

        if not records:
            continue

        with st.expander(
            f"{title} Investigations ({len(records)})",
            expanded=(key == "featured")
        ):
            for action in records:

                with st.container(border=True):

                    st.markdown(
                        "**" +
                        str(action.get(
                            "complaint_category",
                            "Unknown category"
                        )) +
                        "**"
                    )

                    system = action.get(
                        "infrastructure_system"
                    )

                    if system:
                        st.caption(
                            f"Infrastructure system: {system}"
                        )

                    st.write(
                        action.get(
                            "recommended_action",
                            "Investigation unavailable."
                        )
                    )

                    reports = action.get(
                        "category_complaints"
                    )
                    share = action.get(
                        "category_share_pct"
                    )

                    if reports is not None:
                        label = (
                            f"{int(reports):,} historical "
                            "category complaints"
                        )

                        if share is not None:
                            label += (
                                f" | {float(share):.2f}% "
                                "of hotspot complaints"
                            )

                        st.caption(label)

                    evidence_fields = [
                        (
                            "Category-specific evidence",
                            "category_evidence_summary"
                        ),
                        (
                            "Hotspot-wide context",
                            "hotspot_context_summary"
                        ),
                        (
                            "Interpretation caution",
                            "interpretation_caution"
                        )
                    ]

                    for heading, field in evidence_fields:
                        value = action.get(field)

                        if (
                            isinstance(value, str)
                            and value.strip()
                        ):
                            st.markdown(f"**{heading}**")
                            st.caption(value)

    st.caption(
        "Repeated reports do not prove repeated "
        "physical failures. Hotspot-wide statistics "
        "are not necessarily category-specific. "
        "Closed status does not prove repair."
    )
