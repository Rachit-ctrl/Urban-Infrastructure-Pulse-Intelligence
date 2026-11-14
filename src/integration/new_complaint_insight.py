class NewComplaintInsightService:
    """Combine historical spatial lookup and recommendations."""

    SCHEMA_VERSION = "new_complaint_insight_v1"

    def __init__(self, predictor, engine, payload_builder):
        self.predictor = predictor
        self.engine = engine
        self.payload_builder = payload_builder

    def evaluate(
        self, latitude, longitude, complaint_category=None
    ):
        if (
            complaint_category is not None
            and not isinstance(complaint_category, str)
        ):
            raise TypeError(
                "complaint_category must be a string or None."
            )

        category = (
            complaint_category.strip().upper()
            if complaint_category is not None
            else None
        ) or None

        spatial = self.predictor.associate(
            latitude, longitude
        )

        result = {
            "schema_version": self.SCHEMA_VERSION,
            "status": spatial["status"],
            "cluster_id": spatial["cluster_id"],
            "historical_period": "2023-2025",
            "complaint_category": category,
            "spatial_lookup": spatial,
            "hotspot_report": None,
            "candidate_hotspots": [],
            "category_context": None,
            "warnings": [
                "This lookup uses historical 2023-2025 data.",
                "Association is not an infrastructure risk score.",
                "Entered coordinates have not been independently "
                "verified against NYC borough boundaries."
            ]
        }

        # ------------------------------------------
        # One uniquely associated historical hotspot
        # ------------------------------------------

        if spatial["status"] == "established_cluster":

            cluster_id = int(spatial["cluster_id"])

            report = self.payload_builder(
                self.engine, cluster_id
            )

            if report["status"] != "established_cluster":
                raise ValueError(
                    "Spatial and recommendation data disagree."
                )

            result["hotspot_report"] = report

            # Match the new complaint's category to
            # historical category-specific evidence.
            if category is not None:

                matched_actions = []

                for tier, actions in (
                    report["recommendations"].items()
                ):
                    for action in actions:
                        if action["complaint_category"] == category:
                            matched_actions.append(action)

                if len(matched_actions) > 1:
                    raise ValueError(
                        "Duplicate category recommendations."
                    )

                if matched_actions:
                    action = matched_actions[0]

                    result["category_context"] = {
                        "status": "historically_reported",
                        "category": category,
                        "category_complaints": (
                            action["category_complaints"]
                        ),
                        "category_share_pct": (
                            action["category_share_pct"]
                        ),
                        "presentation_tier": (
                            action["presentation_tier"]
                        ),
                        "suggested_investigation": (
                            action["recommended_action"]
                        ),
                        "evidence_summary": (
                            action["category_evidence_summary"]
                        )
                    }

                else:
                    result["category_context"] = {
                        "status": "no_historical_category_match",
                        "category": category,
                        "suggested_investigation": None
                    }

                    result["warnings"].append(
                        "This complaint category has no "
                        "historical reports in the associated "
                        "hotspot profile."
                    )

            result["warnings"].extend(
                report["warnings"]
            )

        # ------------------------------------------
        # Multiple possible historical hotspots
        # ------------------------------------------

        elif spatial["status"] == "ambiguous_association":

            for candidate in spatial["candidate_clusters"]:

                cid = int(candidate["cluster_id"])

                report = self.engine.get_hotspot_report(cid)

                if report["status"] != "established_cluster":
                    raise ValueError(
                        "Unknown candidate hotspot."
                    )

                summary = report["hotspot_summary"]

                result["candidate_hotspots"].append({
                    "cluster_id": cid,
                    "nearest_core_distance_m": (
                        candidate["nearest_core_distance_m"]
                    ),
                    "core_points_within_radius": (
                        candidate["core_points_within_radius"]
                    ),
                    "primary_borough": (
                        summary["primary_borough"]
                    ),
                    "total_complaints": (
                        summary["total_complaints"]
                    ),
                    "dominant_category": (
                        summary["dominant_category"]
                    )
                })

            result["warnings"].append(
                "Multiple historical clusters are nearby. "
                "No unique hotspot or recommendation is "
                "automatically selected."
            )

        # ------------------------------------------
        # No historical hotspot association
        # ------------------------------------------

        elif spatial["status"] == "outside_existing_clusters":

            result["warnings"].append(
                "No established historical hotspot was found. "
                "This does not mean the infrastructure issue "
                "is unimportant or low risk."
            )

        else:
            raise ValueError(
                "Unexpected spatial lookup status."
            )

        return result
