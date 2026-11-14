class InfrastructureGUIAdapter:
    """Prepare safe, consistent GUI-ready responses."""

    SCHEMA_VERSION = "infrastructure_gui_contract_v1"

    def __init__(self, insight_service):
        self.insight_service = insight_service

    def evaluate(
        self,
        latitude,
        longitude,
        complaint_category,
        incident_zip=None
    ):
        response = self.insight_service.evaluate(
            latitude,
            longitude,
            complaint_category,
            incident_zip
        )
        return self.format_response(response)

    @classmethod
    def format_response(cls, result):

        geo = result["geographic_validation"]
        zip_info = result["zip_coordinate_consistency"]
        area = result["kmeans_area_insight"]

        spatial = result.get("dbscan_spatial_insight")

        if spatial is None:
            spatial = {}

        spatial_status = spatial.get(
            "status",
            "skipped_geographic_review"
        )

        report = spatial.get("hotspot_report") or {}

        # Never show an established hotspot report
        # for noise, ambiguous, or review statuses.
        if spatial_status != "established_cluster":
            report = {}

        source_recommendations = (
            report.get("recommendations") or {}
        )

        recommendations = {
            "featured": list(
                source_recommendations.get(
                    "featured", []
                )
            ),
            "supporting": list(
                source_recommendations.get(
                    "supporting", []
                )
            ),
            "limited_evidence": list(
                source_recommendations.get(
                    "limited_evidence", []
                )
            )
        }

        # Enforce profile visibility centrally.
        show_area = (
            area.get("status") == "historical_area_match"
            and area.get(
                "historical_profile_display_allowed"
            ) is True
            and area.get("k3_cluster") is not None
            and area.get("reporting_profile") is not None
        )

        kmeans_view = {
            "status": area.get("status"),
            "display_allowed": show_area,
            "area_id": (
                area.get("area_id")
                if show_area else None
            ),
            "candidate_area_id": area.get(
                "candidate_area_id"
            ),
            "cluster_id": (
                area.get("k3_cluster")
                if show_area else None
            ),
            "reporting_profile": (
                area.get("reporting_profile")
                if show_area else None
            ),
            "geographic_support": area.get(
                "geographic_support"
            ),
            "model_version": area.get(
                "model_version"
            ),
            "postal_zip_verified": False
        }

        spatial_view = {
            "status": spatial_status,
            "cluster_id": (
                spatial.get("cluster_id")
                if spatial_status == "established_cluster"
                else None
            ),
            "hotspot": report.get("hotspot"),
            "category_context": (
                spatial.get("category_context")
                if spatial_status == "established_cluster"
                else None
            ),
            "candidate_hotspots": (
                spatial.get("candidate_hotspots", [])
                if spatial_status == "ambiguous_association"
                else []
            ),
            "nearest_core_distance_m": (
                spatial.get("spatial_lookup") or {}
            ).get("nearest_core_distance_m"),
            "model_version": (
                spatial.get("spatial_lookup") or {}
            ).get("model_version")
        }

        warnings = []

        for source in (
            result.get("warnings", []),
            spatial.get("warnings", []),
            report.get("warnings", []),
            area.get("warnings", []),
            zip_info.get("warnings", [])
        ):
            for warning in source:
                if warning not in warnings:
                    warnings.append(warning)

        return {
            "schema_version": cls.SCHEMA_VERSION,
            "historical_period": result[
                "historical_period"
            ],
            "status": result["status"],
            "complaint_input": result[
                "complaint_input"
            ],
            "geography": {
                "status": geo["status"],
                "borough": geo.get(
                    "coordinate_borough"
                ),
                "candidate_boroughs": geo.get(
                    "candidate_boroughs", []
                ),
                "boundary_distance_m": geo.get(
                    "boundary_distance_m"
                )
            },
            "zip_validation": {
                "status": zip_info["status"],
                "incident_zip": zip_info.get(
                    "incident_zip"
                ),
                "coordinate_zctas": zip_info.get(
                    "coordinate_zctas", []
                ),
                "zcta_polygon_consistent": zip_info.get(
                    "zcta_polygon_consistent", False
                ),
                "postal_zip_verified": False,
                "reference_version": zip_info.get(
                    "reference_version"
                )
            },
            "dbscan": spatial_view,
            "kmeans": kmeans_view,
            "recommendations": recommendations,
            "warnings": warnings
        }
