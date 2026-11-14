class CombinedInfrastructureInsightService:
    """
    Combine validated DBSCAN spatial insights with
    historical K-Means borough-ZIP reporting profiles.

    Does not retrain either model.
    """

    SCHEMA_VERSION = "combined_infrastructure_insight_v1"
    HISTORICAL_PERIOD = "2023-2025"

    def __init__(self, validated_service, kmeans_lookup):
        self.validated_service = validated_service
        self.kmeans_lookup = kmeans_lookup

    def evaluate(
        self,
        latitude,
        longitude,
        complaint_category,
        incident_zip=None
    ):

        # First validate category and geography,
        # and retrieve spatial insights when permitted.
        validated = self.validated_service.evaluate(
            latitude,
            longitude,
            complaint_category
        )

        geo = validated["geographic_validation"]

        normalized_zip = None

        if incident_zip is not None:
            normalized_zip = (
                self.kmeans_lookup.normalize_zip(
                    incident_zip
                )
            )

        area_result = {
            "status": "zip_not_provided",
            "area_id": None,
            "borough": geo["coordinate_borough"],
            "incident_zip": normalized_zip,
            "k3_cluster": None,
            "k2_cluster": None,
            "reporting_profile": None,
            "model_version": (
                self.kmeans_lookup.MODEL_VERSION
            ),
            "historical_period": self.HISTORICAL_PERIOD,
            "lookup_basis": "not_performed",
            "warnings": []
        }

        # Only use a coordinate-derived borough
        # that passed geographic validation.
        if geo["status"] != "within_borough":

            area_result["status"] = (
                "area_lookup_skipped_geographic_review"
            )

            area_result["warnings"].append(
                "K-Means borough-ZIP lookup was skipped "
                "because geographic validation requires review."
            )

        elif normalized_zip is None:

            area_result["warnings"].append(
                "No ZIP code was supplied. Historical "
                "K-Means area classification is unavailable."
            )

        else:

            area_result = self.kmeans_lookup.lookup(
                geo["coordinate_borough"],
                normalized_zip
            )

            if area_result["status"] == (
                "historical_area_match"
            ):
                area_result["warnings"].append(
                    "The input coordinates were validated "
                    "against a borough polygon, but the "
                    "supplied ZIP was NOT independently "
                    "verified against those coordinates."
                )

        warnings = list(
            validated.get("warnings", [])
        )

        warnings.extend(
            area_result.get("warnings", [])
        )

        warnings.append(
            "Both models describe historical reporting "
            "patterns from 2023-2025, not current "
            "infrastructure condition."
        )

        warnings.append(
            "No physical damage, maintenance urgency, "
            "or infrastructure risk score is predicted."
        )

        # Deduplicate while preserving warning order.
        warnings = list(dict.fromkeys(warnings))

        return {
            "schema_version": self.SCHEMA_VERSION,
            "status": validated["status"],
            "historical_period": self.HISTORICAL_PERIOD,

            "complaint_input": {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "complaint_category": (
                    validated["complaint_category"]
                ),
                "incident_zip": normalized_zip
            },

            "geographic_validation": geo,

            "dbscan_spatial_insight": (
                validated["insight"]
            ),

            "kmeans_area_insight": area_result,

            "warnings": warnings
        }
