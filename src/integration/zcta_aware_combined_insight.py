from copy import deepcopy


class ZCTAAwareCombinedInsightService:
    """Combine spatial insight and ZIP geographic evidence."""

    SCHEMA_VERSION = "zcta_aware_combined_insight_v1"

    def __init__(self, combined_service, zcta_validator):
        self.combined_service = combined_service
        self.zcta_validator = zcta_validator

    def evaluate(
        self,
        latitude,
        longitude,
        complaint_category,
        incident_zip=None
    ):
        # Original combined service validates inputs and
        # retrieves independently available DBSCAN insights.
        result = deepcopy(
            self.combined_service.evaluate(
                latitude,
                longitude,
                complaint_category,
                incident_zip
            )
        )

        geo = result["geographic_validation"]
        complaint = result["complaint_input"]
        area = result["kmeans_area_insight"]

        if geo["status"] == "within_borough":

            zip_evidence = self.zcta_validator.evaluate(
                complaint["latitude"],
                complaint["longitude"],
                complaint["incident_zip"]
            )

        else:

            zip_evidence = {
                "schema_version": (
                    self.zcta_validator.SCHEMA_VERSION
                ),
                "status": "skipped_geographic_review",
                "incident_zip": complaint["incident_zip"],
                "coordinate_zctas": [],
                "reference_has_zip": False,
                "boundary_distance_m": None,
                "zcta_polygon_consistent": False,
                "postal_zip_verified": False,
                "reference_version": (
                    self.zcta_validator.REFERENCE_VERSION
                ),
                "warnings": [
                    "ZIP geographic consistency was not "
                    "assessed because borough validation "
                    "requires review."
                ]
            }

        zip_status = zip_evidence["status"]

        # Never equate ZCTA matching with USPS verification.
        area["postal_zip_verified"] = False
        area["zcta_evidence_status"] = zip_status
        area["candidate_area_id"] = None
        area["historical_profile_display_allowed"] = False
        area["geographic_support"] = "not_established"

        if area["status"] == "historical_area_match":

            if zip_status == "zcta_consistent":

                area["geographic_support"] = (
                    "mapped_zcta_consistent"
                )

                area["historical_profile_display_allowed"] = True

            elif zip_status == "zcta_unavailable":

                # A historical area match may still be useful,
                # but must be clearly labeled as unverified.
                area["geographic_support"] = (
                    "supplied_zip_unverified"
                )

                area["historical_profile_display_allowed"] = True

                area["warnings"].append(
                    "This historical K-Means profile uses "
                    "a supplied ZIP without ZCTA geographic "
                    "confirmation."
                )

            elif zip_status in {
                "zcta_mismatch",
                "boundary_review_required"
            }:

                # Preserve the candidate identifier for
                # review, not the unverified cluster/profile.
                area["candidate_area_id"] = area["area_id"]

                area["area_id"] = None
                area["k3_cluster"] = None
                area["k2_cluster"] = None
                area["reporting_profile"] = None

                area["status"] = (
                    "area_held_for_zip_review"
                )

                area["geographic_support"] = (
                    "requires_zip_review"
                )

                area["warnings"].append(
                    "The historical area candidate is held "
                    "for ZIP geographic review. No K-Means "
                    "classification is displayed."
                )

        warnings = (
            list(result.get("warnings", []))
            + list(zip_evidence.get("warnings", []))
            + list(area.get("warnings", []))
        )

        warnings.append(
            "The ZCTA reference is a geographic snapshot "
            "and does not establish historical postal "
            "boundaries for 2023-2025."
        )

        result["schema_version"] = self.SCHEMA_VERSION
        result["zip_coordinate_consistency"] = zip_evidence
        result["kmeans_area_insight"] = area
        result["warnings"] = list(dict.fromkeys(warnings))

        return result
