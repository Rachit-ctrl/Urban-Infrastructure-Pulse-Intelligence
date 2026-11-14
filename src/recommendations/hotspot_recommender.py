from pathlib import Path
from numbers import Integral
import pandas as pd


class HotspotRecommendationEngine:
    """Read validated historical DBSCAN investigation outputs."""

    TIERS = ("Featured", "Supporting", "Limited Evidence")

    SUMMARY_FIELDS = (
        "primary_borough",
        "total_complaints",
        "dominant_category",
        "recurring_reporting_pct",
        "complaints_2024",
        "complaints_2025",
        "yoy_change_pct",
        "trend_interpretation_caution",
        "closed_status_pct",
        "not_closed_status_complaints",
        "model_version",
        "data_start_year",
        "data_end_year",
    )

    ACTION_FIELDS = (
        "complaint_category",
        "infrastructure_system",
        "category_complaints",
        "category_share_pct",
        "presentation_tier",
        "recommended_action",
        "category_evidence_summary",
        "hotspot_context_summary",
        "interpretation_caution",
    )

    def __init__(self, data_dir):
        data_dir = Path(data_dir)

        self.hotspots = pd.read_csv(
            data_dir / "dbscan_primary_f_gui_hotspots.csv"
        )

        self.recommendations = pd.read_csv(
            data_dir / "dbscan_primary_f_recommendations_v1.csv"
        )

        if len(self.hotspots) != 2846:
            raise ValueError("Unexpected hotspot population.")

        if len(self.recommendations) != 21981:
            raise ValueError("Unexpected recommendation population.")

        if self.hotspots["dbscan_cluster_id"].duplicated().any():
            raise ValueError("Duplicate hotspot identifiers.")

        pair_columns = [
            "dbscan_cluster_id",
            "complaint_category",
        ]

        if self.recommendations.duplicated(pair_columns).any():
            raise ValueError("Duplicate hotspot-category actions.")

        if set(self.hotspots["dbscan_cluster_id"]) != set(
            self.recommendations["dbscan_cluster_id"]
        ):
            raise ValueError("Hotspot coverage mismatch.")

        self.hotspot_index = self.hotspots.set_index(
            "dbscan_cluster_id"
        )

    @staticmethod
    def _plain(value):
        """Convert dataframe scalar values for GUI consumption."""
        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            return value.item()

        return value

    def get_hotspot_report(self, cluster_id):
        """Return all evidence tiers for an existing hotspot."""

        if isinstance(cluster_id, bool) or not isinstance(
            cluster_id, Integral
        ):
            raise TypeError("cluster_id must be an integer.")

        cluster_id = int(cluster_id)

        if cluster_id == -1:
            return {
                "cluster_id": -1,
                "status": "outside_existing_clusters",
                "message": (
                    "This location is not assigned to an "
                    "established historical DBSCAN cluster. "
                    "This does not imply low infrastructure risk."
                ),
            }

        if cluster_id not in self.hotspot_index.index:
            return {
                "cluster_id": cluster_id,
                "status": "unknown_cluster",
                "message": (
                    "No historical hotspot profile exists "
                    "for this identifier."
                ),
            }

        hotspot = self.hotspot_index.loc[cluster_id]

        summary = {
            field: self._plain(hotspot[field])
            for field in self.SUMMARY_FIELDS
        }

        actions = self.recommendations.loc[
            self.recommendations["dbscan_cluster_id"]
            == cluster_id
        ].sort_values(
            ["tier_order", "category_complaints",
             "complaint_category"],
            ascending=[True, False, True],
        )

        actions_by_tier = {}

        for tier in self.TIERS:
            subset = actions.loc[
                actions["presentation_tier"] == tier,
                list(self.ACTION_FIELDS),
            ]

            actions_by_tier[tier] = [
                {
                    key: self._plain(value)
                    for key, value in record.items()
                }
                for record in subset.to_dict("records")
            ]

        return {
            "cluster_id": cluster_id,
            "status": "established_cluster",
            "historical_period": "2023-2025",
            "hotspot_summary": summary,
            "actions_by_tier": actions_by_tier,
            "featured_action_count": len(
                actions_by_tier["Featured"]
            ),
            "available_action_count": len(actions),
            "interpretation_note": (
                "Evidence tiers describe historical reporting, "
                "not infrastructure severity. All categories, "
                "including Limited Evidence, remain available. "
                "Current conditions require independent review."
            ),
        }
