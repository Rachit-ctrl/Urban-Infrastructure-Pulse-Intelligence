import pandas as pd


SCHEMA_VERSION = "hotspot_gui_contract_v1"


def _plain(value):
    """Convert pandas/numpy scalars to JSON-safe values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def build_hotspot_gui_payload(engine, cluster_id):
    """
    Convert a historical hotspot backend report into a
    consistent JSON-compatible GUI payload.

    This function does not classify new coordinates.
    """

    report = engine.get_hotspot_report(cluster_id)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": report["status"],
        "cluster_id": report["cluster_id"],
        "model_version": "dbscan_f_75m_min40_v1",
        "historical_period": "2023-2025",
        "hotspot": None,
        "recommendations": {
            "featured": [],
            "supporting": [],
            "limited_evidence": []
        },
        "warnings": []
    }

    # Noise and unknown cluster IDs
    if report["status"] != "established_cluster":

        payload["warnings"].append(
            report["message"]
        )

        return payload

    summary = report["hotspot_summary"]

    # Obtain geographic and display-specific features
    row = engine.hotspot_index.loc[cluster_id]

    payload["model_version"] = _plain(
        row["model_version"]
    )

    payload["hotspot"] = {
        "borough": summary["primary_borough"],
        "latitude": _plain(row["center_latitude"]),
        "longitude": _plain(row["center_longitude"]),
        "total_complaints": summary["total_complaints"],
        "dominant_category": summary["dominant_category"],
        "dominant_category_pct": _plain(
            row["dominant_category_pct"]
        ),
        "recurring_reporting_pct": (
            summary["recurring_reporting_pct"]
        ),
        "complaints_2024": summary["complaints_2024"],
        "complaints_2025": summary["complaints_2025"],
        "yoy_change_pct": summary["yoy_change_pct"],
        "trend_caution": bool(
            summary["trend_interpretation_caution"]
        ),
        "closed_status_pct": summary["closed_status_pct"],
        "not_closed_status_complaints": (
            summary["not_closed_status_complaints"]
        ),
        "cross_borough": bool(row["cross_borough"])
    }

    # Preserve every category-level investigation
    actions = report["actions_by_tier"]

    payload["recommendations"] = {
        "featured": actions["Featured"],
        "supporting": actions["Supporting"],
        "limited_evidence": actions["Limited Evidence"]
    }

    # Historical interpretation safeguards
    payload["warnings"].append(
        "Historical NYC 311 reporting evidence from "
        "2023-2025; not live operational information."
    )

    payload["warnings"].append(
        "Complaint counts and recommendation tiers do "
        "not establish infrastructure severity or "
        "a confirmed physical defect."
    )

    payload["warnings"].append(
        "Recorded closure does not confirm physical "
        "repair or current infrastructure condition."
    )

    if payload["hotspot"]["trend_caution"]:
        payload["warnings"].append(
            "Historical percentage trend has a small "
            "or unavailable 2024 baseline."
        )

    if (
        row["mismatch_complaints"] > 0
        or row["unresolved_complaints"] > 0
    ):
        payload["warnings"].append(
            "Some hotspot complaints have geographic "
            "consistency review flags."
        )

    if not payload["recommendations"]["featured"]:
        payload["warnings"].append(
            "No category meets the Featured evidence "
            "threshold. Other reported issues remain "
            "available for review."
        )

    return payload
