from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def required_artifacts():
    root = PROJECT_ROOT

    return {
        "DBSCAN core lookup": (
            root / "models/dbscan/"
            "primary_f_core_lookup_v1.joblib"
        ),
        "Hotspot tables": (
            root / "outputs/tables/dbscan"
        ),
        "NYC borough boundaries": (
            root / "models/geography/"
            "nyc_borough_boundaries_gthc-hcne_v1.geojson"
        ),
        "Action taxonomy": (
            root / "outputs/tables/dbscan/"
            "dbscan_action_taxonomy_v1.csv"
        ),
        "K-Means area profiles": (
            root / "outputs/tables/kmeans/"
            "kmeans_full_data_k3_descriptive_profiles.csv"
        ),
        "NYC ZCTA reference": (
            root / "models/geography/"
            "nyc_zcta_35j5-n34v_snapshot_2026_v1.geojson"
        ),
    }


def verify_artifacts():
    paths = required_artifacts()

    missing = [
        name for name, path in paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing application artifacts: "
            + ", ".join(missing)
        )

    return paths


@lru_cache(maxsize=1)
def get_gui_adapter():

    paths = verify_artifacts()

    from src.models.dbscan_predictor import (
        DBSCANSpatialPredictor
    )

    from src.recommendations.hotspot_recommender import (
        HotspotRecommendationEngine
    )

    from src.recommendations.hotspot_gui_contract import (
        build_hotspot_gui_payload
    )

    from src.integration.new_complaint_insight import (
        NewComplaintInsightService
    )

    from src.integration.validated_new_complaint import (
        ValidatedNewComplaintService
    )

    from src.integration.kmeans_area_lookup import (
        KMeansAreaLookup
    )

    from src.integration.combined_infrastructure_insight import (
        CombinedInfrastructureInsightService
    )

    from src.integration.zcta_coordinate_validator import (
        ZCTACoordinateValidator
    )

    from src.integration.zcta_aware_combined_insight import (
        ZCTAAwareCombinedInsightService
    )

    from src.integration.infrastructure_gui_contract import (
        InfrastructureGUIAdapter
    )

    predictor = DBSCANSpatialPredictor(
        paths["DBSCAN core lookup"]
    )

    recommender = HotspotRecommendationEngine(
        paths["Hotspot tables"]
    )

    complaint_service = NewComplaintInsightService(
        predictor,
        recommender,
        build_hotspot_gui_payload
    )

    validated_service = ValidatedNewComplaintService(
        complaint_service,
        paths["NYC borough boundaries"],
        paths["Action taxonomy"]
    )

    kmeans_lookup = KMeansAreaLookup(
        paths["K-Means area profiles"]
    )

    combined_service = CombinedInfrastructureInsightService(
        validated_service,
        kmeans_lookup
    )

    zcta_validator = ZCTACoordinateValidator(
        paths["NYC ZCTA reference"],
        kmeans_lookup.normalize_zip,
        boundary_tolerance_m=10.0
    )

    zcta_aware_service = ZCTAAwareCombinedInsightService(
        combined_service,
        zcta_validator
    )

    return InfrastructureGUIAdapter(
        zcta_aware_service
    )
