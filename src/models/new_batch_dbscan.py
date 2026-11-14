"""New-batch same-category spatial concentration analysis.

This engine never retrains historical DBSCAN.
Candidates are not confirmed infrastructure defects.

Candidate IDs are local to the batch and parameters.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6371008.8

BOROUGHS = {
    "BRONX",
    "BROOKLYN",
    "MANHATTAN",
    "QUEENS",
    "STATEN ISLAND"
}

REQUIRED = {
    "source_row",
    "complaint_id",
    "latitude",
    "longitude",
    "borough",
    "complaint_category",
    "processing_status",
    "dbscan_status"
}


def analyze_new_batch_spatial(
    records,
    supported_categories,
    eps_m=125,
    min_samples=5,
    min_report_count=5,
    min_distinct_locations=2
):
    """Analyze already processed geographic complaint records.

    ZIP consistency is not required for spatial
    analysis when geographic processing has succeeded.

    Time-window selection must be done upstream.
    """

    if not isinstance(records, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame.")

    if records.columns.duplicated().any():
        raise ValueError("Duplicate input columns.")

    missing = REQUIRED - set(records.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if records["source_row"].duplicated().any():
        raise ValueError("Duplicate source rows.")

    if (
        isinstance(eps_m, bool)
        or not isinstance(eps_m, (int, float))
        or not np.isfinite(eps_m)
        or eps_m <= 0
    ):
        raise ValueError(
            "eps_m must be finite and positive."
        )

    parameters = [
        ("min_samples", min_samples, 2),
        ("min_report_count", min_report_count, 2),
        (
            "min_distinct_locations",
            min_distinct_locations,
            1
        )
    ]

    for name, value, minimum in parameters:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise ValueError(
                f"{name} must be an integer >= {minimum}."
            )

    categories = {
        str(category).strip().upper()
        for category in supported_categories
        if pd.notna(category)
        and str(category).strip()
    }

    if not categories:
        raise ValueError(
            "Supported categories are empty."
        )

    # ----------------------------------------------
    # 2. Normalize geographic inputs
    # ----------------------------------------------

    work = records.copy().reset_index(drop=True)

    for column in [
        "borough",
        "complaint_category",
        "processing_status",
        "dbscan_status"
    ]:
        work[column] = (
            work[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    work["borough"] = work[
        "borough"
    ].str.upper()

    work["complaint_category"] = work[
        "complaint_category"
    ].str.upper()

    work["lat_num"] = pd.to_numeric(
        work["latitude"],
        errors="coerce"
    )

    work["lon_num"] = pd.to_numeric(
        work["longitude"],
        errors="coerce"
    )

    lat = work["lat_num"].to_numpy(
        dtype=float,
        na_value=np.nan
    )

    lon = work["lon_num"].to_numpy(
        dtype=float,
        na_value=np.nan
    )

    invalid_coords = (
        ~np.isfinite(lat)
        | ~np.isfinite(lon)
        | (lat < -90)
        | (lat > 90)
        | (lon < -180)
        | (lon > 180)
    )

    # ----------------------------------------------
    # 3. Preserve all excluded records with reasons
    # ----------------------------------------------

    work["record_eligibility"] = "eligible"

    rules = [
        (
            work["processing_status"].ne(
                "processed"
            ),
            "not_processed"
        ),
        (
            ~work["borough"].isin(BOROUGHS),
            "invalid_borough"
        ),
        (
            pd.Series(invalid_coords),
            "invalid_coordinates"
        ),
        (
            ~work["complaint_category"].isin(
                categories
            ),
            "unsupported_category"
        )
    ]

    for mask, reason in rules:
        work.loc[
            mask
            & work["record_eligibility"].eq(
                "eligible"
            ),
            "record_eligibility"
        ] = reason

    work["raw_candidate_id"] = pd.Series(
        [None] * len(work),
        dtype="object"
    )

    work["new_batch_candidate_id"] = pd.Series(
        [None] * len(work),
        dtype="object"
    )

    work["new_batch_status"] = "excluded_input"

    eligible = work[
        "record_eligibility"
    ].eq("eligible")

    work.loc[
        eligible, "new_batch_status"
    ] = "unclustered"

    # ----------------------------------------------
    # 4. Run DBSCAN separately for each category
    # ----------------------------------------------

    groups = (
        work.loc[eligible]
        .groupby(
            "complaint_category",
            sort=True
        ).groups
    )

    for category_index, (
        category, group_indices
    ) in enumerate(groups.items()):

        indices = np.asarray(
            list(group_indices),
            dtype=int
        )

        if len(indices) < min_samples:
            continue

        coordinates = work.loc[
            indices,
            ["lat_num", "lon_num"]
        ].to_numpy(dtype=float)

        model = DBSCAN(
            eps=float(eps_m) / EARTH_RADIUS_M,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree"
        )

        labels = model.fit_predict(
            np.radians(coordinates)
        )

        for label in sorted(
            set(labels) - {-1}
        ):

            members = indices[
                labels == label
            ]

            candidate_id = (
                f"NB-{category_index:02d}"
                f"-{int(label):03d}"
            )

            work.loc[
                members,
                "raw_candidate_id"
            ] = candidate_id

    # ----------------------------------------------
    # 5. Apply independent reporting safeguards
    # ----------------------------------------------

    raw_mask = work[
        "raw_candidate_id"
    ].notna()

    raw = work.loc[raw_mask].copy()

    if len(raw):

        raw["location_pair"] = list(zip(
            raw["lat_num"],
            raw["lon_num"]
        ))

        raw_groups = (
            raw.groupby("raw_candidate_id")
            .agg(
                report_count=(
                    "source_row", "size"
                ),
                distinct_locations=(
                    "location_pair", "nunique"
                )
            )
        )

        reportable_ids = set(
            raw_groups.index[
                (
                    raw_groups["report_count"]
                    >= min_report_count
                )
                & (
                    raw_groups["distinct_locations"]
                    >= min_distinct_locations
                )
            ]
        )

    else:
        reportable_ids = set()

    # Raw clusters remain traceable even when
    # they do not meet reporting requirements.

    work.loc[
        raw_mask,
        "new_batch_status"
    ] = "below_reporting_threshold"

    reportable_mask = (
        raw_mask
        & work["raw_candidate_id"].isin(
            reportable_ids
        )
    )

    work.loc[
        reportable_mask,
        "new_batch_status"
    ] = "candidate_concentration"

    work.loc[
        reportable_mask,
        "new_batch_candidate_id"
    ] = work.loc[
        reportable_mask,
        "raw_candidate_id"
    ]

    # ----------------------------------------------
    # 6. Summarize reportable concentrations
    # ----------------------------------------------

    candidate = work.loc[
        reportable_mask
    ].copy()

    profile_columns = [
        "new_batch_candidate_id",
        "complaint_category",
        "incoming_complaints",
        "distinct_locations",
        "boroughs",
        "historical_established",
        "historical_outside",
        "historical_ambiguous",
        "center_latitude",
        "center_longitude",
        "entirely_outside_historical_clusters"
    ]

    if candidate.empty:
        profiles = pd.DataFrame(
            columns=profile_columns
        )

    else:

        candidate["location_pair"] = list(zip(
            candidate["lat_num"],
            candidate["lon_num"]
        ))

        for column, status in [
            (
                "historical_established",
                "established_cluster"
            ),
            (
                "historical_outside",
                "outside_existing_clusters"
            ),
            (
                "historical_ambiguous",
                "ambiguous_association"
            )
        ]:
            candidate[column] = (
                candidate["dbscan_status"].eq(
                    status
                )
            )

        profiles = (
            candidate.groupby(
                "new_batch_candidate_id",
                sort=True
            )
            .agg(
                complaint_category=(
                    "complaint_category",
                    "first"
                ),
                incoming_complaints=(
                    "source_row", "size"
                ),
                distinct_locations=(
                    "location_pair", "nunique"
                ),
                boroughs=(
                    "borough", "nunique"
                ),
                historical_established=(
                    "historical_established", "sum"
                ),
                historical_outside=(
                    "historical_outside", "sum"
                ),
                historical_ambiguous=(
                    "historical_ambiguous", "sum"
                ),
                center_latitude=(
                    "lat_num", "mean"
                ),
                center_longitude=(
                    "lon_num", "mean"
                )
            )
            .reset_index()
        )

        profiles[
            "entirely_outside_historical_clusters"
        ] = profiles[
            "historical_outside"
        ].eq(
            profiles["incoming_complaints"]
        )

        profiles = profiles.sort_values(
            [
                "incoming_complaints",
                "new_batch_candidate_id"
            ],
            ascending=[False, True]
        ).reset_index(drop=True)

    # ----------------------------------------------
    # 7. Account for every original record
    # ----------------------------------------------

    stats = {
        "total_records": len(work),
        "eligible_records": int(
            eligible.sum()
        ),
        "excluded_records": int(
            (~eligible).sum()
        ),
        "raw_clusters": int(
            work["raw_candidate_id"].nunique()
        ),
        "reportable_candidates": len(profiles),
        "candidate_complaints": int(
            work["new_batch_status"].eq(
                "candidate_concentration"
            ).sum()
        ),
        "below_reporting_threshold": int(
            work["new_batch_status"].eq(
                "below_reporting_threshold"
            ).sum()
        ),
        "unclustered": int(
            work["new_batch_status"].eq(
                "unclustered"
            ).sum()
        )
    }

    total_outcomes = sum([
        stats["candidate_complaints"],
        stats["below_reporting_threshold"],
        stats["unclustered"],
        stats["excluded_records"]
    ])

    if total_outcomes != len(work):
        raise AssertionError(
            "Spatial results did not reconcile."
        )

    return {
        "records": work,
        "candidate_profiles": profiles,
        "stats": stats,
        "parameters": {
            "eps_m": eps_m,
            "min_samples": min_samples,
            "min_report_count": min_report_count,
            "min_distinct_locations":
                min_distinct_locations
        }
    }
