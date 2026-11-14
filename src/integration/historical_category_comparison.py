
import numpy as np
import pandas as pd


def compare_historical_current_categories(
    backend_records,
    historical_features,
    reporting_day,
    expected_analyzed=None,
):
    """Compare verified-day complaint mix with historical area mix.

    Returns descriptive comparisons, not anomaly decisions.
    Does not train, update, or save any ML model.
    """

    required = {
        "created_date",
        "processing_status",
        "zip_validation_status",
        "borough",
        "incident_zip",
        "complaint_category",
    }

    missing = required - set(backend_records.columns)

    if missing:
        raise ValueError(
            f"Missing backend columns: {sorted(missing)}"
        )

    records = backend_records.copy(deep=True)
    history = historical_features.copy(deep=True)

    # Use the same verified reporting-day rule
    # used in Phase 21.1.

    selected = records.loc[
        records["processing_status"]
        .astype("string").str.strip().eq("processed")
        &
        records["zip_validation_status"]
        .astype("string").str.strip().eq("zcta_consistent")
        &
        records["created_date"]
        .astype("string").str.slice(0, 10)
        .eq(str(reporting_day))
    ].copy()

    if expected_analyzed is not None:
        if len(selected) != int(expected_analyzed):
            raise ValueError(
                "Historical comparison selected "
                f"{len(selected)} records, but the "
                "main pipeline selected "
                f"{expected_analyzed}. "
                "Check eligibility filters."
            )

    if selected.empty:
        return {
            "status": "no_verified_records",
            "eligible_reports": 0,
            "matched_reports": 0,
            "unmatched_areas": [],
            "categories_without_history": [],
            "summary": pd.DataFrame(),
            "details": pd.DataFrame(),
        }

    # Reconstruct the same borough-ZIP area ID.

    zip_code = (
        selected["incident_zip"]
        .astype("string").str.strip()
    )

    if not zip_code.str.fullmatch(
        r"\d{5}"
    ).fillna(False).all():
        raise ValueError(
            "Verified records contain invalid ZIP codes."
        )

    selected["area_id"] = (
        selected["borough"]
        .astype("string").str.strip().str.upper()
        + "_" + zip_code
    )

    selected["complaint_category"] = (
        selected["complaint_category"]
        .astype("string").str.strip().str.upper()
    )

    if selected[
        "complaint_category"
    ].isna().any():
        raise ValueError("Missing complaint categories.")

    if "area_id" not in history.columns:
        raise ValueError("Missing historical area_id.")

    if history["area_id"].duplicated().any():
        raise ValueError("Duplicate historical areas.")

    history["area_id"] = (
        history["area_id"]
        .astype("string").str.strip().str.upper()
    )

    share_cols = sorted(
        col for col in history.columns
        if col.startswith("share_")
    )

    if not share_cols:
        raise ValueError(
            "No historical category-share features."
        )

    history[share_cols] = (
        history[share_cols].apply(
            pd.to_numeric, errors="raise"
        )
    )

    values = history[
        share_cols
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(values).all()
        or not ((values >= 0) & (values <= 1)).all()
    ):
        raise ValueError("Invalid historical shares.")

    if not np.allclose(
        history[share_cols].sum(axis=1),
        1.0,
        atol=0.01
    ):
        raise ValueError(
            "Historical category shares do not sum to 1."
        )

    # Compare only areas having a historical baseline.

    available = set(history["area_id"])

    unmatched = sorted(
        set(selected["area_id"]) - available
    )

    matched = selected.loc[
        selected["area_id"].isin(available)
    ].copy()

    if matched.empty:
        return {
            "status": "no_historical_area_overlap",
            "eligible_reports": int(len(selected)),
            "matched_reports": 0,
            "unmatched_areas": unmatched,
            "categories_without_history": [],
            "summary": pd.DataFrame(),
            "details": pd.DataFrame(),
        }

    area_counts = matched.groupby(
        "area_id"
    ).size()

    area_ids = sorted(area_counts.index)

    mapping = {
        col: col.removeprefix("share_")
        .replace("_", " ").upper()
        for col in share_cols
    }

    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            "Duplicate normalized historical categories."
        )

    historical_categories = set(mapping.values())

    current_categories = set(
        matched["complaint_category"].unique()
    )

    # A category missing from the historical
    # feature set is represented as zero share,
    # and separately disclosed in the diagnostics.

    categories_without_history = sorted(
        current_categories - historical_categories
    )

    categories = sorted(
        historical_categories | current_categories
    )

    history_index = (
        history.set_index("area_id").loc[area_ids]
    )

    historical_shares = pd.DataFrame(
        {
            mapping[col]:
                history_index[col].to_numpy(dtype=float)
            for col in share_cols
        },
        index=area_ids
    ).reindex(
        columns=categories,
        fill_value=0.0
    )

    counts = pd.crosstab(
        matched["area_id"],
        matched["complaint_category"]
    ).reindex(
        index=area_ids,
        columns=categories,
        fill_value=0
    )

    current_shares = counts.div(
        counts.sum(axis=1),
        axis=0
    )

    difference = (
        current_shares - historical_shares
    )

    mix_distance = (
        difference.abs().sum(axis=1) / 2
    )

    summary_rows = []
    detail_rows = []

    for area_id in area_ids:

        old_top = historical_shares.loc[
            area_id
        ].idxmax()

        new_top = current_shares.loc[
            area_id
        ].idxmax()

        increase = difference.loc[
            area_id
        ].idxmax()

        summary_rows.append({
            "area_id": area_id,
            "incoming_reports": int(
                area_counts.loc[area_id]
            ),
            "historical_top_category": old_top,
            "historical_top_share_pct": round(
                100 * historical_shares.loc[
                    area_id, old_top
                ], 2
            ),
            "current_top_category": new_top,
            "current_top_share_pct": round(
                100 * current_shares.loc[
                    area_id, new_top
                ], 2
            ),
            "top_category_changed": old_top != new_top,
            "largest_share_increase_category": increase,
            "largest_share_increase_pp": round(
                100 * float(
                    difference.loc[area_id, increase]
                ), 2
            ),
            "category_mix_distance": round(
                float(mix_distance.loc[area_id]), 4
            ),
        })

        for category in categories:

            detail_rows.append({
                "area_id": area_id,
                "complaint_category": category,
                "historical_share_pct": round(
                    100 * historical_shares.loc[
                        area_id, category
                    ], 3
                ),
                "current_share_pct": round(
                    100 * current_shares.loc[
                        area_id, category
                    ], 3
                ),
                "share_change_pp": round(
                    100 * float(
                        difference.loc[
                            area_id, category
                        ]
                    ), 3
                ),
                "current_reports": int(
                    counts.loc[area_id, category]
                ),
            })

    return {
        "status": "compared",
        "eligible_reports": int(len(selected)),
        "matched_reports": int(len(matched)),
        "unmatched_areas": unmatched,
        "categories_without_history":
            categories_without_history,
        "summary": pd.DataFrame(summary_rows),
        "details": pd.DataFrame(detail_rows),
    }
