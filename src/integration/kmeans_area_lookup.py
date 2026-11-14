from pathlib import Path
from numbers import Integral
import pandas as pd


class KMeansAreaLookup:
    """Retrieve historical borough-ZIP reporting profiles."""

    MODEL_VERSION = "baseline_pca90_k3_v1"
    HISTORICAL_PERIOD = "2023-2025"

    VALID_BOROUGHS = {
        "BRONX",
        "BROOKLYN",
        "MANHATTAN",
        "QUEENS",
        "STATEN ISLAND",
    }

    def __init__(self, profile_path):

        profiles = pd.read_csv(Path(profile_path))

        required = {
            "area_id",
            "k2_cluster",
            "k3_cluster",
            "reporting_profile",
        }

        if not required.issubset(profiles.columns):
            raise ValueError(
                "K-Means profile columns are missing."
            )

        if len(profiles) != 238:
            raise ValueError(
                "Unexpected K-Means area population."
            )

        if profiles["area_id"].duplicated().any():
            raise ValueError("Duplicate area identifiers.")

        if profiles[list(required)].isna().any().any():
            raise ValueError("Missing K-Means profile values.")

        if not profiles["reporting_profile"].astype(
            str
        ).str.strip().ne("").all():
            raise ValueError("Empty reporting-profile label.")

        if set(profiles["k3_cluster"]) != {0, 1, 2}:
            raise ValueError(
                "Unexpected primary K-Means labels."
            )

        self.records = {}

        for row in profiles.itertuples(index=False):

            area_id = str(row.area_id).strip()

            if "_" not in area_id:
                raise ValueError(
                    f"Invalid area identifier: {area_id}"
                )

            borough_part, zip_part = area_id.rsplit(
                "_", 1
            )

            borough = self.normalize_borough(
                borough_part
            )

            if area_id == "MANHATTAN_83":
                # Preserve the original historical identifier.
                # Do not convert 83 into an invented ZIP.
                zip_code = zip_part
            else:
                zip_code = self.normalize_zip(zip_part)

            key = (borough, zip_code)

            if key in self.records:
                raise ValueError(
                    "Duplicate normalized borough-ZIP area."
                )

            self.records[key] = {
                "area_id": area_id,
                "k2_cluster": int(row.k2_cluster),
                "k3_cluster": int(row.k3_cluster),
                "reporting_profile": str(
                    row.reporting_profile
                ).strip(),
            }

        if len(self.records) != 238:
            raise ValueError(
                "Historical area lookup is incomplete."
            )

    @classmethod
    def normalize_borough(cls, borough):

        if not isinstance(borough, str):
            raise TypeError(
                "Borough must be a string."
            )

        normalized = " ".join(
            borough.strip()
            .replace("_", " ")
            .upper()
            .split()
        )

        if normalized not in cls.VALID_BOROUGHS:
            raise ValueError(
                "Unsupported NYC borough: "
                + repr(borough)
            )

        return normalized

    @staticmethod
    def normalize_zip(zip_code):

        if isinstance(zip_code, bool):
            raise TypeError(
                "ZIP code cannot be boolean."
            )

        if isinstance(zip_code, Integral):
            zip_code = str(int(zip_code))

        elif isinstance(zip_code, str):
            zip_code = zip_code.strip()

        else:
            raise TypeError(
                "ZIP code must be a five-digit string "
                "or an integer."
            )

        if (
            len(zip_code) != 5
            or not zip_code.isascii()
            or not zip_code.isdigit()
        ):
            raise ValueError(
                "ZIP code must contain exactly five digits."
            )

        return zip_code

    def lookup(self, borough, incident_zip):
        """Return a stored historical area classification."""

        normalized_borough = self.normalize_borough(
            borough
        )

        normalized_zip = self.normalize_zip(
            incident_zip
        )

        key = (
            normalized_borough,
            normalized_zip
        )

        base = {
            "borough": normalized_borough,
            "incident_zip": normalized_zip,
            "model_version": self.MODEL_VERSION,
            "historical_period": self.HISTORICAL_PERIOD,
            "lookup_basis": "borough_and_supplied_zip",
        }

        if key not in self.records:

            return {
                **base,
                "status": "area_not_in_training",
                "area_id": None,
                "k3_cluster": None,
                "k2_cluster": None,
                "reporting_profile": None,
                "message": (
                    "No historical K-Means profile exists "
                    "for this borough-ZIP combination."
                ),
                "warnings": [
                    "No K-Means reporting pattern is "
                    "assigned to this area."
                ],
            }

        record = self.records[key]

        return {
            **base,
            "status": "historical_area_match",
            **record,
            "message": (
                "Historical K-Means reporting pattern "
                "retrieved successfully."
            ),
            "warnings": [
                "The supplied ZIP code has not been "
                "independently verified against "
                "the input coordinates.",
                "This classification describes "
                "aggregated historical area reporting, "
                "not the condition or risk of an "
                "individual infrastructure complaint.",
            ],
        }

    def lookup_by_area_id(self, area_id):
        """Retrieve an exact saved historical area identifier."""

        if not isinstance(area_id, str):
            raise TypeError("Area ID must be a string.")

        normalized_id = area_id.strip().upper()

        matching = [
            record
            for record in self.records.values()
            if record["area_id"] == normalized_id
        ]

        if len(matching) != 1:
            raise KeyError(
                "Unknown historical area ID: "
                + repr(normalized_id)
            )

        record = matching[0]

        # Exceptional historical identifier
        if normalized_id == "MANHATTAN_83":

            return {
                "status": "historical_area_match_nonstandard_zip",
                "area_id": record["area_id"],
                "borough": "MANHATTAN",
                "incident_zip": None,
                "historical_zip_token": "83",
                "k2_cluster": record["k2_cluster"],
                "k3_cluster": record["k3_cluster"],
                "reporting_profile": record["reporting_profile"],
                "model_version": self.MODEL_VERSION,
                "historical_period": self.HISTORICAL_PERIOD,
                "lookup_basis": "exact_historical_area_id",
                "message": (
                    "Existing historical area profile retrieved."
                ),
                "warnings": [
                    "This historical area identifier contains "
                    "a nonstandard ZIP token (83).",
                    "The token has not been validated as a "
                    "five-digit postal ZIP code.",
                    "Do not automatically infer this area "
                    "from an individual complaint's ZIP.",
                    "The profile describes historical reporting "
                    "patterns, not infrastructure risk."
                ]
            }

        # Standard saved historical area
        borough, zip_code = normalized_id.rsplit("_", 1)

        result = self.lookup(borough, zip_code)
        result["lookup_basis"] = "exact_historical_area_id"

        return result
