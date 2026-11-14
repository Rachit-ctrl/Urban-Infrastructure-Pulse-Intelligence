from pathlib import Path
from numbers import Real
import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


class ValidatedNewComplaintService:
    """Validate new inputs before historical spatial lookup."""

    SCHEMA_VERSION = "validated_new_complaint_v1"

    EXPECTED_BOROUGHS = {
        "BRONX", "BROOKLYN", "MANHATTAN",
        "QUEENS", "STATEN ISLAND"
    }

    def __init__(
        self,
        insight_service,
        boundary_path,
        taxonomy_path,
        boundary_tolerance_m=10.0
    ):
        self.insight_service = insight_service

        if not isinstance(boundary_tolerance_m, Real):
            raise TypeError("Invalid boundary tolerance.")

        if not math.isfinite(float(boundary_tolerance_m)):
            raise ValueError("Boundary tolerance must be finite.")

        if boundary_tolerance_m < 0:
            raise ValueError("Boundary tolerance cannot be negative.")

        self.boundary_tolerance_m = float(
            boundary_tolerance_m
        )

        # Load official NYC borough polygons
        self.boroughs = gpd.read_file(
            Path(boundary_path)
        )[["boroname", "geometry"]].copy()

        if self.boroughs.crs is None:
            raise ValueError("Borough CRS is missing.")

        self.boroughs = self.boroughs.to_crs(
            "EPSG:4326"
        )

        self.boroughs["boroname"] = (
            self.boroughs["boroname"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if (
            len(self.boroughs) != 5
            or set(self.boroughs["boroname"])
            != self.EXPECTED_BOROUGHS
        ):
            raise ValueError("Unexpected borough reference.")

        if (
            not self.boroughs.geometry.is_valid.all()
            or self.boroughs.geometry.is_empty.any()
        ):
            raise ValueError("Invalid borough geometry.")

        # Metric polygons for distance-to-boundary checks
        self.boroughs_metric = (
            self.boroughs.to_crs("EPSG:32618")
        )

        # Load supported complaint categories
        taxonomy = pd.read_csv(
            Path(taxonomy_path)
        )

        self.supported_categories = set(
            taxonomy["complaint_category"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if len(self.supported_categories) != 15:
            raise ValueError("Unexpected category taxonomy.")

    @staticmethod
    def _validate_number(name, value, minimum, maximum):

        if isinstance(value, bool) or not isinstance(
            value, Real
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} is outside coordinate bounds."
            )

        return value

    def evaluate(self, latitude, longitude, complaint_category):

        lat = self._validate_number(
            "latitude", latitude, -90, 90
        )

        lon = self._validate_number(
            "longitude", longitude, -180, 180
        )

        if not isinstance(complaint_category, str):
            raise TypeError(
                "Complaint category must be a string."
            )

        category = complaint_category.strip().upper()

        if category not in self.supported_categories:
            raise ValueError(
                "Unsupported infrastructure complaint category: "
                + repr(category)
            )

        # Shapely points use longitude, latitude.
        point = Point(lon, lat)

        matched = self.boroughs.loc[
            self.boroughs.geometry.intersects(point),
            "boroname"
        ].tolist()

        matched = sorted(set(matched))

        geographic_status = "within_borough"
        borough = None
        boundary_distance_m = None

        if len(matched) == 0:
            geographic_status = "outside_borough_polygons"

        elif len(matched) > 1:
            geographic_status = "multiple_boroughs"

        else:
            borough = matched[0]

            # Measure proximity to the matched borough edge.
            point_metric = gpd.GeoSeries(
                [point], crs="EPSG:4326"
            ).to_crs("EPSG:32618").iloc[0]

            metric_geometry = (
                self.boroughs_metric.loc[
                    self.boroughs_metric["boroname"]
                    == borough,
                    "geometry"
                ].iloc[0]
            )

            boundary_distance_m = float(
                point_metric.distance(
                    metric_geometry.boundary
                )
            )

            if (
                boundary_distance_m
                <= self.boundary_tolerance_m
            ):
                geographic_status = "near_borough_boundary"

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "status": "geographic_review_required",
            "complaint_category": category,
            "geographic_validation": {
                "status": geographic_status,
                "coordinate_borough": borough,
                "candidate_boroughs": matched,
                "boundary_distance_m": (
                    round(boundary_distance_m, 3)
                    if boundary_distance_m is not None
                    else None
                )
            },
            "insight": None,
            "warnings": []
        }

        if geographic_status != "within_borough":

            payload["warnings"].append(
                "Geographic review required. The location "
                "may be outside mapped land boundaries, "
                "on a borough boundary, or within the "
                "configured boundary review tolerance."
            )

            payload["warnings"].append(
                "No automatic historical hotspot "
                "association was made. This is not "
                "a low-risk or invalid-complaint judgment."
            )

            return payload

        # Validated inputs: call existing integrated service
        insight = self.insight_service.evaluate(
            lat, lon, category
        )

        payload["status"] = insight["status"]
        payload["insight"] = insight

        payload["warnings"].append(
            "Coordinate-in-borough validation does not "
            "establish physical infrastructure condition."
        )

        return payload
