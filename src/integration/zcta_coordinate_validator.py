from pathlib import Path
from numbers import Real
from shapely.ops import unary_union
from shapely.geometry import Point
import geopandas as gpd
import math


class ZCTACoordinateValidator:
    """Check supplied ZIP against mapped ZCTA geography."""

    SCHEMA_VERSION = "zcta_coordinate_consistency_v1"
    REFERENCE_VERSION = "nyc_zcta_35j5-n34v_snapshot_2026_v1"

    def __init__(
        self,
        zcta_path,
        zip_normalizer,
        boundary_tolerance_m=10.0
    ):
        self.zip_normalizer = zip_normalizer

        if (
            isinstance(boundary_tolerance_m, bool)
            or not isinstance(boundary_tolerance_m, Real)
            or not math.isfinite(float(boundary_tolerance_m))
            or boundary_tolerance_m < 0
        ):
            raise ValueError("Invalid boundary tolerance.")

        self.boundary_tolerance_m = float(
            boundary_tolerance_m
        )

        polygons = gpd.read_file(Path(zcta_path))

        if "zcta5" not in polygons.columns:
            raise ValueError("Missing zcta5 identifier.")

        if polygons.crs is None:
            raise ValueError("Missing geographic CRS.")

        if polygons.empty:
            raise ValueError("Empty ZCTA reference.")

        if (
            polygons.geometry.isna().any()
            or polygons.geometry.is_empty.any()
            or not polygons.geometry.is_valid.all()
        ):
            raise ValueError("Invalid ZCTA geometries.")

        polygons = polygons.to_crs("EPSG:4326")

        polygons["zcta5"] = (
            polygons["zcta5"]
            .astype("string")
            .str.strip()
        )

        valid = (
            polygons["zcta5"]
            .str.fullmatch(r"[0-9]{5}")
            .fillna(False)
        )

        if not valid.all():
            raise ValueError(
                "ZCTA reference contains invalid ZIP tokens."
            )

        self.polygon_count = len(polygons)

        # Work in meters for boundary proximity checks.
        metric = polygons[
            ["zcta5", "geometry"]
        ].to_crs("EPSG:32618")

        # Merge features sharing the same ZCTA code.
        self.geometry_by_zip = {}

        for zip_code, group in metric.groupby("zcta5"):
            self.geometry_by_zip[str(zip_code)] = (
                unary_union(group.geometry.tolist())
            )

        self.reference_zips = set(
            self.geometry_by_zip
        )

    @staticmethod
    def _coordinate(name, value, lower, upper):

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise TypeError(f"{name} must be numeric.")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if not lower <= value <= upper:
            raise ValueError(
                f"{name} outside coordinate bounds."
            )

        return value

    def evaluate(
        self,
        latitude,
        longitude,
        incident_zip=None
    ):

        lat = self._coordinate(
            "latitude", latitude, -90, 90
        )

        lon = self._coordinate(
            "longitude", longitude, -180, 180
        )

        if incident_zip is None:
            return {
                "schema_version": self.SCHEMA_VERSION,
                "status": "zip_not_provided",
                "incident_zip": None,
                "coordinate_zctas": [],
                "reference_has_zip": False,
                "boundary_distance_m": None,
                "zcta_polygon_consistent": False,
                "postal_zip_verified": False,
                "reference_version": self.REFERENCE_VERSION,
                "warnings": [
                    "No ZIP was supplied for geographic comparison."
                ]
            }

        zip_code = self.zip_normalizer(
            incident_zip
        )

        point = Point(lon, lat)

        point_m = (
            gpd.GeoSeries(
                [point], crs="EPSG:4326"
            )
            .to_crs("EPSG:32618")
            .iloc[0]
        )

        matched = sorted(
            code
            for code, geom in self.geometry_by_zip.items()
            if geom.intersects(point_m)
        )

        reference_has_zip = (
            zip_code in self.reference_zips
        )

        distance = None
        status = "zcta_unavailable"

        if reference_has_zip:

            expected_geometry = (
                self.geometry_by_zip[zip_code]
            )

            distance = float(
                point_m.distance(
                    expected_geometry.boundary
                )
            )

            # Check boundaries of all intersecting ZCTAs.
            boundary_distances = [distance]

            for code in matched:
                boundary_distances.append(
                    float(
                        point_m.distance(
                            self.geometry_by_zip[
                                code
                            ].boundary
                        )
                    )
                )

            near_boundary = (
                min(boundary_distances)
                <= self.boundary_tolerance_m
            )

            if len(matched) > 1 or near_boundary:
                status = "boundary_review_required"

            elif zip_code in matched:
                status = "zcta_consistent"

            elif matched:
                status = "zcta_mismatch"

        warnings = []

        if status == "zcta_consistent":
            warnings.append(
                "Coordinates are consistent with the "
                "supplied ZIP's mapped ZCTA, but this "
                "is not USPS postal verification."
            )

        elif status == "zcta_mismatch":
            warnings.append(
                "Coordinates intersect a different "
                "mapped ZCTA. Review the supplied ZIP; "
                "ZCTAs are not exact postal delivery areas."
            )

        elif status == "boundary_review_required":
            warnings.append(
                "Coordinate is near a ZCTA boundary "
                "or intersects multiple ZCTAs. "
                "Geographic review is required."
            )

        else:
            warnings.append(
                "ZCTA reference does not provide "
                "sufficient geographic evidence for "
                "this supplied ZIP and location."
            )

        return {
            "schema_version": self.SCHEMA_VERSION,
            "status": status,
            "incident_zip": zip_code,
            "coordinate_zctas": matched,
            "reference_has_zip": reference_has_zip,
            "boundary_distance_m": (
                round(distance, 3)
                if distance is not None
                else None
            ),
            "zcta_polygon_consistent": (
                status == "zcta_consistent"
            ),
            "postal_zip_verified": False,
            "reference_version": self.REFERENCE_VERSION,
            "warnings": warnings
        }
