from pathlib import Path
from numbers import Real

import joblib
import numpy as np
from sklearn.neighbors import BallTree


class DBSCANSpatialPredictor:
    """Associate coordinates with historical DBSCAN core clusters."""

    def __init__(self, artifact_path):

        artifact = joblib.load(Path(artifact_path))

        if artifact["model_version"] != "dbscan_f_75m_min40_v1":
            raise ValueError("Unsupported DBSCAN artifact version.")

        if artifact["coordinate_order"] != [
            "latitude", "longitude"
        ]:
            raise ValueError("Unexpected coordinate order.")

        if artifact["metric"] != "haversine":
            raise ValueError("Unexpected spatial distance metric.")

        self.model_version = artifact["model_version"]
        self.eps_meters = float(artifact["eps_meters"])
        self.earth_radius_km = float(
            artifact["earth_radius_km"]
        )

        self.core_coordinates = np.asarray(
            artifact["core_coordinates_radians"],
            dtype=np.float64
        )

        self.core_labels = np.asarray(
            artifact["core_cluster_labels"],
            dtype=np.int64
        )

        if self.core_coordinates.shape != (40977, 2):
            raise ValueError("Unexpected core coordinate shape.")

        if len(self.core_labels) != 40977:
            raise ValueError("Unexpected core label count.")

        if np.any(self.core_labels == -1):
            raise ValueError("Core points cannot have noise labels.")

        if len(np.unique(self.core_labels)) != 2846:
            raise ValueError("Unexpected historical cluster count.")

        if not np.isfinite(self.core_coordinates).all():
            raise ValueError("Nonfinite core coordinates.")

        self.eps_radians = (
            self.eps_meters
            / (self.earth_radius_km * 1000)
        )

        self.tree = BallTree(
            self.core_coordinates,
            metric="haversine"
        )

    @staticmethod
    def _validate_coordinates(latitude, longitude):

        for name, value, minimum, maximum in [
            ("latitude", latitude, -90, 90),
            ("longitude", longitude, -180, 180)
        ]:

            if isinstance(value, (bool, np.bool_)) or (
                not isinstance(value, Real)
            ):
                raise TypeError(
                    f"{name} must be a real numeric value."
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} is outside valid coordinate bounds."
                )

    def associate(self, latitude, longitude):
        """Find historical core clusters within the saved radius."""

        self._validate_coordinates(latitude, longitude)

        query = np.radians(
            np.array(
                [[float(latitude), float(longitude)]],
                dtype=np.float64
            )
        )

        indices, distances = self.tree.query_radius(
            query,
            r=self.eps_radians,
            return_distance=True,
            sort_results=True
        )

        neighbor_indices = indices[0]
        neighbor_distances_m = (
            distances[0] * self.earth_radius_km * 1000
        )

        nearest_distance_rad = self.tree.query(
            query, k=1
        )[0][0, 0]

        nearest_core_distance_m = round(
            float(
                nearest_distance_rad
                * self.earth_radius_km
                * 1000
            ),
            3
        )

        base = {
            "model_version": self.model_version,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "search_radius_m": self.eps_meters,
            "nearest_core_distance_m": nearest_core_distance_m,
            "matching_core_points": int(
                len(neighbor_indices)
            ),
            "historical_period": "2023-2025",
        }

        if len(neighbor_indices) == 0:
            return {
                **base,
                "status": "outside_existing_clusters",
                "cluster_id": None,
                "candidate_clusters": [],
                "message": (
                    "No established historical DBSCAN core "
                    "location found within 75 meters. "
                    "This does not indicate low risk."
                )
            }

        matches = self.core_labels[neighbor_indices]

        candidates = []

        for cluster_id in np.unique(matches):

            mask = matches == cluster_id

            candidates.append({
                "cluster_id": int(cluster_id),
                "core_points_within_radius": int(
                    mask.sum()
                ),
                "nearest_core_distance_m": round(
                    float(neighbor_distances_m[mask].min()),
                    3
                )
            })

        candidates.sort(
            key=lambda x: (
                x["nearest_core_distance_m"],
                x["cluster_id"]
            )
        )

        if len(candidates) == 1:
            return {
                **base,
                "status": "established_cluster",
                "cluster_id": candidates[0]["cluster_id"],
                "candidate_clusters": candidates,
                "message": (
                    "Location is associated with one "
                    "established historical reporting cluster."
                )
            }

        return {
            **base,
            "status": "ambiguous_association",
            "cluster_id": None,
            "candidate_clusters": candidates,
            "message": (
                "The location is within 75 meters of core "
                "locations belonging to multiple historical "
                "clusters. No unique cluster is assigned."
            )
        }
