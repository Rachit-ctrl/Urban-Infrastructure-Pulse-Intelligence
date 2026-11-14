
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "deployment"
    / "baseline_pca90_k3_pipeline_v1.joblib"
)


class KMeansReportingPredictor:
    """Predict descriptive NYC 311 reporting profiles."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Deployment bundle not found: {self.model_path}"
            )

        # Load only trusted project-generated joblib files.
        self.bundle = joblib.load(self.model_path)

        required_keys = [
            "feature_order",
            "feature_groups",
            "log_features",
            "group_scalers",
            "pca",
            "pca_components_used",
            "kmeans",
            "profile_mapping",
        ]

        missing = [
            key for key in required_keys
            if key not in self.bundle
        ]

        if missing:
            raise ValueError(
                f"Deployment bundle is missing: {missing}"
            )

        if (
            self.bundle.get("representation") != "Baseline PCA90"
            or self.bundle.get("k") != 3
        ):
            raise ValueError("Unexpected model specification.")

    def predict(self, area_features: pd.DataFrame):
        """Return cluster IDs and reporting profiles."""

        if not isinstance(area_features, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        data = area_features.copy()

        if data.empty:
            raise ValueError("Input DataFrame is empty.")

        groups = self.bundle["feature_groups"]

        required = set(self.bundle["log_features"])

        required.update(groups["persistence"])
        required.update(groups["composition"])

        # Log-transformed features are generated internally.
        required.difference_update(
            f"{name}_log" for name in self.bundle["log_features"]
        )

        missing = sorted(required - set(data.columns))

        if missing:
            raise ValueError(
                f"Missing source features: {missing}"
            )

        # Validate raw input columns.
        for column in required:
            data[column] = pd.to_numeric(
                data[column], errors="raise"
            )

            values = data[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid numerical values in {column}"
                )

        # Apply original logarithmic transformations.
        for feature in self.bundle["log_features"]:
            values = data[feature].to_numpy(dtype=float)

            if (values < 0).any():
                raise ValueError(
                    f"Negative values in {feature}"
                )

            data[f"{feature}_log"] = np.log1p(values)

        # Apply the saved fitted group scalers.
        X = pd.DataFrame(index=data.index)

        for group_name in (
            "intensity",
            "temporal",
            "persistence",
        ):
            columns = groups[group_name]
            scaler = self.bundle["group_scalers"][group_name]

            X[columns] = (
                scaler.transform(data[columns])
                / np.sqrt(len(columns))
            )

        # Validate and transform complaint composition.
        share_columns = groups["composition"]

        shares = data[share_columns].to_numpy(dtype=float)

        if (shares < -1e-10).any():
            raise ValueError("Negative complaint shares detected.")

        if not np.allclose(
            shares.sum(axis=1),
            1.0,
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError(
                "Complaint category shares must sum to 1."
            )

        X[share_columns] = np.sqrt(
            np.clip(shares, 0, None)
        )

        # Preserve exact training feature order.
        X = X[self.bundle["feature_order"]]

        # Apply saved PCA and K-Means estimators.
        X_pca = self.bundle["pca"].transform(X)

        X_pca = X_pca[
            :, :self.bundle["pca_components_used"]
        ]

        labels = self.bundle["kmeans"].predict(X_pca)

        result = pd.DataFrame(
            {
                "cluster_id": labels.astype(int),
                "reporting_profile": [
                    self.bundle["profile_mapping"][int(label)]
                    for label in labels
                ],
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
            },
            index=data.index,
        )

        if "area_id" in data.columns:
            result.insert(
                0, "area_id", data["area_id"].to_numpy()
            )

        return result
