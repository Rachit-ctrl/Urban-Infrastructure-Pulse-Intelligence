from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

MASTER_DATA_FILE = (
    PROCESSED_DATA_DIR
    /"nyc311_infrastructure_cleaned.csv"
)

QUALITY_REPORT_FILE = (
    REFERENCE_DATA_DIR
    / "nyc311_data_quality_report.csv"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

EDA_FIGURES_DIR = FIGURES_DIR / "eda"
EDA_TABLES_DIR = TABLES_DIR / "eda"


# ============================================================
# MODEL DIRECTORIES
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"

KMEANS_DIR = MODELS_DIR / "kmeans"
PCA_DIR = MODELS_DIR / "pca"
DBSCAN_DIR = MODELS_DIR / "dbscan"


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_NAME = "Urban Infrastructure Pulse Intelligence"

EXPECTED_RECORDS = 685_284

STUDY_START_DATE = "2023-01-01"
STUDY_END_DATE = "2025-12-31"

RANDOM_STATE = 42


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_project_directories():

    directories = [
        EDA_FIGURES_DIR,
        EDA_TABLES_DIR,
        REPORTS_DIR,
        KMEANS_DIR,
        PCA_DIR,
        DBSCAN_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


if __name__ == "__main__":

    create_project_directories()

    print("Project configuration initialized.")
    print("Project root:", PROJECT_ROOT)
    print("Master dataset:", MASTER_DATA_FILE)