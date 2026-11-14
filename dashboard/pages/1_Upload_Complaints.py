"""CSV upload and validation page for Urban Pulse."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.upload_preflight import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_ROWS,
    run_upload_preflight
)


TEMPLATE_PATH = (
    PROJECT_ROOT
    / "outputs/templates/new_complaints_template.csv"
)

TAXONOMY_PATH = (
    PROJECT_ROOT
    / "outputs/tables/dbscan/dbscan_action_taxonomy_v1.csv"
)


st.set_page_config(
    page_title="Upload Complaints | Urban Pulse",
    page_icon="📄",
    layout="wide"
)

st.title("Upload New Complaints")

st.caption(
    "Phase 20.2 — CSV upload validation"
)

st.info(
    "This page checks uploaded complaint data "
    "using the existing project validation contract. "
    "K-Means, DBSCAN and anomaly screening are "
    "not run during this preflight stage."
)

if not TEMPLATE_PATH.is_file():
    st.error(
        "The upload template is missing."
    )
    st.stop()

if not TAXONOMY_PATH.is_file():
    st.error(
        "The category taxonomy is missing."
    )
    st.stop()

categories = sorted({
    str(value).strip().upper()
    for value in pd.read_csv(
        TAXONOMY_PATH
    )["complaint_category"].dropna()
})

st.download_button(
    "Download CSV Template",
    data=TEMPLATE_PATH.read_bytes(),
    file_name="new_complaints_template.csv",
    mime="text/csv"
)

st.markdown(
    "Upload a CSV containing complaint IDs, "
    "timestamps, coordinates and categories."
)

st.caption(
    f"Maximum file size: "
    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
    f"Maximum complaint rows: {MAX_UPLOAD_ROWS:,}."
)

uploaded = st.file_uploader(
    "Choose a new complaints CSV",
    type=["csv"],
    accept_multiple_files=False
)

if uploaded is not None:

    st.write(
        "**Uploaded file:**",
        uploaded.name
    )

    if st.button(
        "Validate Uploaded Complaints",
        type="primary"
    ):

        try:
            result = run_upload_preflight(
                uploaded.getvalue(),
                categories
            )

        except Exception as exc:
            st.error(
                "Upload validation could not "
                f"be completed: {exc}"
            )

        else:
            st.success(
                "CSV validation finished. "
                "No models were executed."
            )

            st.metric(
                "Submitted complaint rows",
                result["row_count"]
            )

            if result["status_counts"]:

                st.subheader(
                    "Validation status breakdown"
                )

                counts = pd.DataFrame([
                    {
                        "Validation status": key,
                        "Complaints": value
                    }
                    for key, value
                    in result[
                        "status_counts"
                    ].items()
                ])

                st.dataframe(
                    counts,
                    hide_index=True,
                    use_container_width=True
                )

            st.subheader(
                "Row-level validation results"
            )

            st.dataframe(
                result["validation_results"],
                hide_index=True,
                use_container_width=True
            )

            csv_output = (
                result["validation_results"]
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "Download Validation Results",
                data=csv_output,
                file_name=(
                    "upload_validation_results.csv"
                ),
                mime="text/csv"
            )

            st.warning(
                "Validation is not geographic model "
                "processing. These complaints have not "
                "yet been assigned K-Means patterns, "
                "DBSCAN concentrations or anomaly flags."
            )

st.divider()

st.caption(
    "Urban Infrastructure Pulse Intelligence "
    "| Upload validation only | "
    "No historical models modified"
)
