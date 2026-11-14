"""Urban Infrastructure Pulse — primary application.

Launch:
python -m streamlit run dashboard/urban_pulse_app.py

Investigation recommendations are the default page.
"""

from pathlib import Path

import streamlit as st


HERE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Urban Infrastructure Pulse",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "Investigation Intelligence": [
        st.Page(
            str(
                HERE / "pages"
                / "3_Investigation_Recommendations.py"
            ),
            title="Investigation Recommendations",
            icon="🔎",
            default=True
        ),
        st.Page(
            str(
                HERE / "pages"
                / "4_Model_Explorer.py"
            ),
            title="Model Explorer",
            icon="📊"
        )
    ],

    "Complaint Processing": [
        st.Page(
            str(
                HERE / "pages"
                / "1_Upload_Complaints.py"
            ),
            title="Validate CSV Upload",
            icon="📄"
        ),
        st.Page(
            str(
                HERE / "pages"
                / "2_Process_New_Batch.py"
            ),
            title="Process New Batch",
            icon="⚙️"
        )
    ]
}

selected_page = st.navigation(
    pages,
    position="sidebar"
)

selected_page.run()
