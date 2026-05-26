"""U10 Soccer Analysis App — Main Streamlit Entry Point."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import CUSTOM_CSS

# Page configuration
st.set_page_config(
    page_title="U10 Soccer Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Navigation
pages = {
    "Analysis": [
        st.Page("pages/match_analysis.py", title="Match Analysis", icon="⚽"),
        st.Page("pages/training_session.py", title="Training Session", icon="🏋️"),
        st.Page("pages/telecam.py", title="TeleCam", icon="🎥"),
        st.Page("pages/season_progress.py", title="Season Progress", icon="📈"),
    ],
    "Data": [
        st.Page("pages/playermaker.py", title="PlayerMaker 2.0", icon="📡"),
    ],
    "Setup": [
        st.Page("pages/roster_manager.py", title="Roster Manager", icon="👥"),
        st.Page("pages/field_calibration.py", title="Field Setup", icon="🗺️"),
        st.Page("pages/camera_setup.py", title="Camera Setup", icon="📷"),
    ],
}

pg = st.navigation(pages)
pg.run()
