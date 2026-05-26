"""Global constants and paths for Soccer Analyzer."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIELDS_DIR = DATA_DIR / "fields"
SEASONS_DIR = DATA_DIR / "seasons"
PHOTOS_DIR = DATA_DIR / "photos"
DB_PATH = DATA_DIR / "roster.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
FIELDS_DIR.mkdir(exist_ok=True)
SEASONS_DIR.mkdir(exist_ok=True)
PHOTOS_DIR.mkdir(exist_ok=True)

# Device detection
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"

# Default field dimensions (U10 7v7)
DEFAULT_FIELD_LENGTH_M = 50.0
DEFAULT_FIELD_WIDTH_M = 35.0

# Detection defaults
DEFAULT_SAMPLE_RATE = 3        # every 3rd frame — keeps IoU overlap for tracking
DEFAULT_CONFIDENCE = 0.3       # lower to catch small distant players
DEFAULT_MODEL_SIZE = "s"
YOLO_IMGSZ = 1920  # inference resolution — 1920 leverages 4K input
DEFAULT_PLAYERS_PER_TEAM = 7

# Tracker
DEFAULT_TRACKER = "bytetrack"  # "bytetrack" | "botsort" | "botsort_noid"
YOLO_VERSION = "8"             # "8" | "11"

# Sprint threshold for U10 (m/s)
SPRINT_THRESHOLD = 4.0
MAX_SPEED_CAP = 10.0

# Multi-segment
TRACK_ID_OFFSET = 10000          # offset per segment to avoid ID collisions
SEGMENT_LABELS = ["Full Game", "Half 1", "Half 2", "OT 1", "OT 2", "Penalties"]

# Identification thresholds
ID_AUTO_ASSIGN_THRESHOLD = 0.80
ID_CONFIRMATION_THRESHOLD = 0.50
FACE_MATCH_THRESHOLD = 0.4
MIN_BBOX_HEIGHT_FACE = 60
MIN_GAIT_FRAMES = 60

# Fisheye / Calibration
CALIBRATION_DIR = DATA_DIR / "calibration"
CALIBRATION_DIR.mkdir(exist_ok=True)
DEFAULT_CALIBRATION_FILE = CALIBRATION_DIR / "fisheye.npz"
DEFAULT_CHECKERBOARD = (9, 6)

# Flag-based field detection
DEFAULT_FLAG_COLOR = "red"
FLAG_COLOR_RANGES = {
    "red": ((0, 80, 80), (10, 255, 255), (170, 80, 80), (180, 255, 255)),  # red wraps hue 0/180
    "orange": ((5, 80, 80), (25, 255, 255)),
    "pink": ((145, 80, 100), (175, 255, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
}
MIN_FLAG_AREA = 50

# Training / Drill settings
SESSION_TYPES = ["game", "scrimmage", "drill"]
DRILL_MIN_DURATION_S = 30
DRILL_TRANSITION_SPEED = 1.0       # m/s threshold for "idle"
DRILL_TRANSITION_MIN_S = 10        # seconds below threshold = transition
TRAINING_FIELD_LENGTH_M = 40.0
TRAINING_FIELD_WIDTH_M = 30.0

# Color palette
BG_PRIMARY = "#0d1117"
BG_CARD = "#161b22"
BORDER = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED = "#8b949e"
ACCENT_GREEN = "#00c853"
TEAM_A_BLUE = "#2196f3"
TEAM_B_RED = "#f44336"
PITCH_GRASS = "#1a2a1a"
PITCH_LINES = "#4a7a4a"

# Initial roster
INITIAL_ROSTER = [
    {"name": "Ben Adam", "jersey": 3},
    {"name": "Vince Sharma", "jersey": 5},
    {"name": "Maverick Cardoso", "jersey": 6},
    {"name": "Liam Gibala", "jersey": 7},
    {"name": "Ben Hahn", "jersey": 8},
    {"name": "Nolan Bowser", "jersey": 9},
    {"name": "Khalid Yaacoub", "jersey": 10},
    {"name": "Issa Hassoun", "jersey": 11},
    {"name": "Liam Garland", "jersey": 14},
    {"name": "Arian Rezaeian", "jersey": 15},
    {"name": "Alexander Kerr", "jersey": 16},
    {"name": "Jason Qian", "jersey": 17},
    {"name": "David Shallvari", "jersey": 18},
    {"name": "Jaedyn Duncan", "jersey": 19},
    {"name": "Luca Perrotta", "jersey": 20},
    {"name": "Gabriel Zaidan", "jersey": 21},
]

# Custom CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@300;400;500&display=swap');

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Barlow', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    color: #00c853;
    font-size: 2.5rem;
}

.metric-label {
    font-family: 'Barlow', sans-serif;
    font-weight: 300;
    color: #8b949e;
    font-size: 0.85rem;
    text-transform: uppercase;
}

div[data-testid="stMetricValue"] {
    color: #00c853;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
}

div[data-testid="stSidebar"] {
    background-color: #161b22;
}

.stButton>button {
    background-color: #00c853;
    color: #0d1117;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
    border: none;
    border-radius: 4px;
}

.stButton>button:hover {
    background-color: #00e676;
    color: #0d1117;
}

/* Make all input/widget labels white, bold, and larger */
label[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] p,
.stSelectbox label p,
.stTextInput label p,
.stDateInput label p,
.stSlider label p,
.stNumberInput label p,
.stRadio label p {
    color: #e6edf3 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* st.metric — brighter labels, green values */
div[data-testid="stMetricLabel"] label,
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Selectbox / text_input / number_input — light text on dark bg */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #e6edf3 !important;
}
div[data-baseweb="select"] > div {
    background-color: #161b22 !important;
    border-color: #30363d !important;
}
/* Dropdown menu items */
li[role="option"] {
    color: #e6edf3 !important;
    background-color: #161b22 !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #21262d !important;
}
ul[role="listbox"] {
    background-color: #161b22 !important;
}

/* Text input fields */
input[data-testid="stTextInput"],
div[data-testid="stTextInput"] input,
.stTextInput input {
    color: #e6edf3 !important;
    background-color: #161b22 !important;
    border-color: #30363d !important;
}

/* Number input fields */
div[data-testid="stNumberInput"] input {
    color: #e6edf3 !important;
    background-color: #161b22 !important;
    border-color: #30363d !important;
}

/* Date input */
div[data-testid="stDateInput"] input {
    color: #e6edf3 !important;
    background-color: #161b22 !important;
    border-color: #30363d !important;
}

/* Slider min/max and current value text */
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #8b949e !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    color: #e6edf3 !important;
}

/* Caption text — slightly brighter */
div[data-testid="stCaptionContainer"] p {
    color: #8b949e !important;
}

/* Expander header text */
div[data-testid="stExpander"] summary span {
    color: #e6edf3 !important;
}
/* Expander header — all states (hover, focus, active) */
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary:hover,
div[data-testid="stExpander"] summary:focus,
div[data-testid="stExpander"] summary:active,
div[data-testid="stExpander"] summary:hover span,
div[data-testid="stExpander"] summary:focus span,
div[data-testid="stExpander"] summary:active span,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary:hover p {
    color: #e6edf3 !important;
    background-color: #161b22 !important;
}
div[data-testid="stExpander"] summary svg {
    color: #8b949e !important;
    fill: #8b949e !important;
}
/* Expander content area — dark background */
div[data-testid="stExpander"] details {
    background-color: #161b22 !important;
    border-color: #30363d !important;
}
div[data-testid="stExpander"] details[open] > div {
    background-color: #161b22 !important;
}

/* Sidebar — full dark mode */
div[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
}
section[data-testid="stSidebar"] a[aria-current="page"] {
    background-color: #21262d !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] a:hover {
    background-color: #21262d !important;
}
/* Sidebar nav section headers */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] {
    color: #8b949e !important;
}

/* Top header bar — dark mode */
header[data-testid="stHeader"] {
    background-color: #0d1117 !important;
    border-bottom: 1px solid #21262d !important;
}
header[data-testid="stHeader"] button {
    color: #c9d1d9 !important;
}
/* Top toolbar (deploy, stop, etc) */
div[data-testid="stToolbar"] {
    color: #c9d1d9 !important;
}
div[data-testid="stToolbar"] button {
    color: #c9d1d9 !important;
}
/* Status widget (running spinner area) */
div[data-testid="stStatusWidget"] {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border-color: #30363d !important;
}
div[data-testid="stStatusWidget"] label,
div[data-testid="stStatusWidget"] button {
    color: #c9d1d9 !important;
}

/* Radio options text */
.stRadio label span,
.stRadio div[role="radiogroup"] label p {
    color: #c9d1d9 !important;
}

/* Dataframe/table text */
.stDataFrame {
    color: #e6edf3 !important;
}

/* st.info, st.success, st.warning — ensure readable text */
div[data-testid="stAlert"] p {
    color: #e6edf3 !important;
}

/* File uploader text */
div[data-testid="stFileUploader"] label p,
div[data-testid="stFileUploader"] section {
    color: #e6edf3 !important;
}
</style>
"""
