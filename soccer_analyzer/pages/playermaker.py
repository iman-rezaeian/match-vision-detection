"""PlayerMaker — Import sensor data from screenshots."""

import streamlit as st
import numpy as np
import re
import json
import os
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CUSTOM_CSS, DATA_DIR
from database.roster_db import RosterDB
from database.match_db import MatchDB

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

# --- Page setup ---
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("📱 PlayerMaker Import")

roster_db = RosterDB()
match_db = MatchDB()

# --- Sidebar: select match + player ---
players = roster_db.get_all_players()
matches = match_db.get_all_matches()

if not players:
    st.warning("No players in roster. Add players in **Roster Manager** first.")
    st.stop()

# Player selector
player_options = {f"#{p['jersey_number']} {p['name']}": p for p in players}
selected_player_label = st.selectbox(
    "Player", list(player_options.keys()),
    help="Select the player whose PlayerMaker data to import"
)
selected_player = player_options[selected_player_label]

# Match selector
if matches:
    match_options = {
        f"{m['date']} vs {m['opponent']} ({m.get('result', '?')})": m
        for m in matches
    }
    col_match, col_new = st.columns([3, 1])
    with col_match:
        selected_match_label = st.selectbox(
            "Match", list(match_options.keys()),
            help="Select the match to attach sensor data to"
        )
        selected_match = match_options[selected_match_label]
    with col_new:
        st.markdown("<br>", unsafe_allow_html=True)
        create_new_match = st.checkbox("New match")
else:
    create_new_match = True

if create_new_match or not matches:
    st.subheader("New Match")
    col_d, col_o, col_r = st.columns(3)
    with col_d:
        new_date = st.date_input("Date")
    with col_o:
        new_opponent = st.text_input("Opponent")
    with col_r:
        new_result = st.text_input("Result", placeholder="W 3-1")
    selected_match = None  # will be created on save

st.divider()

# --- Screenshot upload ---
st.subheader("📸 Upload PlayerMaker Screenshot")
st.caption("Take a screenshot of the PlayerMaker app summary screen and upload it here.")

uploaded = st.file_uploader(
    "Screenshot", type=["png", "jpg", "jpeg", "heic"],
    key="pm_screenshot",
    label_visibility="collapsed"
)

# --- Known PlayerMaker metric patterns ---
METRIC_PATTERNS = {
    "distance_km":       (r"(?:distance|dist)\s*[:\-]?\s*([\d.]+)\s*km", float),
    "sprint_distance_m": (r"sprint\s*(?:dist(?:ance)?)\s*[:\-]?\s*([\d.]+)\s*m", float),
    "top_speed_kmh":     (r"(?:top|max)\s*speed\s*[:\-]?\s*([\d.]+)\s*km/?h", float),
    "ball_touches":      (r"(?:ball\s*)?touches\s*[:\-]?\s*(\d+)", int),
    "touches_left":      (r"left\s*(?:foot)?\s*[:\-]?\s*(\d+)", int),
    "touches_right":     (r"right\s*(?:foot)?\s*[:\-]?\s*(\d+)", int),
    "first_touch_score": (r"(?:1st|first)\s*touch\s*[:\-]?\s*([\d.]+)", float),
    "time_on_ball_s":    (r"time\s*on\s*ball\s*[:\-]?\s*([\d.]+)\s*s", float),
    "release_time_s":    (r"release\s*(?:time)?\s*[:\-]?\s*([\d.]+)\s*s", float),
    "kick_power_kmh":    (r"kick(?:ing)?\s*(?:power|velocity)\s*[:\-]?\s*([\d.]+)", float),
    "two_footed_pct":    (r"two[\s\-]*foot(?:ed)?\s*[:\-]?\s*([\d.]+)\s*%?", float),
    "weak_foot_pct":     (r"weak\s*foot\s*[:\-]?\s*([\d.]+)\s*%?", float),
    "accelerations":     (r"accel(?:eration)?s?\s*[:\-]?\s*(\d+)", int),
    "decelerations":     (r"decel(?:eration)?s?\s*[:\-]?\s*(\d+)", int),
    "direction_changes": (r"(?:change|direction)\s*(?:of\s*)?(?:direction|changes)\s*[:\-]?\s*(\d+)", int),
    "match_score":       (r"match\s*score\s*[:\-]?\s*([\d.]+)", float),
}


def extract_metrics_from_text(text: str) -> dict:
    """Extract known PlayerMaker metrics from OCR text using regex."""
    results = {}
    text_lower = text.lower()
    for key, (pattern, dtype) in METRIC_PATTERNS.items():
        match = re.search(pattern, text_lower)
        if match:
            try:
                results[key] = dtype(match.group(1))
            except (ValueError, IndexError):
                pass
    return results


def preprocess_for_ocr(img: Image.Image) -> np.ndarray:
    """Preprocess dark-themed PlayerMaker screenshots for better OCR."""
    import cv2
    arr = np.array(img)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # PlayerMaker has white/colored text on dark bg — invert
    # Check if image is predominantly dark
    if np.median(gray) < 128:
        gray = cv2.bitwise_not(gray)

    # Increase contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Threshold for cleaner text
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded screenshot", use_container_width=True)

    # --- OCR extraction ---
    if _EASYOCR_AVAILABLE:
        with st.spinner("Running OCR..."):
            preprocessed = preprocess_for_ocr(img)
            reader = easyocr.Reader(["en"], gpu=False)
            raw_results = reader.readtext(preprocessed)
            raw_text = " ".join([r[1] for r in raw_results])

        with st.expander("🔍 Raw OCR Text", expanded=False):
            st.text(raw_text)

        auto_metrics = extract_metrics_from_text(raw_text)
        if auto_metrics:
            st.success(f"Auto-extracted {len(auto_metrics)} metrics from screenshot")
        else:
            st.info("Could not auto-extract metrics. Please enter values manually below.")
    else:
        st.warning("EasyOCR not installed. Enter values manually below.")
        auto_metrics = {}
        raw_text = ""

    st.divider()

    # --- Editable metrics form ---
    st.subheader("📊 Metrics")
    st.caption("Review and correct auto-extracted values, or enter manually.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Movement**")
        distance_km = st.number_input(
            "Distance (km)", min_value=0.0, max_value=20.0, step=0.01,
            value=auto_metrics.get("distance_km", 0.0),
            format="%.2f", key="pm_dist"
        )
        sprint_distance_m = st.number_input(
            "Sprint Distance (m)", min_value=0.0, max_value=5000.0, step=1.0,
            value=float(auto_metrics.get("sprint_distance_m", 0.0)),
            format="%.0f", key="pm_sprint"
        )
        top_speed_kmh = st.number_input(
            "Top Speed (km/h)", min_value=0.0, max_value=40.0, step=0.1,
            value=auto_metrics.get("top_speed_kmh", 0.0),
            format="%.1f", key="pm_speed"
        )
        accelerations = st.number_input(
            "Accelerations", min_value=0, max_value=500, step=1,
            value=auto_metrics.get("accelerations", 0),
            key="pm_accel"
        )
        decelerations = st.number_input(
            "Decelerations", min_value=0, max_value=500, step=1,
            value=auto_metrics.get("decelerations", 0),
            key="pm_decel"
        )
        direction_changes = st.number_input(
            "Direction Changes", min_value=0, max_value=1000, step=1,
            value=auto_metrics.get("direction_changes", 0),
            key="pm_dir"
        )

    with col2:
        st.markdown("**Ball Skills**")
        ball_touches = st.number_input(
            "Ball Touches (total)", min_value=0, max_value=500, step=1,
            value=auto_metrics.get("ball_touches", 0),
            key="pm_touches"
        )
        touches_left = st.number_input(
            "Touches (Left)", min_value=0, max_value=500, step=1,
            value=auto_metrics.get("touches_left", 0),
            key="pm_left"
        )
        touches_right = st.number_input(
            "Touches (Right)", min_value=0, max_value=500, step=1,
            value=auto_metrics.get("touches_right", 0),
            key="pm_right"
        )
        first_touch_score = st.number_input(
            "First Touch Score", min_value=0.0, max_value=10.0, step=0.1,
            value=auto_metrics.get("first_touch_score", 0.0),
            format="%.1f", key="pm_1st"
        )
        kick_power_kmh = st.number_input(
            "Kick Power (km/h)", min_value=0.0, max_value=150.0, step=0.1,
            value=auto_metrics.get("kick_power_kmh", 0.0),
            format="%.1f", key="pm_kick"
        )

    with col3:
        st.markdown("**Control**")
        time_on_ball_s = st.number_input(
            "Time on Ball (s)", min_value=0.0, max_value=3600.0, step=0.1,
            value=auto_metrics.get("time_on_ball_s", 0.0),
            format="%.1f", key="pm_tob"
        )
        release_time_s = st.number_input(
            "Release Time (s)", min_value=0.0, max_value=10.0, step=0.01,
            value=auto_metrics.get("release_time_s", 0.0),
            format="%.2f", key="pm_release"
        )
        two_footed_pct = st.number_input(
            "Two-Footed %", min_value=0.0, max_value=100.0, step=0.1,
            value=auto_metrics.get("two_footed_pct", 0.0),
            format="%.1f", key="pm_2foot"
        )
        weak_foot_pct = st.number_input(
            "Weak Foot %", min_value=0.0, max_value=100.0, step=0.1,
            value=auto_metrics.get("weak_foot_pct", 0.0),
            format="%.1f", key="pm_weak"
        )
        match_score = st.number_input(
            "Match Score", min_value=0.0, max_value=100.0, step=0.1,
            value=auto_metrics.get("match_score", 0.0),
            format="%.1f", key="pm_mscore"
        )

    st.divider()

    # --- Save ---
    if st.button("💾 Save Sensor Data", type="primary", use_container_width=True):
        # Create match if needed
        if selected_match is None:
            if not new_opponent:
                st.error("Please enter an opponent name.")
                st.stop()
            match_id = match_db.save_match(
                date=str(new_date),
                opponent=new_opponent,
                result=new_result or None,
            )
            st.success(f"Created match: {new_date} vs {new_opponent}")
        else:
            match_id = selected_match["id"]

        # Save screenshot
        screenshot_path = None
        screenshots_dir = DATA_DIR / "screenshots" / "playermaker"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        fname = f"pm_{match_id}_{selected_player['jersey_number']}.png"
        save_path = screenshots_dir / fname
        img.save(str(save_path))
        screenshot_path = str(save_path)

        # Build data dict
        data = {
            "distance_km": distance_km,
            "sprint_distance_m": sprint_distance_m,
            "top_speed_kmh": top_speed_kmh,
            "ball_touches": ball_touches,
            "touches_left": touches_left,
            "touches_right": touches_right,
            "first_touch_score": first_touch_score,
            "time_on_ball_s": time_on_ball_s,
            "release_time_s": release_time_s,
            "kick_power_kmh": kick_power_kmh,
            "two_footed_pct": two_footed_pct,
            "weak_foot_pct": weak_foot_pct,
            "accelerations": accelerations,
            "decelerations": decelerations,
            "direction_changes": direction_changes,
            "match_score": match_score,
        }

        sensor_id = match_db.save_sensor_data(
            match_id=match_id,
            player_id=selected_player["id"],
            data=data,
            source="playermaker",
            screenshot_path=screenshot_path,
        )
        st.success(f"Saved! Sensor data ID: {sensor_id}")
        st.balloons()

# --- History ---
st.divider()
st.subheader("📈 Sensor Data History")

history = match_db.get_player_sensor_history(selected_player["id"])
if history.empty:
    st.info(f"No sensor data recorded for {selected_player_label} yet.")
else:
    # Show summary table
    display_cols = [
        "date", "opponent", "distance_km", "top_speed_kmh",
        "ball_touches", "first_touch_score", "match_score",
        "two_footed_pct", "weak_foot_pct"
    ]
    available = [c for c in display_cols if c in history.columns]
    st.dataframe(
        history[available].rename(columns={
            "date": "Date", "opponent": "Opponent",
            "distance_km": "Dist (km)", "top_speed_kmh": "Top Speed",
            "ball_touches": "Touches", "first_touch_score": "1st Touch",
            "match_score": "Score", "two_footed_pct": "2-Foot %",
            "weak_foot_pct": "Weak Foot %",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Trend charts
    if len(history) >= 2:
        import matplotlib.pyplot as plt
        from config import ACCENT_GREEN

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        fig.patch.set_facecolor("#0d1117")
        x_labels = history["date"] + "\nvs " + history["opponent"]

        chart_configs = [
            ("distance_km", "Distance (km)", axes[0, 0]),
            ("top_speed_kmh", "Top Speed (km/h)", axes[0, 1]),
            ("ball_touches", "Ball Touches", axes[1, 0]),
            ("match_score", "Match Score", axes[1, 1]),
        ]
        for col_name, title, ax in chart_configs:
            if col_name in history.columns:
                vals = history[col_name].fillna(0)
                ax.plot(range(len(vals)), vals, color=ACCENT_GREEN,
                        marker="o", linewidth=2)
                ax.set_title(title, color="white", fontsize=11)
                ax.set_xticks(range(len(x_labels)))
                ax.set_xticklabels(x_labels, rotation=45, ha="right",
                                   fontsize=7, color="#8b949e")
                ax.tick_params(colors="#8b949e")
                ax.set_facecolor("#161b22")
                for spine in ax.spines.values():
                    spine.set_color("#30363d")
                ax.grid(axis="y", color="#21262d", linewidth=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
