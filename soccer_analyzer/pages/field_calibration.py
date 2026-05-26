"""Field Calibration — Streamlit page for homography setup."""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CUSTOM_CSS, DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M
from database.roster_db import RosterDB
from pipeline.homography import FieldHomography


def get_db():
    if "roster_db" not in st.session_state:
        st.session_state["roster_db"] = RosterDB()
    return st.session_state["roster_db"]


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("🗺️ Field Setup")

db = get_db()

# --- Select or Create Field ---
st.header("Select Field")

fields = db.get_all_fields()
field_names = [f["name"] for f in fields]
field_options = ["➕ New Field"] + field_names

selected_field = st.selectbox("Choose a field", field_options)

if selected_field == "➕ New Field":
    # --- New Field Calibration ---
    st.header("New Field Calibration")

    col1, col2 = st.columns(2)
    with col1:
        field_name = st.text_input("Field Name", placeholder="e.g. Riverside Park Field 2")
        field_length = st.slider("Field Length (m)", 30.0, 70.0, DEFAULT_FIELD_LENGTH_M, 1.0)
    with col2:
        field_width = st.slider("Field Width (m)", 20.0, 50.0, DEFAULT_FIELD_WIDTH_M, 1.0)

    st.subheader("Calibration Method")
    method = st.radio("Choose calibration method",
                      ["Auto (from video dimensions)", "Manual (click 4 points)"],
                      horizontal=True)

    if method == "Auto (from video dimensions)":
        st.info("Auto calibration estimates the field from video dimensions. "
                "Best for BallerCam placed at mid-field on the sideline.")

        video_file = st.file_uploader("Upload a video to get dimensions",
                                       type=["mp4", "mov", "avi"])

        if video_file and field_name:
            # Save temp video to get dimensions
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, first_frame = cap.read()
            cap.release()

            st.markdown(f"**Video dimensions:** {frame_w} × {frame_h} px")

            if ret:
                # Show first frame
                frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="First frame of video", use_container_width=True)

            # Auto calibrate
            homography = FieldHomography(field_length, field_width)
            homography.calibrate_auto(frame_h, frame_w)

            st.success("Auto calibration computed!")
            st.markdown(f"**Source points (px):** {homography.src_points.tolist()}")
            st.markdown(f"**Destination points (m):** {homography.dst_points.tolist()}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Field Calibration", use_container_width=True):
                    homography.save(field_name, db)
                    st.success(f"Saved calibration for '{field_name}'!")
                    st.rerun()

            with col2:
                if st.button("🔍 Test Calibration", use_container_width=True):
                    if ret:
                        # Draw grid overlay
                        test_frame = first_frame.copy()
                        lines = homography.get_grid_overlay(frame_h, frame_w, 10.0)
                        for line in lines:
                            if len(line) >= 2:
                                pt1 = (int(line[0][0]), int(line[0][1]))
                                pt2 = (int(line[1][0]), int(line[1][1]))
                                cv2.line(test_frame, pt1, pt2, (0, 255, 0), 2)

                        frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Grid overlay (10m spacing)",
                                 use_container_width=True)

            # Cleanup
            import os
            os.unlink(tmp_path)

    else:  # Manual calibration
        st.info("Upload a frame and click the 4 corners of the field in order: "
                "**top-left → top-right → bottom-right → bottom-left**")

        video_file = st.file_uploader("Upload video or image for calibration",
                                       type=["mp4", "mov", "avi", "jpg", "png"])

        if video_file and field_name:
            import tempfile

            # Get first frame
            suffix = Path(video_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            if suffix.lower() in [".jpg", ".jpeg", ".png"]:
                first_frame = cv2.imread(tmp_path)
            else:
                cap = cv2.VideoCapture(tmp_path)
                ret, first_frame = cap.read()
                cap.release()

            if first_frame is not None:
                frame_h, frame_w = first_frame.shape[:2]
                frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Click coordinates below",
                         use_container_width=True)

                st.markdown("Enter the pixel coordinates of the 4 field corners:")

                # Manual point input
                points = []
                corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

                for i, corner in enumerate(corner_names):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.number_input(f"{corner} X", 0, frame_w, frame_w // 4 * (i % 2 + 1),
                                            key=f"pt_{i}_x")
                    with col2:
                        y = st.number_input(f"{corner} Y", 0, frame_h,
                                            frame_h // 4 if i < 2 else frame_h * 3 // 4,
                                            key=f"pt_{i}_y")
                    points.append([x, y])

                if st.button("💾 Save Manual Calibration", use_container_width=True):
                    homography = FieldHomography(field_length, field_width)
                    homography.calibrate_manual(points)
                    homography.save(field_name, db)
                    st.success(f"Saved manual calibration for '{field_name}'!")
                    st.rerun()

            import os
            os.unlink(tmp_path)

else:
    # --- Existing Field ---
    st.header(f"Field: {selected_field}")

    field_data = db.get_field(selected_field)
    if field_data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Length", f"{field_data['field_length_m']:.0f} m")
        col2.metric("Width", f"{field_data['field_width_m']:.0f} m")
        col3.metric("Created", field_data.get("created_at", "")[:10])

        if field_data["src_points"]:
            st.markdown(f"**Source Points (px):** {field_data['src_points']}")
            st.markdown(f"**Destination Points (m):** {field_data['dst_points']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Use This Field", use_container_width=True):
                st.session_state["selected_field"] = selected_field
                st.success(f"Using '{selected_field}' for match analysis.")

        with col2:
            if st.button("🔄 Recalibrate", use_container_width=True):
                st.session_state["recalibrate"] = selected_field
                st.rerun()
