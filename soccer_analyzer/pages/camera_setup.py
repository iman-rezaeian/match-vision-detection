"""Camera Setup — Fisheye calibration & flag detection UI."""

import streamlit as st
import sys
import cv2
import numpy as np
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CALIBRATION_DIR, DEFAULT_CHECKERBOARD, DEFAULT_FLAG_COLOR, FLAG_COLOR_RANGES
from pipeline.fisheye import FisheyeCalibration, calibrate_from_video
from pipeline.flag_detector import FlagDetector


st.header("📷 Camera Setup")

tab_calibrate, tab_flags, tab_test = st.tabs(["Fisheye Calibration", "Flag Detection", "Test Undistort"])

# ─── Tab 1: Fisheye Calibration ───────────────────────────────────────────────
with tab_calibrate:
    st.subheader("Fisheye Lens Calibration")
    st.markdown("""
    Record a short video (~30s) slowly moving a checkerboard pattern in front of your fisheye lens.
    The calibration will compute lens distortion parameters to undistort your training footage.
    """)

    col1, col2 = st.columns(2)
    with col1:
        board_cols = st.number_input("Checkerboard columns (inner corners)", min_value=3, max_value=20,
                                     value=DEFAULT_CHECKERBOARD[0])
    with col2:
        board_rows = st.number_input("Checkerboard rows (inner corners)", min_value=3, max_value=20,
                                     value=DEFAULT_CHECKERBOARD[1])

    calib_video = st.file_uploader("Upload checkerboard video", type=["mov", "mp4", "avi", "mkv"],
                                   key="calib_video")

    calib_name = st.text_input("Calibration name", value="fisheye_calibration",
                               help="Saved to data/calibration/<name>.npz")

    if calib_video and st.button("Run Calibration", type="primary"):
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mov", delete=False) as tmp:
            tmp.write(calib_video.read())
            tmp_path = tmp.name

        with st.spinner("Finding checkerboard corners..."):
            try:
                result = calibrate_from_video(
                    video_path=tmp_path,
                    checkerboard=(int(board_cols), int(board_rows)),
                    max_frames=50,
                    skip_frames=10,
                )

                if result is None:
                    st.error("Calibration failed — no checkerboard corners detected. "
                             "Try a different video with better lighting and slower movement.")
                else:
                    # Save calibration
                    output_path = CALIBRATION_DIR / f"{calib_name}.npz"
                    np.savez(
                        str(output_path),
                        camera_matrix=result["K"],
                        dist_coeffs=result["D"],
                        image_size=result["image_size"],
                    )
                    st.success(f"Calibration saved to `{output_path}`")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("RMS Error", f"{result['rms']:.4f}")
                    col_b.metric("Frames Used", result["n_frames_used"])
                    col_c.metric("Image Size", f"{result['image_size'][0]}x{result['image_size'][1]}")

            except Exception as e:
                st.error(f"Calibration error: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    # Show existing calibrations
    st.divider()
    st.subheader("Saved Calibrations")
    calib_files = sorted(CALIBRATION_DIR.glob("*.npz"))
    if calib_files:
        for f in calib_files:
            data = np.load(str(f))
            size = data.get("image_size", [0, 0])
            st.text(f"  • {f.name} — {int(size[0])}x{int(size[1])}")
    else:
        st.info("No calibrations saved yet. Record a checkerboard video and calibrate above.")


# ─── Tab 2: Flag Detection ────────────────────────────────────────────────────
with tab_flags:
    st.subheader("Flag/Cone Detection Test")
    st.markdown("""
    Upload a frame from your training video to verify that the colored flags/cones
    at field corners are detected correctly.
    """)

    flag_color = st.selectbox("Flag color", options=list(FLAG_COLOR_RANGES.keys()),
                              index=list(FLAG_COLOR_RANGES.keys()).index(DEFAULT_FLAG_COLOR))

    min_area = st.slider("Minimum flag area (px²)", 50, 2000, 200)

    test_image = st.file_uploader("Upload a test frame", type=["jpg", "png", "jpeg"],
                                  key="flag_test_img")

    if test_image:
        file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detector = FlagDetector(flag_color=flag_color, min_area=min_area)
        centroids = detector.detect(frame)

        st.write(f"**Detected {len(centroids)} flag(s)**")

        if len(centroids) >= 4:
            corners = detector.assign_corners(centroids[:4], frame.shape)
            vis = detector.visualize(frame, centroids[:4], corners)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detected corners", use_column_width=True)

            st.json({k: {"x": int(v[0]), "y": int(v[1])} for k, v in corners.items()})
        elif len(centroids) > 0:
            # Show what we found even if < 4
            vis = frame.copy()
            for i, (cx, cy) in enumerate(centroids):
                cv2.circle(vis, (int(cx), int(cy)), 10, (0, 255, 0), 3)
                cv2.putText(vis, str(i), (int(cx) + 12, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Partial detections", use_column_width=True)
            st.warning(f"Need 4 flags for homography, found {len(centroids)}. "
                       "Adjust min_area or check flag placement.")
        else:
            st.warning("No flags detected. Try adjusting the color or min_area settings.")


# ─── Tab 3: Test Undistort ─────────────────────────────────────────────────────
with tab_test:
    st.subheader("Test Undistortion")
    st.markdown("Upload a fisheye frame to see the undistorted result.")

    calib_files = sorted(CALIBRATION_DIR.glob("*.npz"))
    if not calib_files:
        st.info("No calibration files found. Complete calibration in the first tab.")
    else:
        selected_calib = st.selectbox("Calibration file",
                                      options=[f.name for f in calib_files])
        calib_path = CALIBRATION_DIR / selected_calib

        test_frame = st.file_uploader("Upload fisheye frame", type=["jpg", "png", "jpeg"],
                                      key="undistort_test")

        if test_frame:
            file_bytes = np.asarray(bytearray(test_frame.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            fisheye = FisheyeCalibration(str(calib_path))
            undistorted = fisheye.undistort(frame)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         caption="Original (fisheye)", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB),
                         caption="Undistorted", use_column_width=True)
