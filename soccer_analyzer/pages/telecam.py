"""TeleCam — Virtual broadcast camera from wide-angle 4K footage."""

import gc
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.telecam import TeleCam, CropWindow
from pipeline.video_export import export_telecam_video
from pipeline.multi_segment import MultiSegmentProcessor, CACHE_DIR


st.title("🎥 TeleCam — Virtual Broadcast View")

# ---- Gate: need match results with segments --------------------------------
if "match_results" not in st.session_state:
    st.info("Run **Match Analysis** first. TeleCam uses the player tracking data to auto-pan the camera.")
    st.stop()

results = st.session_state["match_results"]
video_meta = results["video_meta"]
segments = results["segments"]
fps = video_meta.get("fps", 30.0)
frame_w = video_meta.get("frame_w", 3840)
frame_h = video_meta.get("frame_h", 2160)

# Load full Stage 1 detections (all tracks, not filtered) for better trajectory
match_key = results.get("match_key", "")
stage1_cached = MultiSegmentProcessor.load_stage1_cache(segments, match_key)
if stage1_cached is not None:
    detections_df = stage1_cached[0]
else:
    detections_df = results["detections_df"]

if frame_w < 1920 or frame_h < 1080:
    st.error(f"Source resolution ({frame_w}×{frame_h}) is too small for a 1080p crop. Need at least 1920×1080 source.")
    st.stop()

st.caption(
    f"Source: **{frame_w}×{frame_h}** @ {fps:.0f}fps · "
    f"**{len(segments)}** segment(s) · "
    f"**{len(detections_df):,}** detections"
)

# ---- Settings sidebar ------------------------------------------------------
st.sidebar.header("TeleCam Settings")

crop_presets = {
    "1080p (1920×1080)": (1920, 1080),
    "720p (1280×720)": (1280, 720),
}
preset = st.sidebar.selectbox("Output resolution", list(crop_presets.keys()), index=0)
crop_w, crop_h = crop_presets[preset]

max_zoom_x = frame_w / crop_w
max_zoom_y = frame_h / crop_h
st.sidebar.caption(f"Effective zoom: **{min(max_zoom_x, max_zoom_y):.1f}×**")

smoothing = st.sidebar.slider(
    "Pan smoothing",
    min_value=0.01, max_value=0.20, value=0.08, step=0.01,
    help="Lower = smoother/slower panning. Higher = more responsive.",
)
vertical_bias = st.sidebar.slider(
    "Vertical centering",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="0.0 = top of field, 0.5 = center, 1.0 = bottom of field",
)

copy_audio = st.sidebar.checkbox("Include audio", value=True)

# ---- Segment selection -----------------------------------------------------
seg_labels = [s.label for s in segments]
selected_seg_idx = st.selectbox("Select segment to export", range(len(segments)),
                                format_func=lambda i: seg_labels[i])
seg = segments[selected_seg_idx]

seg_df = detections_df[detections_df["segment_id"] == seg.segment_id].copy()

if seg_df.empty:
    st.warning(f"No detections in {seg.label}. Cannot compute camera trajectory.")
    st.stop()

start_frame = int(seg_df["frame"].min())
end_frame = int(seg_df["frame"].max())
duration_s = (end_frame - start_frame) / fps

st.caption(
    f"**{seg.label}**: frames {start_frame}–{end_frame} "
    f"({duration_s / 60:.1f} min, {end_frame - start_frame + 1:,} frames)"
)

# ---- Optional ball detection -----------------------------------------------
use_ball = st.sidebar.checkbox("Detect ball (improves tracking)", value=True,
                                help="Runs YOLO ball detection on sampled frames. Adds ~2 min per segment.")
ball_df = None
if use_ball:
    cache_key = f"telecam_ball_{seg.segment_id}_{start_frame}_{end_frame}"
    if cache_key in st.session_state:
        ball_df = st.session_state[cache_key]
        st.sidebar.caption(f"Ball: {len(ball_df)} detections (cached)")
    else:
        with st.spinner("Detecting ball positions..."):
            from ultralytics import YOLO
            from config import DEVICE
            model = YOLO("yolov8s.pt")
            cap = cv2.VideoCapture(seg.video_path)
            ball_records = []
            sample_rate = 5  # every 5th frame
            for fn in range(start_frame, end_frame + 1, sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame = cap.read()
                if not ret:
                    continue
                results_yolo = model(frame, conf=0.25, classes=[32], device=DEVICE,
                                     imgsz=1920, verbose=False)
                if len(results_yolo) > 0 and results_yolo[0].boxes is not None and len(results_yolo[0].boxes) > 0:
                    boxes = results_yolo[0].boxes
                    # Pick highest confidence ball
                    best_idx = boxes.conf.argmax()
                    bbox = boxes.xyxy[best_idx].cpu().numpy()
                    bx = (bbox[0] + bbox[2]) / 2
                    by = (bbox[1] + bbox[3]) / 2
                    ball_records.append({"frame": fn, "x_px": float(bx), "y_px": float(by)})
            cap.release()
            del model
            import gc; gc.collect()
            ball_df = pd.DataFrame(ball_records) if ball_records else None
            st.session_state[cache_key] = ball_df
        if ball_df is not None and not ball_df.empty:
            st.sidebar.caption(f"Ball: {len(ball_df)} detections found")
        else:
            st.sidebar.caption("Ball: not detected (using player clusters only)")

# ---- Compute trajectory ----------------------------------------------------
telecam = TeleCam(
    crop=CropWindow(width=crop_w, height=crop_h),
    smoothing=smoothing,
    vertical_bias=vertical_bias,
)

with st.spinner("Computing camera trajectory..."):
    trajectory_df = telecam.compute_trajectory(
        seg_df, frame_w, frame_h, fps, start_frame, end_frame,
        ball_df=ball_df,
    )

st.success(f"Trajectory computed: {len(trajectory_df):,} frames")

# ---- Preview ---------------------------------------------------------------
st.subheader("Preview")
st.caption("Drag the slider to scrub through the virtual camera view.")

preview_pos = st.slider(
    "Position",
    min_value=0, max_value=len(trajectory_df) - 1,
    value=len(trajectory_df) // 2,
    key="preview_slider",
)
preview_row = trajectory_df.iloc[preview_pos]
preview_frame_num = int(preview_row["frame"])
preview_cx = int(preview_row["cx"])
preview_cy = int(preview_row["cy"])

# Read the preview frame from video
cap = cv2.VideoCapture(seg.video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_num)
ret, raw_frame = cap.read()
cap.release()

if ret:
    cropped, wide_annotated = telecam.preview_frame(raw_frame, preview_cx, preview_cy)
    col_wide, col_crop = st.columns([1, 1])
    with col_wide:
        st.caption("Wide view (crop window shown)")
        st.image(cv2.cvtColor(wide_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_crop:
        st.caption("TeleCam output")
        st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.warning("Could not read preview frame from video.")

# ---- Export -----------------------------------------------------------------
st.subheader("Export")

if st.button("🎬 Export TeleCam Video", type="primary", use_container_width=True):
    progress = st.progress(0, text="Rendering TeleCam video...")

    def export_progress(done, total):
        frac = min(done / total, 1.0) if total > 0 else 0
        progress.progress(frac, text=f"Rendering: frame {done:,}/{total:,}")

    base = os.path.splitext(os.path.basename(seg.video_path))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{base}_{seg.label}_telecam.mp4")

    try:
        result_path = export_telecam_video(
            source_path=seg.video_path,
            trajectory_df=trajectory_df,
            telecam=telecam,
            output_path=out_path,
            start_frame=start_frame,
            end_frame=end_frame,
            copy_audio=copy_audio,
            progress_callback=export_progress,
        )
        progress.progress(1.0, text="Rendering complete ✓")

        file_size_mb = os.path.getsize(result_path) / 1e6
        st.success(f"Exported: **{os.path.basename(result_path)}** ({file_size_mb:.0f} MB)")

        with open(result_path, "rb") as f:
            st.download_button(
                "⬇️ Download TeleCam Video",
                data=f.read(),
                file_name=os.path.basename(result_path),
                mime="video/mp4",
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Export failed: {e}")
    finally:
        gc.collect()
