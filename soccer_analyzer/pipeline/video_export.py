"""Video export — render a cropped TeleCam video from 4K source."""

import cv2
import subprocess
import shutil
import tempfile
import os
import numpy as np
import pandas as pd
from typing import Optional, Callable

from pipeline.telecam import TeleCam


def export_telecam_video(
    source_path: str,
    trajectory_df: pd.DataFrame,
    telecam: TeleCam,
    output_path: Optional[str] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    copy_audio: bool = True,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Render a TeleCam video by cropping each frame of the source.

    Args:
        source_path: path to the original 4K video
        trajectory_df: DataFrame from TeleCam.compute_trajectory() with [frame, cx, cy]
        telecam: TeleCam instance (used for crop dimensions)
        output_path: where to write the output .mp4 (default: temp dir)
        start_frame: first frame to include
        end_frame: last frame to include (default: last in trajectory)
        copy_audio: if True, mux audio from source into output
        progress_callback: optional (current_frame, total_frames) callback

    Returns:
        Path to the rendered .mp4 file.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(source_path))[0]
        output_path = os.path.join(tempfile.gettempdir(), f"{base}_telecam.mp4")

    if end_frame is None:
        end_frame = int(trajectory_df["frame"].max())

    # Build a lookup: frame_num → (cx, cy)
    traj_map = {}
    for _, row in trajectory_df.iterrows():
        fn = int(row["frame"])
        if start_frame <= fn <= end_frame:
            traj_map[fn] = (int(row["cx"]), int(row["cy"]))

    crop_w = telecam.crop.width
    crop_h = telecam.crop.height
    total_frames = end_frame - start_frame + 1

    # Open source video
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate crop fits in source
    if crop_w > source_w or crop_h > source_h:
        raise ValueError(
            f"Crop {crop_w}×{crop_h} exceeds source {source_w}×{source_h}"
        )

    # Write to a temp file first, then mux audio
    tmp_video = output_path + ".tmp.mp4"

    writer = cv2.VideoWriter(
        tmp_video,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (crop_w, crop_h),
    )
    if not writer.isOpened():
        # Fallback codec
        writer = cv2.VideoWriter(
            tmp_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (crop_w, crop_h),
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    last_cx = source_w // 2
    last_cy = source_h // 2

    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy = traj_map.get(frame_num, (last_cx, last_cy))
        last_cx, last_cy = cx, cy

        x1, y1, x2, y2 = telecam.get_crop_box(cx, cy)
        cropped = frame[y1:y2, x1:x2]

        writer.write(cropped)

        if progress_callback and frame_num % 30 == 0:
            done = frame_num - start_frame
            progress_callback(done, total_frames)

    writer.release()
    cap.release()

    # Mux audio from source if requested
    if copy_audio and shutil.which("ffmpeg"):
        _mux_audio(source_path, tmp_video, output_path, start_frame, end_frame, fps)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            os.remove(tmp_video)
        else:
            os.rename(tmp_video, output_path)
    else:
        os.rename(tmp_video, output_path)

    if progress_callback:
        progress_callback(total_frames, total_frames)

    return output_path


def _mux_audio(source_path: str, video_path: str, output_path: str,
               start_frame: int, end_frame: int, fps: float):
    """Combine the cropped video with original audio using ffmpeg."""
    start_s = start_frame / fps
    duration_s = (end_frame - start_frame + 1) / fps

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", source_path,
        "-i", video_path,
        "-t", f"{duration_s:.3f}",
        "-map", "1:v:0",
        "-map", "0:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=600
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
