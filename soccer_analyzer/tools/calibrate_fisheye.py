#!/usr/bin/env python3
"""
Fisheye Lens Calibration Tool

Usage:
    python tools/calibrate_fisheye.py /path/to/checkerboard_video.mov
    python tools/calibrate_fisheye.py /path/to/checkerboard_video.mov --output data/calibration/neewer_fisheye.npz
    python tools/calibrate_fisheye.py /path/to/checkerboard_video.mov --board 10x7 --skip 10

Instructions:
    1. Print a checkerboard pattern (default 9×6 inner corners) on A3/A4 paper
    2. Tape it to a rigid flat surface (cardboard, clipboard)
    3. Record 15-20 seconds of video holding the board at various:
       - Angles (tilt left/right/up/down)
       - Distances (close + medium + far)
       - Positions (center, edges, corners of the frame)
    4. Run this tool on the video
    5. Use the output .npz file with FisheyeCalibration class
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent so pipeline imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from pipeline.fisheye import calibrate_from_video, FisheyeCalibration


def main():
    parser = argparse.ArgumentParser(description="Calibrate fisheye lens from checkerboard video")
    parser.add_argument("video", help="Path to checkerboard calibration video")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output .npz file path (default: data/calibration/<video_name>.npz)")
    parser.add_argument("--board", type=str, default="9x6",
                        help="Checkerboard inner corners as WxH (default: 9x6)")
    parser.add_argument("--max-frames", type=int, default=40,
                        help="Max frames to use for calibration (default: 40)")
    parser.add_argument("--skip", type=int, default=15,
                        help="Process every Nth frame (default: 15)")
    parser.add_argument("--preview", action="store_true",
                        help="Show undistortion preview after calibration")
    args = parser.parse_args()

    # Parse board size
    try:
        cols, rows = map(int, args.board.split("x"))
        checkerboard = (cols, rows)
    except ValueError:
        print(f"Error: Invalid board format '{args.board}'. Use WxH (e.g., 9x6)")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        video_name = Path(args.video).stem
        output_path = Path(__file__).parent.parent / "data" / "calibration" / f"{video_name}.npz"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FISHEYE CALIBRATION")
    print(f"{'='*60}")
    print(f"  Video: {args.video}")
    print(f"  Board: {checkerboard[0]}×{checkerboard[1]} inner corners")
    print(f"  Output: {output_path}")
    print(f"  Max frames: {args.max_frames}, skip: {args.skip}")
    print()

    # Run calibration
    result = calibrate_from_video(
        args.video,
        checkerboard=checkerboard,
        max_frames=args.max_frames,
        skip_frames=args.skip,
    )

    # Save calibration
    np.savez(
        str(output_path),
        K=result["K"],
        D=result["D"],
        rms=np.array([result["rms"]]),
        image_size=np.array(result["image_size"]),
    )
    print(f"\n  ✓ Saved calibration to: {output_path}")
    print(f"    RMS error: {result['rms']:.4f} (good < 1.0, ideal < 0.5)")

    # Optional preview
    if args.preview:
        print("\n  Generating undistortion preview...")
        calib = FisheyeCalibration(output_path)

        cap = cv2.VideoCapture(args.video)
        # Skip to middle of video for preview
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()

        if ret:
            undistorted = calib.undistort(frame)
            # Save side-by-side comparison
            h, w = frame.shape[:2]
            scale = 1280 / w if w > 1280 else 1.0
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                undistorted = cv2.resize(undistorted, (int(w * scale), int(h * scale)))

            comparison = np.hstack([frame, undistorted])
            preview_path = output_path.with_suffix(".preview.jpg")
            cv2.imwrite(str(preview_path), comparison)
            print(f"  ✓ Preview saved: {preview_path}")
            print(f"    Left=original, Right=undistorted")


if __name__ == "__main__":
    main()
