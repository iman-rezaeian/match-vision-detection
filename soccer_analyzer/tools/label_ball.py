#!/usr/bin/env python3
"""
Ball Labeling Tool — click-to-label ball positions for TrackNetV3 training.

Usage:
    python tools/label_ball.py /path/to/video.mov --output data/ball_labels.csv
    python tools/label_ball.py /path/to/video.mov --output data/ball_labels.csv --start 100 --skip 3

Controls:
    Left Click   — mark ball center at cursor position
    'n' / Right Arrow — next frame (ball not visible / skip)
    'p' / Left Arrow  — go back one frame (undo last)
    'v'          — mark ball as NOT visible this frame
    'q' / Escape — save and quit
    's'          — save progress (continue labeling)
    'z'          — undo last label (remove current frame's label)

Output CSV columns: frame, x, y, visible
    visible=1 means ball at (x,y), visible=0 means ball not in frame
"""

import sys
import os
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


class BallLabeler:
    WINDOW_NAME = "Ball Labeler — Click ball center | 'n'=skip | 'v'=not visible | 'q'=quit"

    def __init__(self, video_path: str, output_path: str, start_frame: int = 0, skip: int = 1):
        self.video_path = video_path
        self.output_path = Path(output_path)
        self.skip = skip
        self.start_frame = start_frame

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Display scaling for 4K on laptop screen
        self.display_width = min(1280, self.width)
        self.scale = self.display_width / self.width
        self.display_height = int(self.height * self.scale)

        self.labels: dict = {}  # frame_num -> (x, y, visible)
        self.current_frame_num = start_frame
        self.click_pos = None

        # Load existing labels if resuming
        self._load_existing()

    def _load_existing(self):
        """Load previously saved labels to resume labeling."""
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame = int(row["frame"])
                    x = int(row["x"])
                    y = int(row["y"])
                    visible = int(row["visible"])
                    self.labels[frame] = (x, y, visible)
            print(f"Loaded {len(self.labels)} existing labels from {self.output_path}")

    def _save(self):
        """Save all labels to CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_frames = sorted(self.labels.keys())
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "x", "y", "visible"])
            for frame_num in sorted_frames:
                x, y, visible = self.labels[frame_num]
                writer.writerow([frame_num, x, y, visible])
        print(f"Saved {len(self.labels)} labels to {self.output_path}")

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click — record ball position in original coordinates."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coords back to original resolution
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.click_pos = (orig_x, orig_y)

    def _get_frame(self, frame_num: int):
        """Seek to frame and read it."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _render_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """Draw overlay info on display frame."""
        display = cv2.resize(frame, (self.display_width, self.display_height))

        # Draw existing label if this frame already labeled
        if frame_num in self.labels:
            x, y, visible = self.labels[frame_num]
            if visible:
                dx, dy = int(x * self.scale), int(y * self.scale)
                cv2.circle(display, (dx, dy), 8, (0, 255, 0), 2)
                cv2.circle(display, (dx, dy), 2, (0, 255, 0), -1)
                cv2.putText(display, f"Ball: ({x},{y})", (dx + 12, dy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display, "NOT VISIBLE", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # HUD
        labeled_count = len(self.labels)
        progress = f"Frame {frame_num}/{self.total_frames} | Labeled: {labeled_count} | Skip: {self.skip}"
        cv2.putText(display, progress, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Time
        time_s = frame_num / self.fps if self.fps > 0 else 0
        time_str = f"{int(time_s // 60):02d}:{time_s % 60:05.2f}"
        cv2.putText(display, time_str, (10, self.display_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display

    def run(self):
        """Main labeling loop."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        print(f"\nBall Labeler")
        print(f"  Video: {self.video_path}")
        print(f"  Frames: {self.total_frames} ({self.total_frames / self.fps:.1f}s)")
        print(f"  Resolution: {self.width}x{self.height} → display {self.display_width}x{self.display_height}")
        print(f"  Starting at frame {self.start_frame}, skip={self.skip}")
        print(f"  Output: {self.output_path}")
        print(f"\nControls: click=mark | n/→=next | p/←=prev | v=not visible | z=undo | s=save | q=quit\n")

        self.current_frame_num = self.start_frame
        history = [self.current_frame_num]

        while True:
            frame = self._get_frame(self.current_frame_num)
            if frame is None:
                print("Reached end of video.")
                break

            self.click_pos = None
            display = self._render_frame(frame, self.current_frame_num)
            cv2.imshow(self.WINDOW_NAME, display)

            while True:
                key = cv2.waitKey(30) & 0xFF

                # Check for click
                if self.click_pos is not None:
                    x, y = self.click_pos
                    self.labels[self.current_frame_num] = (x, y, 1)
                    print(f"  Frame {self.current_frame_num}: ball at ({x}, {y})")
                    # Advance to next frame
                    self.current_frame_num += self.skip
                    history.append(self.current_frame_num)
                    break

                # 'n' or Right Arrow — skip (no label)
                if key == ord("n") or key == 83 or key == 3:
                    self.current_frame_num += self.skip
                    history.append(self.current_frame_num)
                    break

                # 'p' or Left Arrow — go back
                if key == ord("p") or key == 81 or key == 2:
                    if len(history) > 1:
                        history.pop()
                        self.current_frame_num = history[-1]
                    break

                # 'v' — ball not visible
                if key == ord("v"):
                    self.labels[self.current_frame_num] = (0, 0, 0)
                    print(f"  Frame {self.current_frame_num}: NOT VISIBLE")
                    self.current_frame_num += self.skip
                    history.append(self.current_frame_num)
                    break

                # 'z' — undo (remove label for current frame)
                if key == ord("z"):
                    if self.current_frame_num in self.labels:
                        del self.labels[self.current_frame_num]
                        print(f"  Frame {self.current_frame_num}: label removed")
                    display = self._render_frame(frame, self.current_frame_num)
                    cv2.imshow(self.WINDOW_NAME, display)
                    continue

                # 's' — save progress
                if key == ord("s"):
                    self._save()
                    continue

                # 'q' or Escape — quit
                if key == ord("q") or key == 27:
                    self._save()
                    cv2.destroyAllWindows()
                    self.cap.release()
                    print(f"\nDone! {len(self.labels)} frames labeled.")
                    return

                # Check end of video
                if self.current_frame_num >= self.total_frames:
                    print("Reached end of video.")
                    self._save()
                    cv2.destroyAllWindows()
                    self.cap.release()
                    return

        self._save()
        cv2.destroyAllWindows()
        self.cap.release()
        print(f"\nDone! {len(self.labels)} frames labeled.")


def main():
    parser = argparse.ArgumentParser(description="Ball labeling tool for TrackNetV3 training")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="data/ball_labels.csv",
                        help="Output CSV path (default: data/ball_labels.csv)")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting frame number")
    parser.add_argument("--skip", type=int, default=3,
                        help="Label every Nth frame (default: 3)")

    args = parser.parse_args()

    labeler = BallLabeler(
        video_path=args.video,
        output_path=args.output,
        start_frame=args.start,
        skip=args.skip,
    )
    labeler.run()


if __name__ == "__main__":
    main()
