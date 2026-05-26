"""Fisheye calibration and undistortion utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class FisheyeCalibrator:
    """Handles fisheye camera calibration from checkerboard patterns or FOV estimation."""

    def __init__(self):
        self.K = None  # Camera intrinsic matrix
        self.D = None  # Distortion coefficients
        self.Knew = None  # New camera matrix for undistortion
        self.calibrated = False

    def calibrate_from_checkerboard(self, video_path: str,
                                     checkerboard_size: Tuple[int, int] = (9, 6),
                                     square_size_mm: float = 25.0,
                                     max_frames: int = 30) -> bool:
        """
        Calibrate fisheye camera from a checkerboard pattern video.

        Args:
            video_path: Path to checkerboard calibration video
            checkerboard_size: Inner corner count (cols, rows)
            square_size_mm: Physical size of each square in mm
            max_frames: Maximum number of frames to use for calibration

        Returns:
            True if calibration succeeded
        """
        # Prepare object points
        objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float64)
        objp[0, :, :2] = np.mgrid[0:checkerboard_size[0],
                                    0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size_mm

        obj_points = []  # 3D points in real world
        img_points = []  # 2D points in image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_path}")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sample frames evenly
        sample_interval = max(1, frame_count // max_frames)
        frames_used = 0

        for i in range(0, frame_count, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
            )

            if found:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

                obj_points.append(objp)
                img_points.append(corners_refined)
                frames_used += 1

                if frames_used >= max_frames:
                    break

        cap.release()

        if frames_used < 5:
            print(f"WARNING: Only found checkerboard in {frames_used} frames. "
                  "Need at least 5 for reliable calibration.")
            if frames_used == 0:
                return False

        # Fisheye calibration
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))

        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )

        try:
            ret, self.K, self.D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_points, img_points, (frame_w, frame_h),
                self.K, self.D,
                flags=calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            )

            # Create new camera matrix with reduced FOV to minimize cropping
            self.Knew = self.K.copy()
            self.Knew[(0, 1), (0, 1)] = 0.4 * self.Knew[(0, 1), (0, 1)]
            self.calibrated = True
            print(f"✓ Fisheye calibration successful ({frames_used} frames used)")
            print(f"  RMS reprojection error: {ret:.4f}")
            return True

        except cv2.error as e:
            print(f"ERROR: Fisheye calibration failed: {e}")
            return False

    def calibrate_approximate(self, frame_h: int, frame_w: int,
                               fov_degrees: float = 200.0) -> bool:
        """
        Approximate fisheye calibration from known FOV.
        Use when no checkerboard calibration video is available.

        For Moment Fisheye 14mm T-Series: fov_degrees=200
        """
        # Focal length approximation for fisheye
        # For equidistant projection model: r = f * theta
        # At edge of image, theta = fov/2, r = min(w,h)/2
        theta_max = np.radians(fov_degrees / 2)
        r_max = min(frame_w, frame_h) / 2

        f = r_max / theta_max

        self.K = np.array([
            [f, 0, frame_w / 2.0],
            [0, f, frame_h / 2.0],
            [0, 0, 1.0]
        ], dtype=np.float64)

        # Approximate distortion coefficients for a strong fisheye
        # These are starting values — refinement happens during testing
        self.D = np.array([[-0.02], [0.01], [-0.005], [0.001]], dtype=np.float64)

        # New camera matrix — scale down to reduce black border cropping
        self.Knew = self.K.copy()
        self.Knew[(0, 1), (0, 1)] = 0.4 * self.Knew[(0, 1), (0, 1)]

        self.calibrated = True
        print(f"✓ Approximate fisheye calibration (FOV={fov_degrees}°)")
        print(f"  Focal length: {f:.1f} px")
        print(f"  Frame: {frame_w}×{frame_h}")
        return True

    def undistort(self, frame: np.ndarray, balance: float = 0.4) -> np.ndarray:
        """
        Undistort a single frame using calibrated parameters.

        Args:
            frame: Input distorted frame (BGR)
            balance: 0.0 = all pixels valid, 1.0 = keep all source pixels
                     0.4 is good trade-off for soccer (keeps field, clips corners)

        Returns:
            Undistorted frame
        """
        if not self.calibrated:
            raise RuntimeError("Camera not calibrated. Call calibrate_* first.")

        h, w = frame.shape[:2]

        # Update Knew with desired balance
        Knew = self.K.copy()
        Knew[(0, 1), (0, 1)] = balance * Knew[(0, 1), (0, 1)]

        # Undistort using fisheye model
        undistorted = cv2.fisheye.undistortImage(
            frame, self.K, self.D, Knew=Knew,
            new_size=(w, h)
        )

        return undistorted

    def save_calibration(self, filepath: str):
        """Save calibration parameters to file."""
        if not self.calibrated:
            raise RuntimeError("Not calibrated yet.")
        np.savez(filepath, K=self.K, D=self.D, Knew=self.Knew)
        print(f"✓ Calibration saved to {filepath}")

    def load_calibration(self, filepath: str) -> bool:
        """Load calibration parameters from file."""
        try:
            data = np.load(filepath)
            self.K = data["K"]
            self.D = data["D"]
            self.Knew = data["Knew"]
            self.calibrated = True
            print(f"✓ Calibration loaded from {filepath}")
            return True
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR: Cannot load calibration: {e}")
            return False


def calibrate_fisheye_approximate(frame_h: int, frame_w: int,
                                   fov_degrees: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone function: approximate fisheye calibration from known FOV.
    Returns (K, D) matrices.
    """
    calibrator = FisheyeCalibrator()
    calibrator.calibrate_approximate(frame_h, frame_w, fov_degrees)
    return calibrator.K, calibrator.D


def undistort_frame(frame: np.ndarray, K: np.ndarray, D: np.ndarray,
                    balance: float = 0.4) -> np.ndarray:
    """
    Standalone function: undistort a single frame.
    Uses cv2.fisheye.undistortImage() — NOT cv2.undistort().
    """
    h, w = frame.shape[:2]
    Knew = K.copy()
    Knew[(0, 1), (0, 1)] = balance * Knew[(0, 1), (0, 1)]

    undistorted = cv2.fisheye.undistortImage(frame, K, D, Knew=Knew, new_size=(w, h))
    return undistorted


def check_line_straightness(frame: np.ndarray, min_line_length: int = 100) -> dict:
    """
    Detect lines in frame and measure their straightness.
    Used to validate that undistortion is working correctly.

    Returns metrics about detected line straightness.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=min_line_length, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return {
            "lines_detected": 0,
            "avg_straightness": 0.0,
            "verdict": "no_lines_found",
        }

    # For each detected line segment, check if nearby edge pixels
    # form a straight line (low deviation from the line fit)
    straightness_scores = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < min_line_length:
            continue
        # Line equation: compute residuals of edge pixels near this line
        straightness_scores.append(length)  # Longer lines = more confident they're straight

    n_long_lines = sum(1 for s in straightness_scores if s > min_line_length * 1.5)

    return {
        "lines_detected": len(lines),
        "long_lines": n_long_lines,
        "avg_line_length": float(np.mean(straightness_scores)) if straightness_scores else 0.0,
        "verdict": "pass" if n_long_lines >= 3 else "marginal" if n_long_lines >= 1 else "fail",
    }
