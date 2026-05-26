"""Fisheye lens undistortion using checkerboard calibration."""

import cv2
import numpy as np
from pathlib import Path


class FisheyeCalibration:
    """Load and apply fisheye undistortion from a saved calibration file."""

    def __init__(self, calibration_path: Path):
        """
        Args:
            calibration_path: Path to .npz file with K, D matrices from calibrate_fisheye.py
        """
        calibration_path = Path(calibration_path)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

        data = np.load(str(calibration_path))
        # Support both key naming conventions
        self.K = data["camera_matrix"] if "camera_matrix" in data else data["K"]
        self.D = data["dist_coeffs"] if "dist_coeffs" in data else data["D"]
        self.new_K = None
        self.map1 = None
        self.map2 = None
        self._frame_size = None

    def init_undistort(self, frame_size: tuple, balance: float = 0.5):
        """
        Pre-compute undistortion maps for a given frame size.

        Args:
            frame_size: (width, height) of the input frames
            balance: 0.0 = crop all black edges, 1.0 = keep all pixels (with black borders)
        """
        self._frame_size = frame_size
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, frame_size, np.eye(3), balance=balance
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K, frame_size, cv2.CV_16SC2
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a single frame using pre-computed maps.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Undistorted frame with same dimensions
        """
        h, w = frame.shape[:2]
        frame_size = (w, h)

        # Lazy-init maps on first call or if frame size changes
        if self.map1 is None or self._frame_size != frame_size:
            self.init_undistort(frame_size)

        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort a set of 2D points from distorted to rectified coordinates.

        Args:
            points: (N, 2) array of (x, y) pixel coordinates in distorted image

        Returns:
            (N, 2) array of undistorted pixel coordinates
        """
        if self.new_K is None:
            raise RuntimeError("Call init_undistort() or undistort() first")

        pts = points.reshape(-1, 1, 2).astype(np.float64)
        undistorted = cv2.fisheye.undistortPoints(pts, self.K, self.D, P=self.new_K)
        return undistorted.reshape(-1, 2)


def calibrate_from_video(video_path: str, checkerboard: tuple = (9, 6),
                         max_frames: int = 40, skip_frames: int = 15) -> dict:
    """
    Calibrate fisheye camera from a video of a checkerboard pattern.

    Args:
        video_path: Path to calibration video
        checkerboard: (columns, rows) of inner corners
        max_frames: Maximum frames to use for calibration
        skip_frames: Process every Nth frame

    Returns:
        dict with K, D, rms, n_frames_used
    """
    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (3D points in checkerboard space)
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = 0
    used_count = 0
    img_size = None

    while used_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]  # (width, height)

        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(
            gray, checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            used_count += 1
            print(f"  Found checkerboard in frame {frame_count} ({used_count}/{max_frames})")

    cap.release()

    if used_count < 5:
        raise ValueError(
            f"Only found checkerboard in {used_count} frames (need at least 5). "
            f"Try different lighting or hold the board more steadily."
        )

    print(f"\n  Calibrating with {used_count} frames...")

    # Fisheye calibration
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points, img_points, img_size, K, D,
        flags=calibration_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )

    print(f"  Calibration RMS error: {rms:.4f}")
    print(f"  Camera matrix K:\n{K}")
    print(f"  Distortion D: {D.ravel()}")

    return {
        "K": K,
        "D": D,
        "rms": rms,
        "n_frames_used": used_count,
        "image_size": img_size,
    }
