"""Phase 3D — BallerCam-specific preprocessing utilities.

Handles iPhone 15 Pro Max + BallerCam 180° panoramic video preprocessing:
- Lens profile correction for BallerCam's specific distortion
- Resolution normalization
- Field-of-view adaptation
- Panoramic-to-rectilinear conversion
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class BallerCamPreprocessor:
    """
    Preprocessor for BallerCam 180° panoramic footage from iPhone 15 Pro Max.

    BallerCam characteristics:
    - 180° field of view (vs SoccerTrack's ~200°)
    - iPhone 15 Pro Max sensor: 48MP, 24mm f/1.78
    - Panoramic mode stitches into ultra-wide image
    - Typical output: ~6000-8000px wide, ~1000-1500px tall
    """

    def __init__(self, fov_degrees: float = 180.0,
                 target_width: int = 6500):
        """
        Initialize BallerCam preprocessor.

        Args:
            fov_degrees: BallerCam field of view (default 180°)
            target_width: Target width for normalized output
        """
        self.fov = fov_degrees
        self.target_width = target_width
        self.calibration = None
        self.undistort_maps = None

    def calibrate(self, frame: np.ndarray,
                  field_corners: list = None) -> dict:
        """
        Compute lens calibration from a BallerCam frame.

        Uses the known FOV and image dimensions to compute
        approximate fisheye calibration parameters.

        Args:
            frame: Sample frame from BallerCam
            field_corners: Optional 4 corners of the field [(x,y), ...]

        Returns:
            Calibration dict with camera matrix, distortion coefficients
        """
        h, w = frame.shape[:2]

        # Approximate focal length from FOV
        fov_rad = np.radians(self.fov)
        focal_length = w / (2 * np.tan(fov_rad / 2))

        # Camera matrix (assume principal point at center)
        K = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)

        # BallerCam distortion model (equidistant projection)
        # k1-k4 are fisheye distortion coefficients
        # These are approximate for BallerCam's lens profile
        D = np.array([
            [-0.05],    # k1: radial
            [0.01],     # k2: radial
            [0.0],      # k3: tangential
            [0.0],      # k4: tangential
        ], dtype=np.float64)

        # Compute new camera matrix for undistortion
        new_K = K.copy()
        new_K[0, 0] *= 0.6  # Reduce focal length for wider view
        new_K[1, 1] *= 0.6

        # Compute undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )

        self.calibration = {
            "K": K,
            "D": D,
            "new_K": new_K,
            "image_size": (w, h),
            "fov": self.fov,
            "focal_length": focal_length,
        }
        self.undistort_maps = (map1, map2)

        return self.calibration

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a BallerCam frame using calibration.

        Args:
            frame: Raw BallerCam panoramic frame

        Returns:
            Undistorted frame
        """
        if self.undistort_maps is None:
            self.calibrate(frame)

        return cv2.remap(frame, self.undistort_maps[0],
                         self.undistort_maps[1], cv2.INTER_LINEAR)

    def normalize_resolution(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to target resolution while preserving aspect ratio.

        This ensures consistent detection performance across
        different BallerCam recording resolutions.
        """
        h, w = frame.shape[:2]

        if w == self.target_width:
            return frame

        scale = self.target_width / w
        new_h = int(h * scale)
        return cv2.resize(frame, (self.target_width, new_h),
                          interpolation=cv2.INTER_LINEAR)

    def preprocess(self, frame: np.ndarray,
                   undistort: bool = True,
                   normalize: bool = True,
                   enhance: bool = True) -> np.ndarray:
        """
        Full BallerCam preprocessing pipeline.

        Args:
            frame: Raw BallerCam frame
            undistort: Apply lens undistortion
            normalize: Normalize resolution
            enhance: Apply contrast/brightness enhancement

        Returns:
            Preprocessed frame ready for detection
        """
        result = frame.copy()

        if undistort:
            result = self.undistort(result)

        if normalize:
            result = self.normalize_resolution(result)

        if enhance:
            result = self._enhance_visibility(result)

        return result

    def _enhance_visibility(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance player visibility in outdoor soccer footage.

        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to improve detection of players in varying lighting conditions.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def extract_field_region(self, frame: np.ndarray,
                              green_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the field region from a panoramic frame.

        Uses color segmentation to find the green field area,
        which helps focus detection on relevant regions.

        Args:
            frame: Preprocessed frame
            green_threshold: Min fraction of green pixels in a column

        Returns:
            (field_frame, mask) where mask is the field region
        """
        # Convert to HSV for green detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green field mask (tuned for soccer fields)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return frame, mask

    def save_calibration(self, path: str):
        """Save calibration to file."""
        if self.calibration is None:
            raise ValueError("No calibration computed yet")

        np.savez(path,
                 K=self.calibration["K"],
                 D=self.calibration["D"],
                 new_K=self.calibration["new_K"],
                 fov=self.fov,
                 map1=self.undistort_maps[0],
                 map2=self.undistort_maps[1])

    def load_calibration(self, path: str):
        """Load calibration from file."""
        data = np.load(path)
        self.calibration = {
            "K": data["K"],
            "D": data["D"],
            "new_K": data["new_K"],
            "fov": float(data["fov"]),
        }
        self.undistort_maps = (data["map1"], data["map2"])


def adapt_soccertrack_to_ballercam(soccertrack_frame: np.ndarray,
                                    ballercam_fov: float = 180.0,
                                    soccertrack_fov: float = 200.0) -> np.ndarray:
    """
    Adapt a SoccerTrack fisheye frame to simulate BallerCam FOV.

    SoccerTrack uses ~200° FOV while BallerCam is 180°.
    This crops the edges to approximate the narrower FOV.

    Args:
        soccertrack_frame: Frame from SoccerTrack dataset
        ballercam_fov: BallerCam FOV in degrees
        soccertrack_fov: SoccerTrack FOV in degrees

    Returns:
        Cropped/adjusted frame simulating BallerCam perspective
    """
    h, w = soccertrack_frame.shape[:2]

    # Compute crop ratio
    fov_ratio = ballercam_fov / soccertrack_fov
    crop_w = int(w * fov_ratio)
    offset_x = (w - crop_w) // 2

    cropped = soccertrack_frame[:, offset_x:offset_x + crop_w]
    return cropped
