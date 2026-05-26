"""
TrackNetV3 ball tracker — inference wrapper for ball detection and trajectory.

Supports:
    - Pre-trained TrackNetV3 weights (downloaded or fine-tuned)
    - Frame triplet input (current + 2 previous frames)
    - Heatmap → coordinate extraction
    - Trajectory smoothing with visibility estimation

Training pipeline:
    1. Label frames with tools/label_ball.py
    2. Run tools/train_ball_tracker.py (generates fine-tuned weights)
    3. This module loads those weights for inference
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BallDetection:
    """Single ball detection result."""
    x: int
    y: int
    confidence: float
    visible: bool
    frame_num: int


@dataclass
class BallTrajectory:
    """Ball trajectory over multiple frames."""
    detections: List[BallDetection] = field(default_factory=list)

    @property
    def positions(self) -> np.ndarray:
        """Return (N, 2) array of visible positions."""
        visible = [d for d in self.detections if d.visible]
        if not visible:
            return np.empty((0, 2))
        return np.array([[d.x, d.y] for d in visible])

    @property
    def visible_frames(self) -> List[int]:
        return [d.frame_num for d in self.detections if d.visible]

    def get_position_at(self, frame_num: int) -> Optional[Tuple[int, int]]:
        """Get ball position at specific frame, or None if not visible."""
        for d in self.detections:
            if d.frame_num == frame_num:
                return (d.x, d.y) if d.visible else None
        return None


class TrackNetV3Model(nn.Module):
    """
    TrackNetV3 architecture — encoder-decoder with frame triplet input.

    Input: 3 consecutive frames concatenated (9 channels, H, W)
    Output: heatmap (1, H, W) indicating ball probability
    """

    def __init__(self, input_channels: int = 9, output_channels: int = 1):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for TrackNetV3")

        # Encoder
        self.encoder1 = self._conv_block(input_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)

        # Decoder
        self.decoder4 = self._upconv_block(512, 256)
        self.decoder3 = self._upconv_block(256 + 256, 128)
        self.decoder2 = self._upconv_block(128 + 128, 64)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, 1),
            nn.Sigmoid(),
        )

        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Decoder with skip connections
        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        out = self.decoder1(torch.cat([d2, e1], dim=1))

        return out


class BallTracker:
    """
    Ball tracking using TrackNetV3.

    Usage:
        tracker = BallTracker("data/models/ball_tracknet.pt")
        for frame_num, frame in enumerate(frames):
            detection = tracker.predict(frame, frame_num)
    """

    # Input size for TrackNetV3 (width, height)
    INPUT_SIZE = (640, 360)
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for ball tracking. Install with: pip install torch")

        # Device selection
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Model
        self.model = TrackNetV3Model()
        self.weights_path = weights_path

        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"Loaded ball tracker weights from {weights_path}")
        else:
            print("WARNING: No weights loaded — ball tracker will output random predictions.")
            print("         Fine-tune with: python tools/train_ball_tracker.py")

        self.model.to(self.device)
        self.model.eval()

        # Frame buffer (need 3 consecutive frames)
        self.frame_buffer: deque = deque(maxlen=3)
        self.trajectory = BallTrajectory()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame to model input size."""
        resized = cv2.resize(frame, self.INPUT_SIZE)
        # BGR to RGB, normalize to [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb

    def _heatmap_to_position(self, heatmap: np.ndarray, original_size: Tuple[int, int]) -> Optional[Tuple[int, int, float]]:
        """
        Extract ball position from heatmap.
        Returns (x, y, confidence) in original frame coordinates, or None.
        """
        # Find peak
        max_val = heatmap.max()
        if max_val < self.CONFIDENCE_THRESHOLD:
            return None

        # Get peak location
        y_idx, x_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)

        # Scale back to original resolution
        orig_w, orig_h = original_size
        hmap_h, hmap_w = heatmap.shape
        x = int(x_idx * orig_w / hmap_w)
        y = int(y_idx * orig_h / hmap_h)

        return x, y, float(max_val)

    def predict(self, frame: np.ndarray, frame_num: int) -> BallDetection:
        """
        Predict ball position in a single frame.
        Requires at least 3 frames to have been fed (buffers first 2).
        """
        processed = self._preprocess(frame)
        self.frame_buffer.append(processed)

        orig_h, orig_w = frame.shape[:2]

        # Need 3 frames for triplet input
        if len(self.frame_buffer) < 3:
            detection = BallDetection(x=0, y=0, confidence=0.0, visible=False, frame_num=frame_num)
            self.trajectory.detections.append(detection)
            return detection

        # Build triplet: concatenate 3 frames along channel dim
        triplet = np.concatenate(list(self.frame_buffer), axis=-1)  # (H, W, 9)
        # To tensor: (1, 9, H, W)
        tensor = torch.from_numpy(triplet).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmap = self.model(tensor)

        # Extract position from heatmap
        heatmap_np = heatmap.squeeze().cpu().numpy()
        result = self._heatmap_to_position(heatmap_np, (orig_w, orig_h))

        if result is not None:
            x, y, conf = result
            detection = BallDetection(x=x, y=y, confidence=conf, visible=True, frame_num=frame_num)
        else:
            detection = BallDetection(x=0, y=0, confidence=0.0, visible=False, frame_num=frame_num)

        self.trajectory.detections.append(detection)
        return detection

    def get_trajectory(self) -> BallTrajectory:
        """Return full trajectory of all predictions."""
        return self.trajectory

    def reset(self):
        """Reset tracker state for a new video."""
        self.frame_buffer.clear()
        self.trajectory = BallTrajectory()

    def smooth_trajectory(self, window: int = 5) -> List[BallDetection]:
        """
        Apply moving average smoothing to visible ball positions.
        Fills short gaps (≤ window frames) via interpolation.
        """
        detections = self.trajectory.detections
        if not detections:
            return []

        smoothed = []
        positions = {}

        for d in detections:
            if d.visible:
                positions[d.frame_num] = (d.x, d.y)

        sorted_frames = sorted(positions.keys())

        for d in detections:
            if d.visible:
                # Moving average of nearby visible frames
                nearby = [
                    positions[f] for f in sorted_frames
                    if abs(f - d.frame_num) <= window and f in positions
                ]
                if nearby:
                    avg_x = int(np.mean([p[0] for p in nearby]))
                    avg_y = int(np.mean([p[1] for p in nearby]))
                    smoothed.append(BallDetection(
                        x=avg_x, y=avg_y,
                        confidence=d.confidence,
                        visible=True,
                        frame_num=d.frame_num,
                    ))
                else:
                    smoothed.append(d)
            else:
                # Try to interpolate short gaps
                prev_frames = [f for f in sorted_frames if f < d.frame_num]
                next_frames = [f for f in sorted_frames if f > d.frame_num]

                if prev_frames and next_frames:
                    pf = prev_frames[-1]
                    nf = next_frames[0]
                    gap = nf - pf
                    if gap <= window * 2:
                        # Linear interpolation
                        t = (d.frame_num - pf) / gap
                        px, py = positions[pf]
                        nx, ny = positions[nf]
                        ix = int(px + t * (nx - px))
                        iy = int(py + t * (ny - py))
                        smoothed.append(BallDetection(
                            x=ix, y=iy,
                            confidence=0.3,
                            visible=True,
                            frame_num=d.frame_num,
                        ))
                    else:
                        smoothed.append(d)
                else:
                    smoothed.append(d)

        return smoothed
