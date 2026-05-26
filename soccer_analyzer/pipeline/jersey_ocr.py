"""Targeted jersey number classification using PaddleOCR (primary) or EasyOCR (fallback)."""

import cv2
import numpy as np
from typing import Optional, Tuple
from collections import Counter


class JerseyOCR:
    KNOWN_NUMBERS = [3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21]

    def __init__(self, engine: str = "auto"):
        """
        Initialize OCR engine.

        Args:
            engine: "paddle", "easyocr", or "auto" (try paddle first, fallback to easyocr)
        """
        import ssl
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context

        self.engine = engine
        self.paddle_reader = None
        self.easyocr_reader = None

        try:
            if engine in ("paddle", "auto"):
                try:
                    from paddleocr import PaddleOCR
                    self.paddle_reader = PaddleOCR(lang="en")
                    self.engine = "paddle"
                except (ImportError, Exception) as e:
                    if engine == "paddle":
                        raise ImportError(
                            f"PaddleOCR failed to initialize: {e}. "
                            "Install with: pip install paddlepaddle paddleocr"
                        )
                    # Fall through to easyocr

            if self.paddle_reader is None:
                import easyocr
                self.easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.engine = "easyocr"
        finally:
            ssl._create_default_https_context = _orig_ctx

    def extract_jersey_region(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """
        Crop chest/back region: middle 40% height, center 60% width of bbox.
        Return cropped image.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h = y2 - y1
        w = x2 - x1

        # Middle 40% height (jersey number area)
        jersey_y1 = y1 + int(h * 0.25)
        jersey_y2 = y1 + int(h * 0.65)
        # Center 60% width
        jersey_x1 = x1 + int(w * 0.2)
        jersey_x2 = x2 - int(w * 0.2)

        # Clamp
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(frame.shape[0], jersey_y2)
        jersey_x1 = max(0, jersey_x1)
        jersey_x2 = min(frame.shape[1], jersey_x2)

        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return None

        crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        if crop.size == 0:
            return None

        return crop

    def read_number(self, jersey_crop: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Run OCR on jersey crop using configured engine.
        Filter results to only accept numbers in KNOWN_NUMBERS list.
        Return (number, confidence) or (None, 0) if no valid number found.
        """
        if jersey_crop is None or jersey_crop.size == 0:
            return None, 0.0

        try:
            if self.engine == "paddle":
                return self._read_paddle(jersey_crop)
            else:
                return self._read_easyocr(jersey_crop)
        except Exception:
            return None, 0.0

    def _read_paddle(self, jersey_crop: np.ndarray) -> Tuple[Optional[int], float]:
        """Read number using PaddleOCR v3.5+."""
        results = self.paddle_reader.predict(jersey_crop)

        if not results:
            return None, 0.0

        for r in results:
            texts = r.get("rec_texts", [])
            scores = r.get("rec_scores", [])
            for text, confidence in zip(texts, scores):
                text = text.strip()
                # Filter to digits only
                digits = "".join(c for c in text if c.isdigit())
                if not digits:
                    continue
                try:
                    number = int(digits)
                    if number in self.KNOWN_NUMBERS and confidence > 0.3:
                        return number, float(confidence)
                except ValueError:
                    continue

        return None, 0.0

    def _read_easyocr(self, jersey_crop: np.ndarray) -> Tuple[Optional[int], float]:
        """Read number using EasyOCR (fallback)."""
        results = self.easyocr_reader.readtext(jersey_crop, allowlist="0123456789")

        for (bbox, text, confidence) in results:
            text = text.strip()
            if not text:
                continue
            try:
                number = int(text)
                if number in self.KNOWN_NUMBERS and confidence > 0.3:
                    return number, float(confidence)
            except ValueError:
                continue

        return None, 0.0

    def batch_read(self, detections_df, frames: dict, sample_every: int = 15) -> dict:
        """
        Attempt OCR on every Nth frame per track_id.
        Vote on most frequent valid number read.
        Return {track_id: {jersey_number, confidence, read_count}}
        """
        track_ids = detections_df["track_id"].unique()
        results = {}

        for tid in track_ids:
            track_data = detections_df[detections_df["track_id"] == tid]
            readings = []

            # Sample every Nth detection
            sample_indices = range(0, len(track_data), sample_every)

            for idx in sample_indices:
                if idx >= len(track_data):
                    break
                row = track_data.iloc[idx]
                frame_num = int(row["frame"])

                if frame_num not in frames:
                    continue

                frame = frames[frame_num]
                bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])

                # Check bbox is large enough
                bbox_height = row["bbox_y2"] - row["bbox_y1"]
                if bbox_height < 50:
                    continue

                jersey_crop = self.extract_jersey_region(frame, bbox)
                if jersey_crop is not None:
                    number, confidence = self.read_number(jersey_crop)
                    if number is not None:
                        readings.append((number, confidence))

            if readings:
                # Vote on most frequent number
                numbers = [r[0] for r in readings]
                counter = Counter(numbers)
                best_number, count = counter.most_common(1)[0]

                # Average confidence for winning number
                winning_confs = [r[1] for r in readings if r[0] == best_number]
                avg_conf = np.mean(winning_confs)

                results[tid] = {
                    "jersey_number": best_number,
                    "confidence": float(avg_conf),
                    "read_count": count,
                    "total_attempts": len(readings),
                }
            else:
                results[tid] = {
                    "jersey_number": None,
                    "confidence": 0.0,
                    "read_count": 0,
                    "total_attempts": 0,
                }

        return results
