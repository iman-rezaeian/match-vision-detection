# Lazy package — do NOT eagerly import all submodules.
# Each page/module should import what it needs directly:
#   from pipeline.detector import VideoDetector
#   from pipeline.homography import FieldHomography
# This avoids hanging when a submodule tries to download models at import time.

__all__ = [
    "VideoDetector", "FieldHomography", "TeamClassifier",
    "FaceReID", "GaitAnalyzer", "CleatExtractor",
    "PlayerFingerprinter", "JerseyOCR", "PassDetector",
    "StatsCalculator", "FormationDetector",
]
