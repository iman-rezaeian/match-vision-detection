"""Step 2 — Player Detection Accuracy Test.

Validates YOLOv8 player detection against ground truth annotations
using SoccerTrack v2 dataset.
"""

import cv2
import numpy as np
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "soccer_analyzer"))

from utils.metrics import compute_detection_metrics, compute_detection_metrics_batch
from utils.visualization import save_detection_overlay
from utils.soccertrack_loader import (load_soccertrack_annotations, find_video_file,
                                       find_matching_annotation)
from utils.enhanced_detection import (get_model, enhanced_detect,
                                       load_undistort_maps)


def run_detection_test(v2_path: str, output_dir: str,
                       confidence_threshold: float = 0.15,
                       num_test_frames: int = 50) -> dict:
    """
    Run player detection test on SoccerTrack v2 dataset.

    Steps:
    1. Load ground truth annotations (bounding boxes)
    2. Run YOLOv8 (person class only) on test frames
    3. Compare detections against ground truth
    4. Report precision, recall, F1

    Target: recall > 85% for pass

    Returns: dict with detection metrics and verdict
    """
    print("\n" + "=" * 60)
    print("STEP 2: PLAYER DETECTION TEST")
    print("=" * 60)

    v2_dir = Path(v2_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ERROR: ultralytics not installed. pip install ultralytics")
        return {"verdict": "fail", "error": "ultralytics_not_installed"}

    print("  Loading YOLOv8s model...")
    model = get_model("yolov8s.pt")
    print("  Model loaded.")

    # Find video (skip viz_results) and load ground truth
    video_path = find_video_file(v2_dir)
    gt_annotations = None

    if video_path:
        ann_path = find_matching_annotation(video_path, v2_dir)
        if ann_path:
            print(f"  Loading annotations: {ann_path}")
            gt_annotations = load_soccertrack_annotations(str(ann_path))
            print(f"  Ground truth: {len(gt_annotations)} frames with bboxes")
            if gt_annotations:
                sample_frame = next(iter(gt_annotations.values()))
                print(f"  Players per frame (sample): {len(sample_frame)}")

    if video_path is None:
        print("  ⚠️  No video found in SoccerTrack path.")
        print("  Running detection test on synthetic scene...")
        return _run_synthetic_detection_test(model, output, confidence_threshold)

    if gt_annotations is None:
        gt_annotations = _load_ground_truth(v2_dir)

    # Load frames from video
    if video_path:
        print(f"  Video: {video_path}")
        all_detections, all_ground_truth, sample_frames = _test_on_video(
            model, str(video_path), gt_annotations,
            confidence_threshold, num_test_frames
        )
    else:
        # Load frames from image directory
        image_dir = _find_image_dir(v2_dir)
        if image_dir:
            all_detections, all_ground_truth, sample_frames = _test_on_images(
                model, image_dir, gt_annotations,
                confidence_threshold, num_test_frames
            )
        else:
            print("  ERROR: No usable video or images found.")
            return {"verdict": "fail", "error": "no_data_found"}

    # Compute batch metrics
    metrics = compute_detection_metrics_batch(all_detections, all_ground_truth)
    print(f"\n  Detection Results:")
    print(f"    Mean Precision: {metrics['mean_precision']:.3f}")
    print(f"    Mean Recall: {metrics['mean_recall']:.3f}")
    print(f"    Mean F1: {metrics['mean_f1']:.3f}")
    print(f"    False positives/frame: {metrics['false_positives_per_frame']:.1f}")
    print(f"    Frames tested: {metrics['total_frames']}")

    # Save detection overlay for a sample frame
    if sample_frames:
        frame_id, frame, dets, gts = sample_frames[0]
        save_detection_overlay(
            frame, dets, gts,
            str(output / "detection_overlay.png"),
            title=f"Detection Test — Frame {frame_id}"
        )

    # Determine verdict
    recall = metrics["mean_recall"]
    if recall > 0.85:
        verdict = "pass"
    elif recall > 0.70:
        verdict = "marginal"
    else:
        verdict = "fail"

    results = {
        **metrics,
        "confidence_threshold": confidence_threshold,
        "model": "yolov8s",
        "verdict": verdict,
    }

    _print_verdict(verdict, recall)
    return results


def _test_on_video(model, video_path: str, gt_annotations: dict,
                   confidence_threshold: float, num_frames: int):
    """Run detection test on video frames with enhanced pipeline."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Uniform sampling across the entire video for better coverage
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    all_detections = {}
    all_ground_truth = {}
    sample_frames = []
    frames_tested = 0

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue

        # Run enhanced detection (tiling + multiscale + adaptive NMS)
        # Note: Do NOT undistort — GT annotations are in original pixel coords
        detections = enhanced_detect(
            model, frame, confidence=confidence_threshold,
            use_tiling=True, use_multiscale=True,
            use_adaptive_nms=True, undistort_map=None,
        )

        all_detections[int(i)] = detections

        # Get ground truth for this frame
        fi = int(i)
        if gt_annotations and fi in gt_annotations:
            all_ground_truth[fi] = gt_annotations[fi]
        else:
            all_ground_truth[fi] = _get_nearest_gt(gt_annotations, fi)

        # Keep first few frames for visualization
        if len(sample_frames) < 3:
            sample_frames.append((fi, frame.copy(), detections,
                                  all_ground_truth.get(fi, [])))

        frames_tested += 1
        if frames_tested % 10 == 0:
            print(f"    Processed {frames_tested}/{num_frames} frames...")

    cap.release()
    return all_detections, all_ground_truth, sample_frames


def _test_on_images(model, image_dir: Path, gt_annotations: dict,
                    confidence_threshold: float, num_frames: int):
    """Run detection test on image frames."""
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    sample_interval = max(1, len(image_files) // num_frames)

    all_detections = {}
    all_ground_truth = {}
    sample_frames = []
    frames_tested = 0

    for idx in range(0, len(image_files), sample_interval):
        if frames_tested >= num_frames:
            break

        img_path = image_files[idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Run YOLO detection
        frame_w = frame.shape[1]
        imgsz = 1280 if frame_w <= 1920 else min(3200, frame_w // 2)
        results = model(frame, classes=[0], conf=confidence_threshold, imgsz=imgsz, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append((float(x1), float(y1), float(x2), float(y2)))

        all_detections[idx] = detections

        # Get ground truth
        if gt_annotations and idx in gt_annotations:
            all_ground_truth[idx] = gt_annotations[idx]
        else:
            all_ground_truth[idx] = _get_nearest_gt(gt_annotations, idx)

        if len(sample_frames) < 3:
            sample_frames.append((idx, frame.copy(), detections,
                                  all_ground_truth.get(idx, [])))

        frames_tested += 1

    return all_detections, all_ground_truth, sample_frames


def _run_synthetic_detection_test(model, output: Path,
                                   confidence_threshold: float) -> dict:
    """
    Run detection on a synthetic/simple scene to verify model works.
    Used when real dataset is not available.
    """
    print("  Creating synthetic test scene...")

    # Create a green field background with simulated player-like regions
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (34, 120, 50)  # Green field

    # The YOLO model won't detect our synthetic blobs as people,
    # but at least verifies the model runs
    results = model(frame, classes=[0], conf=confidence_threshold, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append((float(x1), float(y1), float(x2), float(y2)))

    print(f"    Detections on green field: {len(detections)} (expected 0)")

    # Model runs successfully — at least validates setup
    results_dict = {
        "mean_precision": 0.0,
        "mean_recall": 0.0,
        "mean_f1": 0.0,
        "false_positives_per_frame": len(detections),
        "total_frames": 1,
        "confidence_threshold": confidence_threshold,
        "model": "yolov8n",
        "verdict": "marginal",
        "notes": "Synthetic test — download SoccerTrack v2 for real validation",
    }

    print("  ⚠️  VERDICT: MARGINAL — Model loads OK but no real test data available")
    return results_dict


def _load_ground_truth(v2_dir: Path) -> dict:
    """Load ground truth annotations from SoccerTrack v2 format."""
    # SoccerTrack v2 uses CSV format with columns:
    # frame, track_id, x, y, w, h, ...
    gt_files = list(v2_dir.rglob("*gt*.txt")) + list(v2_dir.rglob("*gt*.csv"))

    if not gt_files:
        # Try MOT format
        gt_files = list(v2_dir.rglob("gt/gt.txt"))

    if not gt_files:
        return None

    gt_path = gt_files[0]
    print(f"  Ground truth: {gt_path}")

    annotations = {}
    try:
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    frame_id = int(parts[0])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    # Convert to x1, y1, x2, y2
                    bbox = (x, y, x + w, y + h)
                    if frame_id not in annotations:
                        annotations[frame_id] = []
                    annotations[frame_id].append(bbox)
    except Exception as e:
        print(f"  Warning: Error loading GT file: {e}")
        return None

    print(f"  Loaded GT for {len(annotations)} frames")
    return annotations


def _get_nearest_gt(gt_annotations: dict, frame_id: int) -> list:
    """Get ground truth for nearest available frame."""
    if not gt_annotations:
        return []

    available_frames = sorted(gt_annotations.keys())
    if not available_frames:
        return []

    # Find closest frame
    idx = np.searchsorted(available_frames, frame_id)
    if idx == 0:
        return gt_annotations[available_frames[0]]
    if idx >= len(available_frames):
        return gt_annotations[available_frames[-1]]

    # Choose closer of the two adjacent frames
    before = available_frames[idx - 1]
    after = available_frames[idx]
    if abs(frame_id - before) <= abs(frame_id - after):
        return gt_annotations[before]
    return gt_annotations[after]


def _find_video(directory: Path) -> Path:
    """Find a video file in directory."""
    extensions = [".mp4", ".avi", ".mov", ".mkv"]
    for ext in extensions:
        videos = list(directory.rglob(f"*{ext}"))
        if videos:
            return videos[0]
    return None


def _find_image_dir(directory: Path) -> Path:
    """Find a directory with image frames."""
    for subdir in directory.rglob("*"):
        if subdir.is_dir():
            images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            if len(images) > 10:
                return subdir
    return None


def _print_verdict(verdict: str, recall: float):
    """Print colored verdict."""
    if verdict == "pass":
        print(f"\n  ✅ STEP 2 VERDICT: PASS — Recall {recall:.1%} > 85% threshold")
    elif verdict == "marginal":
        print(f"\n  ⚠️  STEP 2 VERDICT: MARGINAL — Recall {recall:.1%} (70-85% range)")
    else:
        print(f"\n  ❌ STEP 2 VERDICT: FAIL — Recall {recall:.1%} < 70% threshold")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test player detection accuracy")
    parser.add_argument("--v2_path", default="data/soccertrack_v2/",
                        help="Path to SoccerTrack v2 data")
    parser.add_argument("--output", default="test_outputs/",
                        help="Output directory for test results")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of test frames")
    args = parser.parse_args()

    results = run_detection_test(args.v2_path, args.output, args.conf, args.frames)
    print(f"\nResults: {results}")
