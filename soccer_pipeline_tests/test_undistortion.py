"""Step 1 — Fisheye Undistortion Test.

Validates that the fisheye undistortion step works correctly
using SoccerTrack v1 fisheye footage.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.fisheye import FisheyeCalibrator, undistort_frame, check_line_straightness
from utils.visualization import save_undistortion_comparison
from utils.soccertrack_loader import find_video_file


def run_undistortion_test(v1_path: str, output_dir: str,
                          fov_degrees: float = 200.0,
                          num_test_frames: int = 10) -> dict:
    """
    Run fisheye undistortion test on SoccerTrack v1 footage.

    Steps:
    1. Load fisheye video
    2. Calibrate using approximate FOV method (or checkerboard if available)
    3. Undistort test frames
    4. Measure field line straightness
    5. Save comparison images

    Returns: dict with verdict and metrics
    """
    print("\n" + "=" * 60)
    print("STEP 1: FISHEYE UNDISTORTION TEST")
    print("=" * 60)

    v1_dir = Path(v1_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Find video file in the v1 directory
    video_path = find_video_file(v1_dir)
    if video_path is None:
        video_path = _find_video(v1_dir)
    if video_path is None:
        print("  ⚠️  No video found in SoccerTrack v1 path.")
        print("  Creating synthetic fisheye test frames instead...")
        return _run_synthetic_test(output, fov_degrees)

    print(f"  Video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return {"verdict": "fail", "error": "cannot_open_video"}

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Resolution: {frame_w}×{frame_h}")
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}")

    # Calibrate
    calibrator = FisheyeCalibrator()

    # Try checkerboard calibration first
    checkerboard_video = _find_checkerboard_video(v1_dir)
    if checkerboard_video:
        print(f"  Found checkerboard video: {checkerboard_video}")
        success = calibrator.calibrate_from_checkerboard(str(checkerboard_video))
        method = "checkerboard"
    else:
        print(f"  No checkerboard video found. Using approximate calibration (FOV={fov_degrees}°)")
        success = calibrator.calibrate_approximate(frame_h, frame_w, fov_degrees)
        method = "approximate"

    if not success:
        cap.release()
        return {"verdict": "fail", "error": "calibration_failed", "method": method}

    # Test undistortion on sample frames
    print(f"\n  Testing undistortion on {num_test_frames} frames...")
    sample_interval = max(1, total_frames // num_test_frames)

    straightness_results = []
    frames_tested = 0

    for i in range(0, total_frames, sample_interval):
        if frames_tested >= num_test_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # Undistort
        undistorted = calibrator.undistort(frame)

        # Check line straightness
        raw_lines = check_line_straightness(frame)
        undist_lines = check_line_straightness(undistorted)
        straightness_results.append({
            "frame": i,
            "raw_lines": raw_lines["lines_detected"],
            "undist_lines": undist_lines["lines_detected"],
            "undist_long_lines": undist_lines["long_lines"],
            "undist_verdict": undist_lines["verdict"],
        })

        # Save first comparison image
        if frames_tested == 0:
            save_undistortion_comparison(
                frame, undistorted,
                str(output / "undistortion_comparison.png"),
                title=f"Fisheye Undistortion ({method} calibration)"
            )

        # Save additional frames
        if frames_tested < 3:
            save_undistortion_comparison(
                frame, undistorted,
                str(output / f"undistortion_frame_{i:05d}.png"),
                title=f"Frame {i}"
            )

        frames_tested += 1

    cap.release()

    # Save calibration for downstream use
    calibrator.save_calibration(str(output / "fisheye_calibration.npz"))

    # Analyze results
    pass_count = sum(1 for r in straightness_results if r["undist_verdict"] == "pass")
    marginal_count = sum(1 for r in straightness_results if r["undist_verdict"] == "marginal")
    fail_count = sum(1 for r in straightness_results if r["undist_verdict"] == "fail")

    pass_rate = pass_count / len(straightness_results) if straightness_results else 0

    # Verdict
    if pass_rate >= 0.7:
        verdict = "pass"
    elif pass_rate >= 0.4 or marginal_count > fail_count:
        verdict = "marginal"
    else:
        verdict = "fail"

    results = {
        "method": method,
        "field_line_straightness": verdict,
        "frames_tested": frames_tested,
        "pass_count": pass_count,
        "marginal_count": marginal_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 3),
        "video_resolution": f"{frame_w}x{frame_h}",
        "verdict": verdict,
        "notes": f"Tested {frames_tested} frames with {method} calibration",
    }

    # Print summary
    print(f"\n  Results:")
    print(f"    Method: {method}")
    print(f"    Frames tested: {frames_tested}")
    print(f"    Pass: {pass_count} | Marginal: {marginal_count} | Fail: {fail_count}")
    print(f"    Pass rate: {pass_rate:.0%}")
    _print_verdict(verdict)

    return results


def _run_synthetic_test(output: Path, fov_degrees: float) -> dict:
    """
    Run undistortion test on a synthetically generated fisheye image.
    Used when no real dataset is available yet.
    """
    print("  Generating synthetic fisheye grid pattern...")

    # Create a clean grid image (simulating a flat field with lines)
    h, w = 1080, 1920
    clean = np.zeros((h, w, 3), dtype=np.uint8)
    clean[:] = (34, 42, 26)  # Dark green background

    # Draw straight grid lines
    for x in range(0, w, 100):
        cv2.line(clean, (x, 0), (x, h), (74, 122, 74), 2)
    for y in range(0, h, 100):
        cv2.line(clean, (0, y), (w, y), (74, 122, 74), 2)

    # Apply synthetic fisheye distortion
    K = np.array([[500, 0, w / 2], [0, 500, h / 2], [0, 0, 1]], dtype=np.float64)
    D = np.array([[-0.3], [0.1], [-0.02], [0.005]], dtype=np.float64)

    # Create distorted version using inverse fisheye mapping
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    # Apply inverse to get distorted image
    distorted = cv2.remap(clean, map1, map2, cv2.INTER_LINEAR)

    # Now undistort it back
    calibrator = FisheyeCalibrator()
    calibrator.K = K
    calibrator.D = D
    calibrator.Knew = K.copy()
    calibrator.Knew[(0, 1), (0, 1)] = 0.4 * calibrator.Knew[(0, 1), (0, 1)]
    calibrator.calibrated = True

    undistorted = calibrator.undistort(distorted, balance=0.5)

    # Save comparison
    save_undistortion_comparison(
        distorted, undistorted,
        str(output / "undistortion_comparison_synthetic.png"),
        title="Synthetic Fisheye Test (Grid Pattern)"
    )

    # Check lines
    raw_result = check_line_straightness(distorted)
    undist_result = check_line_straightness(undistorted)

    print(f"    Raw (distorted): {raw_result['lines_detected']} lines, "
          f"{raw_result['long_lines']} long")
    print(f"    Undistorted: {undist_result['lines_detected']} lines, "
          f"{undist_result['long_lines']} long")

    # For synthetic test, undistortion should work well since we know exact params
    verdict = "pass" if undist_result["long_lines"] > raw_result["long_lines"] else "marginal"

    results = {
        "method": "synthetic_test",
        "field_line_straightness": verdict,
        "frames_tested": 1,
        "pass_count": 1 if verdict == "pass" else 0,
        "marginal_count": 1 if verdict == "marginal" else 0,
        "fail_count": 0,
        "pass_rate": 1.0 if verdict == "pass" else 0.5,
        "video_resolution": f"{w}x{h}",
        "verdict": verdict,
        "notes": "Synthetic test — download SoccerTrack v1 for real validation",
    }

    _print_verdict(verdict)
    return results


def _find_video(directory: Path) -> Path:
    """Find a video file in directory."""
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"]
    for ext in extensions:
        videos = list(directory.rglob(f"*{ext}"))
        if videos:
            return videos[0]
    return None


def _find_checkerboard_video(directory: Path) -> Path:
    """Look for a checkerboard calibration video."""
    for pattern in ["*checker*", "*calib*", "*board*"]:
        for ext in [".mp4", ".avi", ".mov"]:
            matches = list(directory.rglob(f"{pattern}{ext}"))
            if matches:
                return matches[0]
    return None


def _print_verdict(verdict: str):
    """Print colored verdict."""
    if verdict == "pass":
        print(f"\n  ✅ STEP 1 VERDICT: PASS — Field lines are straight after undistortion")
    elif verdict == "marginal":
        print(f"\n  ⚠️  STEP 1 VERDICT: MARGINAL — Some curvature remains at edges")
    else:
        print(f"\n  ❌ STEP 1 VERDICT: FAIL — Lines still curved after undistortion")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test fisheye undistortion")
    parser.add_argument("--v1_path", default="data/soccertrack_v1/",
                        help="Path to SoccerTrack v1 data")
    parser.add_argument("--output", default="test_outputs/",
                        help="Output directory for test results")
    parser.add_argument("--fov", type=float, default=200.0,
                        help="Fisheye field of view in degrees")
    args = parser.parse_args()

    results = run_undistortion_test(args.v1_path, args.output, args.fov)
    print(f"\nResults: {results}")
