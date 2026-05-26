#!/usr/bin/env python3
"""Soccer Pipeline Accuracy Test — Main Runner.

Runs all pipeline validation steps and produces a final accuracy report
with a PASS/FAIL verdict and hardware purchase recommendation.

Usage:
    python test_pipeline.py --v1_path data/soccertrack_v1/ --v2_path data/soccertrack_v2/
    python test_pipeline.py --v2_path data/soccertrack_v2/ --output results/
    python test_pipeline.py --synthetic  # Run with synthetic data only (no datasets needed)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_undistortion import run_undistortion_test
from test_detection import run_detection_test
from test_homography import run_homography_test
from test_tracking import run_tracking_test
from test_integration import run_integration_test
from utils.metrics import compute_overall_verdict
from utils.visualization import save_metrics_summary


def main():
    parser = argparse.ArgumentParser(
        description="Soccer Pipeline Accuracy Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full test with both datasets
  python test_pipeline.py --v1_path data/soccertrack_v1/ --v2_path data/soccertrack_v2/

  # Test with only SoccerTrack v2 (skip undistortion)
  python test_pipeline.py --v2_path data/soccertrack_v2/

  # Quick synthetic test (no datasets needed)
  python test_pipeline.py --synthetic

  # Custom field dimensions (7v7 field)
  python test_pipeline.py --v2_path data/ --field_length 68 --field_width 48

Pass/Fail Criteria:
  | Metric               | Pass    | Marginal  | Fail    |
  |---------------------|---------|-----------|---------|
  | Field line straight | Visual  | Edge curve| Curved  |
  | Detection recall    | > 85%   | 70-85%    | < 70%   |
  | Position error mean | < 1.5m  | 1.5-2.5m  | > 2.5m  |
  | MOTA tracking       | > 0.60  | 0.45-0.60 | < 0.45  |
  | IDF1 score          | > 0.55  | 0.40-0.55 | < 0.40  |
  | ID switches/min     | < 20    | 20-40     | > 40    |
        """
    )

    parser.add_argument("--v1_path", type=str, default=None,
                        help="Path to SoccerTrack v1 fisheye data (for undistortion test)")
    parser.add_argument("--v2_path", type=str, default=None,
                        help="Path to SoccerTrack v2 data (for detection/tracking/homography)")
    parser.add_argument("--output", type=str, default="test_outputs/",
                        help="Output directory for test results and visualizations")
    parser.add_argument("--field_length", type=float, default=105.0,
                        help="Field length in meters (default: 105 for full-size)")
    parser.add_argument("--field_width", type=float, default=68.0,
                        help="Field width in meters (default: 68 for full-size)")
    parser.add_argument("--fov", type=float, default=200.0,
                        help="Fisheye field of view in degrees (for undistortion)")
    parser.add_argument("--confidence", type=float, default=0.15,
                        help="YOLO detection confidence threshold")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames to test per step")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run with synthetic data only (no datasets needed)")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="YOLO model weights (e.g. yolov8s.pt or path to fine-tuned)")
    parser.add_argument("--skip_undistortion", action="store_true",
                        help="Skip undistortion test (if no fisheye data)")
    parser.add_argument("--skip_integration", action="store_true",
                        help="Skip integration test")

    args = parser.parse_args()

    # Validate paths
    if not args.synthetic:
        if args.v1_path is None and args.v2_path is None:
            print("ERROR: Must specify --v1_path and/or --v2_path, or use --synthetic")
            print("       Run with --help for usage details.")
            sys.exit(1)

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use synthetic paths if none provided
    v1_path = args.v1_path or str(output_dir / "synthetic_v1")
    v2_path = args.v2_path or str(output_dir / "synthetic_v2")

    # Header
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   SOCCER PIPELINE ACCURACY TEST SUITE                    ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<47} ║")
    print(f"║   Field: {args.field_length}m × {args.field_width}m" +
          " " * (48 - len(f"{args.field_length}m × {args.field_width}m")) + "║")
    if args.v1_path:
        v1_display = str(args.v1_path)[:44]
        print(f"║   V1: {v1_display:<51} ║")
    if args.v2_path:
        v2_display = str(args.v2_path)[:44]
        print(f"║   V2: {v2_display:<51} ║")
    if args.synthetic:
        print(f"║   Mode: SYNTHETIC (no real datasets)                    ║")
    print("╚" + "═" * 58 + "╝")

    start_time = time.time()
    all_results = {}

    # === STEP 1: Undistortion ===
    if not args.skip_undistortion:
        try:
            undist_results = run_undistortion_test(
                v1_path, str(output_dir / "step1_undistortion"), args.fov
            )
            all_results["undistortion"] = undist_results
        except Exception as e:
            print(f"\n  ❌ Step 1 EXCEPTION: {e}")
            all_results["undistortion"] = {"verdict": "fail", "error": str(e)}
    else:
        print("\n  ⏭️  Skipping undistortion test")
        all_results["undistortion"] = {"verdict": "pass", "notes": "skipped"}

    # === STEP 2: Detection ===
    try:
        det_results = run_detection_test(
            v2_path, str(output_dir / "step2_detection"),
            args.confidence, args.frames
        )
        all_results["player_detection"] = det_results
    except Exception as e:
        print(f"\n  ❌ Step 2 EXCEPTION: {e}")
        all_results["player_detection"] = {"verdict": "fail", "error": str(e)}

    # === STEP 3: Homography ===
    try:
        homo_results = run_homography_test(
            v2_path, str(output_dir / "step3_homography"),
            args.field_length, args.field_width
        )
        all_results["homography"] = homo_results
    except Exception as e:
        print(f"\n  ❌ Step 3 EXCEPTION: {e}")
        all_results["homography"] = {"verdict": "fail", "error": str(e)}

    # === STEP 4: Tracking ===
    # Tracking needs more frames for accurate per-minute switch rate
    tracking_frames = min(args.frames * 5, 750)  # Use up to 30s of video
    try:
        track_results = run_tracking_test(
            v2_path, str(output_dir / "step4_tracking"),
            args.confidence, tracking_frames
        )
        all_results["tracking"] = track_results
    except Exception as e:
        print(f"\n  ❌ Step 4 EXCEPTION: {e}")
        all_results["tracking"] = {"verdict": "fail", "error": str(e)}

    # === STEP 5: Integration ===
    if not args.skip_integration:
        try:
            integ_results = run_integration_test(
                v2_path, str(output_dir / "step5_integration"),
                args.field_length, args.field_width,
                min(args.frames, 150)
            )
            all_results["integration"] = integ_results
        except Exception as e:
            print(f"\n  ❌ Step 5 EXCEPTION: {e}")
            all_results["integration"] = {"verdict": "fail", "error": str(e)}
    else:
        print("\n  ⏭️  Skipping integration test")
        all_results["integration"] = {"verdict": "pass", "notes": "skipped"}

    # === OVERALL VERDICT ===
    total_time = time.time() - start_time

    # Compute overall verdict
    detection_metrics = all_results.get("player_detection", {})
    homography_metrics = all_results.get("homography", {})
    tracking_metrics = all_results.get("tracking", {})
    undist_verdict = all_results.get("undistortion", {}).get("verdict", "pass")

    overall = compute_overall_verdict(
        detection_metrics, homography_metrics, tracking_metrics, undist_verdict
    )

    # Build final report
    report = {
        "test_date": datetime.now().isoformat(),
        "test_duration_s": round(total_time, 1),
        "configuration": {
            "field_length_m": args.field_length,
            "field_width_m": args.field_width,
            "fov_degrees": args.fov,
            "confidence_threshold": args.confidence,
            "frames_tested": args.frames,
            "v1_path": args.v1_path,
            "v2_path": args.v2_path,
            "synthetic_mode": args.synthetic,
        },
        "results": all_results,
        "overall_verdict": overall["overall_verdict"],
        "per_component_verdicts": overall["per_component"],
        "recommendation": overall["recommendation"],
    }

    # Save report
    report_path = output_dir / "accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  📄 Report saved: {report_path}")

    # Save visual summary
    try:
        summary_metrics = {
            "player_detection": detection_metrics,
            "homography": homography_metrics,
            "tracking": tracking_metrics,
            "overall_verdict": overall["overall_verdict"],
        }
        save_metrics_summary(summary_metrics, str(output_dir / "accuracy_summary.png"))
    except Exception as e:
        print(f"  Warning: Could not save visual summary: {e}")

    # Print final verdict
    _print_final_verdict(overall, total_time, report_path)

    # Exit with appropriate code
    if overall["overall_verdict"] == "PASS":
        sys.exit(0)
    elif overall["overall_verdict"] == "MARGINAL":
        sys.exit(0)  # Don't fail CI for marginal
    else:
        sys.exit(1)


def _print_final_verdict(overall: dict, total_time: float, report_path: Path):
    """Print the final test summary."""
    print("\n\n" + "╔" + "═" * 58 + "╗")
    print("║                    FINAL RESULTS                         ║")
    print("╠" + "═" * 58 + "╣")

    # Per-component
    for component, verdict in overall["per_component"].items():
        icon = "✅" if verdict == "pass" else "⚠️" if verdict == "marginal" else "❌"
        name = component.replace("_", " ").title()
        print(f"║   {icon} {name:<25} {verdict.upper():<24} ║")

    print("╠" + "═" * 58 + "╣")

    # Overall
    verdict = overall["overall_verdict"]
    if verdict == "PASS":
        print("║                                                          ║")
        print("║   ✅✅✅  OVERALL VERDICT: PASS  ✅✅✅                   ║")
        print("║                                                          ║")
        print("║   🎉 Pipeline accuracy meets all thresholds.             ║")
        print("║   💰 SAFE TO PURCHASE HARDWARE ($290 CAD)                ║")
    elif verdict == "MARGINAL":
        print("║                                                          ║")
        print("║   ⚠️⚠️⚠️  OVERALL VERDICT: MARGINAL  ⚠️⚠️⚠️              ║")
        print("║                                                          ║")
        print("║   Some metrics are borderline. Review before purchase.   ║")
        rec = overall["recommendation"]
        print(f"║   {rec:<55} ║")
    else:
        print("║                                                          ║")
        print("║   ❌❌❌  OVERALL VERDICT: FAIL  ❌❌❌                   ║")
        print("║                                                          ║")
        print("║   Pipeline does not meet accuracy requirements.          ║")
        print("║   DO NOT PURCHASE HARDWARE until issues are resolved.    ║")
        rec = overall["recommendation"]
        if len(rec) > 55:
            print(f"║   {rec[:55]} ║")
            print(f"║   {rec[55:]:<55} ║")
        else:
            print(f"║   {rec:<55} ║")

    print("╠" + "═" * 58 + "╣")
    print(f"║   Total test time: {total_time:.1f}s" +
          " " * (38 - len(f"{total_time:.1f}s")) + "║")
    print(f"║   Report: {str(report_path)[:46]:<47} ║")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()
