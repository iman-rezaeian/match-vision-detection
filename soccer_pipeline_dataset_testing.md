# Soccer CV Pipeline — Dataset Testing Guide
### Prompt for Claude Opus — Validate Pipeline Before Hardware Purchase

---

## CONTEXT

I am building a local computer vision pipeline on a MacBook (Apple Silicon M-series) to analyze U10 youth soccer games. Before purchasing hardware (Moment Fisheye 14mm T-Series lens + iPhone 15 Pro Max mount, ~$360 CAD), I need to validate my pipeline works correctly using free open-source datasets.

My recording setup will be:
- **iPhone 15 Pro Max** mounted on a 14ft tripod at mid-field
- **Moment Fisheye 14mm T-Series lens** — 200° field of view
- **Key property:** This is a TRUE fisheye lens with significant barrel distortion. It is NOT a clean panoramic. Straight lines near edges bend noticeably toward the center of the frame.

My pipeline (already specced in a separate document) uses:
- YOLOv8 + ByteTrack for player detection and tracking
- OpenCV homography for pixel → real field coordinate mapping
- mplsoccer for heatmaps and passing networks
- MediaPipe for gait signatures
- InsightFace for face recognition

---

## THE TWO DATASETS TO USE

---

### Dataset 1 — SoccerTrack v1 (Fisheye Camera)

**Purpose:** Validate that the fisheye undistortion step works correctly before any tracking.

**GitHub:** https://github.com/Damstagram/SoccerTrack
**Also at:** https://github.com/changjo/SoccerTrack

**What it contains:**
- Real soccer game footage recorded with an **8K fisheye camera** — same type of barrel distortion my Moment lens will produce
- Also contains drone (top-down) footage — ignore this for our purposes
- Player position annotations for the first 30 minutes
- CSV ground truth files with bounding boxes and track IDs

**Why we need it:**
The Moment Fisheye 14mm produces significant barrel distortion — straight lines bend toward the center, and player positions near the edges of the frame are geometrically inaccurate. Before any player tracking can work reliably, every frame must be undistorted using OpenCV's fisheye calibration model. SoccerTrack v1 gives us real fisheye footage to test this on.

**Download:**
```python
# From the GitHub README — download links in the repo
# Dataset available at links provided in the SoccerTrack repo
# Requires registration or direct download from repo releases
```

---

### Dataset 2 — SoccerTrack v2 (Panoramic Camera with Ground Truth)

**Purpose:** Measure actual numerical tracking accuracy against known correct answers.

**Website:** https://atomscott.github.io/SoccerTrack-v2/
**GitHub:** https://github.com/AtomScott/SoccerTrack (same team, v2 branch)

**What it contains:**
- 10 full-length 4K panoramic soccer matches
- Per-frame Game State Reconstruction (GSR) annotations — exact player positions every frame
- Ball action spotting annotations (passes, shots, goals)
- Camera calibration parameters included
- Open access — no NDA or registration required

**Why we need it:**
V2 has ground truth annotations meaning we can compute exact accuracy metrics:
- How often does our tracker correctly identify the same player across frames?
- What is our position error in meters vs ground truth?
- How well does our homography map pixel positions to real field coordinates?

Without ground truth we can only eyeball results. With it we get real numbers like "78% tracking accuracy" which tells us if the pipeline is good enough for coaching use before spending money on hardware.

**Install SportsLabKit (built by same team as SoccerTrack v2):**
```bash
pip install sportsLabKit
```

SportsLabKit has native support for loading and working with SoccerTrack v2 data format.

---

## THE TESTING PIPELINE — WHAT TO BUILD

Build a testing script that runs in this exact order:

---

### Step 1 — Fisheye Undistortion (on SoccerTrack v1 footage)

This is the most critical step. Every downstream component depends on having undistorted frames.

```python
import cv2
import numpy as np

def calibrate_fisheye_from_checkerboard(checkerboard_video_path, 
                                         checkerboard_size=(9,6)):
    """
    Option A: Calibrate from a checkerboard pattern video.
    Best accuracy. Record 30 seconds of a checkerboard pattern
    from the same camera + lens before the game.
    
    Returns K (camera matrix) and D (distortion coefficients)
    for use with cv2.fisheye.undistortImage()
    """

def calibrate_fisheye_approximate(frame_h, frame_w, fov_degrees=200):
    """
    Option B: Approximate calibration from known FOV.
    Use when no checkerboard is available.
    Less accurate but workable for testing.
    
    For Moment Fisheye 14mm T-Series: fov_degrees=200
    Returns approximate K and D matrices.
    """
    # Focal length approximation for fisheye
    f = frame_w / (2 * np.tan(np.radians(fov_degrees / 2)))
    K = np.array([[f, 0, frame_w/2],
                  [0, f, frame_h/2],
                  [0, 0, 1]], dtype=np.float64)
    D = np.zeros((4, 1))  # Will be refined during testing
    return K, D

def undistort_frame(frame, K, D):
    """
    Apply fisheye undistortion to a single frame.
    Returns undistorted frame ready for player detection.
    
    Uses cv2.fisheye.undistortImage() NOT cv2.undistort()
    These are different — fisheye model is required for 200° FOV lenses.
    """
    h, w = frame.shape[:2]
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]  # Scale factor to reduce cropping
    undistorted = cv2.fisheye.undistortImage(frame, K, D, Knew=Knew)
    return undistorted
```

**Validation test for Step 1:**
- Run undistortion on 10 frames from SoccerTrack v1 fisheye footage
- Visually confirm: field lines that were curved in raw footage are now straight
- Save side-by-side comparison images: raw vs undistorted
- If field lines are not straight → adjust D coefficients until they are

---

### Step 2 — Player Detection (on undistorted frames)

```python
from ultralytics import YOLO
import supervision as sv

def test_player_detection(undistorted_frame, confidence=0.35):
    """
    Run YOLOv8 on undistorted frame.
    
    Validation criteria:
    - All visible players should be detected (check against ground truth count)
    - No false positives (referee, coaches on sideline flagged)
    - Bounding boxes should tightly fit players not extend into crowd
    
    Returns detections with confidence scores.
    """
    model = YOLO("yolov8n.pt")
    results = model(undistorted_frame, classes=[0], conf=confidence)
    return results
```

**Validation test for Step 2:**
- Run on 50 frames from SoccerTrack v1
- Count detected players vs ground truth player count per frame
- Target: detect >85% of visible players per frame
- Log frames where detection count is significantly wrong

---

### Step 3 — Homography Calibration (pixel → field meters)

```python
def test_homography_accuracy(detections_df, ground_truth_df, H):
    """
    Compare pipeline's field coordinates against ground truth positions.
    
    For each frame:
    1. Match detected player to nearest ground truth player
    2. Compute position error in meters
    3. Report mean error, median error, 95th percentile error
    
    Acceptable accuracy for coaching use: mean error < 1.5 meters
    Players are ~0.5m wide so 1.5m error still allows meaningful analysis.
    """
    errors = []
    for frame_id in detections_df['frame'].unique():
        detected = detections_df[detections_df['frame'] == frame_id]
        truth = ground_truth_df[ground_truth_df['frame'] == frame_id]
        # Hungarian matching between detected and ground truth positions
        # Compute Euclidean distance error per matched pair
        ...
    return {
        'mean_error_m': np.mean(errors),
        'median_error_m': np.median(errors),
        'p95_error_m': np.percentile(errors, 95),
        'pct_within_1m': (np.array(errors) < 1.0).mean() * 100,
        'pct_within_2m': (np.array(errors) < 2.0).mean() * 100,
    }
```

**Validation test for Step 3:**
- Run homography on SoccerTrack v2 (has camera calibration parameters included)
- Use included calibration as starting point for H matrix
- Report position error metrics
- Target: mean error < 1.5 meters for coaching-grade analysis

---

### Step 4 — Multi-Object Tracking (ByteTrack ID consistency)

```python
def test_tracking_consistency(video_path, ground_truth_tracking_csv):
    """
    Measure how consistently ByteTrack maintains player IDs over time.
    
    Key metrics:
    - ID switches: how many times does a player get a new track ID?
    - Track fragmentation: how many separate track segments per player?
    - Identity preservation through occlusion: does ID survive when 
      player is temporarily hidden by others?
    
    Use SoccerTrack v2 ground truth track IDs for comparison.
    Report using standard MOT metrics: MOTA, MOTP, IDF1
    """
    # pip install motmetrics
    import motmetrics as mm
    ...
```

**Validation test for Step 4:**
- Run full tracking pipeline on 5 complete minutes from SoccerTrack v2
- Compute MOT metrics against ground truth
- Target metrics for coaching-grade use:
  - MOTA (Multiple Object Tracking Accuracy) > 0.60
  - IDF1 (ID F1 Score) > 0.55
  - ID switches < 20 per minute of footage
- These are realistic targets for a single wide-angle camera — not broadcast quality

---

### Step 5 — End-to-End Integration Test

```python
def run_full_pipeline_test(video_clip_path, ground_truth_path,
                            output_dir="test_outputs/"):
    """
    Run complete pipeline on a 5-minute clip from SoccerTrack v2.
    
    Steps:
    1. Load video
    2. Apply fisheye undistortion to each frame
    3. Run YOLOv8 detection
    4. Run ByteTrack tracking
    5. Apply homography to get field coordinates
    6. Classify teams by jersey color
    7. Generate heatmaps for each tracked player
    8. Generate passing network
    9. Calculate stats (distance, speed, zones)
    10. Compare all outputs against ground truth
    
    Saves to output_dir:
    - undistortion_comparison.png (raw vs undistorted side by side)
    - detection_sample.png (frame with bounding boxes)
    - tracking_trajectories.png (all player paths on pitch)
    - heatmap_player_1.png through heatmap_player_N.png
    - passing_network.png
    - stats_table.csv
    - accuracy_report.json (all metrics)
    """
```

---

## ACCURACY REPORT FORMAT

The end-to-end test should produce a JSON report like this:

```json
{
  "test_date": "2026-05-03",
  "dataset": "SoccerTrack_v2_match_001",
  "clip_duration_s": 300,
  "fisheye_undistortion": {
    "method": "approximate|checkerboard",
    "field_line_straightness": "pass|fail",
    "notes": ""
  },
  "player_detection": {
    "mean_recall": 0.87,
    "mean_precision": 0.91,
    "false_positives_per_frame": 0.3
  },
  "homography": {
    "mean_position_error_m": 1.2,
    "median_position_error_m": 0.9,
    "p95_position_error_m": 2.8,
    "pct_within_1m": 62.0,
    "pct_within_2m": 88.0
  },
  "tracking": {
    "MOTA": 0.68,
    "MOTP": 0.71,
    "IDF1": 0.61,
    "id_switches_per_minute": 14.2,
    "track_fragments_per_player": 2.1
  },
  "overall_verdict": "PASS|FAIL|MARGINAL",
  "recommendation": "Safe to purchase hardware|Needs improvement before purchase"
}
```

---

## PASS/FAIL CRITERIA

| Metric | Pass (buy hardware) | Marginal (investigate) | Fail (fix first) |
|---|---|---|---|
| Field line straightness after undistortion | Visually straight | Slight curve at edges | Still curved |
| Player detection recall | > 85% | 70-85% | < 70% |
| Position error mean | < 1.5m | 1.5-2.5m | > 2.5m |
| MOTA tracking | > 0.60 | 0.45-0.60 | < 0.45 |
| IDF1 score | > 0.55 | 0.40-0.55 | < 0.40 |
| ID switches/min | < 20 | 20-40 | > 40 |

---

## KNOWN CHALLENGES TO WATCH FOR

**1. Fisheye undistortion crop:**
After undistorting a 200° fisheye image, significant portions of the frame corners become black (invalid pixels). Adjust the `Knew` scaling factor in `undistort_frame()` to control the trade-off between keeping more field vs more distortion at edges. Typical value: 0.3-0.5.

**2. Team classification on SoccerTrack v1:**
The fisheye footage may have different jersey colors than your actual U10 team. KMeans team classification still works but verify the two clusters make sense visually before trusting results.

**3. Scale differences between datasets and real use:**
SoccerTrack v1 is an 8K fisheye camera — much higher resolution than iPhone. SoccerTrack v2 is 4K panoramic — different distortion profile than fisheye. Neither is a perfect match to iPhone 15 Pro Max + Moment lens. Results will be slightly better on the datasets than on real iPhone footage. Factor in ~10-15% accuracy reduction for real game footage.

**4. Frame rate differences:**
SoccerTrack footage may be 25fps. Your iPhone records at 30fps (or 60fps). The pipeline's speed calculations use fps — make sure to read the correct fps from the video metadata, not hardcode it.

**5. Field size:**
SoccerTrack games are full 11v11 on a full-size pitch (~105m × 68m). Your U10 7v7 field is 50m × 35m. The homography step must use the correct field dimensions. The tracking pipeline itself doesn't care about field size but the homography and all downstream coordinate calculations do.

---

## ENVIRONMENT SETUP

```bash
# Create virtual environment
python3 -m venv test_env
source test_env/bin/activate

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics supervision opencv-python numpy pandas matplotlib
pip install mplsoccer scipy scikit-learn

# SoccerTrack v2 / SportsLabKit
pip install sportsLabKit

# MOT evaluation metrics
pip install motmetrics

# Optional: for loading SoccerTrack v1 annotations
pip install pandas openpyxl
```

---

## FILE STRUCTURE FOR TESTS

```
soccer_pipeline_tests/
├── test_pipeline.py              # Main test runner
├── test_undistortion.py          # Step 1 — fisheye undistortion only
├── test_detection.py             # Step 2 — YOLOv8 detection only  
├── test_homography.py            # Step 3 — homography accuracy
├── test_tracking.py              # Step 4 — ByteTrack MOT metrics
├── test_integration.py           # Step 5 — full end-to-end
├── utils/
│   ├── fisheye.py                # Undistortion helpers
│   ├── metrics.py                # Accuracy computation
│   └── visualization.py         # Test output plots
├── data/
│   ├── soccertrack_v1/           # Downloaded v1 fisheye clips
│   └── soccertrack_v2/           # Downloaded v2 panoramic clips
└── test_outputs/                 # All generated test results
```

---

## INSTRUCTIONS FOR CLAUDE OPUS

Build all test files completely. Run them in order (Step 1 through Step 5). After each step, print clear PASS/FAIL/MARGINAL verdict with the key metric values.

The goal is a single command that runs everything:

```bash
python test_pipeline.py --v1_path data/soccertrack_v1/ \
                        --v2_path data/soccertrack_v2/ \
                        --output test_outputs/ \
                        --field_length 50 \
                        --field_width 35
```

And produces a final `accuracy_report.json` with the overall verdict.

If overall verdict is PASS → print:
```
✅ Pipeline validated. Safe to purchase:
   - Moment Fisheye 14mm T-Series lens (~$200 CAD)
   - Moment iPhone 15 Pro Max T-Series Case (~$55 CAD)
   - Joby GripTight PRO 2 Mount (~$35 CAD)
   Total: ~$290 CAD
```

If overall verdict is FAIL or MARGINAL → print which specific step failed and what to investigate before purchasing hardware.
