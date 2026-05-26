# Soccer Analyzer — Full Implementation Plan

## Overview

Three workstreams running in parallel:
1. **Pipeline Upgrades** — Bring every module to SOTA
2. **Training Mode** — New pipeline for practice/training sessions
3. **Diagnostics Suite** — Three diagnostic CLIs matching the three app modes

---

## WORKSTREAM 1: Pipeline SOTA Upgrades

### Phase 1A: Tracker Upgrade (P0) — BoT-SORT

**Why:** #1 bottleneck. 25,593 track fragments for ~14 players in 5 min. Everything downstream (team assignment, player ID, stats, passes, formations) degrades with fragmented tracks.

**Current:** ByteTrack (IoU-only) via `supervision==0.28`

**Target:** BoT-SORT with appearance Re-ID via `boxmot` library

**Changes:**
| File | Action |
|------|--------|
| `requirements.txt` | Add `boxmot>=10.0.77` |
| `pipeline/detector.py` | Replace `sv.ByteTrack` with `boxmot.BotSort` tracker |
| `pipeline/detector.py` | Add Re-ID feature extraction (lightweight OSNet-x0.25 from boxmot) |
| `config.py` | Add `DEFAULT_TRACKER = "botsort"`, `TRACKER_REID_WEIGHTS = "osnet_x0_25_msmt17"` |
| `diagnostics.py` | Add `--tracker` arg (bytetrack/botsort/deepocsort) for A/B comparison |

**Implementation:**
```python
# detector.py — new tracker init
from boxmot import BotSort

class VideoDetector:
    def __init__(self, ..., tracker_type="botsort"):
        if tracker_type == "botsort":
            self.tracker = BotSort(
                reid_weights=Path("osnet_x0_25_msmt17.pt"),
                device=self.device,
                track_high_thresh=self.confidence,
                track_low_thresh=0.1,
                new_track_thresh=self.confidence,
                track_buffer=120 // self.sample_rate,  # ~20s buffer
                match_thresh=0.8,  # appearance + IoU fusion
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                frame_rate=30 // self.sample_rate,
            )
        else:
            # fallback to ByteTrack
            self.tracker = sv.ByteTrack(...)
```

**Validation:**
- Run diagnostic on same 5-min clip with ByteTrack vs BoT-SORT
- Target: <50 unique track IDs (from 25,593)
- Measure: track continuity (avg track length), ID switches

---

### Phase 1B: Detection Upgrade (P1) — YOLOv11

**Why:** Better recall on small/distant players. Drop-in replacement.

**Current:** YOLOv8s (`yolov8s.pt`)

**Target:** YOLOv11x or YOLOv11l (depending on speed tolerance)

**Changes:**
| File | Action |
|------|--------|
| `config.py` | `DEFAULT_MODEL_SIZE = "11s"` (new naming), add `YOLO_VERSION = "11"` |
| `pipeline/detector.py` | Model loading logic: `YOLO(f"yolo11{size}.pt")` |
| Root directory | Download `yolo11s.pt` (or `yolo11l.pt`) on first run |

**Validation:**
- Compare detection count (especially far-side players) between v8s and v11s
- Target: 5-10% more detections at distance, same or fewer false positives

---

### Phase 1C: Ball Tracking (P0) — TrackNetV3

**Why:** Enables real pass detection, possession, telecam ball-following, shot detection.

**Current:** No ball tracking. Pass detection is positional heuristic.

**Target:** TrackNetV3 (temporal CNN, 3-frame input)

**Changes:**
| File | Action |
|------|--------|
| `pipeline/ball_tracker.py` | **NEW** — TrackNetV3 model wrapper |
| `pipeline/ball_tracker.py` | Inference: sliding 3-frame window, heatmap → argmax → (x,y) |
| `pipeline/ball_tracker.py` | Post-processing: trajectory smoothing, occlusion interpolation |
| `pipeline/passes.py` | Refactor: use ball position for possession + pass events |
| `config.py` | `BALL_CONFIDENCE = 0.5`, `BALL_SMOOTH_WINDOW = 5` |
| `data/models/` | TrackNetV3 weights (fine-tuned on user's data) |
| `tools/label_ball.py` | **NEW** — Labeling tool for ball position ground truth |

**Training data needed (from tonight's session):**
- 300-500 labeled frames (ball center x,y) from user's camera angle
- Labeling tool: simple Streamlit/OpenCV click-to-label interface
- Fine-tune from SoccerNet pretrained weights

**Architecture:**
```
Input: 3 consecutive frames (720×1280×3×3 = 720×1280×9)
Encoder: VGG-like backbone (conv blocks with BN)
Decoder: Transposed convolutions → 720×1280×1 heatmap
Output: Gaussian blob at ball center (σ=2.5)
Loss: Weighted focal loss (handles class imbalance — ball is tiny)
```

**Phased rollout:**
1. Implement inference wrapper (use pretrained SoccerNet weights)
2. Test on user's video — likely poor accuracy without fine-tuning
3. Build labeling tool
4. User labels 300-500 frames
5. Fine-tune (MPS GPU, ~1-2 hours)
6. Integrate into pipeline

---

### Phase 1D: Jersey OCR Upgrade (P2) — PaddleOCR

**Why:** Better digit recognition, faster, more accurate on low-res crops.

**Current:** EasyOCR (general purpose, slow)

**Target:** PaddleOCR v4 (lightweight, optimized for digits)

**Changes:**
| File | Action |
|------|--------|
| `requirements.txt` | Replace `easyocr` with `paddleocr>=2.7` + `paddlepaddle` |
| `pipeline/jersey_ocr.py` | Swap OCR engine, keep same crop/voting logic |

**Implementation:**
```python
from paddleocr import PaddleOCR

class JerseyOCR:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            det=True,
            rec=True,
            cls=False,
            show_log=False,
        )
```

---

### Phase 1E: Pose Upgrade (P3) — RTMPose (Optional)

**Why:** Better keypoints on small/distant players for gait signatures.

**Current:** MediaPipe Pose (complexity=1)

**Target:** RTMPose-L (from mmpose)

**Caveat:** Questionable ROI at U10 sideline distance. Defer unless gait accuracy is clearly a bottleneck after tracker fix.

**Changes (if proceeding):**
| File | Action |
|------|--------|
| `requirements.txt` | Add `mmpose`, `mmdet`, `mmengine` |
| `pipeline/gait.py` | Replace MediaPipe with RTMPose inference |

---

## WORKSTREAM 2: Training Mode

### Architecture

Three distinct **session modes**, each with its own pipeline, page, and diagnostics:

| Mode | Input | Players | Duration | Field Setup |
|------|-------|---------|----------|-------------|
| **Game** | Wide-angle 4K | 14 (7v7) | 40-60 min | Full field, density-based homography |
| **Scrimmage** | Fisheye 170° | 14 (7v7) | 15-30 min | Flag-based homography, same teams |
| **Training/Drills** | Fisheye 170° | 16 (all) | 60-90 min | Flag-based homography, drill zones |

---

### Phase 2A: Fisheye Calibration Utility

**Why:** New NEEWER 170° fisheye lens introduces barrel distortion that must be removed before any tracking/homography.

**New files:**
| File | Purpose |
|------|---------|
| `pipeline/fisheye.py` | Fisheye undistortion using checkerboard calibration |
| `tools/calibrate_fisheye.py` | CLI: capture checkerboard → compute intrinsics → save |
| `data/calibration/` | Store camera intrinsic matrices per lens |
| `pages/camera_setup.py` | Streamlit page for calibration workflow |

**Implementation:**
```python
# pipeline/fisheye.py
import cv2
import numpy as np
from pathlib import Path

class FisheyeCalibration:
    def __init__(self, calibration_path: Path):
        data = np.load(calibration_path)
        self.K = data["K"]           # 3x3 intrinsic matrix
        self.D = data["D"]           # distortion coefficients
        self.new_K = None
        self.map1 = None
        self.map2 = None

    def init_undistort(self, frame_size: tuple):
        """Pre-compute undistortion maps for given frame size."""
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, frame_size, np.eye(3), balance=0.5
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K, frame_size, cv2.CV_16SC2
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Undistort a single frame using pre-computed maps."""
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
```

**User workflow:**
1. Print checkerboard pattern (9×6 or 10×7)
2. Record 15-20 second video waving checkerboard at various angles
3. Run calibration tool → saves `.npz` intrinsics file
4. One-time per lens — reuse forever

---

### Phase 2B: Flag-Based Homography

**Why:** Training fields have no lines. Neon flags at known positions provide ground-truth reference points.

**New/modified files:**
| File | Action |
|------|---------|
| `pipeline/homography.py` | Add `FlagHomography` class alongside existing auto-calibration |
| `pipeline/flag_detector.py` | **NEW** — Detect neon-colored flags using HSV thresholding |

**Implementation:**
```python
# pipeline/flag_detector.py
class FlagDetector:
    """Detect neon orange/pink flags using color thresholding."""

    def __init__(self, flag_color="orange"):
        # HSV ranges for neon colors
        self.color_ranges = {
            "orange": ((5, 150, 150), (25, 255, 255)),
            "pink": ((145, 100, 150), (175, 255, 255)),
            "yellow": ((20, 150, 150), (35, 255, 255)),
        }
        self.range = self.color_ranges[flag_color]

    def detect(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Return sorted list of flag centroids (x, y)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.range[0], self.range[1])
        # morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # find contours → centroids
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:  # min area filter
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        return sorted(centroids, key=lambda p: (p[1], p[0]))  # top-left first
```

**Flag placement for training:**
- 4 corners of the training area (minimum)
- Optional: midline markers for larger fields
- Known real-world positions (e.g., 40m × 30m rectangle)

---

### Phase 2C: Training Session Page

**New file:** `pages/training_session.py`

**UI Flow:**
```
┌─────────────────────────────────────────────────────┐
│  Training Session Analysis                           │
├─────────────────────────────────────────────────────┤
│  Session Type: [Scrimmage ▼] [Drills ▼]            │
│  Video: [Upload / Select]                            │
│  Camera: [Fisheye calibration: indoor_neewer.npz ▼] │
│  Field: [Flag-based ▼]  Flag color: [Orange ▼]      │
│  Field size: [40m × 30m]                             │
├─────────────────────────────────────────────────────┤
│  ▶ Stage 1/4: Undistort + Detect + Track            │
│  ▶ Stage 2/4: Team Assignment (scrimmage only)       │
│  ▶ Stage 3/4: Player Identification                  │
│  ▶ Stage 4/4: Session-Specific Analysis              │
│    - Scrimmage: Same as game (stats/passes/formation)│
│    - Drills: Drill segmentation + individual metrics │
└─────────────────────────────────────────────────────┘
```

---

### Phase 2D: Scrimmage Mode Pipeline

**Identical to game mode** except:
- Fisheye undistortion pre-step
- Flag-based homography (instead of density-based)
- Shorter duration (15-30 min)
- Both teams wear same-color bibs vs. no bibs (team classification still works)

**Pipeline:**
1. Undistort frames (fisheye)
2. Detect + Track (same as game, BoT-SORT)
3. Team Classification (same KMeans approach)
4. Homography (flag-based)
5. Player Identification (same multi-modal)
6. Stats + Passes + Formation (same)
7. Output: identical to game analysis

---

### Phase 2E: Drill Mode Pipeline

**Different from game mode:**
- All 16 players on field simultaneously
- No "teams" — individual analysis only
- Drill segmentation (auto-detect activity periods)
- Per-drill metrics (specific to drill type)

**New files:**
| File | Purpose |
|------|---------|
| `pipeline/drill_segmenter.py` | **NEW** — Detect drill boundaries from movement patterns |
| `pipeline/drill_metrics.py` | **NEW** — Drill-specific metrics (agility, sprint, passing accuracy) |

**Drill Segmentation Algorithm:**
```python
class DrillSegmenter:
    """Segment training video into individual drill periods."""

    def segment(self, df: pd.DataFrame, fps: float) -> list[DrillSegment]:
        """
        Hybrid approach:
        1. Compute team-wide average speed per second
        2. Low-activity periods (avg speed < 1.0 m/s for >10s) = transitions
        3. High-activity periods between transitions = drills
        4. Merge short drills (<30s) with adjacent
        """
        ...

@dataclass
class DrillSegment:
    start_frame: int
    end_frame: int
    duration_s: float
    avg_intensity: float  # avg speed during drill
    player_count: int     # active players
    drill_type: str       # "sprint", "passing", "possession", "unknown"
```

**Drill Type Classification (rule-based):**
| Pattern | Classification |
|---------|---------------|
| Linear high-speed, short duration | Sprint drill |
| Clustered players, frequent short passes | Passing drill |
| Small area, high density | Possession/rondo |
| Spread out, moderate speed | Tactical/positioning |
| Repeated short sprints with rest | Agility/fitness |

**Per-Drill Metrics:**
- Sprint drills: top speed, acceleration, deceleration, sprint time
- Passing drills: pass count, accuracy (if ball tracked), tempo
- Possession: touches, time in possession zone, movement off-ball
- General: distance covered, intensity (% time >sprint threshold), work rate

---

### Phase 2F: Training Database

**Modified file:** `database/match_db.py` → rename to `database/session_db.py`

**Schema additions:**
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    date TEXT,
    type TEXT CHECK(type IN ('game', 'scrimmage', 'drill')),
    duration_min REAL,
    video_path TEXT,
    field_config TEXT,
    notes TEXT,
    created_at TEXT
);

CREATE TABLE drill_segments (
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id),
    segment_index INTEGER,
    start_frame INTEGER,
    end_frame INTEGER,
    duration_s REAL,
    drill_type TEXT,
    avg_intensity REAL
);

CREATE TABLE drill_metrics (
    id INTEGER PRIMARY KEY,
    segment_id INTEGER REFERENCES drill_segments(id),
    player_id TEXT,
    metric_name TEXT,
    metric_value REAL
);
```

---

## WORKSTREAM 3: Diagnostics Suite

### Three Diagnostic Modules

Each diagnostic module is a **self-contained CLI** that exercises the exact same pipeline as its corresponding app page, producing visual/statistical output for validation.

| Diagnostic | App Page | Pipeline |
|-----------|----------|----------|
| `diagnostics_game.py` | `pages/match_analysis.py` | Full game pipeline |
| `diagnostics_scrimmage.py` | `pages/training_session.py` (scrimmage mode) | Fisheye + game pipeline |
| `diagnostics_drill.py` | `pages/training_session.py` (drill mode) | Fisheye + drill segmentation |

---

### Phase 3A: Refactor Existing `diagnostics.py` → `diagnostics_game.py`

**Rename** `diagnostics.py` to `diagnostics_game.py` and clean up:
- Keep all 13 steps
- Add `--tracker` flag (bytetrack / botsort / deepocsort)
- Add `--yolo-version` flag (8 / 11)
- Ensure steps match app page pipeline exactly:
  1. Detection + Tracking
  2. Team Classification
  3. Homography + Field Filter
  4. Player Identification
  5. Stats + Passes + Formation

**Output structure:**
```
diagnostics_game/
├── 01_detections/          # annotated frames, bbox stats
├── 02_tracking/            # track timeline, fragmentation metrics
├── 03_team_assignment/     # colored frames, cluster plots
├── 04_homography/          # field overlay, coordinate scatter
├── 05_player_id/           # face matches, OCR results, confidence
├── 06_stats/               # speed/distance tables, heatmaps
├── 07_passes/              # pass map, network graph
├── 08_formation/           # formation snapshot, timeline
└── summary.json            # full pipeline metrics
```

---

### Phase 3B: `diagnostics_scrimmage.py` — Scrimmage Diagnostics

**New file.** Same as game diagnostics but with:
- **Step 0:** Fisheye undistortion (visualize before/after)
- **Step 0.5:** Flag detection (visualize detected flags, compute homography)
- Steps 1-8: Same as game

**CLI:**
```
python diagnostics_scrimmage.py "/path/to/scrimmage.mov" \
    --calibration data/calibration/neewer_fisheye.npz \
    --flag-color orange \
    --field-size 40x30 \
    --my-team red \
    --frames 60
```

**Additional output:**
```
diagnostics_scrimmage/
├── 00_undistort/           # before/after fisheye correction
├── 00_flags/              # detected flag positions, homography overlay
├── 01_detections/         # (same as game)
...
└── summary.json
```

---

### Phase 3C: `diagnostics_drill.py` — Drill/Training Diagnostics

**New file.** Different pipeline:
- **Step 0:** Fisheye undistortion
- **Step 0.5:** Flag detection + homography
- **Step 1:** Detection + Tracking (all players, no team split)
- **Step 2:** Player Identification (all 16 players)
- **Step 3:** Drill Segmentation (visualize detected segments)
- **Step 4:** Per-drill metrics (per player × per drill)
- **Step 5:** Individual development metrics (speed, agility, work rate)

**CLI:**
```
python diagnostics_drill.py "/path/to/training.mov" \
    --calibration data/calibration/neewer_fisheye.npz \
    --flag-color orange \
    --field-size 40x30 \
    --frames 120
```

**Output:**
```
diagnostics_drill/
├── 00_undistort/
├── 00_flags/
├── 01_detections/
├── 02_player_id/
├── 03_segmentation/        # drill boundaries timeline, intensity chart
├── 04_drill_metrics/       # per-drill tables and charts
├── 05_individual/          # per-player development cards
└── summary.json
```

---

## Implementation Order (Chronological)

### Tonight (with new fisheye data):
1. **Fisheye calibration** — Print checkerboard, record calibration video, compute intrinsics
2. **Record training session** — Full practice with flags at corners

### Sprint 1 (Days 1-3): Foundation
| # | Task | Dependencies |
|---|------|-------------|
| 1.1 | Install `boxmot`, implement BoT-SORT in `detector.py` | None |
| 1.2 | Add `--tracker` arg to existing `diagnostics.py` | 1.1 |
| 1.3 | Run comparison diagnostic: ByteTrack vs BoT-SORT on existing 5-min clip | 1.2 |
| 1.4 | Download/integrate YOLOv11s model | None |
| 1.5 | Build `pipeline/fisheye.py` + `tools/calibrate_fisheye.py` | Tonight's data |
| 1.6 | Build `pipeline/flag_detector.py` | Tonight's data |

### Sprint 2 (Days 4-6): Training Pipeline
| # | Task | Dependencies |
|---|------|-------------|
| 2.1 | Flag-based homography in `pipeline/homography.py` | 1.6 |
| 2.2 | Build `pipeline/drill_segmenter.py` | 1.1 (tracker) |
| 2.3 | Build `pipeline/drill_metrics.py` | 2.2 |
| 2.4 | Refactor `diagnostics.py` → `diagnostics_game.py` | 1.1, 1.4 |
| 2.5 | Build `diagnostics_scrimmage.py` | 1.5, 1.6, 2.1 |
| 2.6 | Build `diagnostics_drill.py` | 2.2, 2.3 |

### Sprint 3 (Days 7-9): Ball Tracking
| # | Task | Dependencies |
|---|------|-------------|
| 3.1 | Build `tools/label_ball.py` (labeling interface) | None |
| 3.2 | Label 300-500 frames from existing game video | 3.1 |
| 3.3 | Implement `pipeline/ball_tracker.py` (TrackNetV3 inference) | None |
| 3.4 | Fine-tune TrackNetV3 on labeled data | 3.2, 3.3 |
| 3.5 | Refactor `pipeline/passes.py` to use ball position | 3.4 |
| 3.6 | Integrate ball into telecam (70/30 ball/cluster) | 3.4 |

### Sprint 4 (Days 10-12): App Pages + Polish
| # | Task | Dependencies |
|---|------|-------------|
| 4.1 | Build `pages/training_session.py` (Streamlit UI) | 2.1-2.3 |
| 4.2 | Build `pages/camera_setup.py` (calibration UI) | 1.5 |
| 4.3 | Upgrade Jersey OCR → PaddleOCR | None |
| 4.4 | Database schema update (`session_db.py`) | 2.3 |
| 4.5 | Run all 3 diagnostics end-to-end, fix bugs | All above |
| 4.6 | Update app navigation (add Training section) | 4.1, 4.2 |

### Sprint 5 (Days 13-15): Validation + Optional
| # | Task | Dependencies |
|---|------|-------------|
| 5.1 | Full game diagnostic with all upgrades | All |
| 5.2 | Full scrimmage diagnostic with tonight's data | All |
| 5.3 | Full drill diagnostic with tonight's data | All |
| 5.4 | (Optional) RTMPose upgrade for gait | All working |
| 5.5 | (Optional) Season progress page for training metrics | 4.4 |

---

## File Changes Summary

### New Files (14)
```
pipeline/ball_tracker.py         # TrackNetV3 wrapper
pipeline/fisheye.py              # Fisheye undistortion
pipeline/flag_detector.py        # Neon flag detection
pipeline/drill_segmenter.py      # Drill boundary detection
pipeline/drill_metrics.py        # Per-drill metrics
tools/calibrate_fisheye.py       # Checkerboard calibration CLI
tools/label_ball.py              # Ball labeling interface
pages/training_session.py        # Training session Streamlit page
pages/camera_setup.py            # Camera calibration Streamlit page
diagnostics_scrimmage.py         # Scrimmage diagnostic CLI
diagnostics_drill.py             # Drill diagnostic CLI
database/session_db.py           # Session/drill database
data/calibration/.gitkeep        # Calibration storage
data/models/.gitkeep             # Model weights storage
```

### Modified Files (10)
```
pipeline/detector.py             # BoT-SORT + YOLOv11
pipeline/homography.py           # Add FlagHomography class
pipeline/passes.py               # Use ball position when available
pipeline/jersey_ocr.py           # PaddleOCR swap
pipeline/telecam.py              # Ball-priority mode
config.py                        # New constants for all modes
requirements.txt                 # New dependencies
app.py                           # Navigation update
diagnostics.py → diagnostics_game.py  # Rename + enhance
database/match_db.py             # Extend or replace with session_db
```

---

## Config Additions

```python
# config.py additions

# === Tracker ===
DEFAULT_TRACKER = "botsort"          # "bytetrack" | "botsort" | "deepocsort"
TRACKER_REID_WEIGHTS = "osnet_x0_25_msmt17"
BOTSORT_MATCH_THRESH = 0.8
BOTSORT_APPEARANCE_THRESH = 0.25

# === Detection ===
YOLO_VERSION = "11"                  # "8" | "11"

# === Ball Tracking ===
BALL_MODEL_PATH = DATA_DIR / "models" / "tracknetv3.pt"
BALL_CONFIDENCE = 0.5
BALL_SMOOTH_WINDOW = 5
BALL_MAX_SPEED = 35.0                # m/s (max ball speed to filter outliers)

# === Fisheye ===
CALIBRATION_DIR = DATA_DIR / "calibration"
DEFAULT_CHECKERBOARD = (9, 6)        # inner corners

# === Flags ===
DEFAULT_FLAG_COLOR = "orange"
FLAG_COLORS = {
    "orange": ((5, 150, 150), (25, 255, 255)),
    "pink": ((145, 100, 150), (175, 255, 255)),
    "yellow": ((20, 150, 150), (35, 255, 255)),
}
MIN_FLAG_AREA = 200                  # px²

# === Training ===
SESSION_TYPES = ["game", "scrimmage", "drill"]
DRILL_MIN_DURATION_S = 30            # min drill length
DRILL_TRANSITION_SPEED = 1.0         # m/s threshold for "idle"
DRILL_TRANSITION_MIN_S = 10          # seconds below threshold = transition
TRAINING_FIELD_LENGTH_M = 40.0
TRAINING_FIELD_WIDTH_M = 30.0
```

---

## Dependencies Update

```
# New in requirements.txt
boxmot>=10.0.77                      # BoT-SORT, Deep OC-SORT, StrongSORT
paddlepaddle>=2.6.0                  # PaddleOCR backend
paddleocr>=2.7.0                     # Jersey digit OCR upgrade
# ultralytics already installed (supports v11)
```

**Remove:**
```
easyocr                              # Replaced by PaddleOCR
```

---

## Tonight's Checklist (Before Recording)

- [ ] Print checkerboard pattern (A3 or A4 taped to cardboard) — 9×6 inner corners
- [ ] Buy/set up 4 neon orange flags (or cones) at field corners
- [ ] Measure training area dimensions (length × width in meters)
- [ ] Mount iPhone + fisheye lens on tripod at midfield sideline
- [ ] Record 1: Checkerboard calibration (15-20s, wave board at various angles/distances)
- [ ] Record 2: Full training session (drills + scrimmage, don't stop recording between)
- [ ] Note flag positions relative to field corners
- [ ] Note which drills happen when (rough timestamps)

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Track fragments (5 min) | 25,593 | < 50 |
| Track avg length | ~0.5s | > 30s |
| Pass detection accuracy | ~20% (guess) | > 70% (with ball) |
| Player ID accuracy | ~60% | > 80% |
| Drill segmentation | N/A | > 90% boundary detection |
| Diagnostic runtime (5 min clip) | OOM/killed | < 3 min |
| End-to-end pipeline modes | 1 (game) | 3 (game/scrimmage/drill) |
