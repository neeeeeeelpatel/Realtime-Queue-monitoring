# Automated Queue Monitoring System – Project Documentation

## 1. Project Overview

### 1.1 Purpose

The **Automated Queue Monitoring System** is a computer vision application that:

- Counts people in a queue from video
- Tracks individuals across frames
- Compares different detection/tracking methods
- Evaluates performance against ground truth

### 1.2 Problem Addressed

Manual counting of people in queues is slow and error-prone. This system automates counting using object detection and tracking, with temporal filtering to reduce false positives (e.g. people passing by).

### 1.3 Main Components

| Component | Description |
|-----------|-------------|
| **Detection** | YOLOv8 detects people in each frame |
| **Tracking** | Centroid or DeepSORT assigns stable IDs across frames |
| **ROI** | User-defined polygon for the queue area |
| **Temporal Filter** | Only counts people who stay in ROI for ≥2 seconds |
| **Evaluation** | Metrics vs. manual ground truth |

### 1.4 Workflow

```
Video Input → CLAHE Preprocessing → YOLOv8 Detection → Tracking (Centroid/DeepSORT)
     → ROI Filtering → Temporal Filter (2s rule) → Count Output → Evaluation
```

---

## 2. Libraries and Models Used

### 2.1 Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **ultralytics** | 8.x | YOLOv8 object detection |
| **torch** | 2.x | Deep learning backend |
| **torchvision** | 0.23+ | Image transforms and utilities |
| **opencv-python** | 4.x | Video I/O, CLAHE, ROI, drawing |
| **numpy** | 2.x | Arrays and math |
| **pandas** | 2.x | CSV and data handling |
| **matplotlib** | 3.x | ROI selection, plots |
| **scikit-learn** | 1.x | Metrics (if used) |
| **deep-sort-realtime** | 1.3+ | DeepSORT tracking (optional) |

### 2.2 Models

| Model | Description |
|-------|-------------|
| **YOLOv8n** | YOLOv8 nano, person detection (class 0) |
| **MobileNet** | DeepSORT embedder for appearance features |

### 2.3 Installation

```bash
pip install ultralytics torch torchvision matplotlib numpy opencv-python pandas scikit-learn deep-sort-realtime
```

---

## 3. Code Block Explanations

### 3.1 Configuration (Lines 39–64)

```python
VIDEO_PATH = "video.mp4"
SKIP_FRAMES = 4
IMGSZ = 640
CONF = 0.35
```

- **VIDEO_PATH**: Input video file
- **SKIP_FRAMES**: Process every Nth frame (4 → ~25% of frames)
- **IMGSZ**: YOLO input size (640×640)
- **CONF**: Detection confidence threshold (0.35)

### 3.2 CLAHE Preprocessing (Lines 69–75)

```python
def apply_clahe_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
```

- Converts BGR → LAB
- Applies CLAHE on L channel to improve contrast
- Merges back and converts to BGR  
Improves detection in low-light or uneven lighting.

### 3.3 ROI Check (Lines 81–82)

```python
def inside_roi_xy(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0
```

- Uses OpenCV `pointPolygonTest` to check if (x, y) is inside the polygon
- Returns `True` if point is inside or on the boundary

### 3.4 Centroid Tracker (Lines 84–139)

```python
class CentroidTracker:
    def update(self, centroids):
        # 1. If no detections: increment "disappeared" count, remove stale objects
        # 2. If no existing objects: assign new IDs to all centroids
        # 3. Else: compute distance matrix between existing objects and new centroids
        # 4. Greedy matching: pair closest objects (within max_distance)
        # 5. Unmatched objects: increment disappeared or remove
        # 6. Unmatched centroids: create new object IDs
```

- Keeps object IDs across frames using centroid positions
- Uses Euclidean distance for matching
- `max_disappeared`: frames before an object is removed
- `max_distance`: max distance for a valid match

### 3.5 ROI Selection (Lines 144–205)

```python
def select_roi_with_jump(video_path, jump_frame=None):
    # 1. Use HARD_CODED_ROI if set
    # 2. Load saved ROI from roi_points.json if exists
    # 3. Open video, seek to jump_frame (or 0)
    # 4. Find first non-black frame (mean gray > 8)
    # 5. Display frame, user clicks polygon vertices
    # 6. Save ROI to JSON
```

- Skips black frames
- Lets user draw a polygon for the queue area
- Saves ROI for reuse

### 3.6 Main Pipeline – Proposed Method (Lines 210–299)

```python
def run_pipeline():
    # 1. Load ROI, init YOLO, open video
    frames_required_for_2s = ceil(2.0 * fps / SKIP_FRAMES)  # e.g., ~12 frames at 24fps
    tracker = CentroidTracker(...)
    
    for each processed frame:
        # 2. CLAHE → YOLO → get boxes & centroids
        # 3. tracker.update(cents)
        # 4. Assign boxes to tracker IDs (nearest centroid)
        # 5. For each ID in ROI: append processed_idx to inside_processed_frames
        # 6. Temporal filter: if ID has ≥frames_2s consecutive frames in ROI → confirm
        # 7. count_inside = people currently in ROI
        # 8. Draw boxes, ROI, count; save frame; append to rows
```

- **2-second rule**: Only counts IDs that appear in ROI for at least `frames_required_for_2s` consecutive processed frames.

### 3.7 Baseline: No Temporal Filter (Lines 305–408)

```python
def run_baseline_no_temporal():
    # Same as proposed, but:
    # Instead of temporal filter, immediately add ID to confirmed when in ROI
    all_ids_seen.add(oid)  # No 2-second wait
```

- Counts every unique ID that ever enters the ROI
- Higher recall, more false positives (passers-by)

### 3.8 Baseline: DeepSORT (Lines 415–547)

```python
def run_baseline_deepsort():
    deepsort = DeepSort(embedder="mobilenet", ...)
    # YOLO detections → DeepSORT format [x1,y1,w,h], conf
    tracks = deepsort.update_tracks(detections, frame=frame)
    # Apply same 2-second temporal filter on track IDs
```

- Uses DeepSORT with MobileNet for appearance features
- More robust to occlusions and re-identification
- Heavier than centroid tracking

### 3.9 Evaluation Metrics (Lines 330–355, 772–804)

| Metric | Formula |
|--------|---------|
| **MAE** | Mean of \|GT − Pred\| |
| **RMSE** | √(mean((GT − Pred)²)) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2 × Precision × Recall / (Precision + Recall) |
| **Queue Accuracy** | 1 − (total_error / total_GT_people) |

---

## 4. Results Summary

| Method | Unique Count | Queue Accuracy | F1 Score |
|--------|--------------|----------------|----------|
| **Proposed (Centroid + Temporal)** | 8 | 69.39% | 0.819 |
| **YOLOv8 No Temporal** | 12 | 69.39% | 0.819 |
| **YOLOv8 + DeepSORT** | 8 | 66.33% | 0.798 |

- Proposed method matches DeepSORT in unique count with simpler tracking
- No temporal filter overcounts (12 vs 8)
- Proposed method is suitable for real-time use on limited hardware

---

## 5. Usage Commands

```bash
python queue_monitor.py --run_pipeline          # Proposed method
python queue_monitor.py --baseline_no_temporal  # No temporal filter
python queue_monitor.py --baseline_deepsort     # DeepSORT baseline
python queue_monitor.py --compare               # Run all and compare
python queue_monitor.py --create_gt             # Create ground truth
python queue_monitor.py --evaluate              # Evaluate vs ground truth
```

---

## 6. Output Files

| File/Folder | Description |
|-------------|-------------|
| `processed_frames/` | Annotated frames (proposed) |
| `baseline_no_temporal/` | Annotated frames (no temporal) |
| `baseline_deepsort/` | Annotated frames (DeepSORT) |
| `predictions_*.csv` | Per-frame predictions |
| `ground_truth.csv` | Manual counts for evaluation |
| `comparison_outputs/method_comparison.csv` | Metrics for all methods |
| `evaluation_outputs/` | Plots and summary metrics |

---

## 7. Project Structure

```
Major project/
├── queue_monitor.py           # Full implementation (939 lines)
├── queue_monitor_simple.py    # Simplified version (316 lines)
├── video.mp4                  # Input video
├── roi_points.json            # Saved ROI polygon
├── ground_truth.csv           # Manual annotations
├── yolov8n.pt                 # YOLO weights (auto-downloaded)
├── processed_frames/          # Output frames
├── baseline_no_temporal/
├── baseline_deepsort/
├── comparison_outputs/
├── evaluation_outputs/
└── PROJECT_DOCUMENTATION.md   # This document
```
