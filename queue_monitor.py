# queue_monitor_final_with_accuracy.py
# Final script: pipeline + GT helper + evaluation with queue occupancy accuracy.
# Includes baseline comparisons for academic evaluation.
#
# Usage:
#   python queue_monitor.py --run_pipeline              # Proposed method (temporal filtering)
#   python queue_monitor.py --baseline_no_temporal      # Baseline: YOLOv8 without temporal filtering
#   python queue_monitor.py --baseline_deepsort         # Baseline: YOLOv8 + DeepSORT
#   python queue_monitor.py --compare                   # Run all methods and compare
#   python queue_monitor.py --create_gt
#   python queue_monitor.py --evaluate
#
# Requirements:
#   pip install ultralytics torch torchvision matplotlib numpy opencv-python pandas scikit-learn deep-sort-realtime

import os
import sys
import json
import math
import argparse
import time
import cv2

# #region agent log
_DBG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor", "debug-8e732b.log")
def _dbg(location, message, data=None, hypothesis_id=""):
    import json as _j, time as _t
    os.makedirs(os.path.dirname(_DBG_LOG_PATH), exist_ok=True)
    entry = {"sessionId":"8e732b","timestamp":int(_t.time()*1000),"location":location,"message":message,"data":data or {},"hypothesisId":hypothesis_id}
    with open(_DBG_LOG_PATH, "a") as _f: _f.write(_j.dumps(entry)+"\n")
# #endregion
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from math import ceil
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("Warning: deep-sort-realtime not installed. DeepSORT baseline unavailable.")

# -----------------------------
# USER SETTINGS
# -----------------------------
VIDEO_PATH = "queue.mp4"
SKIP_FRAMES = 4
IMGSZ = 640
CONF = 0.35
SAVE_EVERY_PROCESSED = 3       # set to 1 if you want every processed frame saved for GT
OUT_DIR = "processed_frames"
ROI_SAVE_PATH = "roi_points.json"
PRED_CSV = "predictions_per_processed_frame.csv"
GT_CSV = "ground_truth.csv"
EVAL_OUT_DIR = "evaluation_outputs"
HARD_CODED_ROI = None          # e.g. [[100,150],[500,150],[500,400],[100,400]]
ROI_FRAME_INDEX = 50           # jump target for ROI selection (increase if still black)

# Baseline output directories and files
BASELINE_NO_TEMPORAL_DIR = "baseline_no_temporal"
BASELINE_DEEPSORT_DIR = "baseline_deepsort"
PRED_CSV_NO_TEMPORAL = "predictions_no_temporal.csv"
PRED_CSV_DEEPSORT = "predictions_deepsort.csv"
COMPARISON_OUT_DIR = "comparison_outputs"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUT_DIR, exist_ok=True)
os.makedirs(BASELINE_NO_TEMPORAL_DIR, exist_ok=True)
os.makedirs(BASELINE_DEEPSORT_DIR, exist_ok=True)
os.makedirs(COMPARISON_OUT_DIR, exist_ok=True)

# -----------------------------
# helpers
# -----------------------------
def apply_clahe_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def box_centroid(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

def inside_roi_xy(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0

MIN_DEPARTURES = 3

def fmt_time(seconds):
    """Format seconds into 'Xm Ys' or 'Ys' string."""
    s = max(0, int(seconds))
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.inside_processed_frames = defaultdict(lambda: deque(maxlen=5000))
    def update(self, centroids):
        if len(centroids) == 0:
            remove_ids=[]
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid]+=1
                if self.disappeared[oid] > self.max_disappeared:
                    remove_ids.append(oid)
            for oid in remove_ids:
                self.objects.pop(oid,None)
                self.disappeared.pop(oid,None)
                self.inside_processed_frames.pop(oid,None)
            return self.objects
        if len(self.objects) == 0:
            for c in centroids:
                self.objects[self.next_object_id] = c
                self.disappeared[self.next_object_id] = 0
                self.next_object_id += 1
            return self.objects
        existing_ids = list(self.objects.keys())
        existing_centroids = list(self.objects.values())
        D = np.zeros((len(existing_centroids), len(centroids)), dtype=float)
        for i, ec in enumerate(existing_centroids):
            for j, nc in enumerate(centroids):
                D[i,j] = math.hypot(ec[0]-nc[0], ec[1]-nc[1])
        used_rows, used_cols = set(), set()
        row_idx = D.min(axis=1).argsort()
        for row in row_idx:
            col = D[row].argmin()
            if row in used_rows or col in used_cols: continue
            if D[row,col] > self.max_distance: continue
            oid = existing_ids[row]
            self.objects[oid] = centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row); used_cols.add(col)
        unused_rows = set(range(len(existing_centroids))) - used_rows
        for row in unused_rows:
            oid = existing_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.objects.pop(oid,None)
                self.disappeared.pop(oid,None)
                self.inside_processed_frames.pop(oid,None)
        unused_cols = set(range(len(centroids))) - used_cols
        for col in unused_cols:
            self.objects[self.next_object_id] = centroids[col]
            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1
        return self.objects

# -----------------------------
# ROI selection (interactive OpenCV popup)
# Click to add vertices, 'd' to undo, Enter/q to finish
# -----------------------------
_roi_points = []
_roi_frame = None

def _roi_mouse_cb(event, x, y, flags, param):
    global _roi_points, _roi_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        _roi_points.append([x, y])
        display = _roi_frame.copy()
        for i, pt in enumerate(_roi_points):
            cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if len(_roi_points) > 1:
            cv2.polylines(display, [np.array(_roi_points, np.int32)], len(_roi_points)>2, (0,255,0), 2)
        cv2.imshow("Select ROI - Click points, ENTER when done, D to undo", display)

def select_roi_with_jump(video_path, jump_frame=None):
    global _roi_points, _roi_frame
    if HARD_CODED_ROI is not None:
        print("Using HARD_CODED_ROI")
        with open(ROI_SAVE_PATH, "w") as f: json.dump(HARD_CODED_ROI, f)
        return HARD_CODED_ROI
    if os.path.exists(ROI_SAVE_PATH):
        try:
            with open(ROI_SAVE_PATH, "r") as f:
                roi = json.load(f)
            if isinstance(roi, list) and len(roi) >= 3:
                print("Loaded ROI from", ROI_SAVE_PATH)
                return roi
        except Exception:
            pass
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    selected_frame = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    def frame_is_nonblack(frm):
        return np.mean(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)) > 8
    if jump_frame is not None:
        start = min(int(max(0, jump_frame)), max(0, total-1)) if total > 0 else int(max(0, jump_frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(150):
            ret, frame = cap.read()
            if not ret: break
            if frame_is_nonblack(frame):
                selected_frame = frame
                print(f"Selected frame at index {start + i} for ROI.")
                break
    if selected_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(500):
            ret, frame = cap.read()
            if not ret: break
            if frame_is_nonblack(frame):
                selected_frame = frame
                print(f"Selected frame at index {i} for ROI.")
                break
    cap.release()
    if selected_frame is None:
        raise RuntimeError("Failed to find a non-black frame. Set HARD_CODED_ROI or increase ROI_FRAME_INDEX.")
    _roi_points = []
    _roi_frame = selected_frame.copy()
    win = "Select ROI - Click points, ENTER when done, D to undo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 600)
    cv2.setMouseCallback(win, _roi_mouse_cb)
    cv2.imshow(win, _roi_frame)
    print(">>> ROI POPUP OPENED - Click to add points, 'd' to undo, Enter when done <<<")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('q'):
            break
        elif key == ord('d') and _roi_points:
            _roi_points.pop()
            display = _roi_frame.copy()
            for i, pt in enumerate(_roi_points):
                cv2.circle(display, tuple(pt), 5, (0,0,255), -1)
                cv2.putText(display, str(i+1), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            if len(_roi_points) > 1:
                cv2.polylines(display, [np.array(_roi_points, np.int32)], len(_roi_points)>2, (0,255,0), 2)
            cv2.imshow(win, display)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if len(_roi_points) < 3:
        raise RuntimeError("You must select at least 3 points for the ROI.")
    with open(ROI_SAVE_PATH, "w") as f: json.dump(_roi_points, f)
    print(f"Saved ROI ({len(_roi_points)} points) to {ROI_SAVE_PATH}")
    return _roi_points

# -----------------------------
# Pipeline run
# -----------------------------
def run_pipeline():
    roi_points = select_roi_with_jump(VIDEO_PATH, jump_frame=ROI_FRAME_INDEX)
    roi_poly = np.array(roi_points, dtype=np.int32)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    try:
        model.to(device)
        print("Model moved to", device)
    except Exception as e:
        print("Warning: failed to move model to device:", e)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print("Video FPS:", fps, "Total frames:", total_frames)
    frames_required_for_2s_processed = max(1, int(ceil(3.0*fps/SKIP_FRAMES)))
    # #region agent log
    _dbg("queue_monitor.py:275", "Pipeline config", {"fps": fps, "total_frames": total_frames, "skip_frames": SKIP_FRAMES, "frames_required_for_2s": frames_required_for_2s_processed, "roi_points": roi_points, "total_processed_frames_estimate": total_frames // SKIP_FRAMES}, "H3_H5")
    # #endregion
    tracker = CentroidTracker(max_disappeared=int(fps/2), max_distance=80)
    ids_confirmed=set(); id_first_processed={}
    processed_idxs=[]; roi_counts=[]; confirmed_counts=[]
    last_frame_for_display=None
    frame_idx=-1; processed_idx=0; save_counter=0
    rows=[]
    # Wait-time estimation state
    entry_time_wt = {}
    departed_durations = []
    prev_confirmed_in_roi = set()
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % SKIP_FRAMES != 0: continue
        processed_idx += 1
        last_frame_for_display = frame.copy()
        proc = apply_clahe_bgr(frame)
        results = model.predict(source=proc, imgsz=IMGSZ, conf=CONF, classes=[0], device=device, verbose=False)
        r = results[0]
        boxes=[]; cents=[]
        if hasattr(r, "boxes") and len(r.boxes)>0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (x1,y1,x2,y2) in xyxy:
                bx=(int(x1),int(y1),int(x2),int(y2))
                boxes.append(bx); cents.append(box_centroid(bx))
        # #region agent log
        if processed_idx <= 5 or processed_idx % 20 == 0:
            _dbg("queue_monitor.py:289", "YOLO detections", {"processed_idx": processed_idx, "num_boxes": len(boxes), "centroids": [list(c) for c in cents[:5]]}, "H1")
        # #endregion
        tracker.update(cents)
        assigned={}
        if len(boxes)>0 and len(cents)>0:
            for oid,c in tracker.objects.items():
                dists=[math.hypot(c[0]-cx, c[1]-cy) for (cx,cy) in cents]
                if not dists: continue
                best_i = int(np.argmin(dists))
                if best_i < len(boxes): assigned[oid]=(boxes[best_i], cents[best_i])
        # #region agent log
        if processed_idx <= 5 or processed_idx % 20 == 0:
            roi_hits = []
            roi_misses = []
            for oid,(box,cent) in assigned.items():
                cx,cy = cent
                isin = inside_roi_xy(cx,cy,roi_poly)
                if isin: roi_hits.append({"oid":oid,"cx":cx,"cy":cy})
                else: roi_misses.append({"oid":oid,"cx":cx,"cy":cy})
            _dbg("queue_monitor.py:300", "ROI check", {"processed_idx": processed_idx, "assigned_count": len(assigned), "inside_roi": len(roi_hits), "outside_roi": len(roi_misses), "hits": roi_hits[:3], "misses": roi_misses[:3]}, "H2")
        # #endregion
        for oid,(box,cent) in assigned.items():
            cx,cy = cent
            if inside_roi_xy(cx,cy,roi_poly):
                tracker.inside_processed_frames[oid].append(processed_idx)
                if oid not in id_first_processed: id_first_processed[oid]=processed_idx
        for oid,dq in list(tracker.inside_processed_frames.items()):
            if len(dq) < frames_required_for_2s_processed: continue
            seq_len=1
            for i in range(len(dq)-1,0,-1):
                if dq[i]-dq[i-1] <= 1: seq_len+=1
                else: break
            # #region agent log
            _dbg("queue_monitor.py:310", "Temporal check", {"oid": oid, "processed_idx": processed_idx, "deque_len": len(dq), "seq_len": seq_len, "threshold": frames_required_for_2s_processed, "last_5_dq": list(dq)[-5:]}, "H3_H4")
            # #endregion
            if seq_len >= frames_required_for_2s_processed and oid not in ids_confirmed:
                ids_confirmed.add(oid)
                print(f"Counted ID {oid} at processed_idx {processed_idx}")
        count_inside = sum(1 for oid, (box, cent) in assigned.items() if oid in ids_confirmed and inside_roi_xy(cent[0], cent[1], roi_poly))
        # ── Wait-time estimation ──
        current_video_time = frame_idx / fps
        current_confirmed_in_roi = set()
        for oid, (box, cent) in assigned.items():
            if oid in ids_confirmed and inside_roi_xy(cent[0], cent[1], roi_poly):
                current_confirmed_in_roi.add(oid)
                if oid not in entry_time_wt:
                    entry_time_wt[oid] = current_video_time
        departed = prev_confirmed_in_roi - current_confirmed_in_roi
        for oid in departed:
            if oid in entry_time_wt:
                departed_durations.append(current_video_time - entry_time_wt[oid])
        prev_confirmed_in_roi = current_confirmed_in_roi
        avg_service = None
        if len(departed_durations) >= MIN_DEPARTURES:
            avg_service = sum(departed_durations) / len(departed_durations)

        processed_idxs.append(processed_idx); roi_counts.append(count_inside); confirmed_counts.append(len(ids_confirmed))
        vis = frame.copy()
        for oid,(box,cent) in assigned.items():
            x1,y1,x2,y2 = box
            color=(0,255,0) if oid in ids_confirmed else (0,0,255)
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            cv2.putText(vis, f"ID-{oid}", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if oid in current_confirmed_in_roi and oid in entry_time_wt:
                elapsed_wt = current_video_time - entry_time_wt[oid]
                if avg_service is not None:
                    remaining = max(0, avg_service - elapsed_wt)
                    wait_label = f"~{fmt_time(remaining)}"
                else:
                    wait_label = "..."
                cv2.putText(vis, wait_label, (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.polylines(vis,[roi_poly],True,(255,255,0),2)
        cv2.putText(vis, f"In Queue: {count_inside}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        if avg_service is not None:
            cv2.putText(vis, f"Avg Wait: {fmt_time(avg_service)}", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        save_counter += 1
        if save_counter % SAVE_EVERY_PROCESSED == 0:
            out_path = os.path.join(OUT_DIR, f"proc_{processed_idx:06d}.jpg")
            cv2.imwrite(out_path, vis)
        rows.append({"processed_idx": processed_idx, "pred_count": count_inside, "cumulative_count": len(ids_confirmed)})
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    cap.release()
    pd.DataFrame(rows).to_csv(PRED_CSV, index=False)
    print("Saved predictions to", PRED_CSV)
    print("Done! Unique counted (>=2s):", len(ids_confirmed))
    return {
        "processed_idxs": processed_idxs,
        "roi_counts": roi_counts,
        "confirmed_counts": confirmed_counts,
        "last_frame_for_display": last_frame_for_display,
        "roi_points": roi_points if 'roi_points' in locals() else None,
        "fps": fps
    }

# -----------------------------
# Baseline 1: YOLOv8 without temporal filtering
# Counts all detections in ROI immediately (no 2-second rule)
# -----------------------------
def run_baseline_no_temporal():
    """
    Baseline: YOLOv8 detection without temporal filtering.
    Counts every person detected in ROI without waiting for 2-second confirmation.
    This typically has higher recall but lower precision due to false positives.
    """
    roi_points = select_roi_with_jump(VIDEO_PATH, jump_frame=ROI_FRAME_INDEX)
    roi_poly = np.array(roi_points, dtype=np.int32)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[Baseline No-Temporal] Device:", device)
    
    model = YOLO("yolov8n.pt")
    try:
        model.to(device)
    except Exception as e:
        print("Warning: failed to move model to device:", e)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[Baseline No-Temporal] Video FPS: {fps}, Total frames: {total_frames}")
    
    # Simple tracking without temporal filtering - count unique IDs immediately
    tracker = CentroidTracker(max_disappeared=int(fps/2), max_distance=80)
    all_ids_seen = set()  # Count all IDs that appear in ROI (no temporal filter)
    
    frame_idx = -1
    processed_idx = 0
    save_counter = 0
    rows = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SKIP_FRAMES != 0:
            continue
        processed_idx += 1
        
        proc = apply_clahe_bgr(frame)
        results = model.predict(source=proc, imgsz=IMGSZ, conf=CONF, classes=[0], device=device, verbose=False)
        r = results[0]
        
        boxes = []
        cents = []
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2) in xyxy:
                bx = (int(x1), int(y1), int(x2), int(y2))
                boxes.append(bx)
                cents.append(box_centroid(bx))
        
        tracker.update(cents)
        
        # Assign boxes to tracked objects
        assigned = {}
        if len(boxes) > 0 and len(cents) > 0:
            for oid, c in tracker.objects.items():
                dists = [math.hypot(c[0]-cx, c[1]-cy) for (cx, cy) in cents]
                if not dists:
                    continue
                best_i = int(np.argmin(dists))
                if best_i < len(boxes):
                    assigned[oid] = (boxes[best_i], cents[best_i])
        
        # Count all IDs in ROI immediately (NO temporal filtering)
        for oid, (box, cent) in assigned.items():
            cx, cy = cent
            if inside_roi_xy(cx, cy, roi_poly):
                all_ids_seen.add(oid)  # Immediately count
        
        count_inside = sum(1 for oid, (box, cent) in assigned.items() if oid in all_ids_seen and inside_roi_xy(cent[0], cent[1], roi_poly))
        
        # Visualization
        vis = frame.copy()
        for oid, (box, cent) in assigned.items():
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if oid in all_ids_seen else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"ID-{oid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.polylines(vis, [roi_poly], True, (255, 255, 0), 2)
        cv2.putText(vis, f"In Queue: {count_inside}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        save_counter += 1
        if save_counter % SAVE_EVERY_PROCESSED == 0:
            out_path = os.path.join(BASELINE_NO_TEMPORAL_DIR, f"proc_{processed_idx:06d}.jpg")
            cv2.imwrite(out_path, vis)
        
        rows.append({
            "processed_idx": processed_idx,
            "pred_count": count_inside,
            "cumulative_count": len(all_ids_seen)
        })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    cap.release()
    pd.DataFrame(rows).to_csv(PRED_CSV_NO_TEMPORAL, index=False)
    print(f"[Baseline No-Temporal] Saved predictions to {PRED_CSV_NO_TEMPORAL}")
    print(f"[Baseline No-Temporal] Done! Unique counted (no temporal filter): {len(all_ids_seen)}")
    
    return {"method": "no_temporal", "unique_count": len(all_ids_seen), "rows": rows}


# -----------------------------
# Baseline 2: YOLOv8 + DeepSORT
# Uses DeepSORT for more robust tracking with appearance features
# -----------------------------
def run_baseline_deepsort():
    """
    Baseline: YOLOv8 + DeepSORT tracking.
    Uses DeepSORT's deep appearance features for more robust tracking.
    Evaluated offline - counts unique track IDs that appear in ROI.
    """
    if not DEEPSORT_AVAILABLE:
        raise RuntimeError("DeepSORT not available. Install with: pip install deep-sort-realtime")
    
    roi_points = select_roi_with_jump(VIDEO_PATH, jump_frame=ROI_FRAME_INDEX)
    roi_poly = np.array(roi_points, dtype=np.int32)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[Baseline DeepSORT] Device:", device)
    
    model = YOLO("yolov8n.pt")
    try:
        model.to(device)
    except Exception as e:
        print("Warning: failed to move model to device:", e)
    
    # Initialize DeepSORT tracker
    deepsort = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=100,
        embedder="mobilenet",
        half=False,
        embedder_gpu=torch.cuda.is_available()
    )
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[Baseline DeepSORT] Video FPS: {fps}, Total frames: {total_frames}")
    
    frames_required_for_2s = max(1, int(ceil(3.0 * fps / SKIP_FRAMES)))
    all_track_ids = set()
    track_roi_frames = defaultdict(list)  # track_id -> list of frame indices in ROI
    confirmed_ids = set()
    
    frame_idx = -1
    processed_idx = 0
    save_counter = 0
    rows = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SKIP_FRAMES != 0:
            continue
        processed_idx += 1
        
        proc = apply_clahe_bgr(frame)
        results = model.predict(source=proc, imgsz=IMGSZ, conf=CONF, classes=[0], device=device, verbose=False)
        r = results[0]
        
        # Prepare detections for DeepSORT: [[x1, y1, w, h, conf], ...]
        detections = []
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], confs[i], 'person'))
        
        # Update DeepSORT tracker
        tracks = deepsort.update_tracks(detections, frame=frame)
        
        vis = frame.copy()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            all_track_ids.add(track_id)
            
            if inside_roi_xy(cx, cy, roi_poly):
                track_roi_frames[track_id].append(processed_idx)
                
                # Apply temporal filtering (2-second rule) for fair comparison
                frames_in_roi = track_roi_frames[track_id]
                if len(frames_in_roi) >= frames_required_for_2s:
                    # Check for consecutive frames
                    seq_len = 1
                    for i in range(len(frames_in_roi)-1, 0, -1):
                        if frames_in_roi[i] - frames_in_roi[i-1] <= 1:
                            seq_len += 1
                        else:
                            break
                    if seq_len >= frames_required_for_2s and track_id not in confirmed_ids:
                        confirmed_ids.add(track_id)
                        print(f"[DeepSORT] Counted track {track_id} at processed_idx {processed_idx}")
            
            # Visualization
            color = (0, 255, 0) if track_id in confirmed_ids else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"T-{track_id}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        count_inside = sum(1 for t in tracks if t.is_confirmed() and t.track_id in confirmed_ids and inside_roi_xy((int(t.to_ltrb()[0])+int(t.to_ltrb()[2]))//2, (int(t.to_ltrb()[1])+int(t.to_ltrb()[3]))//2, roi_poly))
        
        cv2.polylines(vis, [roi_poly], True, (255, 255, 0), 2)
        cv2.putText(vis, f"In Queue: {count_inside}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        save_counter += 1
        if save_counter % SAVE_EVERY_PROCESSED == 0:
            out_path = os.path.join(BASELINE_DEEPSORT_DIR, f"proc_{processed_idx:06d}.jpg")
            cv2.imwrite(out_path, vis)
        
        rows.append({
            "processed_idx": processed_idx,
            "pred_count": count_inside,
            "cumulative_count": len(confirmed_ids)
        })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    cap.release()
    pd.DataFrame(rows).to_csv(PRED_CSV_DEEPSORT, index=False)
    print(f"[Baseline DeepSORT] Saved predictions to {PRED_CSV_DEEPSORT}")
    print(f"[Baseline DeepSORT] Done! Unique counted (with 2s filter): {len(confirmed_ids)}")
    
    return {"method": "deepsort", "unique_count": len(confirmed_ids), "rows": rows}


# -----------------------------
# Compare all methods
# -----------------------------
def compare_all_methods():
    """
    Run all three methods and generate comparison metrics/plots.
    1. Proposed: YOLOv8 + Centroid Tracker + Temporal Filtering
    2. Baseline 1: YOLOv8 without temporal filtering
    3. Baseline 2: YOLOv8 + DeepSORT
    """
    print("="*60)
    print("RUNNING COMPARISON OF ALL METHODS")
    print("="*60)
    
    results = {}
    
    # Run proposed method
    print("\n[1/3] Running PROPOSED method (YOLOv8 + Centroid + Temporal Filter)...")
    run_pipeline()
    proposed_df = pd.read_csv(PRED_CSV)
    results['proposed'] = {
        'name': 'Proposed (Centroid + Temporal)',
        'unique_count': proposed_df['cumulative_count'].iloc[-1] if len(proposed_df) > 0 else 0,
        'csv': PRED_CSV
    }
    
    # Run baseline without temporal filtering
    print("\n[2/3] Running BASELINE 1 (YOLOv8 without temporal filtering)...")
    run_baseline_no_temporal()
    no_temporal_df = pd.read_csv(PRED_CSV_NO_TEMPORAL)
    results['no_temporal'] = {
        'name': 'YOLOv8 (No Temporal Filter)',
        'unique_count': no_temporal_df['cumulative_count'].iloc[-1] if len(no_temporal_df) > 0 else 0,
        'csv': PRED_CSV_NO_TEMPORAL
    }
    
    # Run DeepSORT baseline
    print("\n[3/3] Running BASELINE 2 (YOLOv8 + DeepSORT)...")
    if DEEPSORT_AVAILABLE:
        run_baseline_deepsort()
        deepsort_df = pd.read_csv(PRED_CSV_DEEPSORT)
        results['deepsort'] = {
            'name': 'YOLOv8 + DeepSORT',
            'unique_count': deepsort_df['cumulative_count'].iloc[-1] if len(deepsort_df) > 0 else 0,
            'csv': PRED_CSV_DEEPSORT
        }
    else:
        print("DeepSORT not available, skipping...")
        results['deepsort'] = {'name': 'YOLOv8 + DeepSORT', 'unique_count': 'N/A', 'csv': None}
    
    # Generate comparison if ground truth exists
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for method, data in results.items():
        print(f"{data['name']}: {data['unique_count']} unique people counted")
    
    # If ground truth exists, compute metrics for each method
    if os.path.exists(GT_CSV):
        print("\n" + "="*60)
        print("EVALUATION AGAINST GROUND TRUTH")
        print("="*60)
        
        gt = pd.read_csv(GT_CSV)
        comparison_data = []
        
        for method, data in results.items():
            if data['csv'] and os.path.exists(data['csv']):
                preds = pd.read_csv(data['csv'])
                merged = pd.merge(gt, preds, on="processed_idx", how="left", suffixes=("_gt", "_pred"))
                merged['pred_count'] = merged['pred_count'].fillna(0).astype(int)
                
                gt_list = merged['gt_count'].tolist()
                pred_list = merged['pred_count'].tolist()
                
                errors = [abs(g-p) for g, p in zip(gt_list, pred_list)]
                MAE = sum(errors) / len(errors) if errors else 0
                RMSE = math.sqrt(sum((g-p)**2 for g, p in zip(gt_list, pred_list)) / len(gt_list)) if gt_list else 0
                
                tp = sum(min(g, p) for g, p in zip(gt_list, pred_list))
                fp = sum(max(0, p-g) for g, p in zip(gt_list, pred_list))
                fn = sum(max(0, g-p) for g, p in zip(gt_list, pred_list))
                
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
                
                total_error = sum(errors)
                total_people = sum(gt_list)
                queue_accuracy = (1 - total_error / total_people) * 100 if total_people > 0 else 0
                
                comparison_data.append({
                    'Method': data['name'],
                    'MAE': round(MAE, 3),
                    'RMSE': round(RMSE, 3),
                    'Precision': round(precision, 3),
                    'Recall': round(recall, 3),
                    'F1': round(f1, 3),
                    'Queue_Accuracy': round(queue_accuracy, 2)
                })
                
                print(f"\n{data['name']}:")
                print(f"  MAE: {MAE:.3f}, RMSE: {RMSE:.3f}")
                print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
                print(f"  Queue Accuracy: {queue_accuracy:.2f}%")
        
        # Save comparison table
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            comp_df.to_csv(os.path.join(COMPARISON_OUT_DIR, "method_comparison.csv"), index=False)
            print(f"\nSaved comparison table to {COMPARISON_OUT_DIR}/method_comparison.csv")
            
            # Generate comparison bar plots
            generate_comparison_plots(comparison_data)
    else:
        print("\nNo ground truth file found. Run --create_gt first for detailed evaluation.")
    
    return results


def generate_comparison_plots(comparison_data):
    """Generate bar plots comparing all methods."""
    if not comparison_data:
        return
    
    methods = [d['Method'] for d in comparison_data]
    # Shorten method names for better display
    short_names = []
    for m in methods:
        if 'Proposed' in m:
            short_names.append('Proposed')
        elif 'No Temporal' in m:
            short_names.append('No Temporal')
        elif 'DeepSORT' in m:
            short_names.append('DeepSORT')
        else:
            short_names.append(m[:15])
    
    metrics = ['MAE', 'RMSE', 'Precision', 'Recall', 'F1', 'Queue_Accuracy']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green for proposed, blue for no-temporal, red for deepsort
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [d[metric] for d in comparison_data]
        bars = ax.bar(short_names, values, color=colors[:len(values)])
        ax.set_title(metric.replace('_', ' '))
        ax.set_ylabel(metric)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        # Rotate x labels for better fit
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('Method Comparison: Proposed vs Baselines', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_OUT_DIR, "method_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved comparison plot to {COMPARISON_OUT_DIR}/method_comparison.png")


# -----------------------------
# Ground-truth helper
# -----------------------------
def create_ground_truth_from_saved_frames(processed_dir=OUT_DIR, pred_csv=PRED_CSV, gt_csv=GT_CSV):
    if not os.path.exists(pred_csv):
        print("Predictions CSV not found. Run --run_pipeline first.")
        return
    preds = pd.read_csv(pred_csv)
    rows = []
    for _, r in preds.iterrows():
        idx = int(r['processed_idx'])
        img_name = f"proc_{idx:06d}.jpg"
        img_path = os.path.join(processed_dir, img_name)
        rows.append({"processed_idx": idx, "pred_count": int(r['pred_count']), "image_path": img_path})
    gt_rows=[]
    print("Starting interactive ground truth creation.")
    print("Controls: type number + Enter to save; 's' to skip; 'q' to quit and save progress.")
    for item in rows:
        idx = item['processed_idx']; p = item['pred_count']; ip = item['image_path']
        if not os.path.exists(ip):
            print(f"Image missing, skipping: {ip}")
            continue
        img = cv2.imread(ip)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8,5)); plt.imshow(img_rgb); plt.title(f"processed_idx={idx}  pred={p}"); plt.axis('off'); plt.show()
        val = input(f"GT count for processed_idx {idx} (pred={p}) >> ")
        plt.close()
        if val.strip().lower() == 'q':
            print("Quitting annotation, saving progress.")
            break
        if val.strip().lower() == 's' or val.strip()=='':
            print("Skipped", idx)
            continue
        try:
            g = int(val.strip())
        except Exception:
            print("Invalid input, skipping.")
            continue
        gt_rows.append({"processed_idx": idx, "gt_count": g, "image_path": ip})
    if len(gt_rows) == 0:
        print("No ground-truth entries collected.")
        return
    gdf = pd.DataFrame(gt_rows)
    gdf.to_csv(gt_csv, index=False)
    print("Saved ground truth to", gt_csv)
    return gdf

# -----------------------------
# Evaluation (with queue occupancy accuracy)
# -----------------------------
def evaluate_and_plot(pred_csv=PRED_CSV, gt_csv=GT_CSV, out_dir=EVAL_OUT_DIR):
    if not os.path.exists(pred_csv):
        raise RuntimeError("Predictions CSV not found. Run --run_pipeline first.")
    if not os.path.exists(gt_csv):
        raise RuntimeError("Ground truth CSV not found. Run --create_gt first.")
    preds = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)
    merged = pd.merge(gt, preds, on="processed_idx", how="left", suffixes=("_gt","_pred"))
    merged['pred_count'] = merged['pred_count'].fillna(0).astype(int)
    gt_list = merged['gt_count'].tolist()
    pred_list = merged['pred_count'].tolist()
    processed_idxs = merged['processed_idx'].tolist()
    # basic metrics
    errors = [abs(g-p) for g,p in zip(gt_list,pred_list)]
    MAE = sum(errors)/len(errors)
    RMSE = math.sqrt(sum((g-p)**2 for g,p in zip(gt_list,pred_list))/len(gt_list))
    within1 = sum(1 for e in errors if e<=1)/len(errors)
    within2 = sum(1 for e in errors if e<=2)/len(errors)
    # confusion aggregate
    tp_list = [min(g,p) for g,p in zip(gt_list,pred_list)]
    fp_list = [max(0,p-g) for g,p in zip(gt_list,pred_list)]
    fn_list = [max(0,g-p) for g,p in zip(gt_list,pred_list)]
    TP = sum(tp_list); FP = sum(fp_list); FN = sum(fn_list); TN = 0
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * (precision*recall) / (precision + recall + 1e-9)
    # --- Queue Occupancy Accuracy (recommended) ---
    total_error = sum(errors)
    total_people = sum(gt_list)
    if total_people > 0:
        queue_accuracy = 1 - (total_error / total_people)
    else:
        queue_accuracy = None
    # print metrics
    print("\n=== OCCUPANCY METRICS ===")
    print("Frames evaluated:", len(gt_list))
    print(f"MAE: {MAE:.3f}  RMSE: {RMSE:.3f}")
    print(f"Within ±1: {within1*100:.2f}%  Within ±2: {within2*100:.2f}%")
    print("\n=== COUNTING CONFUSION (aggregate) ===")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print("\n=== CLASSIFICATION METRICS ===")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    if queue_accuracy is not None:
        print(f"\nQueue Occupancy Accuracy (1 - total_error/total_GT_people): {queue_accuracy*100:.2f}%")
    else:
        print("\nQueue Occupancy Accuracy: undefined (no GT people found).")
    # save numeric summary
    summary = {
        "frames_evaluated": len(gt_list),
        "MAE": MAE, "RMSE": RMSE, "within1": within1, "within2": within2,
        "TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "f1": f1,
        "queue_accuracy": queue_accuracy
    }
    pd.Series(summary).to_csv(os.path.join(out_dir, "summary_metrics.csv"))
    print("Saved numeric summary to", os.path.join(out_dir, "summary_metrics.csv"))
    # ------------------- plots -------------------
    # GT vs Pred over time
    plt.figure(figsize=(10,4))
    plt.plot(processed_idxs, pred_list, label='Pred')
    plt.plot(processed_idxs, gt_list, label='GT', linestyle='--')
    plt.xlabel('processed_idx'); plt.ylabel('count'); plt.title('GT vs Pred over processed frames'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "gt_vs_pred_over_time.png")); plt.close()
    # abs error over time
    plt.figure(figsize=(10,3))
    plt.plot(processed_idxs, errors, label='abs_error')
    plt.xlabel('processed_idx'); plt.ylabel('abs_error'); plt.title('Absolute Error over time'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "abs_error_over_time.png")); plt.close()
    # scatter
    plt.figure(figsize=(5,5))
    plt.scatter(gt_list, pred_list, alpha=0.6)
    mx = max(max(gt_list), max(pred_list))+1
    plt.plot([0,mx],[0,mx], linestyle='--')
    plt.xlabel('GT'); plt.ylabel('Pred'); plt.title('GT vs Pred scatter'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "gt_vs_pred_scatter.png")); plt.close()
    # confusion aggregate heatmap-like
    plt.figure(figsize=(4,3))
    mat = np.array([[TP, FP],[FN, 0]])
    plt.imshow(mat, interpolation='nearest')
    for (i,j), val in np.ndenumerate(mat):
        plt.text(j, i, str(int(val)), ha='center', va='center', color='white' if mat.max()>0 else 'black')
    plt.xticks([0,1], ['TP','FP']); plt.yticks([0,1], ['TP-row','FN-row'])
    plt.title('Aggregate counts (TP/FP/FN)'); plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion_aggregate.png")); plt.close()
    # error histogram
    plt.figure(figsize=(6,3))
    plt.hist(errors, bins=range(0, max(errors)+2))
    plt.xlabel('Absolute Error'); plt.ylabel('Frequency'); plt.title('Error Distribution (abs error)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "error_histogram.png")); plt.close()
    # precision/recall/f1 bar
    plt.figure(figsize=(5,3))
    plt.bar(['precision','recall','f1'], [precision, recall, f1])
    plt.ylim(0,1.05)
    plt.title('Aggregate classification metrics'); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "prf_bar.png")); plt.close()
    # unique total bar (sum-of-frames approximation)
    GT_unique_approx = int(sum(gt_list))
    Pred_unique_approx = int(sum(pred_list))
    plt.figure(figsize=(4,3))
    plt.bar(['GT_unique_sum','Pred_unique_sum'], [GT_unique_approx, Pred_unique_approx])
    plt.title('Total people counted (sum over frames) - approximate'); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "unique_count_bar.png")); plt.close()
    print("Saved evaluation plots to", out_dir)
    return summary

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Queue Monitoring System with Baseline Comparisons")
    p.add_argument("--run_pipeline", action="store_true",
                   help="Run proposed method (YOLOv8 + Centroid Tracker + Temporal Filtering)")
    p.add_argument("--baseline_no_temporal", action="store_true",
                   help="Run baseline: YOLOv8 without temporal filtering")
    p.add_argument("--baseline_deepsort", action="store_true",
                   help="Run baseline: YOLOv8 + DeepSORT tracker")
    p.add_argument("--compare", action="store_true",
                   help="Run all methods and generate comparison")
    p.add_argument("--create_gt", action="store_true",
                   help="Create ground truth annotations interactively")
    p.add_argument("--evaluate", action="store_true",
                   help="Evaluate predictions against ground truth")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.run_pipeline:
        print("Running PROPOSED pipeline (ROI selection -> inference)...")
        run_pipeline()
        print("Pipeline finished.")
    
    if args.baseline_no_temporal:
        print("Running BASELINE: YOLOv8 without temporal filtering...")
        run_baseline_no_temporal()
        print("Baseline (no temporal) finished.")
    
    if args.baseline_deepsort:
        print("Running BASELINE: YOLOv8 + DeepSORT...")
        if DEEPSORT_AVAILABLE:
            run_baseline_deepsort()
            print("Baseline (DeepSORT) finished.")
        else:
            print("Error: DeepSORT not installed. Run: pip install deep-sort-realtime")
    
    if args.compare:
        print("Running comparison of all methods...")
        compare_all_methods()
        print("Comparison finished.")
    
    if args.create_gt:
        print("Launching ground-truth creation helper...")
        create_ground_truth_from_saved_frames()
    
    if args.evaluate:
        print("Running evaluation...")
        evaluate_and_plot()
    
    if not any([args.run_pipeline, args.baseline_no_temporal, args.baseline_deepsort,
                args.compare, args.create_gt, args.evaluate]):
        print("Queue Monitoring System with Baseline Comparisons")
        print("="*50)
        print("\nAvailable commands:")
        print("  --run_pipeline        : Proposed method (Centroid + Temporal Filter)")
        print("  --baseline_no_temporal: YOLOv8 without temporal filtering")
        print("  --baseline_deepsort   : YOLOv8 + DeepSORT tracker")
        print("  --compare             : Run all methods and compare")
        print("  --create_gt           : Create ground truth annotations")
        print("  --evaluate            : Evaluate against ground truth")
        print("\nExample: python3 queue_monitor.py --compare")
