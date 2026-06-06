import csv
import json
import os
import time
import cv2

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Write header
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'timestamp', 'track_id', 'class', 'x1', 'y1', 'x2', 'y2', 'confidence'])

    def log(self, frame_id, timestamp, track):
        x1, y1, x2, y2 = track.bbox
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                timestamp,
                track.id,
                track.class_name,
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                f"{track.confidence:.4f}"
            ])


class SessionSummary:
    def __init__(self):
        self.unique_ids = set()
        self.fps_records = []
        self.start_time = time.time()

    def record_track(self, track_id):
        self.unique_ids.add(track_id)

    def record_fps(self, fps):
        if fps > 0:
            self.fps_records.append(fps)

    def write_summary(self, filepath, track_id_classes):
        """
        Write end of session summary JSON.
        track_id_classes: Dictionary mapping track_id -> class_name.
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        avg_fps = sum(self.fps_records) / len(self.fps_records) if self.fps_records else 0.0
        peak_fps = max(self.fps_records) if self.fps_records else 0.0
        
        # Count unique class instances based on unique IDs
        class_counts = {}
        for tid in self.unique_ids:
            cls = track_id_classes.get(tid, "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
        summary = {
            "unique_ids": list(self.unique_ids),
            "total_unique_objects": len(self.unique_ids),
            "class_counts": class_counts,
            "avg_fps": float(f"{avg_fps:.2f}"),
            "peak_fps": float(f"{peak_fps:.2f}"),
            "duration_seconds": float(f"{duration:.2f}"),
            "total_frames": len(self.fps_records)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=4)


class VideoSaveHandler:
    def __init__(self, filepath, fps=30.0):
        self.filepath = filepath
        self.fps = fps
        self.writer = None
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def write_frame(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (w, h))
        self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()
            self.writer = None
