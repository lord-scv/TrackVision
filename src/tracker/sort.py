from src.tracker.base import BaseTracker
from src.tracker.state import Track, TrackState
from src.tracker.kalman import KalmanFilter
from src.tracker.association import iou_distance, linear_assignment
import numpy as np

class SORTTracker(BaseTracker):
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.kf = KalmanFilter()
        self.track_id_counter = 1

    def update(self, detections, frame):
        """
        detections: list of (x1, y1, x2, y2, confidence, class_id, class_name)
        """
        # Step 1: Predict state forward for all active tracks
        predicted_boxes = []
        for track in self.tracks:
            # Convert last bounding box to center, width, height for Kalman
            x1, y1, x2, y2 = track.bbox
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            
            if not hasattr(track, 'kf_mean') or track.kf_mean is None:
                measurement = np.array([cx, cy, w, h], dtype=np.float32)
                track.kf_mean, track.kf_covariance = self.kf.initiate(measurement)
                
            track.kf_mean, track.kf_covariance = self.kf.predict(track.kf_mean, track.kf_covariance)
            
            pred_cx, pred_cy, pred_w, pred_h = track.kf_mean[:4]
            pred_x1 = pred_cx - pred_w / 2.0
            pred_y1 = pred_cy - pred_h / 2.0
            pred_x2 = pred_cx + pred_w / 2.0
            pred_y2 = pred_cy + pred_h / 2.0
            
            predicted_boxes.append([pred_x1, pred_y1, pred_x2, pred_y2])
            
        # Step 2: Hungarian association using IoU distance
        cost_matrix = iou_distance(predicted_boxes, detections)
        max_cost = 1.0 - self.iou_threshold
        
        matches, unmatched_tracks, unmatched_detections = linear_assignment(cost_matrix, max_cost)
        
        # Step 3: Match updates
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            det = detections[d_idx]
            det_bbox = det[:4]
            det_conf = det[4]
            
            # Convert detection box to Kalman measurement
            det_x1, det_y1, det_x2, det_y2 = det_bbox
            det_w = max(1.0, det_x2 - det_x1)
            det_h = max(1.0, det_y2 - det_y1)
            det_cx = det_x1 + det_w / 2.0
            det_cy = det_y1 + det_h / 2.0
            measurement = np.array([det_cx, det_cy, det_w, det_h], dtype=np.float32)
            
            # Kalman measurement update
            track.kf_mean, track.kf_covariance = self.kf.update(
                track.kf_mean, track.kf_covariance, measurement
            )
            
            # Project mean back to bbox
            upd_cx, upd_cy, upd_w, upd_h = track.kf_mean[:4]
            upd_x1 = upd_cx - upd_w / 2.0
            upd_y1 = upd_cy - upd_h / 2.0
            upd_x2 = upd_cx + upd_w / 2.0
            upd_y2 = upd_cy + upd_h / 2.0
            
            track.update([upd_x1, upd_y1, upd_x2, upd_y2], det_conf)
            
            # Handle status conversion
            if track.state == TrackState.TENTATIVE and track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED

        # Step 4: Handle unmatched tracks
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.mark_missed()
            
            if track.time_since_update > self.max_age:
                track.state = TrackState.DELETED
                
        # Step 5: Initialize tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            det_bbox = det[:4]
            det_conf = det[4]
            det_cls_id = det[5]
            det_cls_name = det[6]
            
            new_track = Track(self.track_id_counter, det_bbox, det_cls_name, det_conf, det_cls_id)
            self.track_id_counter += 1
            
            det_x1, det_y1, det_x2, det_y2 = det_bbox
            det_w = max(1.0, det_x2 - det_x1)
            det_h = max(1.0, det_y2 - det_y1)
            det_cx = det_x1 + det_w / 2.0
            det_cy = det_y1 + det_h / 2.0
            measurement = np.array([det_cx, det_cy, det_w, det_h], dtype=np.float32)
            
            new_track.kf_mean, new_track.kf_covariance = self.kf.initiate(measurement)
            
            if self.min_hits <= 1:
                new_track.state = TrackState.CONFIRMED
                
            self.tracks.append(new_track)
            
        # Clear deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        return self.tracks
