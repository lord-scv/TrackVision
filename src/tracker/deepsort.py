from collections import deque
import numpy as np
from src.tracker.base import BaseTracker
from src.tracker.state import Track, TrackState
from src.tracker.kalman import KalmanFilter
from src.tracker.association import iou_distance, cosine_distance, linear_assignment
from src.tracker.embedder import ReIDEmbedder

class DeepSORTTracker(BaseTracker):
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, reid_model="osnet_x0_25"):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.kf = KalmanFilter()
        self.embedder = ReIDEmbedder(model_name=reid_model)
        self.track_id_counter = 1
        
        # DeepSORT association parameters
        self.max_cosine_distance = 0.2
        self.nn_budget = 100

    def update(self, detections, frame):
        """
        detections: list of (x1, y1, x2, y2, confidence, class_id, class_name)
        """
        if len(detections) == 0:
            # Predict and age all tracks
            for track in self.tracks:
                track.kf_mean, track.kf_covariance = self.kf.predict(track.kf_mean, track.kf_covariance)
                track.mark_missed()
                if track.time_since_update > self.max_age:
                    track.state = TrackState.DELETED
            self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
            return self.tracks

        # Extract features for all detections
        det_bboxes = [d[:4] for d in detections]
        det_features = self.embedder.extract(frame, det_bboxes)

        # Step 1: Predict state forward for all active tracks
        for track in self.tracks:
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
            
            track.pred_bbox = [pred_x1, pred_y1, pred_x2, pred_y2]

        # Separate confirmed and tentative tracks
        confirmed_tracks_indices = [i for i, t in enumerate(self.tracks) if t.state == TrackState.CONFIRMED]
        tentative_tracks_indices = [i for i, t in enumerate(self.tracks) if t.state == TrackState.TENTATIVE]

        confirmed_tracks = [self.tracks[i] for i in confirmed_tracks_indices]

        # Step 2: Cascaded matching on confirmed tracks by age
        matches_a = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks_confirmed = []

        for age in range(1, self.max_age + 1):
            tracks_to_match_idx = [
                i for i, t in enumerate(confirmed_tracks) 
                if t.time_since_update == age - 1
            ]
            if len(tracks_to_match_idx) == 0:
                continue
                
            tracks_to_match = [confirmed_tracks[i] for i in tracks_to_match_idx]
            
            # Distance matrix matching (minimum cosine distance over track feature history)
            cost_matrix = np.zeros((len(tracks_to_match), len(unmatched_detections)), dtype=np.float32)
            for r, track in enumerate(tracks_to_match):
                if not hasattr(track, 'features'):
                    track.features = deque(maxlen=self.nn_budget)
                
                track_feats = np.array(list(track.features))
                det_feats = det_features[unmatched_detections]
                
                dist = cosine_distance(track_feats, det_feats)
                cost_matrix[r] = np.min(dist, axis=0)
                
            matches, unmatched_t_local, unmatched_d_local = linear_assignment(
                cost_matrix, self.max_cosine_distance
            )
            
            for r, c in matches:
                track_obj = tracks_to_match[r]
                det_idx = unmatched_detections[c]
                matches_a.append((self.tracks.index(track_obj), det_idx))
                
            unmatched_detections = [unmatched_detections[c] for c in unmatched_d_local]
            
            for r in unmatched_t_local:
                unmatched_tracks_confirmed.append(self.tracks.index(tracks_to_match[r]))

        # Include other confirmed tracks that were not checked
        for i, t in enumerate(confirmed_tracks):
            idx = confirmed_tracks_indices[i]
            if idx not in [m[0] for m in matches_a] and idx not in unmatched_tracks_confirmed:
                unmatched_tracks_confirmed.append(idx)

        # Step 3: Match remaining tracks (tentative + unmatched confirmed) via IoU distance
        remaining_tracks_idx = tentative_tracks_indices + unmatched_tracks_confirmed
        remaining_tracks = [self.tracks[i] for i in remaining_tracks_idx]
        
        cost_matrix_iou = iou_distance(
            [t.pred_bbox for t in remaining_tracks], 
            [detections[d] for d in unmatched_detections]
        )
        matches_b, unmatched_t_b, unmatched_d_b = linear_assignment(
            cost_matrix_iou, 1.0 - self.iou_threshold
        )

        # Combine matches
        final_matches = list(matches_a)
        for r, c in matches_b:
            track_idx = remaining_tracks_idx[r]
            det_idx = unmatched_detections[c]
            final_matches.append((track_idx, det_idx))

        unmatched_tracks = [remaining_tracks_idx[r] for r in unmatched_t_b]
        final_unmatched_detections = [unmatched_detections[c] for c in unmatched_d_b]

        # Step 4: Update matches
        for t_idx, d_idx in final_matches:
            track = self.tracks[t_idx]
            det = detections[d_idx]
            det_bbox = det[:4]
            det_conf = det[4]
            
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
            
            upd_cx, upd_cy, upd_w, upd_h = track.kf_mean[:4]
            upd_x1 = upd_cx - upd_w / 2.0
            upd_y1 = upd_cy - upd_h / 2.0
            upd_x2 = upd_cx + upd_w / 2.0
            upd_y2 = upd_cy + upd_h / 2.0
            
            track.update([upd_x1, upd_y1, upd_x2, upd_y2], det_conf)
            
            # Add feature embedding to track's history pool
            if not hasattr(track, 'features'):
                track.features = deque(maxlen=self.nn_budget)
            track.features.append(det_features[d_idx])
            
            if track.state == TrackState.TENTATIVE and track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED

        # Step 5: Mark unmatched tracks as missed
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.mark_missed()
            if track.time_since_update > self.max_age:
                track.state = TrackState.DELETED

        # Step 6: Spawn tentative tracks for unmatched detections
        for d_idx in final_unmatched_detections:
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
            
            new_track.features = deque(maxlen=self.nn_budget)
            new_track.features.append(det_features[d_idx])
            
            if self.min_hits <= 1:
                new_track.state = TrackState.CONFIRMED
                
            self.tracks.append(new_track)

        # Clear deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        return self.tracks
