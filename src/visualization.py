import cv2
import numpy as np
import time
from src.tracker.state import TrackState

class Visualizer:
    def __init__(self, trail_length=20, draw_velocity=True):
        self.trail_length = trail_length
        self.draw_velocity = draw_velocity

    def get_track_color(self, track_id):
        """
        Deterministic hash-to-color mapping (Section 10 of PRD):
        hue = (track_id * 47) mod 360
        saturation = 0.72
        value = 0.88
        """
        hue = int((track_id * 47) % 360)
        # Convert HSV (Hue [0, 179], Saturation [0, 255], Value [0, 255]) to BGR
        hsv = np.uint8([[[hue // 2, int(0.72 * 255), int(0.88 * 255)]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in bgr)

    def draw(self, frame, tracks, fps, frame_num):
        """
        Draw all tracking elements onto the frame.
        """
        annotated_frame = frame.copy()
        confirmed_count = 0
        
        for track in tracks:
            # We only draw tentative and confirmed tracks, not deleted ones.
            if track.state == TrackState.DELETED:
                continue
                
            if track.state == TrackState.CONFIRMED:
                confirmed_count += 1
                
            color = self.get_track_color(track.id)
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # 1. Bounding Box & Label
            label = f"{track.class_name} #{track.id} {int(track.confidence * 100)}%"
            self._draw_premium_box(annotated_frame, (x1, y1, x2, y2), color, label)
            
            # 2. Draw Trails
            if self.trail_length > 0 and len(track.history) > 1:
                self._draw_fading_trail(annotated_frame, list(track.history)[-self.trail_length:], color)
                
            # 3. Draw Velocity Vector
            if self.draw_velocity and track.velocity != [0.0, 0.0]:
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                vx, vy = track.velocity
                # Scale velocity vector arrow for visibility (5 frames scale)
                scale = 5.0
                end_x = int(centroid_x + vx * scale)
                end_y = int(centroid_y + vy * scale)
                cv2.arrowedLine(annotated_frame, (centroid_x, centroid_y), (end_x, end_y), color, 2, tipLength=0.3)

        # 4. Draw HUD Overlay
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self._draw_hud(annotated_frame, fps, confirmed_count, frame_num, timestamp)
        
        return annotated_frame

    def _draw_premium_box(self, frame, bbox, color, label):
        x1, y1, x2, y2 = bbox
        # Draw thin solid box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Draw sci-fi corner highlights
        length = min(15, int((x2 - x1) * 0.2))
        t = 3  # corner thickness
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, t)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, t)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, t)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, t)
        
        # Draw text label box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        
        # Keep label box within image boundaries
        label_y = max(y1, text_h + 8)
        cv2.rectangle(frame, (x1, label_y - text_h - 6), (x1 + text_w + 6, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 3, label_y - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_fading_trail(self, frame, history, color):
        num_points = len(history)
        for i in range(num_points - 1):
            pt1 = tuple(map(int, history[i]))
            pt2 = tuple(map(int, history[i + 1]))
            # Calculate alpha fading factor
            alpha = (i + 1) / num_points
            # Fade towards background color by scaling BGR intensities
            fade_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pt1, pt2, fade_color, thickness=2, lineType=cv2.LINE_AA)

    def _draw_hud(self, frame, fps, active_count, frame_num, timestamp):
        # Create dark overlay rectangle for HUD box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 115), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Thin HUD border
        cv2.rectangle(frame, (10, 10), (300, 115), (70, 70, 70), 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color_white = (240, 240, 240)
        color_accent = (0, 215, 255)  # Cyan/Gold HUD style accent
        
        cv2.putText(frame, "TRACKVISION CONTROL HUD", (20, 30), font, 0.5, color_accent, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), font, font_scale, color_white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Active Tracks: {active_count}", (20, 68), font, font_scale, color_white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame Number: {frame_num}", (20, 86), font, font_scale, color_white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Timestamp: {timestamp}", (20, 104), font, font_scale, color_white, 1, cv2.LINE_AA)
