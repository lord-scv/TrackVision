from collections import deque
import time

class TrackState:
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"

class Track:
    def __init__(self, track_id, bbox, class_name, confidence, class_id=0):
        self.id = track_id
        self.bbox = list(bbox)  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.state = TrackState.TENTATIVE
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        
        # Position history: last 30 frames of bounding box centroids
        self.history = deque(maxlen=30)
        centroid = self._get_centroid(bbox)
        self.history.append(centroid)
        
        # Velocity vector [vx, vy] (pixel change per frame)
        self.velocity = [0.0, 0.0]
        
        self.first_seen = time.time()
        self.last_seen = time.time()

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def update(self, bbox, confidence=None):
        self.bbox = list(bbox)
        if confidence is not None:
            self.confidence = confidence
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_seen = time.time()
        
        centroid = self._get_centroid(bbox)
        
        if len(self.history) > 0:
            last_c = self.history[-1]
            self.velocity = [centroid[0] - last_c[0], centroid[1] - last_c[1]]
            
        self.history.append(centroid)

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1
        self.velocity = [0.0, 0.0]

    def get_dwell_time(self):
        """Return the dwell time in seconds (difference between first and last seen)."""
        return self.last_seen - self.first_seen

    def to_dict(self):
        return {
            "id": self.id,
            "bbox": self.bbox,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "velocity": self.velocity,
            "age": self.age,
            "state": self.state,
            "dwell_time": self.get_dwell_time(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen
        }
