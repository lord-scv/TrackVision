from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections, frame):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections, where each detection is:
                        (x1, y1, x2, y2, confidence, class_id, class_name)
            frame: The current video frame (numpy array, BGR)
            
        Returns:
            list: List of active Track objects in the current frame.
        """
        pass
