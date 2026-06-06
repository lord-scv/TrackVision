from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame):
        """
        Run object detection on the input frame.
        
        Args:
            frame: numpy.ndarray (BGR format image)
            
        Returns:
            list of tuple: A list of detections in the format:
                (x1, y1, x2, y2, confidence, class_id, class_name)
                where:
                - x1, y1, x2, y2: Bounding box coordinates (float/int)
                - confidence: Detection confidence score (float, 0.0 to 1.0)
                - class_id: Integer class ID
                - class_name: String class name
        """
        pass
