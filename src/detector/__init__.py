from src.detector.yolo import YOLODetector
from src.detector.faster_rcnn import FasterRCNNDetector

def get_detector(backend_name, model_size="small", confidence=0.45, nms_iou=0.50, classes=None):
    """
    Factory function to instantiate object detector backends.
    """
    name = backend_name.lower()
    if name in ["yolov8", "yolo"]:
        return YOLODetector(model_size, confidence, nms_iou, classes)
    elif name in ["faster_rcnn", "fasterrcnn"]:
        return FasterRCNNDetector(model_size, confidence, nms_iou, classes)
    else:
        raise ValueError(f"Unsupported detector backend: {backend_name}")
