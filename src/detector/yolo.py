from ultralytics import YOLO
import torch
import logging
from src.detector.base import BaseDetector

logger = logging.getLogger("YOLODetector")

class YOLODetector(BaseDetector):
    def __init__(self, model_size="small", confidence=0.45, nms_iou=0.5, classes=None):
        """
        model_size: "nano", "small", "medium"
        confidence: confidence threshold
        nms_iou: NMS IoU threshold
        classes: list of class IDs to filter (e.g., [0] for person). If empty/None, keep all.
        """
        size_map = {
            "nano": "yolov8n.pt",
            "small": "yolov8s.pt",
            "medium": "yolov8m.pt"
        }
        model_name = size_map.get(model_size.lower(), "yolov8s.pt")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading YOLOv8 model: {model_name} on device: {self.device}")
        
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.classes = classes if classes else None

    def detect(self, frame):
        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.nms_iou,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if not results:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            cls_name = self.model.names[cls_id]
            
            # Filter by class list if specified
            if self.classes is not None and len(self.classes) > 0 and cls_id not in self.classes:
                continue
                
            detections.append((x1, y1, x2, y2, conf, cls_id, cls_name))
            
        return detections
