import cv2
import torch
import logging
from src.detector.base import BaseDetector

logger = logging.getLogger("FasterRCNNDetector")

# COCO 91 classes list (matches torchvision Faster R-CNN defaults)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class FasterRCNNDetector(BaseDetector):
    def __init__(self, model_size="small", confidence=0.45, nms_iou=0.5, classes=None):
        """
        Faster R-CNN backend using torchvision.
        model_size: "small" (uses MobileNetV3 backbone), "medium" or "large" (uses ResNet50 backbone)
        confidence: confidence threshold
        nms_iou: NMS threshold (not directly exposed in torchvision inference but used for custom scoring if needed)
        classes: list of class IDs to filter. Note that Faster R-CNN IDs might differ slightly from YOLOv8 COCO IDs.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence = confidence
        self.classes = classes if classes else None
        
        logger.info(f"Loading Faster R-CNN model (size: {model_size}) on device: {self.device}")
        
        # Load appropriate model from torchvision
        try:
            if model_size.lower() == "small":
                # Lightweight CPU-friendly model
                try:
                    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
                    self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
                except ImportError:
                    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
                    self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
            else:
                # Heavyweight ResNet50 model
                try:
                    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
                    self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                except ImportError:
                    from torchvision.models.detection import fasterrcnn_resnet50_fpn
                    self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        except Exception as e:
            logger.error(f"Failed to load Faster R-CNN model: {e}. Falling back to MobileNetV3 FPN.")
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame):
        # Convert frame from BGR to RGB, normalize to [0, 1] range, and convert to Tensor
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        
        # Faster R-CNN expects batch dimension
        outputs = self.model([tensor])
        
        detections = []
        if not outputs:
            return detections
            
        output = outputs[0]
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        
        for box, label_id, score in zip(boxes, labels, scores):
            if score < self.confidence:
                continue
                
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            
            # Map label ID to COCO class name
            if label_id < len(COCO_CLASSES):
                cls_name = COCO_CLASSES[label_id]
            else:
                cls_name = f"object_{label_id}"
                
            # Filter class list if specified
            if self.classes is not None and len(self.classes) > 0 and label_id not in self.classes:
                continue
                
            detections.append((x1, y1, x2, y2, score, int(label_id), cls_name))
            
        return detections
