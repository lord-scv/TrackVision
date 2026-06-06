from src.tracker.sort import SORTTracker
from src.tracker.deepsort import DeepSORTTracker

def get_tracker(backend_name, max_age=30, min_hits=3, iou_threshold=0.3, reid_model="osnet_x0_25"):
    """
    Factory function to instantiate tracker backends.
    """
    name = backend_name.lower()
    if name == "sort":
        return SORTTracker(max_age, min_hits, iou_threshold)
    elif name == "deepsort":
        return DeepSORTTracker(max_age, min_hits, iou_threshold, reid_model)
    else:
        raise ValueError(f"Unsupported tracker backend: {backend_name}")
