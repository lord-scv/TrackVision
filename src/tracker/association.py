import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    bbox: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
        
    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_u = area1 + area2 - area_i
    
    if area_u <= 0:
        return 0.0
        
    return area_i / area_u

def iou_distance(tracks, detections):
    """
    Computes the IoU distance matrix between tracks and detections.
    tracks: list of Track objects or boxes [x1, y1, x2, y2]
    detections: list of detection boxes [x1, y1, x2, y2, ...] or Track objects
    Returns:
        cost_matrix of shape (len(tracks), len(detections))
        where cost_matrix[i, j] = 1 - IoU(tracks[i], detections[j])
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        bbox1 = track.bbox if hasattr(track, 'bbox') else track
        for j, det in enumerate(detections):
            # detections can be list of tuples (x1, y1, x2, y2, conf, class_id, class_name)
            # or just bounding boxes.
            bbox2 = det.bbox if hasattr(det, 'bbox') else (det[:4] if isinstance(det, (list, tuple, np.ndarray)) else det)
            cost_matrix[i, j] = 1.0 - iou(bbox1, bbox2)
    return cost_matrix

def cosine_distance(features1, features2):
    """
    Computes L2-normalized cosine distance between two feature matrices.
    features1: shape (N, D)
    features2: shape (M, D)
    Returns:
        (N, M) matrix of cosine distances (1 - cosine_similarity)
    """
    if features1.size == 0 or features2.size == 0:
        return np.empty((features1.shape[0], features2.shape[0]), dtype=np.float32)
        
    # L2 normalize rows
    f1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
    f2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
    
    # Cosine similarity
    sim = np.dot(f1, f2.T)
    return 1.0 - sim

def linear_assignment(cost_matrix, max_distance):
    """
    Solve assignment problem using Hungarian algorithm.
    Gated by max_distance.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_tracks = []
    unmatched_detections = []
    
    matched_cols = set()
    matched_rows = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > max_distance:
            continue
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)
        
    for r in range(cost_matrix.shape[0]):
        if r not in matched_rows:
            unmatched_tracks.append(r)
            
    for c in range(cost_matrix.shape[1]):
        if c not in matched_cols:
            unmatched_detections.append(c)
            
    return matches, unmatched_tracks, unmatched_detections
