import unittest
import numpy as np
from src.tracker.association import iou, iou_distance, cosine_distance, linear_assignment

class TestAssociation(unittest.TestCase):
    def test_iou(self):
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        # Intersection: [5, 5, 10, 10], area = 25
        # Union: 100 (box1) + 100 (box2) - 25 = 175
        # IoU = 25 / 175 = 1 / 7
        self.assertAlmostEqual(iou(box1, box2), 1.0 / 7.0)
        
        # No overlap
        box3 = [20, 20, 30, 30]
        self.assertEqual(iou(box1, box3), 0.0)

    def test_iou_distance(self):
        tracks = [[0, 0, 10, 10]]
        detections = [[5, 5, 15, 15], [20, 20, 30, 30]]
        dist_matrix = iou_distance(tracks, detections)
        self.assertEqual(dist_matrix.shape, (1, 2))
        self.assertAlmostEqual(dist_matrix[0, 0], 1.0 - (1.0 / 7.0))
        self.assertEqual(dist_matrix[0, 1], 1.0)

    def test_cosine_distance(self):
        f1 = np.array([[1.0, 0.0]], dtype=np.float32)
        f2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        dist = cosine_distance(f1, f2)
        self.assertEqual(dist.shape, (1, 2))
        self.assertAlmostEqual(dist[0, 0], 0.0)  # exact match
        self.assertAlmostEqual(dist[0, 1], 1.0)  # orthogonal

    def test_linear_assignment(self):
        # 2x2 cost matrix
        cost = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        matches, unmatched_t, unmatched_d = linear_assignment(cost, max_distance=0.5)
        self.assertEqual(matches, [(0, 0), (1, 1)])
        self.assertEqual(unmatched_t, [])
        self.assertEqual(unmatched_d, [])
        
        # Gating threshold constraint
        cost_gated = np.array([[0.6, 0.9], [0.8, 0.7]], dtype=np.float32)
        matches, unmatched_t, unmatched_d = linear_assignment(cost_gated, max_distance=0.5)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_t, [0, 1])
        self.assertEqual(unmatched_d, [0, 1])

if __name__ == "__main__":
    unittest.main()
