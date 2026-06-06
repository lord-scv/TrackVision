import unittest
import numpy as np
from src.tracker.kalman import KalmanFilter

class TestKalmanFilter(unittest.TestCase):
    def test_initiate(self):
        kf = KalmanFilter()
        measurement = np.array([100, 100, 50, 50], dtype=np.float32)
        mean, covariance = kf.initiate(measurement)
        
        self.assertEqual(mean.shape, (8,))
        self.assertEqual(covariance.shape, (8, 8))
        np.testing.assert_array_equal(mean[:4], measurement)
        np.testing.assert_array_equal(mean[4:], np.zeros(4))

    def test_predict(self):
        kf = KalmanFilter()
        measurement = np.array([100, 100, 50, 50], dtype=np.float32)
        mean, covariance = kf.initiate(measurement)
        
        # Inject velocity for prediction testing
        mean[4:] = np.array([5.0, -2.0, 1.0, 0.0])  # v_cx, v_cy, v_w, v_h
        
        pred_mean, pred_covariance = kf.predict(mean, covariance)
        
        # cx = 100 + 5 = 105, cy = 100 - 2 = 98, w = 50 + 1 = 51, h = 50 + 0 = 50
        np.testing.assert_array_almost_equal(pred_mean[:4], [105, 98, 51, 50])
        self.assertTrue(np.all(np.diag(pred_covariance) >= np.diag(covariance)))

    def test_update(self):
        kf = KalmanFilter()
        measurement = np.array([100, 100, 50, 50], dtype=np.float32)
        mean, covariance = kf.initiate(measurement)
        
        new_measurement = np.array([102, 98, 48, 52], dtype=np.float32)
        updated_mean, updated_covariance = kf.update(mean, covariance, new_measurement)
        
        self.assertEqual(updated_mean.shape, (8,))
        self.assertEqual(updated_covariance.shape, (8, 8))
        self.assertTrue(99 < updated_mean[0] < 103)
        self.assertTrue(97 < updated_mean[1] < 101)

if __name__ == "__main__":
    unittest.main()
