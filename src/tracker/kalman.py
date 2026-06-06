import numpy as np

class KalmanFilter:
    """
    A standard 8D Kalman filter for tracking bounding boxes.
    State vector: [cx, cy, w, h, v_cx, v_cy, v_w, v_h]
    Measurement vector: [cx, cy, w, h]
    """
    def __init__(self):
        # State transition matrix F (8x8)
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0
            
        # Measurement matrix H (4x8)
        self._H = np.zeros((4, 8))
        for i in range(4):
            self._H[i, i] = 1.0
            
        # Process and measurement noise parameters
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        """
        Initialize a track's mean and covariance.
        measurement: [cx, cy, w, h] (box center x, center y, width, height)
        """
        mean = np.zeros(8)
        mean[:4] = measurement
        
        # Initial state covariance
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Predict state forward in time.
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        
        mean = np.dot(self._F, mean)
        covariance = np.dot(self._F, np.dot(covariance, self._F.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state to measurement space.
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        measurement_cov = np.diag(np.square(std))
        
        projected_mean = np.dot(self._H, mean)
        projected_cov = np.dot(self._H, np.dot(covariance, self._H.T)) + measurement_cov
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        """
        Perform Kalman filter update with new measurement.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        
        cov_HT = np.dot(covariance, self._H.T)
        
        # Solve S * K.T = (covariance * H.T).T using linear algebra
        # S is projected_cov (4x4), cov_HT is (8, 4)
        K = np.dot(cov_HT, np.linalg.inv(projected_cov))
        
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(K, innovation)
        new_covariance = covariance - np.dot(K, np.dot(projected_cov, K.T))
        return new_mean, new_covariance
