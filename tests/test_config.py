import unittest
import os
import yaml
from src.config import Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_yaml = "test_config.yaml"
        self.config_data = {
            "input": {"source": 0, "width": 1280, "height": 720},
            "detector": {"backend": "yolov8", "model_size": "small", "confidence": 0.45, "nms_iou": 0.5, "classes": []},
            "tracker": {"backend": "deepsort", "max_age": 30, "min_hits": 3, "iou_threshold": 0.3, "reid_model": "osnet_x0_25"},
            "visualization": {"trail_length": 20, "draw_velocity": True, "headless": False},
            "output": {"save_video": False, "save_csv": True, "output_dir": "./runs/"},
            "api": {"enabled": True, "port": 8080}
        }
        with open(self.test_yaml, "w") as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        if os.path.exists(self.test_yaml):
            os.remove(self.test_yaml)

    def test_load_config(self):
        cfg = Config.load(self.test_yaml)
        self.assertEqual(cfg.input.source, 0)
        self.assertEqual(cfg.detector.backend, "yolov8")
        self.assertEqual(cfg.tracker.max_age, 30)

    def test_update_config(self):
        cfg = Config.load(self.test_yaml)
        cfg.update({"detector": {"confidence": 0.6}, "visualization": {"headless": True}})
        self.assertEqual(cfg.detector.confidence, 0.6)
        self.assertEqual(cfg.visualization.headless, True)

if __name__ == "__main__":
    unittest.main()
