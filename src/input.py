import cv2
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VideoSource")

class VideoSource:
    def __init__(self, source, width=None, height=None):
        """
        source: int (webcam index) or str (file path)
        width: target width (optional, max 1280)
        height: target height (optional, max 720)
        """
        # If source is digit string, convert to int
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source
            
        self.target_width = width
        self.target_height = height
        
        self.cap = None
        self.fps = 0.0
        self.total_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.start_time = None
        
        self._init_capture()

    def _init_capture(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.source}")
            
        # Get original source specs
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0  # default fallback
            
        # Try setting dimensions directly if it is a camera source
        if isinstance(self.source, int):
            if self.target_width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, min(self.target_width, 1280))
            if self.target_height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, min(self.target_height, 720))
                
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Initialized video source: {self.source} | Native Resolution: {w}x{h} | FPS: {self.fps}")

    def get_fps(self):
        return self.fps

    def frame_generator(self):
        """Yields BGR frames, performing resize if target width/height are specified (max processed: 1280x720)."""
        self.start_time = time.time()
        self.processed_frames = 0
        self.dropped_frames = 0
        
        expected_interval = 1.0 / self.fps
        last_frame_time = time.time()
        
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.total_frames += 1
            t1 = time.time()
            elapsed = t1 - last_frame_time
            
            # For live webcam, detect frame drops when reading took longer than expected.
            if isinstance(self.source, int) and elapsed > expected_interval * 2.0:
                dropped = int(elapsed / expected_interval) - 1
                if dropped > 0:
                    self.dropped_frames += dropped
                    drop_rate = (self.dropped_frames / (self.total_frames + self.dropped_frames)) * 100
                    logger.warning(
                        f"Frame lag detected: elapsed={elapsed:.3f}s. "
                        f"Estimated dropped frames: {dropped}. "
                        f"Total dropped: {self.dropped_frames} (rate: {drop_rate:.1f}%)"
                    )
            
            last_frame_time = t1
            
            # Apply resizing if requested (FR-03: max processed resolution 1280x720)
            h, w = frame.shape[:2]
            tw = self.target_width if self.target_width else w
            th = self.target_height if self.target_height else h
            tw = min(tw, 1280)
            th = min(th, 720)
            
            if w != tw or h != th:
                frame = cv2.resize(frame, (tw, th))
                    
            self.processed_frames += 1
            yield frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Video capture source released.")
