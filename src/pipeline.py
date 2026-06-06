import threading
import time
import queue
import logging
import os
import cv2
import numpy as np

from src.input import VideoSource
from src.detector import get_detector
from src.tracker import get_tracker
from src.tracker.state import TrackState
from src.visualization import Visualizer
from src.logger import CSVLogger, SessionSummary, VideoSaveHandler

logger = logging.getLogger("Pipeline")

class StateStore:
    def __init__(self, config):
        self.lock = threading.Lock()
        self.config = config
        self.active = True
        self.config_updated = False
        
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_tracks = []
        
        # Position history: track_id -> list of position updates
        self.track_history = {}
        # Track ID to class name mapping
        self.track_id_classes = {}
        
        # Running session stats
        self.total_unique_ids = set()
        self.fps_log = []
        self.start_time = time.time()
        self.frame_count = 0
        
        # Queue for serving MJPEG frames
        self.stream_queue = queue.Queue(maxsize=1)

    def set_config(self, new_config_dict):
        with self.lock:
            self.config.update(new_config_dict)
            self.config_updated = True

    def get_config(self):
        with self.lock:
            return self.config.to_dict()

    def update_frame(self, raw_frame, annotated_frame, tracks, fps):
        with self.lock:
            self.latest_frame = raw_frame
            self.latest_annotated_frame = annotated_frame
            self.latest_tracks = [t.to_dict() for t in tracks]
            self.fps_log.append(fps)
            self.frame_count += 1
            
            curr_time = time.time()
            for t in tracks:
                if t.state == TrackState.CONFIRMED:
                    tid = t.id
                    self.total_unique_ids.add(tid)
                    self.track_id_classes[tid] = t.class_name
                    
                    if tid not in self.track_history:
                        self.track_history[tid] = []
                    self.track_history[tid].append({
                        "bbox": list(t.bbox),
                        "timestamp": curr_time,
                        "frame_num": self.frame_count
                    })
                    
        # Update MJPEG stream queue, dropping stale frames if queue is full
        if self.stream_queue.full():
            try:
                self.stream_queue.get_nowait()
            except queue.Empty:
                pass
        self.stream_queue.put(annotated_frame)

    def get_latest_data(self):
        with self.lock:
            return {
                "frame_count": self.frame_count,
                "tracks": self.latest_tracks
            }

    def get_track_history(self, track_id):
        with self.lock:
            return list(self.track_history.get(track_id, []))

    def get_stats(self):
        with self.lock:
            avg_fps = sum(self.fps_log) / len(self.fps_log) if self.fps_log else 0.0
            peak_fps = max(self.fps_log) if self.fps_log else 0.0
            
            class_counts = {}
            for tid in self.total_unique_ids:
                cls = self.track_id_classes.get(tid, "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1
                
            return {
                "total_unique_ids_seen": len(self.total_unique_ids),
                "class_breakdown": class_counts,
                "average_fps": float(f"{avg_fps:.2f}"),
                "peak_fps": float(f"{peak_fps:.2f}"),
                "uptime_seconds": float(f"{time.time() - self.start_time:.2f}"),
                "frames_processed": self.frame_count
            }


class TrackingPipeline(threading.Thread):
    def __init__(self, state_store, source_override=None, save_video_override=None, headless_override=None):
        super().__init__()
        self.store = state_store
        self.daemon = True
        
        self.source_override = source_override
        self.save_video_override = save_video_override
        self.headless_override = headless_override
        
        cfg = self.store.config
        
        # Initialize video source
        src_w = getattr(cfg.input, 'width', None)
        src_h = getattr(cfg.input, 'height', None)
        src = self.source_override if self.source_override is not None else cfg.input.source
        
        self.video_source = VideoSource(src, src_w, src_h)
        
        # Instantiate detector & tracker
        self.detector = get_detector(
            cfg.detector.backend,
            cfg.detector.model_size,
            cfg.detector.confidence,
            cfg.detector.nms_iou,
            cfg.detector.classes
        )
        
        self.tracker = get_tracker(
            cfg.tracker.backend,
            cfg.tracker.max_age,
            cfg.tracker.min_hits,
            cfg.tracker.iou_threshold,
            cfg.tracker.reid_model
        )
        
        # visualizer
        self.visualizer = Visualizer(cfg.visualization.trail_length, cfg.visualization.draw_velocity)
        
        # Outdir setup
        out_dir = cfg.output.output_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # CSV log handler
        self.csv_logger = None
        if cfg.output.save_csv:
            self.csv_logger = CSVLogger(os.path.join(out_dir, "detections.csv"))
            
        # Summary & video exporter
        self.session_summary = SessionSummary()
        self.summary_path = os.path.join(out_dir, "summary.json")
        
        self.video_writer = None
        save_vid = self.save_video_override if self.save_video_override is not None else cfg.output.save_video
        if save_vid:
            self.video_writer = VideoSaveHandler(
                os.path.join(out_dir, "output.mp4"),
                self.video_source.get_fps()
            )

    def run(self):
        logger.info("Tracking pipeline thread launched.")
        frame_num = 0
        
        try:
            for frame in self.video_source.frame_generator():
                if not self.store.active:
                    break
                    
                frame_num += 1
                t0 = time.time()
                
                # Check for config reload
                if self.store.config_updated:
                    self._handle_config_reload()
                    
                # 1. Detect
                detections = self.detector.detect(frame)
                
                # 2. Track
                tracks = self.tracker.update(detections, frame)
                
                # Performance metrics
                latency = time.time() - t0
                fps = 1.0 / latency if latency > 0 else 0.0
                
                # 3. Log results to CSV
                curr_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                if self.csv_logger:
                    for track in tracks:
                        if track.state == TrackState.CONFIRMED:
                            self.csv_logger.log(frame_num, curr_timestamp, track)
                            
                # Record to summary
                for track in tracks:
                    if track.state == TrackState.CONFIRMED:
                        self.session_summary.record_track(track.id)
                self.session_summary.record_fps(fps)
                
                # 4. Visualize
                annotated_frame = self.visualizer.draw(frame, tracks, fps, frame_num)
                
                # Save frame to video
                if self.video_writer:
                    self.video_writer.write_frame(annotated_frame)
                    
                # 5. Push to store
                self.store.update_frame(frame, annotated_frame, tracks, fps)
                
                # Draw local GUI window if not headless
                cfg = self.store.config
                headless = self.headless_override if self.headless_override is not None else cfg.visualization.headless
                if not headless:
                    cv2.imshow("TrackVision Local Viewer", annotated_frame)
                    if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
                        logger.info("Exit key pressed in display window.")
                        self.store.active = False
                        break
                        
        except Exception as e:
            logger.error(f"Exception in tracking pipeline loop: {e}", exc_info=True)
        finally:
            self._cleanup()

    def _handle_config_reload(self):
        logger.info("Hot-reloading config settings in pipeline...")
        cfg = self.store.config
        
        # Dynamic detector re-instantiation
        self.detector = get_detector(
            cfg.detector.backend,
            cfg.detector.model_size,
            cfg.detector.confidence,
            cfg.detector.nms_iou,
            cfg.detector.classes
        )
        
        # Save tracks context and re-instantiate tracker
        old_tracks = self.tracker.tracks if hasattr(self.tracker, 'tracks') else []
        self.tracker = get_tracker(
            cfg.tracker.backend,
            cfg.tracker.max_age,
            cfg.tracker.min_hits,
            cfg.tracker.iou_threshold,
            cfg.tracker.reid_model
        )
        self.tracker.tracks = old_tracks
        
        # Visualizer config update
        self.visualizer.trail_length = cfg.visualization.trail_length
        self.visualizer.draw_velocity = cfg.visualization.draw_velocity
        
        self.store.config_updated = False
        logger.info("Hot-reload completed.")

    def _cleanup(self):
        logger.info("Cleaning up tracking pipeline...")
        self.video_source.release()
        if self.video_writer:
            self.video_writer.release()
            
        try:
            self.session_summary.write_summary(self.summary_path, self.store.track_id_classes)
            logger.info(f"Summary JSON successfully written to {self.summary_path}")
        except Exception as e:
            logger.error(f"Could not export summary JSON: {e}")
            
        cv2.destroyAllWindows()
        self.store.active = False
        logger.info("Pipeline thread terminated.")
