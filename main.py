import argparse
import sys
import os
import time
import threading
import logging
import uvicorn

# Add current workspace directory to path to enable package relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.pipeline import StateStore, TrackingPipeline
from src.api import create_app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")

class UvicornServer(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__()
        self.app = app
        self.host = host
        self.port = port
        self.daemon = True
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def main():
    parser = argparse.ArgumentParser(description="TrackVision — Object Detection & Tracking System")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration YAML")
    parser.add_argument("--source", type=str, default=None, help="Inference video source override (path or webcam index)")
    parser.add_argument("--save-video", action="store_true", default=None, help="Save annotated output video flag override")
    parser.add_argument("--headless", action="store_true", default=None, help="Run without display GUI flag override")
    args = parser.parse_args()

    # Load configuration schema
    config_path = args.config
    if not os.path.exists(config_path):
        # Fallback to local default path if running from subdir
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "default.yaml")
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found in: {args.config} or fallback: {config_path}")
            sys.exit(1)

    logger.info(f"Loading config from {config_path}")
    config = Config.load(config_path)

    # Initialize thread-safe StateStore
    store = StateStore(config)

    # Determine headless mode
    is_headless = args.headless if args.headless is not None else config.visualization.headless

    # Create Tracking Pipeline
    pipeline = TrackingPipeline(
        store,
        source_override=args.source,
        save_video_override=args.save_video,
        headless_override=is_headless
    )

    server = None
    if config.api.enabled:
        app = create_app(store)
        port = config.api.port
        host = "0.0.0.0"
        logger.info(f"Starting TrackVision REST API server on {host}:{port}")
        server = UvicornServer(app, host, port)
        server.start()

    try:
        if not is_headless:
            # OpenCV GUI requires running on the main thread for Windows/macOS.
            # Run the pipeline synchronously on the main thread.
            logger.info("Running tracking pipeline on the main thread (GUI viewer active).")
            pipeline.run()
        else:
            # Run the pipeline in a background thread and keep main thread sleeping
            logger.info("Running tracking pipeline in a background thread (headless mode).")
            pipeline.start()
            while store.active:
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping TrackVision...")
        store.active = False
    finally:
        # Gracefully stop API server
        if server:
            logger.info("Stopping REST API server...")
            server.stop()
            server.join(timeout=2.0)
            
        # Clean up pipeline resources
        if pipeline.is_alive():
            pipeline.join(timeout=2.0)
            
        logger.info("TrackVision execution terminated.")


if __name__ == "__main__":
    main()
