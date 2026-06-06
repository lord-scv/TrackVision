from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import queue
import cv2
import logging

logger = logging.getLogger("API")

def create_app(state_store):
    app = FastAPI(
        title="TrackVision API",
        description="Real-time Object Detection & Tracking REST API",
        version="1.0"
    )
    
    # Store the state store reference
    app.state.store = state_store

    def gen_frames():
        while state_store.active:
            try:
                # Retrieve annotated frames from queue
                frame = state_store.stream_queue.get(timeout=0.5)
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except queue.Empty:
                continue

    @app.get("/stream")
    async def get_stream():
        """Serves the live annotated MJPEG stream for browser display."""
        if not state_store.active:
            raise HTTPException(status_code=503, detail="Tracking pipeline is not active.")
        return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/detections/latest")
    async def get_latest_detections():
        """Returns JSON list of tracks in the most recent frame."""
        return state_store.get_latest_data()

    @app.get("/detections/history")
    async def get_track_history(track_id: int):
        """Returns the full position history of a specific track ID."""
        history = state_store.get_track_history(track_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"No history found for track_id: {track_id}")
        return {
            "track_id": track_id,
            "history": history
        }

    @app.post("/config")
    async def hot_reload_config(config_update: dict):
        """Hot-reloads the detector/tracker configurations dynamically."""
        logger.info(f"Received configuration update: {config_update}")
        state_store.set_config(config_update)
        return {
            "status": "success",
            "message": "Configuration hot-reload triggered.",
            "updated_config": state_store.get_config()
        }

    @app.get("/stats")
    async def get_session_stats():
        """Returns aggregate session stats (unique counts, FPS rates, uptime)."""
        return state_store.get_stats()

    return app
