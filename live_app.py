"""
Web UI for analyzing a live YouTube basketball stream in real time.

    python live_app.py
    # then open http://127.0.0.1:8000, paste a YouTube link and press Start

Stats for each run are recorded under output_live/<date-time>/.
"""

import argparse
import os
import threading
import time

from flask import Flask, Response, abort, jsonify, request, send_file

from live import LiveAnalyzer, LiveSession
from configs import LIVE_OUTPUT_PATH

ROOT = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = {"events.csv", "timeline.csv", "summary.json"}
DEFAULT_TEAM_NAMES = {1: "white shirt", 2: "dark blue shirt"}

app = Flask(__name__)

analyzer = None
analyzer_error = None
analyzer_ready = threading.Event()

session = None
session_lock = threading.Lock()


def load_analyzer(device, fast):
    """Load the models once in the background so the first Start is quicker."""
    global analyzer, analyzer_error
    try:
        analyzer = LiveAnalyzer(device, fast=fast)
    except Exception as e:
        analyzer_error = e
    finally:
        analyzer_ready.set()


def get_analyzer():
    analyzer_ready.wait()
    if analyzer is None:
        raise RuntimeError(f"Could not load models: {analyzer_error}")
    return analyzer


def models_status():
    if not analyzer_ready.is_set():
        return "loading"
    return "ready" if analyzer is not None else f"error: {analyzer_error}"


@app.get("/")
def index():
    return send_file(os.path.join(ROOT, "live", "web", "index.html"))


@app.post("/api/start")
def start():
    global session
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify(error="Paste a YouTube link first."), 400
    team_names = {team: (body.get(f"team_{team}") or default).strip()
                  for team, default in DEFAULT_TEAM_NAMES.items()}

    with session_lock:
        # The models are shared, so the previous run must be fully stopped first
        if session is not None:
            session.stop()
        session = LiveSession(url, team_names, get_analyzer, LIVE_OUTPUT_PATH)
        session.start()
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    if session is not None:
        session.stop(wait=False)
    return jsonify(ok=True)


@app.get("/api/state")
def state():
    current = session
    if current is None:
        state = {"status": "idle", "team_names": DEFAULT_TEAM_NAMES}
    else:
        state = current.get_state()
    state["models"] = models_status()
    state["device"] = state.get("device") or (analyzer.device if analyzer is not None else None)
    return jsonify(state)


@app.get("/video_feed")
def video_feed():
    """MJPEG stream of annotated frames, one per analyzed frame."""
    def frames():
        current, version = None, 0
        while True:
            if session is not current:
                current, version = session, 0
            if current is None:
                time.sleep(1)
                continue
            # Times out every few seconds and resends the last frame, which keeps the
            # connection alive and notices when the browser has gone away
            jpeg, version = current.wait_for_frame(version)
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/download/<name>")
def download(name):
    current = session
    if current is None or name not in DOWNLOADS:
        abort(404)
    path = os.path.join(current.output_dir, name)
    if not os.path.exists(path):
        abort(404)
    return send_file(os.path.abspath(path), as_attachment=True,
                     download_name=f"{os.path.basename(current.output_dir)}-{name}")


def main():
    parser = argparse.ArgumentParser(description="Live basketball analysis web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to open it from other devices")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="cuda:0, mps or cpu (default: best available)")
    parser.add_argument("--fast", action="store_true",
                        help="About 2x faster without a GPU: skips the court map, distance and speed")
    args = parser.parse_args()

    # Model and image paths in configs are relative to the project folder
    os.chdir(ROOT)
    threading.Thread(target=load_analyzer, args=(args.device, args.fast), daemon=True).start()

    mode = " (fast mode)" if args.fast else ""
    print(f"\n  Basketball live analysis{mode} running at http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
