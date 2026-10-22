"""
A module that runs one live analysis: stream in, stats and annotated frames out.

Stats are recorded to disk while the session runs:
    events.csv    every pass and interception
    timeline.csv  running totals after every analyzed frame
    summary.json  latest totals, rewritten every few seconds
"""

import csv
import json
import os
import threading
import time
import traceback
from datetime import datetime

import cv2

from .stream_reader import StreamReader

SUMMARY_EVERY_SECONDS = 5
MAX_EVENTS_IN_MEMORY = 50

EVENT_FIELDS = ["wall_clock", "video_time", "type", "team", "team_name", "from_player", "to_player"]
TIMELINE_FIELDS = [
    "wall_clock", "video_time", "frame", "players", "ball_holder", "ball_holder_team",
    "team1_control_pct", "team2_control_pct", "team1_passes", "team2_passes",
    "team1_interceptions", "team2_interceptions",
]


class LiveSession:
    """
    Analyzes one stream on a background thread.

    Args:
        source (str): YouTube link, other video link, or local file path.
        team_names (dict): {1: "white shirt", 2: "dark blue shirt"} jersey descriptions.
        get_analyzer (callable): Returns the shared, already-loaded LiveAnalyzer (may block).
        output_root (str): Folder where this session's recording folder is created.
    """
    def __init__(self, source, team_names, get_analyzer, output_root):
        self.source = source
        self.team_names = team_names
        self.get_analyzer = get_analyzer
        self.output_dir = os.path.join(output_root, datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.status = "starting"
        self.message = "Starting…"
        self.title = source
        self.is_live = False
        self.device = None
        self.stream_fps = None
        self.analysis_fps = 0.0
        self.seconds_per_frame = None
        self.lag_seconds = 0.0
        self.stats = None
        self.events = []

        self.jpeg = None
        self.frame_version = 0
        self._frame_condition = threading.Condition()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self, wait=True):
        """Ask the session to stop; the frame being analyzed is finished first."""
        if self._thread.is_alive() and not self._stop_event.is_set():
            self._set_status("stopping", "Stopping after the current frame…")
        self._stop_event.set()
        if self._reader is not None:
            self._reader.stop()
        if wait and self._thread.ident is not None:
            self._thread.join()

    def is_active(self):
        return self._thread.is_alive()

    def _set_status(self, status, message):
        with self._lock:
            self.status = status
            self.message = message

    def _run(self):
        try:
            self._set_status("loading", "Loading models (first run takes a minute)…")
            analyzer = self.get_analyzer()
            self.device = analyzer.device
            if self._stop_event.is_set():
                self._set_status("stopped", "Stopped")
                return

            self._set_status("connecting", "Connecting to stream…")
            self._reader = StreamReader(self.source)
            self._reader.open()
            self.title = self._reader.title
            self.is_live = self._reader.is_live
            self.stream_fps = round(self._reader.fps, 2)

            analyzer.reset(self.team_names[1], self.team_names[2])
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "events.csv"), "w", newline="") as events_file, \
                 open(os.path.join(self.output_dir, "timeline.csv"), "w", newline="") as timeline_file:
                events_writer = csv.DictWriter(events_file, fieldnames=EVENT_FIELDS)
                timeline_writer = csv.DictWriter(timeline_file, fieldnames=TIMELINE_FIELDS)
                events_writer.writeheader()
                timeline_writer.writeheader()
                self._set_status("running", "Analyzing")
                self._analyze(analyzer, events_writer, events_file, timeline_writer, timeline_file)

            if self._stop_event.is_set():
                self._set_status("stopped", "Stopped")
            elif self._reader.error:
                self._set_status("error", self._reader.error)
            else:
                self._set_status("finished", "Stream ended")
        except Exception as e:
            traceback.print_exc()
            self._set_status("error", str(e))
        finally:
            if self._reader is not None:
                self._reader.stop()
            if self.stats is not None:
                self._write_summary()

    def _analyze(self, analyzer, events_writer, events_file, timeline_writer, timeline_file):
        last_idx = -1
        recent_durations = []
        last_summary = time.monotonic()

        while not self._stop_event.is_set():
            item = self._reader.next_frame(last_idx)
            if item is None:
                if self._reader.finished:
                    break
                continue
            last_idx, video_time, frame = item

            started = time.monotonic()
            annotated, events = analyzer.process(frame, video_time)
            recent_durations = (recent_durations + [time.monotonic() - started])[-10:]
            stats = analyzer.get_stats()
            jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

            wall_clock = datetime.now().isoformat(timespec="seconds")
            for event in events:
                event = {**event, "wall_clock": wall_clock, "team_name": self.team_names[event["team"]]}
                events_writer.writerow(event)
            events_file.flush()
            timeline_writer.writerow({
                "wall_clock": wall_clock,
                "video_time": round(video_time, 2),
                "frame": last_idx,
                "players": stats["players_in_frame"],
                "ball_holder": stats["holder"]["player"],
                "ball_holder_team": stats["holder"]["team"],
                "team1_control_pct": stats["ball_control_pct"][1],
                "team2_control_pct": stats["ball_control_pct"][2],
                "team1_passes": stats["passes"][1],
                "team2_passes": stats["passes"][2],
                "team1_interceptions": stats["interceptions"][1],
                "team2_interceptions": stats["interceptions"][2],
            })
            timeline_file.flush()

            seconds_per_frame = sum(recent_durations) / len(recent_durations)
            with self._lock:
                self.stats = stats
                self.events = (self.events + [{**e, "wall_clock": wall_clock} for e in events])[-MAX_EVENTS_IN_MEMORY:]
                self.seconds_per_frame = seconds_per_frame
                self.analysis_fps = 1 / seconds_per_frame if seconds_per_frame > 0 else 0
                self.lag_seconds = max(0.0, self._reader.current_time - video_time)
            with self._frame_condition:
                self.jpeg = jpeg
                self.frame_version += 1
                self._frame_condition.notify_all()

            if time.monotonic() - last_summary >= SUMMARY_EVERY_SECONDS:
                self._write_summary()
                last_summary = time.monotonic()

    def wait_for_frame(self, after_version, timeout=5.0):
        """Block until a frame newer than `after_version` exists. Returns (jpeg, version)."""
        with self._frame_condition:
            self._frame_condition.wait_for(lambda: self.frame_version > after_version, timeout)
            return self.jpeg, self.frame_version

    def get_state(self):
        with self._lock:
            return {
                "status": self.status,
                "message": self.message,
                "source": self.source,
                "title": self.title,
                "is_live": self.is_live,
                "team_names": self.team_names,
                "device": self.device,
                "stream_fps": self.stream_fps,
                "analysis_fps": round(self.analysis_fps, 3),
                "seconds_per_frame": round(self.seconds_per_frame, 2) if self.seconds_per_frame else None,
                "lag_seconds": round(self.lag_seconds, 1),
                "stats": self.stats,
                "events": list(reversed(self.events)),
                "output_dir": self.output_dir,
            }

    def _write_summary(self):
        state = self.get_state()
        state["saved_at"] = datetime.now().isoformat(timespec="seconds")
        path = os.path.join(self.output_dir, "summary.json")
        with open(path + ".tmp", "w") as f:
            json.dump(state, f, indent=2)
        os.replace(path + ".tmp", path)
