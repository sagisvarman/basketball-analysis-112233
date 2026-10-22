"""
A module for reading live (or recorded) video streams frame by frame.

YouTube links are resolved to a direct stream URL with yt-dlp and decoded with OpenCV
on a background thread. Only the newest frame is kept, so a slow analyzer always works
on what is happening now instead of falling further and further behind the stream.
"""

import os
import threading
import time

import cv2
import yt_dlp

# The models and pixel thresholds were tuned on 720p broadcast footage; at lower
# resolutions most players go undetected. H.264 decodes everywhere OpenCV does.
STREAM_FORMAT = (
    'best[height<=720][vcodec^=avc1]/bestvideo[height<=720][vcodec^=avc1]/'
    'best[height<=720]/best'
)
MAX_RECONNECTS = 6          # Waits 2, 4, 8, 16, 30, 30s: rides out about 90s of network trouble


def resolve_stream(source):
    """
    Turn a YouTube link (or any link yt-dlp understands) into a URL OpenCV can open.

    Args:
        source (str): YouTube link, other video link, or local video file path.

    Returns:
        tuple: (stream_url, title, is_live)
    """
    if os.path.exists(source):
        return source, os.path.basename(source), False

    options = {'format': STREAM_FORMAT, 'quiet': True, 'no_warnings': True, 'noplaylist': True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(source, download=False)
    if 'entries' in info:
        # Search results and playlists: use the first video
        info = next(iter(info['entries']))
    return info['url'], info.get('title') or source, bool(info.get('is_live'))


class StreamReader:
    """
    Reads a video stream on a background thread, keeping only the most recent frame.

    Frames are released at real-time speed for every kind of source: recorded videos
    would otherwise be read as fast as they decode, and live HLS streams arrive in
    segment-sized bursts.
    """
    def __init__(self, source):
        self.source = source
        self.title = source
        self.is_live = False
        self.fps = 30.0
        self.finished = False
        self.error = None
        self.current_time = 0.0

        self._cap = None
        self._latest = None
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = None

    def open(self):
        """Resolve and open the stream, then start reading. Raises on failure."""
        self._cap = self._open_capture()
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 120 else 30.0
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _open_capture(self):
        url, self.title, self.is_live = resolve_stream(self.source)
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream for {self.source}")
        return cap

    def _read_loop(self):
        frame_idx = 0
        reconnects = 0
        start_wall = None

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if not self.is_live or reconnects >= MAX_RECONNECTS:
                    break
                # Live streams drop now and then: fetch a fresh stream URL and carry on
                self._cap.release()
                if self._stop_event.wait(min(2 ** (reconnects + 1), 30)):
                    break
                reconnects += 1
                try:
                    self._cap = self._open_capture()
                except Exception as e:
                    self.error = f"Lost connection to the stream: {e}"
                    continue
                self.error = None
                if not self.is_live:
                    # The broadcast is over; don't replay the recording from the start
                    break
                start_wall = None
                continue
            reconnects = 0

            video_time = frame_idx / self.fps
            if start_wall is None:
                start_wall = time.monotonic() - video_time
            delay = start_wall + video_time - time.monotonic()
            if delay > 0:
                self._stop_event.wait(delay)

            with self._condition:
                self._latest = (frame_idx, video_time, frame)
                self.current_time = video_time
                self._condition.notify_all()
            frame_idx += 1

        self._cap.release()
        with self._condition:
            self.finished = True
            self._condition.notify_all()

    def next_frame(self, after_idx, timeout=1.0):
        """
        Get the newest frame read after `after_idx`.

        Args:
            after_idx (int): Index of the last frame the caller has seen.
            timeout (float): Seconds to wait for a newer frame.

        Returns:
            tuple: (frame_idx, video_time_seconds, frame), or None if nothing newer arrived.
        """
        def has_new_frame():
            return self._latest is not None and self._latest[0] > after_idx

        with self._condition:
            self._condition.wait_for(lambda: self.finished or has_new_frame(), timeout)
            return self._latest if has_new_frame() else None

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
