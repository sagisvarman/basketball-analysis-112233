from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append('../')
from utils import read_stub, save_stub

# ByteTrack's second matching pass keeps a track alive on low-confidence boxes (a player
# half hidden behind another), so the detector must hand those over instead of dropping
# everything under 0.5. Only boxes above TRACK_ACTIVATION_THRESHOLD + 0.1 start a new player.
DETECTION_CONFIDENCE = 0.1
TRACK_ACTIVATION_THRESHOLD = 0.4
# How long ByteTrack keeps looking for a player it lost, in frames of 30fps video
LOST_TRACK_BUFFER = 60
# A track needs this many frames in a row before it gets an ID, so one-frame false
# detections don't use up IDs
MINIMUM_CONSECUTIVE_FRAMES = 3


class PlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and ByteTrack.

    This class combines YOLO object detection with ByteTrack tracking to follow players
    from frame to frame while processing detections in batches. ByteTrack only follows
    players by box overlap; PlayerIdentityKeeper turns its track IDs into one ID per player.
    """
    def __init__(self, model_path):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.tracker = self.new_tracker()

    @staticmethod
    def new_tracker(fps=30, minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES):
        """
        Create a ByteTrack tracker tuned for basketball players.

        Args:
            fps (float): Frame rate of the video, so the lost-track buffer is in seconds.
            minimum_consecutive_frames (int): Frames a new track must be seen before it gets an ID.

        Returns:
            sv.ByteTrack: A fresh tracker.
        """
        return sv.ByteTrack(
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=LOST_TRACK_BUFFER,
            frame_rate=fps,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    @staticmethod
    def get_player_detections(result):
        """
        Convert a YOLO result to supervision Detections, keeping only players.

        Args:
            result: One ultralytics result.

        Returns:
            sv.Detections: Player detections only.
        """
        player_class_id = {name: class_id for class_id, name in result.names.items()}['Player']
        detections = sv.Detections.from_ultralytics(result)
        return detections[detections.class_id == player_class_id]

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size=20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=DETECTION_CONFIDENCE)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, fps=30):
        """
        Get player tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.
            fps (float): Frame rate of the video.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps ByteTrack IDs to their bounding box coordinates.
        """
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        self.tracker = self.new_tracker(fps)

        tracks=[]

        for detection in detections:
            # Only players go into the tracker: the ball, hoop and referees would otherwise
            # compete with players for the same tracks
            player_detections = self.get_player_detections(detection)
            detection_with_tracks = self.tracker.update_with_detections(player_detections)

            tracks.append({
                int(track_id): {"bbox": bbox.tolist()}
                for bbox, track_id in zip(detection_with_tracks.xyxy, detection_with_tracks.tracker_id)
            })

        save_stub(stub_path,tracks)
        return tracks
