from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import read_stub, save_stub, get_center_of_bbox, measure_distance

# The thresholds were tuned on the 1280x720, 30fps sample videos; they're scaled to each video
REFERENCE_FRAME_WIDTH = 1280
REFERENCE_FPS = 30
MAX_BALL_PIXELS_PER_FRAME = 25      # how far the ball can move in one frame, at the reference size and fps
LOST_BALL_SECONDS = 1.0             # after this long without a good sighting the ball could be anywhere...
CONFIRM_BALL_SECONDS = 0.2          # ...so a sighting only counts if the ball is seen again this soon, nearby
MAX_FILLED_GAP_SECONDS = 1.0        # longer gaps stay empty instead of guessing where the ball was


def max_ball_pixels_per_second(frame_width):
    """How far the ball can move in one second, in pixels, in a video this wide."""
    return MAX_BALL_PIXELS_PER_FRAME * REFERENCE_FPS * frame_width / REFERENCE_FRAME_WIDTH


class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path) 

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks=[]

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox =None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                if cls_id == cls_names_inv['Ball']:
                    if max_confidence<confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox":chosen_bbox}

        save_stub(stub_path,tracks)
        
        return tracks

    def remove_wrong_detections(self,ball_positions,frame_width=REFERENCE_FRAME_WIDTH,fps=REFERENCE_FPS):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.

        A detection is kept if the ball could have moved there since the last good
        sighting. When there's no recent good sighting (the start of the video, or the ball
        was lost for a while), a detection is only kept if the next one agrees with it, so
        a single false detection can't become the reference that real ones are rejected against.

        Args:
            ball_positions (list): List of detected ball positions across frames.
            frame_width (int): Width of the video frames in pixels.
            fps (float): Frame rate of the video.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        max_pixels_per_frame = max_ball_pixels_per_second(frame_width) / fps
        lost_frames = LOST_BALL_SECONDS * fps
        confirm_frames = CONFIRM_BALL_SECONDS * fps

        sightings = [(i, get_center_of_bbox(frame[1]['bbox']))
                     for i, frame in enumerate(ball_positions) if frame.get(1, {}).get('bbox')]
        last_good = None

        for n, (i, center) in enumerate(sightings):
            if last_good is not None and i - last_good[0] <= lost_frames:
                last_frame, last_center = last_good
                good = measure_distance(center, last_center) <= max_pixels_per_frame * (i - last_frame)
            else:
                good = False
                if n + 1 < len(sightings):
                    next_frame, next_center = sightings[n + 1]
                    good = (next_frame - i <= confirm_frames and
                            measure_distance(center, next_center) <= max_pixels_per_frame * (next_frame - i))

            if good:
                last_good = (i, center)
            else:
                ball_positions[i] = {}

        return ball_positions

    def interpolate_ball_positions(self,ball_positions,fps=REFERENCE_FPS):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Only short gaps between two sightings are filled. Before the first sighting, after
        the last one and in long gaps (ball out of view, camera cut) the ball stays missing,
        instead of being placed where it wasn't.

        Args:
            ball_positions (list): List of ball positions with potential gaps.
            fps (float): Frame rate of the video.

        Returns:
            list: List of ball positions with short gaps filled; frames without a ball are {}.
        """
        bboxes = [x.get(1,{}).get('bbox') or [np.nan] * 4 for x in ball_positions]
        df_ball_positions = pd.DataFrame(bboxes,columns=['x1','y1','x2','y2'],dtype=float)
        missing = df_ball_positions['x1'].isna().to_numpy()

        # Interpolate missing values, between two sightings only
        df_ball_positions = df_ball_positions.interpolate(limit_area='inside')

        # Empty again the gaps too long to guess
        max_gap = MAX_FILLED_GAP_SECONDS * fps
        gap_start = None
        for i in range(len(missing) + 1):
            if i < len(missing) and missing[i]:
                if gap_start is None:
                    gap_start = i
            elif gap_start is not None:
                if i - gap_start > max_gap:
                    df_ball_positions.iloc[gap_start:i] = np.nan
                gap_start = None

        ball_positions = [{} if np.isnan(x).any() else {1: {"bbox":x}}
                          for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
