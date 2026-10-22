"""
A frame-by-frame version of the main.py pipeline for live streams.

main.py sees the whole video up front (stubs, ball interpolation, look-ups across all
frames). Here every step only uses the current frame plus a little remembered state.
Thresholds are in seconds of video instead of frames, because frames are skipped
whenever the analyzer is slower than the stream.
"""

from collections import deque

import cv2
import supervision as sv
import torch
from PIL import Image

from trackers import PlayerTracker, BallTracker
from court_keypoint_detector import CourtKeypointDetector
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import PlayerTracksDrawer, BallTracksDrawer, TacticalViewDrawer
from utils import get_center_of_bbox, measure_distance
from configs import PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH, COURT_KEYPOINT_DETECTOR_PATH

# The batch pipeline's frame-based thresholds were tuned on 30fps video
BALL_MAX_PIXELS_PER_SECOND = 25 * 30     # BallTracker.remove_wrong_detections
MIN_POSSESSION_SECONDS = 11 / 30         # BallAquisitionDetector.min_frames
TEAM_REFRESH_SECONDS = 50 / 30           # TeamAssigner clears its cache every 50 frames
SPEED_WINDOW_SECONDS = 15 / 30           # SpeedAndDistanceCalculator.calculate_speed window
MAX_PLAYER_SPEED_KMH = 40                # Faster than this is a tracking glitch, not a sprint
MAX_TRACKING_GAP_SECONDS = 0.2           # ByteTrack matches by box overlap, which is gone after ~0.2s

# BGR; the web UI uses the same colors for each team
TEAM_1_COLOR = [11, 158, 245]
TEAM_2_COLOR = [248, 189, 56]


def pick_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class LiveAnalyzer:
    """
    Runs detection, tracking, team assignment and stat keeping on one frame at a time.

    Models are loaded once; call reset() before each new stream.

    Fast mode is for computers without a GPU. It skips the ball model (the player model
    also detects the ball) and the court keypoint model, which leaves out the court map,
    distance and speed, and roughly halves the time per frame.
    """
    def __init__(self, device=None, fast=False):
        self.device = device or pick_device()
        self.fast = fast

        self.player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
        if not fast:
            self.ball_tracker = BallTracker(BALL_DETECTOR_PATH)
            self.court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

        self.team_assigner = TeamAssigner()
        self.team_assigner.load_model()
        self.team_assigner.model.to(self.device).eval()

        self.ball_aquisition_detector = BallAquisitionDetector()
        self.tactical_view_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
        self.speed_and_distance_calculator = SpeedAndDistanceCalculator(
            self.tactical_view_converter.width,
            self.tactical_view_converter.height,
            self.tactical_view_converter.actual_width_in_meters,
            self.tactical_view_converter.actual_height_in_meters
        )

        self.player_tracks_drawer = PlayerTracksDrawer(team_1_color=TEAM_1_COLOR, team_2_color=TEAM_2_COLOR)
        self.ball_tracks_drawer = BallTracksDrawer()
        self.tactical_view_drawer = TacticalViewDrawer(team_1_color=TEAM_1_COLOR, team_2_color=TEAM_2_COLOR)

        self.reset()

    def reset(self, team_1_class_name="white shirt", team_2_class_name="dark blue shirt"):
        """Forget all tracking state and stats, and set the jersey descriptions for each team."""
        self.player_tracker.tracker = sv.ByteTrack()
        self.id_offset = 0
        self.max_player_id = 0
        self.team_assigner.team_1_class_name = team_1_class_name
        self.team_assigner.team_2_class_name = team_2_class_name
        self.team_assigner.player_team_dict = {}
        self.last_team_refresh = None

        self.last_ball = None
        self.streak_player = -1
        self.streak_start = None
        self.last_holder = -1
        self.last_holder_team = -1
        self.holder = -1
        self.holder_team = -1

        self.player_positions = {}
        self.player_steps = {}
        self.player_distance = {}
        self.player_speed = {}
        self.player_team = {}
        self.player_last_seen = {}

        self.frames = 0
        self.players_in_frame = 0
        self.control_frames = {1: 0, 2: 0}
        self.passes = {1: 0, 2: 0}
        self.interceptions = {1: 0, 2: 0}
        self.video_time = 0.0

    def process(self, frame, video_time):
        """
        Analyze one frame and update the running stats.

        Args:
            frame (numpy.ndarray): The video frame.
            video_time (float): Seconds since the stream started.

        Returns:
            tuple: (annotated_frame, events) where events is a list of pass/interception
                dicts that happened on this frame.
        """
        # When frames are too far apart (slow hardware), players can't be followed from one
        # analyzed frame to the next: each frame is treated as an independent sample instead
        continuous = self.frames == 0 or video_time - self.video_time <= MAX_TRACKING_GAP_SECONDS
        if not continuous:
            self.start_new_tracker()

        player_result = self.predict(self.player_tracker.model, frame)
        ball_result = player_result if self.fast else self.predict(self.ball_tracker.model, frame)

        players = self.track_players(player_result)
        ball_bbox = self.detect_ball(ball_result, video_time)
        tactical_positions = {} if self.fast else self.get_tactical_positions(frame, players)
        assignment = self.assign_teams(frame, players, video_time)
        events = self.update_possession(players, ball_bbox, assignment, video_time, continuous)
        self.update_distances(tactical_positions, video_time)

        self.frames += 1
        self.players_in_frame = len(players)
        if self.holder_team in self.control_frames:
            self.control_frames[self.holder_team] += 1
        for player_id in players:
            self.player_last_seen[player_id] = video_time
        self.video_time = video_time

        return self.draw(frame, players, ball_bbox, tactical_positions, assignment), events

    def start_new_tracker(self):
        """
        Restart ByteTrack, which only reports a new player once it matches them in the next
        frame. Numbering continues from the previous tracker so player IDs stay unique.
        """
        self.player_tracker.tracker = sv.ByteTrack()
        self.id_offset = self.max_player_id
        self.streak_player = -1
        self.last_holder = -1

    def predict(self, model, frame):
        return model.predict(frame, conf=0.5, device=self.device, verbose=False)[0]

    def track_players(self, result):
        player_class_id = {name: class_id for class_id, name in result.names.items()}['Player']

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == player_class_id]
        detections = self.player_tracker.tracker.update_with_detections(detections)

        players = {int(track_id) + self.id_offset: {"bbox": bbox.tolist()}
                   for bbox, track_id in zip(detections.xyxy, detections.tracker_id)}
        self.max_player_id = max([self.max_player_id, *players])
        return players

    def detect_ball(self, result, video_time):
        ball_class_id = {name: class_id for class_id, name in result.names.items()}['Ball']

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == ball_class_id]
        if len(detections) == 0:
            return None
        bbox = detections.xyxy[detections.confidence.argmax()].tolist()

        # The ball can't have moved further than it could travel since the last good sighting
        if self.last_ball is not None:
            last_bbox, last_time = self.last_ball
            max_distance = BALL_MAX_PIXELS_PER_SECOND * (video_time - last_time)
            if measure_distance(last_bbox[:2], bbox[:2]) > max_distance:
                return None
        self.last_ball = (bbox, video_time)
        return bbox

    def get_tactical_positions(self, frame, players):
        keypoints = self.predict(self.court_keypoint_detector.model, frame).keypoints
        # Crowd shots, replays and ads have no court in view
        if keypoints is None or len(keypoints) == 0:
            return {}
        keypoints = self.tactical_view_converter.validate_keypoints([keypoints])
        return self.tactical_view_converter.transform_players_to_tactical_view(keypoints, [players])[0]

    def assign_teams(self, frame, players, video_time):
        assigner = self.team_assigner
        if self.last_team_refresh is None or video_time - self.last_team_refresh >= TEAM_REFRESH_SECONDS:
            assigner.player_team_dict = {}
            self.last_team_refresh = video_time

        # Classify every new player in one CLIP batch instead of one call per player
        new_ids, crops = [], []
        for player_id, track in players.items():
            if player_id in assigner.player_team_dict:
                continue
            x1, y1, x2, y2 = (max(0, int(v)) for v in track["bbox"])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            new_ids.append(player_id)
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

        if crops:
            classes = [assigner.team_1_class_name, assigner.team_2_class_name]
            inputs = assigner.processor(text=classes, images=crops, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                best_classes = assigner.model(**inputs).logits_per_image.argmax(dim=1).tolist()
            for player_id, best_class in zip(new_ids, best_classes):
                assigner.player_team_dict[player_id] = 1 if best_class == 0 else 2

        assignment = {player_id: assigner.player_team_dict[player_id]
                      for player_id in players if player_id in assigner.player_team_dict}
        self.player_team.update(assignment)
        return assignment

    def update_possession(self, players, ball_bbox, assignment, video_time, continuous=True):
        """
        Live port of BallAquisitionDetector.detect_ball_possession plus pass/interception detection.

        For a frame that isn't continuous with the previous one, the closest player counts as
        holding the ball (good enough for sampling ball control), but passes and interceptions
        are only detected when possession can be followed from frame to frame.
        """
        self.holder = -1
        self.holder_team = -1
        if ball_bbox is None:
            return []

        candidate = self.ball_aquisition_detector.find_best_candidate_for_possession(
            get_center_of_bbox(ball_bbox), players, ball_bbox
        )
        if candidate == -1:
            self.streak_player = -1
            return []
        if not continuous:
            self.holder = candidate
            self.holder_team = assignment.get(candidate, -1)
            return []
        if candidate != self.streak_player:
            self.streak_player = candidate
            self.streak_start = video_time
        if video_time - self.streak_start < MIN_POSSESSION_SECONDS:
            return []

        self.holder = candidate
        self.holder_team = assignment.get(candidate, -1)

        events = []
        if self.last_holder != -1 and self.holder != self.last_holder:
            if self.holder_team == self.last_holder_team and self.holder_team != -1:
                self.passes[self.holder_team] += 1
                events.append(self.make_event("pass", self.holder_team, video_time))
            elif self.holder_team != -1 and self.last_holder_team != -1:
                self.interceptions[self.holder_team] += 1
                events.append(self.make_event("interception", self.holder_team, video_time))

        self.last_holder = self.holder
        self.last_holder_team = self.holder_team
        return events

    def make_event(self, event_type, team, video_time):
        return {
            "type": event_type,
            "team": team,
            "video_time": round(video_time, 2),
            "from_player": self.last_holder,
            "to_player": self.holder,
        }

    def update_distances(self, tactical_positions, video_time):
        for player_id, position in tactical_positions.items():
            previous = self.player_positions.get(player_id)
            self.player_positions[player_id] = (video_time, position)
            if previous is None:
                continue

            previous_time, previous_position = previous
            elapsed = video_time - previous_time
            if elapsed <= 0:
                continue
            meters = self.speed_and_distance_calculator.calculate_meter_distance(previous_position, position)
            if meters / elapsed * 3.6 > MAX_PLAYER_SPEED_KMH:
                continue

            self.player_distance[player_id] = self.player_distance.get(player_id, 0) + meters
            steps = self.player_steps.setdefault(player_id, deque())
            steps.append((video_time, meters, elapsed))
            while video_time - steps[0][0] > SPEED_WINDOW_SECONDS:
                steps.popleft()
            self.player_speed[player_id] = sum(s[1] for s in steps) / sum(s[2] for s in steps) * 3.6

    def draw(self, frame, players, ball_bbox, tactical_positions, assignment):
        frame = self.player_tracks_drawer.draw([frame], [players], [assignment], [self.holder])[0]
        if ball_bbox is not None:
            frame = self.ball_tracks_drawer.draw([frame], [{1: {"bbox": ball_bbox}}])[0]
        if self.fast:
            return frame
        converter = self.tactical_view_converter
        return self.tactical_view_drawer.draw([frame],
                                              converter.court_image_path,
                                              converter.width,
                                              converter.height,
                                              converter.key_points,
                                              [tactical_positions],
                                              [assignment],
                                              [self.holder])[0]

    def get_stats(self, recent_seconds=5, max_players=12):
        """Running totals for the web UI and the saved summary."""
        frames = max(self.frames, 1)
        team_distance = {1: 0.0, 2: 0.0}
        for player_id, distance in self.player_distance.items():
            team = self.player_team.get(player_id)
            if team in team_distance:
                team_distance[team] += distance

        recent_players = [player_id for player_id, seen in self.player_last_seen.items()
                          if self.video_time - seen <= recent_seconds]
        recent_players.sort(key=lambda player_id: self.player_distance.get(player_id, 0), reverse=True)

        return {
            "fast_mode": self.fast,
            "video_time": round(self.video_time, 2),
            "frames_analyzed": self.frames,
            "players_in_frame": self.players_in_frame,
            "ball_control_pct": {team: round(100 * count / frames, 1) for team, count in self.control_frames.items()},
            "passes": dict(self.passes),
            "interceptions": dict(self.interceptions),
            "team_distance_m": {team: round(distance, 1) for team, distance in team_distance.items()},
            "holder": {"player": self.holder, "team": self.holder_team},
            "players": [{
                "id": player_id,
                "team": self.player_team.get(player_id, -1),
                "distance_m": round(self.player_distance.get(player_id, 0), 1),
                "speed_kmh": round(self.player_speed.get(player_id, 0), 1),
            } for player_id in recent_players[:max_players]],
        }
