import os
import sys
import pathlib
from collections import deque

import numpy as np
import pandas as pd

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import measure_distance

# Court positions jitter from frame to frame (the court keypoints move a little each
# frame). Measured frame by frame, that jitter adds up to many meters of "running" even
# for a player standing still, so positions are smoothed first: a median (drops one-off
# bad positions), then an average, each over this long
SMOOTHING_SECONDS = 0.5
MAX_GAP_SECONDS = 5 / 30            # a player missing longer than this starts again: the jump isn't counted
MAX_PLAYER_SPEED_KMH = 40           # faster than this is a tracking glitch, not running
SPEED_WINDOW_SECONDS = 15 / 30      # speed is the average over this long...
MIN_SPEED_SECONDS = 5 / 30          # ...once the player has been seen moving for this long in it


class PositionSmoother:
    """
    Smooths one player's court positions as they arrive, for the live app: the median of
    the last SMOOTHING_SECONDS of positions, then the average of those medians.
    """
    def __init__(self):
        self.positions = deque()
        self.medians = deque()
        self.last_time = None
        self.last_position = None

    def add(self, video_time, position):
        """
        Args:
            video_time (float): Seconds since the start of the video.
            position (tuple): (x, y) court position in meters.

        Returns:
            numpy.ndarray: The smoothed (x, y) position in meters.
        """
        self.positions.append((video_time, np.asarray(position, dtype=float)))
        while video_time - self.positions[0][0] > SMOOTHING_SECONDS:
            self.positions.popleft()
        median = np.median([p for _, p in self.positions], axis=0)

        self.medians.append((video_time, median))
        while video_time - self.medians[0][0] > SMOOTHING_SECONDS:
            self.medians.popleft()

        self.last_time = video_time
        self.last_position = np.mean([m for _, m in self.medians], axis=0)
        return self.last_position


class SpeedAndDistanceCalculator():
    def __init__(self,
                 width_in_pixels,
                 height_in_pixels,
                 width_in_meters,
                 height_in_meters,
                 ):

        self.width_in_pixels = width_in_pixels
        self.height_in_pixels= height_in_pixels

        self.width_in_meters = width_in_meters
        self.height_in_meters= height_in_meters

    def to_meters(self, pixel_position):
        """Convert a position on the tactical court image to meters."""
        pixel_x, pixel_y = pixel_position
        return (pixel_x * self.width_in_meters / self.width_in_pixels,
                pixel_y * self.height_in_meters / self.height_in_pixels)

    def calculate_distance(self,
                            tactical_player_positions,
                            fps=30
                            ):
        """
        Distance each player covers in each frame, in meters, from smoothed court positions.

        Args:
            tactical_player_positions (list): Per frame, {player_id: (x, y)} on the tactical court image.
            fps (float): Frame rate of the video.

        Returns:
            list: Per frame, {player_id: meters covered since their previous position}.
        """
        window = max(1, int(round(SMOOTHING_SECONDS * fps)))
        max_frame_gap = max(1, int(round(MAX_GAP_SECONDS * fps)))
        output_distances = [{} for _ in tactical_player_positions]

        positions_by_player = {}
        for frame_number, tactical_player_position_frame in enumerate(tactical_player_positions):
            for player_id, position in tactical_player_position_frame.items():
                positions_by_player.setdefault(player_id, []).append((frame_number, self.to_meters(position)))

        for player_id, samples in positions_by_player.items():
            # Players keep their ID after going missing; when they come back somewhere else,
            # that jump isn't distance run, so each unbroken stretch is measured on its own
            stretches = [[samples[0]]]
            for sample in samples[1:]:
                if sample[0] - stretches[-1][-1][0] <= max_frame_gap:
                    stretches[-1].append(sample)
                else:
                    stretches.append([sample])

            for stretch in stretches:
                frames = [frame_number for frame_number, _ in stretch]
                smoothed = (pd.DataFrame([position for _, position in stretch])
                            .rolling(window, center=True, min_periods=1).median()
                            .rolling(window, center=True, min_periods=1).mean()
                            .to_numpy())
                for i in range(1, len(stretch)):
                    meter_distance = measure_distance(smoothed[i], smoothed[i - 1])
                    seconds = (frames[i] - frames[i - 1]) / fps
                    if meter_distance / seconds * 3.6 > MAX_PLAYER_SPEED_KMH:
                        continue
                    output_distances[frames[i]][player_id] = meter_distance

        return output_distances

    def calculate_meter_distance(self,previous_pixel_position, current_pixel_position):
        """Straight-line distance in meters between two positions on the tactical court image."""
        return measure_distance(self.to_meters(current_pixel_position), self.to_meters(previous_pixel_position))

    def calculate_speed(self, distances, fps=30):
        """
        Calculate player speeds based on distances covered over the last SPEED_WINDOW_SECONDS.

        Args:
            distances (list): List of dictionaries containing distance per player per frame,
                            as output by calculate_distance method.
            fps (float): Frames per second of the video, used to calculate elapsed time.

        Returns:
            list: List of dictionaries where each dictionary maps player_id to their
                speed in km/h at that frame.
        """
        speeds = []
        window_frames = max(1, int(round(SPEED_WINDOW_SECONDS * fps)))
        min_frames_present = max(1, int(round(MIN_SPEED_SECONDS * fps)))

        # Calculate speed for each frame and player
        for frame_idx in range(len(distances)):
            speeds.append({})
            # For each player in current frame
            for player_id in distances[frame_idx].keys():
                # Look back over the window, or less at the beginning
                start_frame = max(0, frame_idx - window_frames + 1)

                total_distance = 0
                frames_present = 0
                last_frame_present = None

                # Calculate total distance in the window
                for i in range(start_frame, frame_idx + 1):
                    if player_id in distances[i]:
                        if last_frame_present is not None:
                            total_distance += distances[i][player_id]
                            frames_present += 1
                        last_frame_present = i

                # Calculate speed only if player was present long enough
                if frames_present >= min_frames_present:
                    # Calculate time in hours (convert frames to hours)
                    time_in_seconds = frames_present / fps
                    time_in_hours = time_in_seconds / 3600

                    # Calculate speed in km/h
                    if time_in_hours > 0:
                        speed_kmh = (total_distance / 1000) / time_in_hours
                        speeds[frame_idx][player_id] = speed_kmh
                    else:
                        speeds[frame_idx][player_id] = 0
                else:
                    speeds[frame_idx][player_id] = 0

        return speeds
