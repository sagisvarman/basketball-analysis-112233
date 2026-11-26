"""
Keeps one ID and one team per player for the whole video.

ByteTrack follows players by how much their boxes overlap from one frame to the next,
so it loses a player whenever they're hidden behind someone, leave the picture or the
camera cuts away, and gives them a brand new ID when they come back. This module sits
on top of it. When ByteTrack starts a new track, it's compared with the players who
went missing recently: same team, close enough to where they were last seen, and
similar looking jersey (CLIP embedding). The best match keeps its old player ID.

Teams are decided per player from all the looks at their jersey so far, instead of from
a single frame, so a player doesn't flip team when someone walks in front of them. The
teams themselves are found by grouping players' jerseys into the two most different-looking
groups, so nobody has to say which colors the teams wear.
"""

import math
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.append('../')
from utils import get_foot_position, measure_distance, read_stub, save_stub

STUB_VERSION = 2

# Matching a new track to a missing player
MAX_MISSING_SECONDS = 20            # missing players are remembered this long
MIN_APPEARANCE_SIMILARITY = 0.75    # cosine similarity of jersey embeddings; below this it's someone else
MAX_TEAM_DISAGREEMENT = 0.5         # gap in Team 1 probability that rules a match out...
TEAM_RULES_OUT_AFTER_SECONDS = 1.0  # ...once the player has been missing this long, and only on a clear look.
                                    # Sooner, being right where they were counts more than one jersey look
# How far a missing player may have moved, in body heights. Generous because the camera pans too
BASE_MOVE_BODY_HEIGHTS = 1.5
MOVE_BODY_HEIGHTS_PER_SECOND = 4.0
POSITION_COST_WEIGHT = 0.3
TIME_COST_WEIGHT = 0.2
TEAM_COST_WEIGHT = 0.3              # team disagreement that doesn't rule a match out still costs
INFEASIBLE = 1e6

# Looking at jerseys
LOOK_EVERY_SECONDS = 0.5            # how often a tracked player's jersey is looked at again
EMBEDDING_MOMENTUM = 0.9            # how slowly a player's remembered appearance changes
TEAM_VOTE_DECAY = 0.9               # older team votes fade, so a wrong start can be corrected
MAX_OVERLAP_FOR_CLEAR_LOOK = 0.3    # share of the box covered by another player
UNCLEAR_LOOK_WEIGHT = 0.2           # a partly hidden player's team vote counts this much

# ByteTrack can swap two crossing players' IDs. Once a player's team is settled, a track
# that keeps showing the other team's jersey is taken off its player and matched again,
# together with the track it probably swapped with
CONFIDENT_TEAM_MARGIN = 0.3         # Team 1 probability below 0.2 or above 0.8 counts as sure
MIN_VOTES_TO_BE_SURE = 2.0          # about two clear looks before a player's team counts as settled
SWAP_LOOKS = 3                      # clear looks in a row at the other team's jersey

# Finding the teams by grouping jerseys
MIN_PLAYERS_TO_GROUP = 8            # players seen clearly before their jerseys are grouped into teams
MIN_CLEAR_LOOKS_TO_GROUP = 2
REGROUP_EVERY_SECONDS = 30          # frame by frame (live): group again as more players are seen
GROUP_PLAYERS_SEEN_WITHIN_SECONDS = 120
MAX_GROUPING_WEIGHT = 10            # so players on screen the whole time don't drown out the rest
SAMPLE_CROPS_FOR_GROUPING = 600     # whole videos: jerseys looked at up front to find the teams


class PlayerProfile:
    """What is remembered about one player: where and when they were last seen, their look and team votes."""
    def __init__(self, player_id):
        self.player_id = player_id
        self.last_bbox = None
        self.last_seen = None
        self.last_look = -math.inf
        self.embedding = None
        self.team_votes = np.zeros(2)
        self.clear_looks = 0
        self.opposite_team_looks = 0
        # Where and when a look last confirmed this player's team. If their track turns out
        # to have swapped onto someone else, this is where the player really was
        self.trusted_bbox = None
        self.trusted_time = None

    def team_1_probability(self):
        total = self.team_votes.sum()
        return float(self.team_votes[0] / total) if total > 0 else None

    def team(self):
        probability = self.team_1_probability()
        if probability is None:
            return None
        return 1 if probability >= 0.5 else 2

    def is_sure_of_team(self):
        probability = self.team_1_probability()
        return (probability is not None and self.team_votes.sum() >= MIN_VOTES_TO_BE_SURE
                and abs(probability - 0.5) >= CONFIDENT_TEAM_MARGIN)

    def add_look(self, embedding, team_1_probability, clear):
        """
        Add one look at the player's jersey.

        Args:
            embedding (numpy.ndarray or None): Unit-length appearance embedding.
            team_1_probability (float or None): How likely the jersey is Team 1's.
            clear (bool): False when another player covers part of this one, so the crop
                may show the wrong jersey.
        """
        if embedding is not None and clear:
            self.clear_looks += 1
        if embedding is not None and (clear or self.embedding is None):
            if self.embedding is None:
                self.embedding = embedding
            else:
                mixed = EMBEDDING_MOMENTUM * self.embedding + (1 - EMBEDDING_MOMENTUM) * embedding
                self.embedding = mixed / np.linalg.norm(mixed)

        if team_1_probability is not None:
            weight = 1.0 if clear else UNCLEAR_LOOK_WEIGHT
            decay = 1 - (1 - TEAM_VOTE_DECAY) * weight
            vote = np.array([team_1_probability, 1 - team_1_probability])
            self.team_votes = self.team_votes * decay + weight * vote


class PlayerIdentityKeeper:
    """
    Turns ByteTrack track IDs into one stable ID per player, and gives each player a team.

    Feed it one frame at a time with update(), or a whole video with process_video().
    """
    def __init__(self, team_assigner):
        """
        Args:
            team_assigner (TeamAssigner): Used to look at jerseys (team and appearance).
        """
        self.team_assigner = team_assigner
        self.reset()

    def reset(self):
        """Forget every player, for a new video."""
        self.players = {}
        self.track_to_player = {}
        self.swapped_off = set()
        self.next_player_id = 1
        self.learn_teams = True
        self.last_team_grouping = -math.inf

    def forget_tracks(self):
        """
        Call after replacing the ByteTrack tracker: its track IDs start over, but the
        players are still remembered, so new tracks are matched to them again.
        """
        self.track_to_player = {}
        self.swapped_off = set()

    def update(self, frame, video_time, tracks):
        """
        Give this frame's tracks their player IDs and teams.

        Args:
            frame (numpy.ndarray): The video frame.
            video_time (float): Seconds since the start of the video.
            tracks (dict): {track_id: {"bbox": [x1, y1, x2, y2]}} from ByteTrack.

        Returns:
            tuple: (players, teams) where players is {player_id: {"bbox": bbox}} and teams
                is {player_id: 1 or 2}.
        """
        known = {}
        new_tracks = []
        for track_id in tracks:
            player_id = self.track_to_player.get(track_id)
            if player_id is None:
                new_tracks.append(track_id)
            else:
                known[track_id] = player_id

        # Look at every new track, and at known players whose last look is getting old
        to_look = new_tracks + [track_id for track_id, player_id in known.items()
                                if video_time - self.players[player_id].last_look >= LOOK_EVERY_SECONDS]
        looks = {}
        if to_look:
            embeddings, probabilities = self.team_assigner.describe_players(
                frame, [tracks[track_id]["bbox"] for track_id in to_look])
            looks = {track_id: (embedding, probability)
                     for track_id, embedding, probability in zip(to_look, embeddings, probabilities)}
        clear = {track_id: self.is_clear(track_id, tracks) for track_id in looks}

        for track_id in self.find_swapped_tracks(known, looks, clear):
            profile = self.players[known[track_id]]
            if profile.trusted_bbox is not None:
                profile.last_bbox, profile.last_seen = profile.trusted_bbox, profile.trusted_time
            self.swapped_off.add((track_id, known[track_id]))
            del self.track_to_player[track_id]
            del known[track_id]
            new_tracks.append(track_id)

        self.match_new_tracks(new_tracks, tracks, looks, clear, video_time, known)

        for track_id, player_id in known.items():
            profile = self.players[player_id]
            bbox = tracks[track_id]["bbox"]
            profile.last_bbox = bbox
            profile.last_seen = video_time
            if track_id in looks:
                # While a swap is suspected, the other team's looks are held back so they
                # don't rewrite this player's team and appearance
                if profile.opposite_team_looks == 0:
                    embedding, probability = looks[track_id]
                    profile.add_look(embedding, probability, clear[track_id])
                    if clear[track_id]:
                        profile.trusted_bbox, profile.trusted_time = bbox, video_time
                profile.last_look = video_time

        self.maybe_group_teams(video_time)

        players, teams = {}, {}
        for track_id, player_id in known.items():
            players[player_id] = {"bbox": tracks[track_id]["bbox"]}
            team = self.players[player_id].team()
            if team is not None:
                teams[player_id] = team
        return players, teams

    def maybe_group_teams(self, video_time):
        """
        Find (or refresh) the teams from the jerseys of the players seen so far, once
        enough of them have been seen clearly.
        """
        assigner = self.team_assigner
        if not self.learn_teams:
            return
        if assigner.grouped and video_time - self.last_team_grouping < REGROUP_EVERY_SECONDS:
            return
        seen = [profile for profile in self.players.values()
                if profile.embedding is not None and profile.clear_looks >= MIN_CLEAR_LOOKS_TO_GROUP
                and video_time - profile.last_seen <= GROUP_PLAYERS_SEEN_WITHIN_SECONDS]
        if len(seen) < MIN_PLAYERS_TO_GROUP:
            return
        self.last_team_grouping = video_time
        if not assigner.fit_teams([profile.embedding for profile in seen],
                                  [min(profile.clear_looks, MAX_GROUPING_WEIGHT) for profile in seen]):
            return

        # Players seen before the teams were known get a first vote from their remembered look
        unvoted = [profile for profile in self.players.values()
                   if profile.embedding is not None and profile.team_votes.sum() == 0]
        if unvoted:
            probabilities = assigner.team_1_probabilities(np.stack([profile.embedding for profile in unvoted]))
            for profile, probability in zip(unvoted, probabilities):
                profile.add_look(None, probability, clear=True)

    def group_teams_from_video(self, video_frames, tracks, fps):
        """
        Find the teams before going through a whole video, from jerseys sampled across all
        of it, so players have their team from the first frame.

        Returns:
            bool: True if the teams were found.
        """
        total_boxes = sum(len(frame_tracks) for frame_tracks in tracks)
        if total_boxes == 0:
            return False
        step = max(1, int(round(fps * LOOK_EVERY_SECONDS)), math.ceil(total_boxes / SAMPLE_CROPS_FOR_GROUPING))

        samples = defaultdict(list)
        for frame_num in range(0, len(video_frames), step):
            frame_tracks = tracks[frame_num]
            clear_tracks = [track_id for track_id in frame_tracks if self.is_clear(track_id, frame_tracks)]
            if not clear_tracks:
                continue
            embeddings, _ = self.team_assigner.describe_players(
                video_frames[frame_num], [frame_tracks[track_id]["bbox"] for track_id in clear_tracks])
            for track_id, embedding in zip(clear_tracks, embeddings):
                if embedding is not None:
                    samples[track_id].append(embedding)

        if len(samples) >= MIN_PLAYERS_TO_GROUP:
            # One averaged look per track is steadier than single crops
            embeddings = [np.mean(looks, axis=0) for looks in samples.values()]
            embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
            weights = [min(len(looks), MAX_GROUPING_WEIGHT) for looks in samples.values()]
        else:
            embeddings = [embedding for looks in samples.values() for embedding in looks]
            weights = None
        return self.team_assigner.fit_teams(embeddings, weights)

    def find_swapped_tracks(self, known, looks, clear):
        """
        Tracks that have clearly shown the other team's jersey several looks in a row, plus
        any other track that is close to that too: when ByteTrack swaps two players, both
        tracks start showing the wrong jersey at the same moment.
        """
        swap_found = False
        for track_id, player_id in known.items():
            if track_id not in looks or not clear[track_id]:
                continue
            profile = self.players[player_id]
            if self.is_other_team(profile, looks[track_id][1]):
                profile.opposite_team_looks += 1
                swap_found = swap_found or profile.opposite_team_looks >= SWAP_LOOKS
            else:
                profile.opposite_team_looks = 0
        if not swap_found:
            return []

        swapped = [track_id for track_id, player_id in known.items()
                   if self.players[player_id].opposite_team_looks >= SWAP_LOOKS - 1]
        for track_id in swapped:
            self.players[known[track_id]].opposite_team_looks = 0
        return swapped

    @staticmethod
    def is_other_team(profile, team_1_probability):
        if team_1_probability is None or not profile.is_sure_of_team():
            return False
        sure_of_look = abs(team_1_probability - 0.5) >= CONFIDENT_TEAM_MARGIN
        return sure_of_look and (team_1_probability >= 0.5) != (profile.team() == 1)

    def match_new_tracks(self, new_tracks, tracks, looks, clear, video_time, known):
        """
        Match new tracks to missing players (one-to-one, lowest total cost), and give the
        rest new player IDs. Adds every new track to `known`.
        """
        visible_players = set(known.values())
        missing = [player_id for player_id, profile in self.players.items()
                   if player_id not in visible_players
                   and video_time - profile.last_seen <= MAX_MISSING_SECONDS]

        matches = {}
        if new_tracks and missing:
            costs = np.full((len(new_tracks), len(missing)), INFEASIBLE)
            for row, track_id in enumerate(new_tracks):
                embedding, probability = looks.get(track_id, (None, None))
                for column, player_id in enumerate(missing):
                    if (track_id, player_id) in self.swapped_off:
                        continue
                    cost = self.match_cost(self.players[player_id], tracks[track_id]["bbox"],
                                           embedding, probability, clear.get(track_id, False), video_time)
                    if cost is not None:
                        costs[row, column] = cost
            for row, column in zip(*linear_sum_assignment(costs)):
                if costs[row, column] < INFEASIBLE:
                    matches[new_tracks[row]] = missing[column]

        for track_id in new_tracks:
            player_id = matches.get(track_id)
            if player_id is None:
                player_id = self.next_player_id
                self.next_player_id += 1
                self.players[player_id] = PlayerProfile(player_id)
            else:
                self.players[player_id].opposite_team_looks = 0
                # The player's old track may come back later (ByteTrack revives lost tracks
                # with their old ID), by then probably on someone else: it must be matched again
                for old_track_id in [t for t, p in self.track_to_player.items() if p == player_id]:
                    del self.track_to_player[old_track_id]
            self.track_to_player[track_id] = player_id
            known[track_id] = player_id

    @staticmethod
    def match_cost(profile, bbox, embedding, team_1_probability, clear, video_time):
        """
        How unlikely it is that a new track is this missing player, or None if it can't be.

        Lower is better: an unlike appearance, a long way from where they were last seen
        and a long time missing all add to the cost.
        """
        if embedding is None or profile.embedding is None:
            return None

        similarity = float(np.dot(profile.embedding, embedding))
        if similarity < MIN_APPEARANCE_SIMILARITY:
            return None

        missing_seconds = max(0.0, video_time - profile.last_seen)

        team_disagreement = 0.0
        profile_probability = profile.team_1_probability()
        if profile_probability is not None and team_1_probability is not None:
            team_disagreement = abs(profile_probability - team_1_probability)
            if (clear and missing_seconds > TEAM_RULES_OUT_AFTER_SECONDS
                    and team_disagreement > MAX_TEAM_DISAGREEMENT):
                return None
        body_height = max(bbox[3] - bbox[1], profile.last_bbox[3] - profile.last_bbox[1], 1)
        allowed_distance = (BASE_MOVE_BODY_HEIGHTS + MOVE_BODY_HEIGHTS_PER_SECOND * missing_seconds) * body_height
        distance = measure_distance(get_foot_position(bbox), get_foot_position(profile.last_bbox))
        if distance > allowed_distance:
            return None

        return ((1 - similarity)
                + POSITION_COST_WEIGHT * distance / allowed_distance
                + TIME_COST_WEIGHT * missing_seconds / MAX_MISSING_SECONDS
                + TEAM_COST_WEIGHT * team_disagreement)

    @staticmethod
    def is_clear(track_id, tracks):
        """False when another player's box covers too much of this one."""
        x1, y1, x2, y2 = tracks[track_id]["bbox"]
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        for other_id, other in tracks.items():
            if other_id == track_id:
                continue
            other_x1, other_y1, other_x2, other_y2 = other["bbox"]
            overlap_width = min(x2, other_x2) - max(x1, other_x1)
            overlap_height = min(y2, other_y2) - max(y1, other_y1)
            if overlap_width > 0 and overlap_height > 0 and overlap_width * overlap_height / area > MAX_OVERLAP_FOR_CLEAR_LOOK:
                return False
        return True

    def process_video(self, video_frames, tracks, fps=30, read_from_stub=False, stub_path=None):
        """
        Give every player in a whole video a stable ID and a team, with optional caching.

        Args:
            video_frames (list): The video frames.
            tracks (list): ByteTrack tracks per frame, from PlayerTracker.get_object_tracks.
            fps (float): Frame rate of the video.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            tuple: (player_tracks, player_assignment), both lists with one dict per frame:
                {player_id: {"bbox": bbox}} and {player_id: team}.
        """
        team_names = [self.team_assigner.team_1_class_name, self.team_assigner.team_2_class_name]
        stub = read_stub(read_from_stub, stub_path)
        if (isinstance(stub, dict) and stub.get("version") == STUB_VERSION
                and stub.get("team_names") == team_names
                and len(stub.get("player_tracks", [])) == len(video_frames)):
            return stub["player_tracks"], stub["player_assignment"]

        if self.team_assigner.model is None:
            self.team_assigner.load_model()
        self.reset()
        # Teams found from the whole video don't need regrouping along the way
        self.learn_teams = not self.group_teams_from_video(video_frames, tracks, fps)

        player_tracks, player_assignment = [], []
        for frame_num, (frame, frame_tracks) in enumerate(zip(video_frames, tracks)):
            players, teams = self.update(frame, frame_num / fps, frame_tracks)
            player_tracks.append(players)
            player_assignment.append(teams)

        save_stub(stub_path, {
            "version": STUB_VERSION,
            "team_names": team_names,
            "player_tracks": player_tracks,
            "player_assignment": player_assignment,
        })
        return player_tracks, player_assignment
