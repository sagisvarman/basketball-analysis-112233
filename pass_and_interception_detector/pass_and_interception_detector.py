from copy import deepcopy

# A change of holder only counts as a pass or interception if the ball was loose for at
# most this long. Longer, and it was out of play or out of view (a timeout, a camera cut),
# not passed. Long enough for a long pass, so rebounds after a quick shot can still count.
MAX_LOOSE_BALL_SECONDS = 3.0


class PassAndInterceptionDetector():
    """
    A class that detects passes between teammates and interceptions by opposing teams.
    """
    def __init__(self):
        pass

    def get_possession_changes(self, ball_acquisition, player_assignment, fps=30):
        """
        Find every frame where a different player gets the ball, soon enough after the
        previous holder had it.

        Args:
            ball_acquisition (list): A list indicating which player has possession of the ball in each frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            fps (float): Frame rate of the video.

        Returns:
            list: (frame, previous_team, current_team) tuples; a team is -1 when unknown.
        """
        max_gap = MAX_LOOSE_BALL_SECONDS * fps
        changes = []
        prev_holder=-1
        previous_frame=-1

        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                prev_holder = ball_acquisition[frame - 1]
                previous_frame= frame - 1

            current_holder = ball_acquisition[frame]

            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                if frame - previous_frame > max_gap:
                    continue
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)
                changes.append((frame, prev_team, current_team))

        return changes

    def detect_passes(self,ball_acquisition,player_assignment,fps=30):
        """
        Detects successful passes between players of the same team.

        Args:
            ball_acquisition (list): A list indicating which player has possession of the ball in each frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            fps (float): Frame rate of the video.

        Returns:
            list: A list where each element indicates if a pass occurred in that frame
                (-1: no pass, 1: Team 1 pass, 2: Team 2 pass).
        """
        passes = [-1] * len(ball_acquisition)
        for frame, prev_team, current_team in self.get_possession_changes(ball_acquisition, player_assignment, fps):
            if prev_team == current_team and prev_team != -1:
                passes[frame] = prev_team
        return passes

    def detect_interceptions(self,ball_acquisition,player_assignment,fps=30):
        """
        Detects interceptions where the ball possession changes between opposing teams.

        Args:
            ball_acquisition (list): A list indicating which player has possession of the ball in each frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            fps (float): Frame rate of the video.

        Returns:
            list: A list where each element indicates if an interception occurred in that frame
                (-1: no interception, 1: Team 1 interception, 2: Team 2 interception).
        """
        interceptions = [-1] * len(ball_acquisition)
        for frame, prev_team, current_team in self.get_possession_changes(ball_acquisition, player_assignment, fps):
            if prev_team != current_team and prev_team != -1 and current_team != -1:
                interceptions[frame] = current_team
        return interceptions
