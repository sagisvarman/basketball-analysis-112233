from PIL import Image
import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# Part of the player box that shows the jersey: skips the head, the shorts and legs, and
# the edges where arms, the floor and other players get in
JERSEY_LEFT, JERSEY_RIGHT = 0.2, 0.8
JERSEY_TOP, JERSEY_BOTTOM = 0.15, 0.55
MIN_CROP_PIXELS = 4

# Grouping jerseys into two teams
GROUPING_RESTARTS = 10
GROUPING_ITERATIONS = 50
OUTLIER_SHARE = 0.1     # the least team-like jerseys (bench, fans, odd crops) are left out of the final grouping


def pick_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def group_in_two(embeddings, weights, seed=0):
    """
    Split unit-length embeddings into two groups by cosine similarity (spherical k-means).

    Args:
        embeddings (numpy.ndarray): (n, d) unit-length vectors.
        weights (numpy.ndarray): (n,) how much each vector counts.
        seed (int): Seed for the random starting points, so results repeat.

    Returns:
        tuple or None: (centers, labels, group_weights), or None if the vectors can't be split
            (fewer than two, or all the same).
    """
    if len(embeddings) < 2:
        return None
    rng = np.random.default_rng(seed)
    probabilities = weights / weights.sum()
    best = None
    for _ in range(GROUPING_RESTARTS):
        # k-means++ start: the second center is likely far from the first
        first = embeddings[rng.choice(len(embeddings), p=probabilities)]
        spread = weights * np.maximum(1 - embeddings @ first, 0) ** 2
        if spread.sum() <= 0:
            return None
        second = embeddings[rng.choice(len(embeddings), p=spread / spread.sum())]
        centers = np.stack([first, second])

        for _ in range(GROUPING_ITERATIONS):
            labels = np.argmax(embeddings @ centers.T, axis=1)
            if len(set(labels.tolist())) < 2:
                centers = None
                break
            new_centers = np.stack([(embeddings[labels == group] * weights[labels == group, None]).sum(0)
                                    for group in (0, 1)])
            new_centers /= np.linalg.norm(new_centers, axis=1, keepdims=True)
            converged = np.allclose(new_centers, centers, atol=1e-6)
            centers = new_centers
            if converged:
                break
        if centers is None:
            continue

        labels = np.argmax(embeddings @ centers.T, axis=1)
        score = float((weights * (embeddings @ centers.T).max(axis=1)).sum())
        if best is None or score > best[0]:
            group_weights = np.array([weights[labels == group].sum() for group in (0, 1)])
            best = (score, centers, labels, group_weights)
    return None if best is None else best[1:]


class TeamAssigner:
    """
    Looks at players' jerseys with a CLIP model.

    For each player crop it gives two things from one model pass:
    - how likely the jersey is Team 1's rather than Team 2's
    - an appearance embedding, used to recognize the same player again later

    Teams are found without being told the jersey colors: fit_teams() splits the jerseys
    it is shown into the two most different-looking groups. If jersey descriptions are
    given (e.g. "white shirt"), they decide which group is Team 1, and are used on their
    own until enough players have been seen to group them.

    Deciding a player's team over many frames is done by PlayerIdentityKeeper, which
    adds up these per-frame votes for each player and decides when to group jerseys.

    Attributes:
        team_1_class_name (str or None): Description of Team 1's jersey, if given.
        team_2_class_name (str or None): Description of Team 2's jersey, if given.
        prototypes (torch.Tensor or None): (2, d) what each team's jersey looks like.
        grouped (bool): True once the prototypes come from grouping players' jerseys.
    """
    def __init__(self,
                 team_1_class_name=None,
                 team_2_class_name=None,
                 device=None,
                 ):
        """
        Initialize the TeamAssigner.

        Args:
            team_1_class_name (str, optional): Description of Team 1's jersey, e.g. "white shirt".
                Leave both descriptions out to have the teams found automatically.
            team_2_class_name (str, optional): Description of Team 2's jersey.
            device (str, optional): Torch device; picks CUDA, then MPS, then CPU by default.
        """
        self.device = device or pick_device()
        self.model = None
        self.text_features = None
        self.prototypes = None
        self.grouped = False
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

    @property
    def has_descriptions(self):
        return bool(self.team_1_class_name and self.team_2_class_name)

    def load_model(self):
        """
        Loads the pre-trained vision model for jersey color classification.
        """
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.set_team_names(self.team_1_class_name, self.team_2_class_name)

    def set_team_names(self, team_1_class_name=None, team_2_class_name=None):
        """
        Set the jersey descriptions (or none, for automatic teams) and forget any grouping.

        Args:
            team_1_class_name (str, optional): Description of Team 1's jersey, e.g. "white shirt".
            team_2_class_name (str, optional): Description of Team 2's jersey, e.g. "red shirt".
        """
        self.team_1_class_name = (team_1_class_name or "").strip() or None
        self.team_2_class_name = (team_2_class_name or "").strip() or None
        self.text_features = None
        self.prototypes = None
        self.grouped = False
        if self.model is None or not self.has_descriptions:
            return
        inputs = self.processor(text=[self.team_1_class_name, self.team_2_class_name],
                                return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.prototypes = self.text_features

    @staticmethod
    def get_jersey_crop(frame, bbox):
        """
        Cut the jersey area out of a player's bounding box.

        Args:
            frame (numpy.ndarray): The video frame (BGR).
            bbox (list): Player bounding box (x1, y1, x2, y2).

        Returns:
            PIL.Image or None: The jersey crop in RGB, or None if the box is too small.
        """
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        frame_height, frame_width = frame.shape[:2]
        left = max(0, int(x1 + JERSEY_LEFT * width))
        right = min(frame_width, int(x1 + JERSEY_RIGHT * width))
        top = max(0, int(y1 + JERSEY_TOP * height))
        bottom = min(frame_height, int(y1 + JERSEY_BOTTOM * height))
        if right - left < MIN_CROP_PIXELS or bottom - top < MIN_CROP_PIXELS:
            return None
        crop = frame[top:bottom, left:right]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def describe_players(self, frame, bboxes):
        """
        Classify jerseys and compute appearance embeddings for several players at once.

        Args:
            frame (numpy.ndarray): The video frame containing the players.
            bboxes (list): Player bounding boxes.

        Returns:
            tuple: (embeddings, team_1_probabilities), lists aligned with `bboxes`. An
                embedding is a unit-length numpy vector; the probability is how likely the
                jersey is Team 1's rather than Team 2's. Embeddings are None for boxes too
                small to look at; probabilities are also None while the teams aren't known yet.
        """
        embeddings = [None] * len(bboxes)
        team_1_probabilities = [None] * len(bboxes)

        crops, crop_indexes = [], []
        for index, bbox in enumerate(bboxes):
            crop = self.get_jersey_crop(frame, bbox)
            if crop is not None:
                crops.append(crop)
                crop_indexes.append(index)
        if not crops:
            return embeddings, team_1_probabilities

        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float().cpu().numpy()
        probabilities = self.team_1_probabilities(image_features)

        for crop_number, index in enumerate(crop_indexes):
            embeddings[index] = image_features[crop_number]
            if probabilities is not None:
                team_1_probabilities[index] = probabilities[crop_number]
        return embeddings, team_1_probabilities

    def team_1_probabilities(self, embeddings):
        """
        How likely each jersey embedding is Team 1's, or None while the teams aren't known.

        Args:
            embeddings (numpy.ndarray): (n, d) unit-length jersey embeddings.

        Returns:
            list or None: One probability per embedding.
        """
        if self.prototypes is None or len(embeddings) == 0:
            return None
        features = torch.as_tensor(np.asarray(embeddings), dtype=self.prototypes.dtype, device=self.prototypes.device)
        with torch.no_grad():
            logits = self.model.logit_scale.exp() * features @ self.prototypes.T
            return logits.softmax(dim=1)[:, 0].float().cpu().tolist()

    def fit_teams(self, embeddings, weights=None):
        """
        Find the two teams by splitting jerseys into the two most different-looking groups.

        Args:
            embeddings (list): Unit-length jersey embeddings, ideally one per player.
            weights (list, optional): How much each embedding counts (e.g. how often that
                player was seen clearly).

        Returns:
            bool: True if the teams were found.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        weights = np.ones(len(embeddings)) if weights is None else np.asarray(weights, dtype=np.float64)
        result = group_in_two(embeddings, weights)
        if result is None:
            return False
        centers, labels, group_weights = result

        # Group again without the jerseys least like either team
        if len(embeddings) >= 10:
            similarity = (embeddings * centers[labels]).sum(axis=1)
            keep = similarity >= np.quantile(similarity, OUTLIER_SHARE)
            trimmed = group_in_two(embeddings[keep], weights[keep])
            if trimmed is not None:
                centers, _, group_weights = trimmed

        if self.should_swap_groups(centers, group_weights):
            centers = centers[::-1].copy()
        self.prototypes = torch.as_tensor(centers, device=self.device)
        self.grouped = True
        return True

    def should_swap_groups(self, centers, group_weights):
        """
        Which group is Team 1: the one closest to Team 1's description if there is one,
        otherwise the one closest to the previous Team 1 so teams don't swap when grouping
        again, otherwise the bigger group.
        """
        if self.text_features is not None:
            reference = self.text_features.float().cpu().numpy()
        elif self.prototypes is not None:
            reference = self.prototypes.float().cpu().numpy()
        else:
            return group_weights[1] > group_weights[0]
        similarity = centers @ reference.T
        return similarity[0, 1] + similarity[1, 0] > similarity[0, 0] + similarity[1, 1]
