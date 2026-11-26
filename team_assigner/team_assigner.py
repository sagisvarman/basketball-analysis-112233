from PIL import Image
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel

# Part of the player box that shows the jersey: skips the head, the shorts and legs, and
# the edges where arms, the floor and other players get in
JERSEY_LEFT, JERSEY_RIGHT = 0.2, 0.8
JERSEY_TOP, JERSEY_BOTTOM = 0.15, 0.55
MIN_CROP_PIXELS = 4


def pick_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class TeamAssigner:
    """
    Looks at players' jerseys with a CLIP model.

    For each player crop it gives two things from one model pass:
    - how likely the jersey matches Team 1's description rather than Team 2's
    - an appearance embedding, used to recognize the same player again later

    Deciding a player's team over many frames is done by PlayerIdentityKeeper, which
    adds up these per-frame votes for each player.

    Attributes:
        team_1_class_name (str): Description of Team 1's jersey appearance.
        team_2_class_name (str): Description of Team 2's jersey appearance.
    """
    def __init__(self,
                 team_1_class_name= "white shirt",
                 team_2_class_name= "dark blue shirt",
                 device=None,
                 ):
        """
        Initialize the TeamAssigner with specified team jersey descriptions.

        Args:
            team_1_class_name (str): Description of Team 1's jersey appearance.
            team_2_class_name (str): Description of Team 2's jersey appearance.
            device (str, optional): Torch device; picks CUDA, then MPS, then CPU by default.
        """
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.device = device or pick_device()
        self.model = None
        self.text_features = None

    def load_model(self):
        """
        Loads the pre-trained vision model for jersey color classification.
        """
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.set_team_names(self.team_1_class_name, self.team_2_class_name)

    def set_team_names(self, team_1_class_name, team_2_class_name):
        """
        Set the jersey descriptions and prepare their text embeddings.

        Args:
            team_1_class_name (str): Description of Team 1's jersey, e.g. "white shirt".
            team_2_class_name (str): Description of Team 2's jersey, e.g. "red shirt".
        """
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        if self.model is None:
            return
        inputs = self.processor(text=[team_1_class_name, team_2_class_name],
                                return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
                jersey is Team 1's rather than Team 2's. Both are None for boxes too small
                to look at.
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
            logits = self.model.logit_scale.exp() * image_features @ self.text_features.T
            probabilities = logits.softmax(dim=1)[:, 0]

        image_features = image_features.float().cpu().numpy()
        probabilities = probabilities.float().cpu().tolist()
        for crop_number, index in enumerate(crop_indexes):
            embeddings[index] = image_features[crop_number]
            team_1_probabilities[index] = probabilities[crop_number]
        return embeddings, team_1_probabilities
