import torch
import cv2
import numpy as np
from PIL import Image
import supervision as sv
import torchvision.transforms as transforms
from typing import List, Tuple
import os

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
groundingdino_path = os.path.join(project_root, "GroundingDINO")
if groundingdino_path not in sys.path:
    sys.path.insert(0, groundingdino_path)

class GroundingDINODetector:
    """
    Object detector using Grounding DINO for waste detection
    """

    def __init__(self,
                 model_config: str = os.path.join(project_root, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                 model_checkpoint: str = os.path.join(project_root, "GroundingDINO/weights/groundingdino_swint_ogc.pth"),
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Grounding DINO detector
        """
        self.device = device
        self.model = self.load_model(model_config, model_checkpoint)
        self.text_prompt = "waste . trash . garbage . rubbish . litter"
        self.box_threshold = 0.35
        self.text_threshold = 0.25

        # Image preprocessing for classifier later
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def set_detection_parameters(self, box_threshold: float = None, text_threshold: float = None):
        if box_threshold is not None:
            self.box_threshold = box_threshold
        if text_threshold is not None:
            self.text_threshold = text_threshold

    def load_model(self, model_config: str, model_checkpoint: str):
        """
        Load Grounding DINO model
        """
        try:
            from GroundingDINO.groundingdino.util.inference import Model

            model = Model(
                model_config_path=model_config,
                model_checkpoint_path=model_checkpoint,
                device=self.device
            )
            print("✅ Grounding DINO model loaded successfully")
            return model
        except ImportError:
            raise ImportError("Grounding DINO not installed. "
                              "Install from https://github.com/IDEA-Research/GroundingDINO")
        except Exception as e:
            raise Exception(f"Failed to load Grounding DINO model: {e}")

    def detect(self, image: np.ndarray) -> Tuple[sv.Detections, np.ndarray, List[str]]:
        """
        Detect waste objects in image
        """
        # Convert BGR to RGB once
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run Grounding DINO (expects BGR image!)
        detections, phrases = self.model.predict_with_caption(
            image=image,  # pass original BGR image
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # Create annotator with INDEX-based colors (since no class_id)
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)

        # Create nice labels
        labels = [
            f"{phrase} {conf:.2f}"
            for phrase, conf in zip(phrases, detections.confidence)
        ]

        # Annotate
        annotated_image = box_annotator.annotate(
            scene=image_rgb.copy(),
            detections=detections
        )

        # Convert back to BGR for OpenCV compatibility
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Return phrases as string labels
        label_strings = [str(p) for p in phrases]

        return detections, annotated_image_bgr, label_strings

    def extract_object_crops(self, image: np.ndarray, detections) -> List[Tuple[np.ndarray, List[int]]]:
        """
        Extract cropped images of detected objects
        """
        crops = []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for bbox in detections.xyxy:
            x1, y1, x2, y2 = bbox.astype(int)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            crop = image_rgb[y1:y2, x1:x2]

            if crop.size > 0:
                crops.append((crop, [x1, y1, x2, y2]))

        return crops

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess crop for classification
        """
        pil_image = Image.fromarray(crop)
        return self.transform(pil_image).unsqueeze(0).to(self.device)