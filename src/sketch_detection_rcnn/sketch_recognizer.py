from typing import Union, Any

import cv2
import os
import json

import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from .roi_heads import SketchROIHeads  # noqa


class SketchRecognizer:

    def __init__(self, model_path: str, device="cpu") -> None:
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(model_path, "cfg.json"))
        cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.DEVICE = device

        self._predictor = DefaultPredictor(cfg)

        # Read classes labels from json file
        with open(os.path.join(model_path, "classes.json"), "r") as file:
            self._labels = json.load(file)

    def predict_form_bytes(self, image: bytes) -> list[dict[str, Union[list, Any]]]:
        npdata = np.fromstring(image, np.uint8)
        image = cv2.imdecode(npdata, cv2.IMREAD_COLOR)

        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = self._predictor(image)

        instances = outputs["instances"]
        boxes = instances.pred_boxes
        scores = instances.scores
        prediction_classes = instances.pred_classes
        keypoints = None

        if hasattr(instances, "pred_arrow_keypoints"):
            keypoints = instances.pred_arrow_keypoints

        predictions = []

        idx = 0
        for box in boxes:
            entry = {
                "box": [tensor.item() for tensor in box],
                "confidence": scores[idx].item(),
                "class": self._labels[prediction_classes[idx].item()],
                "keypoints": keypoints[idx].tolist() if keypoints is not None else None
            }

            predictions.append(entry)
            idx += 1

        return predictions

    def predict_form_file(self, image_path: str):
        with open(image_path, "rb") as file:
            data = file.read()

        return self.predict_form_bytes(data)
