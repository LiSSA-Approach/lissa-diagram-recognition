import os

from detectron2.structures import BoxMode
from src.utils.utils_json import read_json

# Some bounding boxes are very narrow, some arrow heads
# are not fully included in the bounding box. Increase box size to include them.
box_increase = 20


def read_dateset_from(directory: str, filename: str):
    """
    Read dataset from directory and filename.
    Args:
        directory: Directory of dataset.
        filename: Filename of dataset.

    Returns: Dataset records.
    """

    meta = read_json(os.path.join(directory, filename + ".json"))
    dataset_records = []

    for image in meta["images"]:
        image_id = image["id"]

        annotations = list(filter(lambda a: a["image_id"] == image_id, meta["annotations"]))
        for anno in annotations:
            bbox = anno["bbox"]
            anno["bbox_mode"] = BoxMode.XYWH_ABS
            anno["bbox"] = [
                bbox[0] - box_increase,  # x
                bbox[1] - box_increase,  # y
                bbox[2] + box_increase * 2,  # w
                bbox[3] + box_increase * 2,  # h
            ]

        data_record = {
            "file_name": os.path.join(directory, filename, image["file_name"]),
            "image_id": image_id,
            "height": image["height"],
            "width": image["width"],
            "annotations": annotations
        }

        dataset_records.append(data_record)

    return dataset_records
