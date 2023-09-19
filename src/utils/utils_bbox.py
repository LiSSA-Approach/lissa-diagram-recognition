def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box_a: bounding box a
        box_b: bounding box b

    Returns:
        IoU of two bounding boxes
    """

    # Determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # Compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    if box_a_area + box_b_area - inter_area == 0:
        return 0
    else:
        return inter_area / float(box_a_area + box_b_area - inter_area)
