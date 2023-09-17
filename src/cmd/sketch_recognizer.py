import argparse

from src.utils.utils_json import print_json
from src.sketch_detection_rcnn.sketch_recognizer import SketchRecognizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize sketches')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model directory'
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Image path'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        required=False,
        help='Device to use for prediction'
    )

    args = parser.parse_args()

    sketch_recognizer = SketchRecognizer(args.model, device=args.device)
    prediction = sketch_recognizer.predict_form_file(args.image)
    print_json(prediction)
