import argparse
import os

from flask import Flask, jsonify, request

from src.sketch_detection_rcnn.sketch_recognizer import SketchRecognizer

parser = argparse.ArgumentParser(description='Sketch Recognition Server')

parser.add_argument(
    '--port',
    type=int,
    required=False,
    help='Port to run the server on'
)

parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='Model directory'
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

app = Flask("SketchRecognition")
sketch_recognizer = SketchRecognizer(args.model, device=args.device)


@app.route('/', methods=['POST'])
def upload():
    file = request.files['file']
    byt = file.stream.read()
    predictions = sketch_recognizer.predict_form_bytes(byt)

    return jsonify(predictions)


if __name__ == '__main__':
    port = args.port

    if not port:
        port = os.environ.get('PORT', 8000)

    # if port is str, convert to int
    if isinstance(port, str):
        port = int(port)

    app.run(debug=True, host='0.0.0.0', port=port)
