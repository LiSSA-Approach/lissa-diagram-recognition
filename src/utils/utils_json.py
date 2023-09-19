import json
import os


def read_json(filename: str):
    """
    Read json file
    :param filename: json file path
    :return: json data
    """

    json_file = os.path.join(filename)
    with open(json_file) as f:
        img_annotations = json.load(f)

    return img_annotations


def print_json(data: any, tag=str("")):
    """
    Pretty print json data to console
    :param data: json data
    :param tag: optional tag to print before json data
    :return: None
    """

    out = json.dumps(data, indent=2, sort_keys=True)
    if tag != "":
        print(tag, out)
    else:
        print(out)


def write_json(filename: str, data: any):
    """
    Write json data to file
    :param filename: json file path
    :param data: json data
    :return: None
    """

    with open(filename, "w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
