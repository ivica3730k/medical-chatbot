import random

import cv2
import requests

_SUBSCRIPTION_KEY = None
_ENDPOINT = None
_ANALYSE_URL = None


def _get_random_color():
    """
    Returns:

    """
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def load_credentials(endpoint, subscription_key):
    """

    Args:
        endpoint:
        subscription_key:

    Returns:

    """
    global _SUBSCRIPTION_KEY
    global _ENDPOINT
    global _ANALYSE_URL
    _SUBSCRIPTION_KEY = subscription_key
    _ENDPOINT = endpoint
    _ANALYSE_URL = endpoint + "vision/v3.2/detect"


def inference_from_file(image_path):
    """

    Args:
        image_path:

    Returns:

    """
    assert _ANALYSE_URL is not None, "You need to load in your Azure credentials with load_credentials() first!"
    image_data = open(image_path, "rb").read()
    image = cv2.imread(image_path)
    headers = {'Ocp-Apim-Subscription-Key': _SUBSCRIPTION_KEY,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Categories,Description,Color'}
    response = requests.post(
        _ANALYSE_URL, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    analysis = response.json()
    labeled_frame = _draw_on_frame(image, analysis)
    cv2.imshow("Inference results", labeled_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _draw_on_frame(frame, items):
    """

    Args:
        frame:
        items:

    Returns:

    """
    for obj in items["objects"]:
        label = obj["object"]
        score = obj["confidence"]
        rectangle = obj["rectangle"]
        xmin = rectangle["x"]
        ymin = rectangle["y"]
        w = rectangle["w"]
        h = rectangle["h"]
        xmax = xmin + w
        ymax = ymin + h
        color = _get_random_color()
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            color, 1, cv2.LINE_AA)
    return frame
