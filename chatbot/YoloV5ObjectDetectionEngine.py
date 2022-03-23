"""Yolov5 object detection engine based on
https://github.com/ultralytics/yolov5 used to provide object recognition
capabilities locally."""
import random
import sys
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Dict

import cv2
import numpy
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent.absolute()) + "/_yolov5")

from _yolov5.models.experimental import attempt_load
from _yolov5.utils.general import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RED_COLOR = (0, 0, 255)

_detection_network = None
_classes = []
_class_colors = []


class ObjectDetection:

    def __init__(self,
                 model_path: str,
                 input_width: int = 320,
                 conf_threshold: float = 0.25,
                 iou_thres: float = 0.45) -> None:
        """Initialise the object detecion class.

        Args:
            model_path: Path to the yolov5 model on filesystem
            input_width: Input width to the model, default 640
            conf_threshold: Confidence threshold for non-maxima suppression
            iou_thres: IoU threshold for non-maxim suppression
        """
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        self.input_width = input_width
        self.conf_threshold = conf_threshold
        self.iou_thres = iou_thres

    def detect(self, input_image: numpy.array) -> List[Dict]:
        """Run the input image trough YoloV5 Object detection neural network.

        Args:
            input_image: Input image as numpy array

        Returns: Inference results, list of dictionaries containing bounding boxes,labels and scores
        """
        height, width = input_image.shape[:2]
        new_height = int((((self.input_width / width) * height) // 32) * 32)

        img = cv2.resize(input_image, (self.input_width, new_height))
        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred,
                                   conf_thres=self.conf_threshold,
                                   iou_thres=self.iou_thres,
                                   classes=None)
        items = []

        if pred[0] is not None and len(pred):
            for p in pred[0]:
                score = np.round(p[4].cpu().detach().numpy(), 2)
                # label = self.classes[int(p[5])]
                label = int(p[5])
                xmin = int(p[0] * input_image.shape[1] / self.input_width)
                ymin = int(p[1] * input_image.shape[0] / new_height)
                xmax = int(p[2] * input_image.shape[1] / self.input_width)
                ymax = int(p[3] * input_image.shape[0] / new_height)

                item = {
                    'label': label,
                    'bbox': [(xmin, ymin), (xmax, ymax)],
                    'score': score
                }

                items.append(item)

        return items


def _get_random_color() -> Tuple[int, int, int]:
    """Get a random color for the cv2 plots.

    Returns: Tuple, with 3 values ranging from 0 to 255
    """
    return random.randint(0, 255), random.randint(0,
                                                  255), random.randint(0, 255)


def load_network(model_path: str,
                 input_width: int = 640,
                 conf_threshold: float = 0.25,
                 iou_thres: float = 0.45,
                 classes: list = []) -> None:
    """Load the Yolov5 neural network for the inference.

    Classes parameter can be omitted. If not provided, drawing on the frame will use class numbers
    instead of class labels.

    Args:
        model_path: Path to the yolov5 model on filesystem
        input_width: Input width to the model, default 640
        conf_threshold: Confidence threshold for non-maxima suppression
        iou_thres: IoU threshold for non-maxim suppression
        classes: Array holding list of classes
    """
    global _detection_network
    global _classes
    global _class_colors
    _detection_network = ObjectDetection(model_path, input_width,
                                         conf_threshold, iou_thres)
    _classes = classes
    if classes:
        for i in range(0, len(classes)):
            _class_colors.append(_get_random_color())
    else:
        for i in range(0, 1000):
            _class_colors.append(_get_random_color())


def _inference_frame(img: numpy.array) -> List[Dict]:
    """Runs the inference from the supplied cv2 frame.

    The detection network is must be loaded in via load_network() function before attempting inference

    Args:
        img: CV2 Type frame to run the inference on

    Returns: Inference results
    """
    assert _detection_network is not None, "You first need to load in your neural network via load_network()"
    items = _detection_network.detect(img)
    return items


def _draw_on_frame(frame: numpy.array, items: List[Dict]) -> numpy.array:
    """Draws provided inference results on the provided frame.

    Args:
        frame: Frame to draw on
        items: Inference results to draw

    Returns: Frame with inference results drawn on it
    """
    for obj in items:
        label = obj['label']
        color = _class_colors[label]
        if _classes:
            label = _classes[label]
        score = obj['score']
        [(xmin, ymin), (xmax, ymax)] = obj['bbox']
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin, ymin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1,
                            cv2.LINE_AA)
    return frame


def inference_from_file(img_path: str) -> None:
    """Runs the inference on frame after loading it in from the path provided.

    The inference results are displayed to the user via CV2 window

    Args:
        img_path: Path to the image on the filesystem
    """
    img = cv2.imread(img_path)
    items = _inference_frame(img)
    img = _draw_on_frame(img, items)
    cv2.imshow("Inference results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inference_on_camera(camera: str = "/dev/video0") -> None:
    """Runs the inference from the frames incoming via camera provided.

    The inference results are displayed to the user via CV2 window


    Args:
        camera: System path to camera, /dev/video* on Linux
    """
    camera = cv2.VideoCapture(camera)

    while True:
        ok, frame = camera.read()
        if not ok:
            raise IOError
        objs = _inference_frame(frame)
        frame = _draw_on_frame(frame, objs)
        cv2.imshow("Result", frame)
        key = cv2.waitKey(20)
        if key == ord('a'):
            cv2.destroyAllWindows()
            break


# Some code to make private members visible in documentation
__pdoc__ = {
    name: True
    for name, obj in globals().items()
    if name.startswith('_') and callable(obj)
}
