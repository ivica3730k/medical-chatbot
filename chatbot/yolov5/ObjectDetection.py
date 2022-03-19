import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.absolute()))
from models.experimental import attempt_load
from utils.general import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RED_COLOR = (0, 0, 255)

_detection_network = None
_classes = []


class ObjectDetection:
    def __init__(self, model_path, input_width=320, conf_threshold=0.25, iou_thres=0.45):
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        self.input_width = input_width
        self.conf_threshold = conf_threshold
        self.iou_thres = iou_thres

    def detect(self, main_img):
        height, width = main_img.shape[:2]
        new_height = int((((self.input_width / width) * height) // 32) * 32)

        img = cv2.resize(main_img, (self.input_width, new_height))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_threshold, iou_thres=self.iou_thres, classes=None)
        items = []

        if pred[0] is not None and len(pred):
            for p in pred[0]:
                score = np.round(p[4].cpu().detach().numpy(), 2)
                # label = self.classes[int(p[5])]
                label = int(p[5])
                xmin = int(p[0] * main_img.shape[1] / self.input_width)
                ymin = int(p[1] * main_img.shape[0] / new_height)
                xmax = int(p[2] * main_img.shape[1] / self.input_width)
                ymax = int(p[3] * main_img.shape[0] / new_height)

                item = {'label': label,
                        'bbox': [(xmin, ymin), (xmax, ymax)],
                        'score': score
                        }

                items.append(item)

        return items


def load_network(model_path, input_width=320, conf_threshold=0.25, iou_thres=0.45, classes=[]):
    global _detection_network
    global _classes
    _detection_network = ObjectDetection(model_path, input_width, conf_threshold, iou_thres)
    _classes = classes


def _inference_frame(img):
    assert _detection_network is not None, "You first need to load in your neural network via load_network()"
    items = _detection_network.detect(img)
    return items


def _draw_on_frame(frame, items):
    for obj in items:
        label = obj['label']
        if _classes:
            label = _classes[label]
        score = obj['score']
        [(xmin, ymin), (xmax, ymax)] = obj['bbox']
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED_COLOR, 2)
        frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            RED_COLOR, 1, cv2.LINE_AA)
    return frame


def inference_from_file(img_path):
    img = cv2.imread(img_path)
    items = _inference_frame(img)
    img = _draw_on_frame(img, items)
    cv2.imshow("Inference results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inference_on_camera(camera="/dev/video0"):
    camera = cv2.VideoCapture(camera)

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        objs = _inference_frame(frame)
        frame = _draw_on_frame(frame, objs)
        cv2.imshow("Result", frame)
        key = cv2.waitKey(20)
        if key == ord('a'):
            cv2.destroyAllWindows()
            break
