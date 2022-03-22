import logging
import os

import cv2
import numpy as np
import tensorflow as tf

_dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

_model = None
_model_size = ()
_classes = []


def load_model(model_filepath: str = _dir_path + "model.h5", model_size: tuple = (224, 224), classes: list = []):
    """

    Args:
        model_filepath:
        model_size:
        classes:

    Returns:

    """
    global _model
    global _model_size
    global _classes
    _model = tf.keras.models.load_model(model_filepath)
    _model_size = model_size
    _classes = classes
    logging.info(_model.summary())


def classify_from_file(img_path):
    """

    Args:
        img_path:

    Returns:

    """
    assert _model is not None, "You need to load your model first using load_model()"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = tf.image.per_image_standardization(img)
    x = np.expand_dims(img, axis=0)
    single_prediction = _model.predict(x=x)
    for i in range(0, single_prediction.shape[0]):
        class_id = np.argmax(single_prediction[i])
        score = single_prediction.flatten()[np.argmax(single_prediction[i])]
        if len(_classes):
            return _classes[class_id], score
        else:
            return class_id, score


def classify_from_image(img):
    """

    Args:
        img:

    Returns:

    """
    assert _model is not None, "You need to load your model first using load_model()"
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = tf.image.per_image_standardization(img)
    x = np.expand_dims(img, axis=0)
    single_prediction = _model.predict(x=x)
    for i in range(0, single_prediction.shape[0]):
        class_id = np.argmax(single_prediction[i])
        score = single_prediction.flatten()[np.argmax(single_prediction[i])]
        if len(_classes):
            return _classes[class_id], score
        else:
            return class_id, score
