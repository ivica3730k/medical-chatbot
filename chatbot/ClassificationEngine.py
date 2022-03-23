"""Classification Engine used to provide local image classification from pre
trained CNN."""
import logging
import os
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf

_dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

_model = None
_model_size = ()
_classes = []


def load_model(model_filepath: str = _dir_path + "model.h5",
               input_size: tuple = (224, 224),
               classes: list = []) -> None:
    """Loads image classification neural network into program.

    Classes argument can be omitted, in that case classification returns class ID.
    Args:
        model_filepath: Filepath to the model
        input_size: Image size for the input layer
        TODO: Expand the tuple size to include also the channels, i.e. (224,224,3)
        classes: List of classes model can detect, if omitted classification returns class ID

    Returns:
    """
    global _model
    global _model_size
    global _classes
    _model = tf.keras.models.load_model(model_filepath)
    _model_size = input_size
    _classes = classes
    logging.info(_model.summary())


def classify_from_file(img_path: str) -> Tuple[str, float]:
    """Loads the image from the filesystem and returns classification results.

    The classification network must be loaded in previously via load_model() function
    Args:
        img_path: Path to the image on the filesystem

    Returns: Classification results as tuple (class_id / class_name , score)
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
            return str(class_id), score


def classify_from_image(img: np.array) -> Tuple[str, float]:
    """Takes the numpy array as image input and performs classification,
    returning classification results.

    The classification network must be loaded in previously via load_model() function

    Args:
        img: CV2 Style numpy image

    Returns: Classification results as tuple (class_id / class_name , score)
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
            return str(class_id), score


# Some code to make private members visible in documentation
__pdoc__ = {
    name: True
    for name, obj in globals().items()
    if name.startswith('_') and callable(obj)
}
