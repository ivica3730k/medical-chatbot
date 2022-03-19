import logging

from chatbot import AIMLEngine as AIMLBasedLookup
from chatbot import ClassificationEngine as ImageClassificationLookup
from chatbot import KBEngine as KnowledgeBasedLookup
from chatbot import QAEngine as SimilarityBasedLookup
from chatbot.yolov5 import ObjectDetection as ObjectDetection
from get_answer import get_answer

logging.basicConfig(level=logging.CRITICAL)  # change critical to info to display information

if __name__ == "__main__":
    # AIML Based lookup will use data from our xml file, load it in
    AIMLBasedLookup.load_aiml('./dataset/aiml_set.xml')
    # Similarity based lookup will use data from our csv file, load it in
    SimilarityBasedLookup.load_qa_csv('./dataset/thyroid-problems-qa.csv')
    # Knowledge based lookup will use data from our KB txt file, load it in
    KnowledgeBasedLookup.load_knowledge_base('./dataset/kb_set.txt')
    # Load in our image classification model
    ImageClassificationLookup.load_model('./dataset/pneumonia-detection-model-via-loss-3-0.3.h5',
                                         classes=["Normal", "Pneumonia"])
    # Load in yolov5 neural network for general object detection
    try:
        with open("./dataset/coco.names") as f:
            lines = f.read().splitlines()
        ObjectDetection.load_network('yolov5n.pt', input_width=640, iou_thres=0.5, conf_threshold=0.25, classes=lines)
    except:
        ObjectDetection.load_network('yolov5n.pt', input_width=640, iou_thres=0.5, conf_threshold=0.25)
    while True:
        try:
            user_query = input(">>")
            user_query = user_query.replace(".", "Z2IKzn")
            print(get_answer(user_query))
        except:
            break
