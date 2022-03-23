import logging

logging.basicConfig(level=logging.CRITICAL)

from chatbot import AIMLEngine as AIMLBasedLookup
from chatbot import AzureObjectDetectionEngine as AzureObjectDetection
from chatbot import ClassificationEngine as ImageClassificationLookup
from chatbot import KBEngine as KnowledgeBasedLookup
from chatbot import QAEngine as SimilarityBasedLookup
from chatbot import WikiApi
from chatbot import YoloV5ObjectDetectionEngine as YoloV5ObjectDetection
from chatbot import TranslateEngine as AzureTranslation

translate_target: str = None


def get_answer(query):
    global translate_target
    aiml_answer = AIMLBasedLookup.get_response(query)  # first use aiml for lookup
    if aiml_answer.split("#")[0] == "inaiml":
        return aiml_answer.split("#")[1]
    if aiml_answer.split("#")[0] == "wikishort":
        ok, wikipedia_answer = WikiApi.get_from_wiki(aiml_answer.split("#")[1])
        if not ok:
            ok, similarity_answer = SimilarityBasedLookup.get_answer(query, confidence_threshold=0.25)
            if ok:
                return similarity_answer
            else:
                return "Weren't able to find your term on Wikipedia nor our QA dataset, please try to rephrase"
        else:
            return wikipedia_answer
    if aiml_answer.split("#")[0] == "wikilong":
        ok, wikipedia_answer = WikiApi.get_from_wiki(aiml_answer.split("#")[1], sentences=10)
        if not ok:
            ok, similarity_answer = SimilarityBasedLookup.get_answer(query, confidence_threshold=0.25)
            if ok:
                return similarity_answer
            else:
                return "Weren't able to find your term on Wikipedia nor our QA dataset, please try to rephrase"
        else:
            return wikipedia_answer
    if aiml_answer.split("#")[1] in ("PRODUCES", "CAUSES", "INCLUDE", "HELPS"):
        a = aiml_answer.split("#")[1].lower()
        b = aiml_answer.split("#")[0].lower()
        c = aiml_answer.split("#")[2].lower()
        validity = KnowledgeBasedLookup.prove_statement(a, b, c)
        return str(validity)
    if aiml_answer.split("#")[0] == "diagnose":
        fixed_arg = aiml_answer.split("#")[1].replace("Z2IKzn", ".")
        fixed_arg = fixed_arg.replace(" ", "")
        try:
            class_label, score = ImageClassificationLookup.classify_from_file(fixed_arg)
            return "Image provided represent a sample of " + class_label + " x-ray image, determined with " + str(round(
                score * 100.0, 2)) + " % certainty."
        except:
            return "Sorry, I was not able to find your file"
    if aiml_answer.split("#")[0] == "objectdetectiononphoto":
        fixed_arg = aiml_answer.split("#")[1].replace("Z2IKzn", ".")
        fixed_arg = fixed_arg.replace(" ", "")
        try:
            YoloV5ObjectDetection.inference_from_file(fixed_arg)
            return
        except:
            return "Error opening your image"
    if aiml_answer.split("#")[0] == "objectdetectiononcamera":
        fixed_arg = aiml_answer.split("#")[1].replace("Z2IKzn", ".")
        fixed_arg = fixed_arg.replace(" ", "")
        try:
            YoloV5ObjectDetection.inference_on_camera(fixed_arg)
            return
        except IOError:
            return "Error reading camera specified"

    if aiml_answer.split("#")[0] == "objectdetectiononphotoviaazure":
        fixed_arg = aiml_answer.split("#")[1].replace("Z2IKzn", ".")
        fixed_arg = fixed_arg.replace(" ", "")
        try:
            AzureObjectDetection.inference_from_file(fixed_arg)
            return
        except:
            return "You have exhausted your Azure resources, please wait a minute!"

    if aiml_answer.split("#")[0] == "translatetarget":
        fixed_arg = aiml_answer.split("#")[1].replace("Z2IKzn", ".")
        fixed_arg = fixed_arg.replace(" ", "")
        translate_target = fixed_arg.lower()
        return "From now on I'll reply in " + translate_target

    if aiml_answer.split("#")[0] == "translatetargetcroatian":
        translate_target = "hr"
        return "From now on I'll reply in Croatian"

    if aiml_answer.split("#")[0] == "translatetargetnone":
        translate_target = None
        return "From now on I'll reply in english"

    if aiml_answer.split("#")[0] == "notinaiml":  # if answer is not in aiml use Similarity based lookup
        ok, similarity_answer = SimilarityBasedLookup.get_answer(query, confidence_threshold=0.25)
        if ok:
            return similarity_answer
        else:
            return "Sorry, please be more precise with your question"


if __name__ == "__main__":
    # AIML Based lookup will use data from our xml file, load it in
    AIMLBasedLookup.load_aiml('./resources/aiml_set.xml')
    # Similarity based lookup will use data from our csv file, load it in
    SimilarityBasedLookup.load_qa_csv('./resources/thyroid-problems-qa.csv')
    # Knowledge based lookup will use data from our KB txt file, load it in
    KnowledgeBasedLookup.load_knowledge_base('./resources/kb_set.txt')
    # Load in our image classification model
    ImageClassificationLookup.load_model('./resources/pneumonia-detection-model-via-loss-3-0.3.h5',
                                         classes=["Normal", "Pneumonia"])
    # Load in _yolov5 neural network for general object detection
    with open("./resources/coco.names") as f:
        lines = f.read().splitlines()
    YoloV5ObjectDetection.load_network('./resources/yolov5n.pt', input_width=640, iou_thres=0.5, conf_threshold=0.25,
                                       classes=lines)
    # Load in azure computer vision service
    AzureObjectDetection.load_credentials("https://n0781349-cv.cognitiveservices.azure.com/",
                                          "10e1d0d7661e495b8f2bc696f74a34fa")
    # Load in azure translation service
    AzureTranslation.load_credentials("5b235f697e86437fbf9d17d3a58b21e6", location="northeurope")
    while True:
        try:
            user_query = input(">>")
            user_query = user_query.replace(".", "Z2IKzn")
            response = get_answer(user_query)
            if translate_target:
                response = AzureTranslation.translate(response, output_language=translate_target)
            if response:
                print(response)
        except:
            break
