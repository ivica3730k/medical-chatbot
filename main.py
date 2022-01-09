import logging

from chatbot import AIMLEngine as AIMLBasedLookup
from chatbot import KBEngine as KnowledgeBasedLookup
from chatbot import QAEngine as SimilarityBasedLookup
from get_answer import get_answer

logging.basicConfig(level=logging.INFO)  # change critical to info to display information

if __name__ == "__main__":
    # AIML Based lookup will use data from our xml file, load it in
    AIMLBasedLookup.load_aiml('./dataset/aiml_set.xml')
    # Similarity based lookup will use data from our csv file, load it in
    SimilarityBasedLookup.load_qa_csv('./dataset/thyroid-problems-qa.csv')
    # Knowledge based lookup will use data from our KB txt file, load it in
    KnowledgeBasedLookup.load_knowledge_base('./dataset/kb_set.txt')
    while True:
        try:
            user_query = input(">>")
            print(get_answer(user_query))
        except:
            break
