import logging

from chatbot import AIMLEngine as AIMLBasedLookup
from chatbot import QAEngine as SimilarityBasedLookup
from chatbot import WikiApi

logging.basicConfig(level=logging.CRITICAL)  # change critical to info to display information

if __name__ == "__main__":
    # AIML Based lookup will use data from our xml file, load it in
    AIMLBasedLookup.load_aiml('./dataset/aiml_set.xml')
    # Similarity based lookup will use data from our csv file, load it in
    SimilarityBasedLookup.load_qa_csv('./dataset/thyroid-problems-qa.csv')
    while True:

        user_query = input(">>")
        aiml_answer = AIMLBasedLookup.get_response(user_query)  # first use aiml for lookup
        if aiml_answer.split("#")[0] == "inaiml":
            print(aiml_answer.split("#")[1])
        if aiml_answer.split("#")[0] == "wikishort":
            ok, wikipedia_answer = WikiApi.get_from_wiki(aiml_answer.split("#")[1])
            if not ok:
                # if not able to find answer on wikipedia give our QA another go
                ok, similarity_answer = SimilarityBasedLookup.get_answer(user_query, confidence_threshold=0.125)
                if ok:
                    print(similarity_answer)
                else:
                    # print("Sorry, please be more precise with your question")
                    print("Weren't able to find your term on Wikipedia nor our QA dataset, please try to rephrase")
            else:
                print(wikipedia_answer)
        if aiml_answer.split("#")[0] == "wikilong":
            ok, wikipedia_answer = WikiApi.get_from_wiki(aiml_answer.split("#")[1], sentences=10)
            if not ok:
                # if not able to find answer on wikipedia give our QA another go
                ok, similarity_answer = SimilarityBasedLookup.get_answer(user_query, confidence_threshold=0.125)
                if ok:
                    print(similarity_answer)
                else:
                    # print("Sorry, please be more precise with your question")
                    print("Weren't able to find your term on Wikipedia nor our QA dataset, please try to rephrase")
            else:
                print(wikipedia_answer)
        if aiml_answer.split("#")[0] == "notinaiml":  # if answer is not in aiml use Similarity based lookup
            ok, similarity_answer = SimilarityBasedLookup.get_answer(user_query, confidence_threshold=0.125)
            if ok:
                print(similarity_answer)
            else:
                print("Sorry, please be more precise with your question")
