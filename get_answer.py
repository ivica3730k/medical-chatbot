from chatbot import AIMLEngine as AIMLBasedLookup
from chatbot import KBEngine as KnowledgeBasedLookup
from chatbot import QAEngine as SimilarityBasedLookup
from chatbot import WikiApi


def get_answer(user_query):
    aiml_answer = AIMLBasedLookup.get_response(user_query)  # first use aiml for lookup
    if aiml_answer.split("#")[0] == "inaiml":
        return (aiml_answer.split("#")[1])
    if aiml_answer.split("#")[0] == "wikishort":
        ok, wikipedia_answer = WikiApi.get_from_wiki(aiml_answer.split("#")[1])
        if not ok:
            # if not able to find answer on wikipedia give our QA another go
            ok, similarity_answer = SimilarityBasedLookup.get_answer(user_query, confidence_threshold=0.25)
            if ok:
                return (similarity_answer)
            else:
                # return("Sorry, please be more precise with your question")
                return ("Weren't able to find your term on Wikipedia nor our QA dataset, please try to rephrase")
        else:
            return (wikipedia_answer)
    if aiml_answer.split("#")[0] == "wikilong":
        ok, wikipedia_answer = WikiApi.get_from_wiki(aiml_answer.split("#")[1], sentences=10)
        if not ok:
            # if not able to find answer on wikipedia give our QA another go
            ok, similarity_answer = SimilarityBasedLookup.get_answer(user_query, confidence_threshold=0.25)
            if ok:
                return (similarity_answer)
            else:
                # return("Sorry, please be more precise with your question")
                return ("Weren't able to find your term on Wikipedia nor our QA dataset, please try to rephrase")
        else:
            return (wikipedia_answer)
    if aiml_answer.split("#")[1] in ("PRODUCES", "CAUSES", "INCLUDE", "HELPS"):
        a = aiml_answer.split("#")[1].lower()
        b = aiml_answer.split("#")[0].lower()
        c = aiml_answer.split("#")[2].lower()
        validity = KnowledgeBasedLookup.prove_statement(a, b, c)
        return str(validity)
    if aiml_answer.split("#")[0] == "notinaiml":  # if answer is not in aiml use Similarity based lookup
        ok, similarity_answer = SimilarityBasedLookup.get_answer(user_query, confidence_threshold=0.25)
        if ok:
            return (similarity_answer)
        else:
            return ("Sorry, please be more precise with your question")
