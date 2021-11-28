# Similarity-based functionality is implemented in QAPairs module. Import it as SimilarityBasedLookup module
from chatbot import QAPairs as SimilarityBasedLookup

if __name__ == "__main__":
    # Similarity based lookup will use data from our csv file, load it in
    SimilarityBasedLookup.load_qa_csv('./thyroid-problems-qa.csv')
    while True:
        try:
            user_query = input(">>")
            ok, answer = SimilarityBasedLookup.get_answer(user_query, confidence_threshold=0.125)
            if ok:
                print(answer)
            else:
                print("Sorry, answer to your question is not in my QA list")
        except:
            print("Bye")
            break
