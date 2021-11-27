# Similarity-based functionality is implemented in QAPairs module. Import it as SimilarityBasedLookup module
import QAPairs as SimilarityBasedLookup

if __name__ == "__main__":
    # Similarity based lookup will use data from our csv file, load it in
    SimilarityBasedLookup.load_qa_csv('./thyroid-problems-qa.csv')
    while True:
        try:
            user_query = input(">>")
            print(SimilarityBasedLookup.get_answer(user_query))
        except:
            print("Bye")
            break
