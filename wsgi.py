from webchat import app
# AIML Based lookup will use data from our xml file, load it in
app.AIMLBasedLookup.load_aiml('./dataset/aiml_set.xml')
# Similarity based lookup will use data from our csv file, load it in
app.SimilarityBasedLookup.load_qa_csv('./dataset/thyroid-problems-qa.csv')
app.app.run()
