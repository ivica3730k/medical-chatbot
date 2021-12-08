import os

from webchat import app

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

# AIML Based lookup will use data from our xml file, load it in
app.AIMLBasedLookup.load_aiml(dir_path + '/dataset/aiml_set.xml')
# Similarity based lookup will use data from our csv file, load it in
app.SimilarityBasedLookup.load_qa_csv(dir_path + '/dataset/thyroid-problems-qa.csv')
app.app.run()
