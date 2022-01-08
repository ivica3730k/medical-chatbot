import logging
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

from flask import Flask, render_template, request, send_from_directory
from flask import jsonify

sys.path.append('..')
sys.path.append('../')
import chatbot.AIMLEngine as AIMLBasedLookup
import chatbot.QAEngine as SimilarityBasedLookup
import chatbot.KBEngine as KnowledgeBasedLookup
from get_answer import get_answer

logging.basicConfig(level=logging.CRITICAL)  # change critical to info to display information
app = Flask(__name__)
# do the top level import if possible, required for heroku hosting
try:
    # AIML Based lookup will use data from our xml file, load it in
    AIMLBasedLookup.load_aiml(dir_path + '../dataset/aiml_set.xml')
    # Similarity based lookup will use data from our csv file, load it in
    SimilarityBasedLookup.load_qa_csv(dir_path + '../dataset/thyroid-problems-qa.csv')
    # Knowledge based lookup will use data from our KB txt file, load it in
    KnowledgeBasedLookup.load_knowledge_base(dir_path + '../dataset/kb_set.txt')
except:
    pass


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/docs/<path:filename>', methods=['GET', 'POST'])
def index(filename):
    filename = filename or 'index.html'
    if request.method == 'GET':
        return send_from_directory(dir_path + "../docs/html/chatbot", filename)

    return jsonify(request.data)


@app.route("/process", methods=["POST"])
def response():
    msg = request.form["msg"]
    # return(msg)
    msg = get_answer(msg)
    return msg


if __name__ == "__main__":
    # AIML Based lookup will use data from our xml file, load it in
    AIMLBasedLookup.load_aiml(dir_path + '../dataset/aiml_set.xml')
    # Similarity based lookup will use data from our csv file, load it in
    SimilarityBasedLookup.load_qa_csv(dir_path + '../dataset/thyroid-problems-qa.csv')
    # Knowledge based lookup will use data from our KB txt file, load it in
    KnowledgeBasedLookup.load_knowledge_base(dir_path + '../dataset/kb_set.txt')
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, host='0.0.0.0', debug=True)
