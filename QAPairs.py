"""
QAPair module used to perform similarity-based question lookup to provide the user with the best possible answer.
The similarity-based  functionality is based on a set of pre-defined Q/As in a CSV file.
The similarity-based component is based on the bag-of-words model, tf/idf, and cosine similarity.
"""

import csv

import autocorrect
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Initialize the spell checker we are going to use to autocorrect are questions and
# answers when we load them into our QRPair class
_spell = autocorrect.Speller("en")
vectorizer = TfidfVectorizer(stop_words='english')

_questions = []
_answers = []


def load_qa_pair(question: str, answer: str) -> None:
    """
    Load the QA pair into QAPair module
    :param question: Question (as String)
    :param answer: Answer (as String)
    """
    # global _questions
    # global _answers
    _questions.append(_spell(question))
    _questions.append(_spell(answer))


def _get_real_question_id(question: str) -> int:
    """
    Perform the similarity-based lookup for the real question from our QA list based on the user-entered question.

    Similarity based lookup based on bag of words and cosine similarity is used to determine the question the user most
    likely wanted to ask. User question is appended to the question list and sparse matrix is created and passed to the
    pandas data frame. Afterwards the cosine similarity is calculated using sklearn, our question is removed from
    the question list and similarity list (as it's score is always 1.00). Finally, the index with biggest
    score is returned.

    :param question: User question to apply similarity-based lookup on (as String)
    :return: Index of question in _questions list best matching to User question input
    """
    question = _spell(question)
    _questions.append(question)
    sparse_matrix = vectorizer.fit_transform(_questions)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix)
    cs = cosine_similarity(df, df)[len(_questions) - 1]
    cs = np.delete(cs, -1)  # remove our question from the scores
    _questions.pop()
    return np.argmax(cs)


def get_answer(question: str) -> str:
    """
    Interface function used to obtain the answer for the question provided, running similarity-based lookup
    in the background.
    :param question: User question (as String)
    :return: Answer to user question (as String)
    """
    question_id = _get_real_question_id(question)
    return _answers[question_id]


def load_qa_csv(filepath: str) -> None:
    """
    Function used to load qa csv file into module
    :param filepath: Path to csv file
    """
    with open(filepath, encoding="latin") as csvfile:
        for l in csv.reader(csvfile, quotechar='"', delimiter=',',
                            quoting=csv.QUOTE_ALL, skipinitialspace=True):
            _questions.append(_spell(BeautifulSoup(l[0], "lxml").get_text(strip=True)))
            # _answers.append(_spell(BeautifulSoup(l[1], "lxml").get_text(strip=True)))
            _answers.append((BeautifulSoup(l[1], "lxml").get_text(strip=True)))


def print_qa_pairs() -> None:
    """
    Print QA Pairs for debug purposes
    """
    for i in range(0, len(_answers)):
        print(_questions[i], '>>', _answers[i])
