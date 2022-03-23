"""The QAEngine module is used to perform similarity-based question lookup to
provide the user with the best possible answer.

The similarity-based  functionality is based on a set of pre-defined
Q/As in a CSV file. The similarity-based component is based on the bag-
of-words model, tf/idf, and cosine similarity.
"""

import csv
import logging
from typing import Tuple

import autocorrect
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# logging.basicConfig(level=logging.CRITICAL)  # change critical to info to display information

# Initialize the spell checker we are going to use to autocorrect questions
_spell = autocorrect.Speller("en")
vectorizer = TfidfVectorizer(stop_words='english')

_questions = []
_answers = []


def load_qa_pair(question: str, answer: str) -> None:
    """Load the QA pair into QAPair module.

    Args:
        question: Question
        answer: Answer
    """
    _questions.append(
        _spell(BeautifulSoup(question, "lxml").get_text(strip=True)).lower())
    _answers.append((BeautifulSoup(answer, "lxml").get_text(strip=True)))


def _get_real_question_id(question: str,
                          confidence_threshold: float = 0.00
                          ) -> Tuple[bool, int]:
    """Perform the similarity-based lookup for the real question from our QA
    list based on the user-entered question.

    Similarity based lookup based on bag of words and cosine similarity is used to determine the question the user most
    likely wanted to ask. User question is appended to the question list and sparse matrix is created and passed to the
    pandas data frame. Afterwards the cosine similarity is calculated using sklearn, our question is removed from
    the question list and similarity list (as it's score is always 1.00). Finally, the index with biggest
    score is returned. Note, in order to exclude useless answers, the confidence threshold is applied.

    Args:
        question: User question to apply similarity-based lookup on
        confidence_threshold: Confidence threshold for cosine-similarity. Used to exclude useless answer


    Returns: Validity status, Index of question in _questions list best matching to User question input
    """
    question = _spell(question)
    _questions.append(question)
    sparse_matrix = vectorizer.fit_transform(_questions)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix)
    cs = cosine_similarity(df, df)[len(_questions) - 1]
    cs = np.delete(cs, -1)  # remove our question from the scores
    _questions.pop()
    index = np.argmax(cs)
    if cs[index] >= confidence_threshold:
        return True, index
    else:
        return False, index


def get_answer(question: str,
               confidence_threshold: float = 0.25) -> Tuple[bool, str]:
    """Interface function used to obtain the answer for the question provided,
    running similarity-based lookup in the background.

    Args:
        question: User question
        confidence_threshold: Confidence threshold for cosine-similarity. Used to exclude useless answer

    Returns:
        Validity status ,answer to user question
    """
    if not _questions:
        logging.critical("Trying to get answer without QA dataset loaded")
        return False, ""
    question = question.lower()
    question_corrected = _spell(question)
    if question_corrected != question:
        logging.info("Corrected {0} into {1}".format(question,
                                                     question_corrected))
        question = question_corrected
    ok, question_id = _get_real_question_id(question, confidence_threshold)
    if ok:
        return True, _answers[question_id]
    else:
        return False, ""


def load_qa_csv(filepath: str) -> None:
    """Function used to load qa csv file into module.

    Args:
        filepath: Path to csv file
    """
    with open(filepath, encoding="latin") as csvfile:
        for l in csv.reader(csvfile,
                            quotechar='"',
                            delimiter=',',
                            quoting=csv.QUOTE_ALL,
                            skipinitialspace=True):
            load_qa_pair(l[0], l[1])


def print_qa_pairs() -> None:
    """Print QA Pairs for debug purposes."""
    for i in range(0, len(_answers)):
        print(_questions[i], '>>', _answers[i])


# Some code to make private members visible in documentation
__pdoc__ = {
    name: True
    for name, obj in globals().items()
    if name.startswith('_') and callable(obj)
}
