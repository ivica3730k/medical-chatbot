import autocorrect
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize the spell checker we are going to use to autocorrect are questions and
# answers when we load them into our QRPair class
_spell = autocorrect.Speller("en")

questions = [
    "What is a thyroid gland?",
    "How common are thyroid disorders?",
    "How many people have thyroid problems?",
    "Where is thyroid gland?",
    "What does my thyroid gland do?",
    "What do my thyroid hormones do for me?",
    "What can go wrong with my thyroid?",
    "What are the most common symptoms of the most common thyroid disorders that I might experience",
    "What other disorders are there?",
    "What is hypothyroidism?",
    "What is hyperthyroidism?",
    "How common is thyroid cancer?",
    "What is post-partum thyroiditis?",
    "What causes a thyroid disorder?",
    "How are thyroid disorders diagnosed?",
    "Can thyroid disorders be treated?",
    "What are early signs of thyroid problems?",
    "What are hypothyroidism signs and symptoms that can occur?",
    "At what age do thyroid problems tend to start?",
    "How can I cure my thyroid without medication?",
    "What foods help heal thyroid?",
    "Can thyroid be cured by exercise?",
    "Is coffee bad for thyroid?",
    "How can I check my thyroid at home?",
    "Do thyroid problems affect sex life?",
]

vectorizer = TfidfVectorizer(stop_words='english')
questions.append("Can exercise help")
sparse_matrix = vectorizer.fit_transform(questions)
# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix)
# display(df)
cosine_similarity = cosine_similarity(df, df)[len(questions) - 1]  # similarity with first index, our question
print(cosine_similarity)
cosine_similarity = np.delete(cosine_similarity, -1)  # remove our question from the scores
print(questions[(np.argmax(cosine_similarity))])
