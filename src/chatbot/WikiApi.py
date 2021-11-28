import logging

import autocorrect
import requests
import wikipedia

logging.basicConfig(level=logging.INFO)  # change critical to info to display information

# Initialize the spell checker we are going to use to autocorrect are questions and
# answers when we load them into our QRPair class
_spell = autocorrect.Speller("en")


def _get_from_wiki_using_request(topic: str) -> str:
    """
    Simple function used to obtain data on topic from wikipedia without using python wikipedia module

    Args:
        topic: Topic to get information on

    Returns: Details about the topic

    """
    topic = topic.lower()
    topic_corrected = _spell(topic)
    if topic_corrected != topic:
        logging.info("Corrected {0} into {1}".format(topic, topic_corrected))
        topic = topic_corrected
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': topic,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))
    # print(page['extract'])
    return page['extract']


def get_from_wiki_using_api(topic: str, sentences=3) -> str:
    """
    Get the information from wikipedia on provided topic using python wikipedia module

    Args:
        topic: Topic of interest
        sentences: Number of sentences on the topic

    Returns: Details about the topic

    """
    topic = topic.lower()
    topic_corrected = _spell(topic)
    if topic_corrected != topic:
        logging.info("Corrected {0} into {1}".format(topic, topic_corrected))
        topic = topic_corrected
    try:
        return wikipedia.summary(topic, sentences=sentences, auto_suggest=False)
    except wikipedia.DisambiguationError:
        logging.info("Cant process query using wikipedia module, using manual requests route")
        return _get_from_wiki_using_request(topic)


# Some code to make private members visible in documentation
__pdoc__ = {name: True
            for name, obj in globals().items()
            if name.startswith('_') and callable(obj)}
