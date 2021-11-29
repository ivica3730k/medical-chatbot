import logging
from typing import Tuple

import autocorrect
import wikipedia

# logging.basicConfig(level=logging.CRITICAL)  # change critical to info to display information

# Initialize the spell checker we are going to use to autocorrect questions
_spell = autocorrect.Speller("en")


def get_from_wiki(topic: str, sentences=3) -> Tuple[bool, str]:
    """
    Get the information from wikipedia on provided topic using python wikipedia module

    Args:
        topic: Topic of interest
        sentences: Number of sentences on the topic

    Returns: Validity status, Details about the topic

    """
    topic = topic.lower()
    topic_corrected = _spell(topic)
    if topic_corrected != topic:
        logging.info("Corrected {0} into {1}".format(topic, topic_corrected))
        topic = topic_corrected
    try:
        return True, wikipedia.summary(topic, sentences=sentences, auto_suggest=False)
    except (wikipedia.DisambiguationError, wikipedia.PageError):
        return False, ""


# Some code to make private members visible in documentation
__pdoc__ = {name: True
            for name, obj in globals().items()
            if name.startswith('_') and callable(obj)}
