import logging

import aiml
import autocorrect

logging.basicConfig(level=logging.INFO)  # change critical to info to display information
# Initialize the spell checker we are going to use to autocorrect questions
_spell = autocorrect.Speller("en")
kern = aiml.Kernel()
kern.setTextEncoding(None)


def load_aiml(filepath: str) -> None:
    """
    Loads AIML file into the module

    Args:
        filepath: Path to AIML file

    """
    kern.bootstrap(learnFiles=filepath)


def _get_response(query: str) -> str:
    """
    Get the response from the AIML agent

    Args:
        query: User query

    Returns: Response from AIML agent

    """
    # Still not sure on shall autocorrect should be used here
    # query = query.lower()
    # query_corrected = _spell(query)
    # if query_corrected != query:
    #    logging.info("Corrected {0} into {1}".format(query, query_corrected))
    #    query = query_corrected
    return kern.respond(query)


# Some code to make private members visible in documentation
__pdoc__ = {name: True
            for name, obj in globals().items()
            if name.startswith('_') and callable(obj)}
