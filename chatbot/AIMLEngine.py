"""
AIML engine module is used to perform the AIML based functionalities of the chat bot.
The patters of the conversation are loaded in from pre-defined in an xml file.
"""

import aiml
import autocorrect
import logging

# logging.basicConfig(level=logging.CRITICAL)  # change critical to info to display information
# Initialize the spell checker we are going to use to autocorrect questions
_spell = autocorrect.Speller("en")
_aiml_kernel = aiml.Kernel()
_aiml_kernel.setTextEncoding(None)


def load_aiml(filepath: str) -> None:
    """
    Loads AIML file into the module

    Args:
        filepath: Path to AIML file

    """
    _aiml_kernel.bootstrap(learnFiles=filepath)


def get_response(query: str) -> str:
    """
    Get the response from the AIML agent

    Args:
        query: User query

    Returns: Response from AIML agent

    """
    # Still not sure on shall autocorrect should be used here
    query = query.lower()
    query_corrected = _spell(query)
    if query_corrected != query:
        logging.info("Corrected {0} into {1}".format(query, query_corrected))
        query = query_corrected
    return _aiml_kernel.respond(query)


# Some code to make private members visible in documentation
__pdoc__ = {name: True
            for name, obj in globals().items()
            if name.startswith('_') and callable(obj)}
