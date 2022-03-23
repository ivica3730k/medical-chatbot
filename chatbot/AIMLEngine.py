"""AIML engine module is used to perform the AIML based functionalities of the
chat bot.

The patters of the conversation are loaded in from pre-defined in an xml
file.
"""

import aiml

_aiml_kernel = aiml.Kernel()
_aiml_kernel.setTextEncoding(None)


def load_aiml(filepath: str) -> None:
    """Loads AIML file into the module.

    Args:
        filepath: Path to AIML file
    """
    _aiml_kernel.bootstrap(learnFiles=filepath)


def get_response(query: str) -> str:
    """Get the response from the AIML agent.

    Args:
        query: User query

    Returns: Response from AIML agent
    """
    return _aiml_kernel.respond(query)


# Some code to make private members visible in documentation
__pdoc__ = {
    name: True
    for name, obj in globals().items()
    if name.startswith('_') and callable(obj)
}
