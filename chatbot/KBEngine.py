"""
The KBEngine module is used to provide the logical reasoning capabilities with the help of the NLTK library.
Initial logic is loaded into the chatbot from the Knowledge base txt file.
"""

from nltk.inference import ResolutionProver
from nltk.sem import Expression

read_expr = Expression.fromstring
_knowledge_base = []


def load_knowledge_base(filepath: str) -> None:
    """
    Loads knowledge base from external txt file into the module
    Args:
        filepath: Path to the txt KB file
    """
    file = open(filepath, "r")
    lines = file.readlines()
    [_knowledge_base.append(read_expr(row)) for row in lines]


def prove_statement(a: str, b: str, c: str) -> bool:
    """
    Prove statement using NLTK Inference Resolution Prover

    Format for proving > a(b,c)
    Args:
        a: Word
        b: Word
        c: Word

    Returns:
        Validity of statement
    """
    expr = read_expr(a + '(' + b + ',' + c + ')')
    answer_validity = ResolutionProver().prove(expr, _knowledge_base, verbose=False)
    return answer_validity
