import pandas
from nltk.inference import ResolutionProver
from nltk.sem import Expression

read_expr = Expression.fromstring
_knowledge_base = []


def load_knowledge_base(filepath: str) -> None:
    """
    Loads knowledge base from external csv file into the module
    Args:
        filepath: Path to the CSV KB file
    """
    data = pandas.read_csv(filepath, header=None)
    [_knowledge_base.append(read_expr(row)) for row in data[0]]


def prove(subject: str, object: str) -> str:
    """

    Args:
        subject:
        object:

    Returns:

    """
    expr = read_expr(subject + '(' + object + ')')
    answer = ResolutionProver().prove(expr, _knowledge_base, verbose=True)
    print(answer)
    if answer:
        return 'Correct.'
    else:
        return 'It may not be true.'

# def add_negation(subject: str, object: str) -> None:
#     """
#     Adds negation to knowledge_base, not saving to csv file
#
#     Rick is not here.
#     Args:
#         subject: Term subject
#         object: Term object
#
#     """
#     expr = read_expr(subject + '(-' + object + ')')
#     # >>> ADD SOME CODES HERE to make sure expr does not contradict
#     # with the KB before appending, otherwise show an error message.
#     _knowledge_base.append(expr)
#
#
# def add_conjunction(subject: str, object: str) -> None:
#     """
#     Adds conjunction to knowledge_base, not saving to csv file
#
#     She usually eats at home, because she likes cooking.
#     Args:
#         subject: Term subject
#         object: Term object
#
#     """
#     expr = read_expr('-' + subject + '(' + object + ')')
#     # >>> ADD SOME CODES HERE to make sure expr does not contradict
#     # with the KB before appending, otherwise show an error message.
#     _knowledge_base.append(expr)
#
#
# def add_disjunction(subject: str, object: str) -> None:
#     """
#     Adds disjunction to knowledge_base, not saving to csv file
#
#
#     Args:
#         subject: Term subject
#         object: Term object
#
#     """
#     expr = read_expr(subject + '(' + object + ')')
#     # >>> ADD SOME CODES HERE to make sure expr does not contradict
#     # with the KB before appending, otherwise show an error message.
#     _knowledge_base.append(expr)
#
#
# def add_implication(subject: str, object: str) -> None:
#     """
#     Adds implication to knowledge_base, not saving to csv file
#
#     Tim is british
#     Args:
#         subject: Term subject
#         object: Term object
#
#     """
#     expr = read_expr('->' + subject + '(' + object + ')')
#     # >>> ADD SOME CODES HERE to make sure expr does not contradict
#     # with the KB before appending, otherwise show an error message.
#     _knowledge_base.append(expr)
#
