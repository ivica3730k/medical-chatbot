import json
import uuid

import requests

_SUBSCRIPTION_KEY: str = None
_ENDPOINT: str = None
_LOCATION: str = None
_PATH: str = '/translate'


def load_credentials(subscription_key, location="global", endpoint="https://api.cognitive.microsofttranslator.com"):
    """

    Args:
        subscription_key:
        location:
        endpoint:

    Returns:

    """
    global _SUBSCRIPTION_KEY
    global _ENDPOINT
    global _LOCATION
    _SUBSCRIPTION_KEY = subscription_key
    _LOCATION = location
    _ENDPOINT = endpoint


def translate(input_text, input_language="en", output_language="hr"):
    """

    Args:
        input_text:
        input_language:
        output_language:

    Returns:

    """
    assert _SUBSCRIPTION_KEY or _LOCATION or _ENDPOINT is not None, "You need to load in your credentials using " \
                                                                    "load_credentials() function "
    params = {
        'api-version': '3.0',
        'from': input_language,
        'to': [output_language]
    }
    constructed_url = _ENDPOINT + _PATH

    headers = {
        'Ocp-Apim-Subscription-Key': _SUBSCRIPTION_KEY,
        'Ocp-Apim-Subscription-Region': _LOCATION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': input_text
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    a = json.dumps(response)
    b = json.loads(a)
    b = b[0]
    return b["translations"][0]["text"]
