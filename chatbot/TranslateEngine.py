import json
import uuid

import requests

_SUBSCRIPTION_KEY: str = None
_ENDPOINT: str = None
_LOCATION: str = None
_PATH: str = '/translate'


def load_credentials(
        subscription_key: str,
        location: str = "global",
        endpoint: str = "https://api.cognitive.microsofttranslator.com"
) -> None:
    """
    Load in credentials for Azure Translator Service
    Args:
        subscription_key: Subscription key from Azure
        location: Location for the translator, default global
        endpoint: Endpoint for translation service

    """
    global _SUBSCRIPTION_KEY
    global _ENDPOINT
    global _LOCATION
    _SUBSCRIPTION_KEY = subscription_key
    _LOCATION = location
    _ENDPOINT = endpoint


def translate(input_text: str,
              input_language: str = "en",
              output_language: str = "hr") -> str:
    """
    Translate the input text to target language
    Args:
        input_text: Input text
        input_language: Input language code, default en
        output_language: Output language code, default hr

    Returns: Translated text

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

    body = [{'text': input_text}]

    request = requests.post(constructed_url,
                            params=params,
                            headers=headers,
                            json=body)
    response = request.json()
    a = json.dumps(response)
    b = json.loads(a)
    b = b[0]
    return b["translations"][0]["text"]
