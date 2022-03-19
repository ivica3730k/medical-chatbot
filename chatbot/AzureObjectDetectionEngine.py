import requests

_SUBSCRIPTION_KEY = ""
_ENDPOINT = ""
_ANALYSE_URL = ""


def load_credentials(subscription_key, endpoint):
    global _SUBSCRIPTION_KEY
    global _ENDPOINT
    global _ANALYSE_URL
    _SUBSCRIPTION_KEY = subscription_key
    _ENDPOINT = endpoint
    _ANALYSE_URL = endpoint + "vision/v3.2/detect"


def inference_from_file(image_path):
    assert _ANALYSE_URL is not None, "You need to load in your Azure credentials with load_credentials() first!"
    image_data = open(image_path, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': _SUBSCRIPTION_KEY,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Categories,Description,Color'}
    response = requests.post(
        _ANALYSE_URL, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    analysis = response.json()
    return analysis
