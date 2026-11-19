"""Hugging Face's dataset viewer (datasets-server): column types and first rows.

Only https://datasets-server.huggingface.co is called. Any answer but 200 —
the viewer cannot read the dataset (501), it is gated (401), or it does not
exist — is `ViewerUnavailable`: the caller shows the card without a preview."""
import requests

from ..errors import ProviderRateLimited

VIEWER = "https://datasets-server.huggingface.co/"
TIMEOUT = 10


class ViewerUnavailable(Exception):
    """The viewer gave no preview: it cannot read the dataset (501), it is gated (401), the
    dataset is not there (404), or the viewer is down (ViewerDown)."""


class ViewerDown(ViewerUnavailable):
    """The viewer itself did not answer: no connection, a server error other than 501, or an
    answer it could not read. A dataset's copy may stand in for it (#121); a permanent "no"
    from the viewer is not this."""


def get(endpoint, **params):
    try:
        response = requests.get(VIEWER + endpoint, params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise ViewerDown(str(exc)) from exc
    if response.status_code == 429:
        raise ProviderRateLimited("Hugging Face is limiting requests")
    if response.status_code >= 500 and response.status_code != 501:
        raise ViewerDown(f"The viewer answered {response.status_code}")
    if response.status_code != 200:
        raise ViewerUnavailable(f"The viewer answered {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise ViewerDown("Unreadable answer") from exc
