from smartdoc.exceptions import MissingAPIKeyError
from smartdoc.llm.gemini_client import GeminiClient


def test_missing_api_key_has_user_message():
    client = GeminiClient(api_key="", model_name="gemini-2.0-flash")
    try:
        client.require_key()
        assert False, "expected missing key error"
    except MissingAPIKeyError as exc:
        assert "GEMINI_API_KEY" in exc.user_message
