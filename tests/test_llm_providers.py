from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from analyst.providers.anthropic import AnthropicProvider
from analyst.providers.google import GoogleProvider, _google_api_error_message
from analyst.providers.openai import OpenAIProvider
from config.settings import AnalystProviderConfig


def _mock_response(payload: dict[str, Any], *, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.is_error = status >= 400
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload)
    response.request = MagicMock()
    response.text = ""
    return response


@pytest.mark.asyncio
async def test_openai_provider_complete_requests_json_response_format() -> None:
    config = AnalystProviderConfig(openai_api_key="sk-test")
    provider = OpenAIProvider(config)
    payload = {"choices": [{"message": {"content": '{"action": "APPROVE"}'}}]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response(payload))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("analyst.providers.openai.httpx.AsyncClient", return_value=mock_client):
        text = await provider.complete("prompt")
    assert "APPROVE" in text
    call_kwargs = mock_client.post.call_args.kwargs
    body = call_kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_anthropic_provider_prefills_assistant_with_brace() -> None:
    config = AnalystProviderConfig(anthropic_api_key="sk-ant")
    provider = AnthropicProvider(config)
    # Anthropic's response continues the prefilled assistant turn — i.e. it
    # returns the body *after* the opening brace. We test that the client
    # re-attaches the brace so callers see a complete object.
    payload = {"content": [{"type": "text", "text": '"action": "VETO"}'}]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response(payload))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("analyst.providers.anthropic.httpx.AsyncClient", return_value=mock_client):
        text = await provider.complete("prompt")
    assert text.startswith("{")
    assert "VETO" in text
    call_kwargs = mock_client.post.call_args.kwargs
    body = call_kwargs["json"]
    messages = body["messages"]
    assert messages[-1] == {"role": "assistant", "content": "{"}


@pytest.mark.asyncio
async def test_google_provider_complete_sets_json_mime_type() -> None:
    config = AnalystProviderConfig(google_api_key="g-test")
    provider = GoogleProvider(config)
    payload = {"candidates": [{"content": {"parts": [{"text": '{"action": "SHRINK"}'}]}}]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response(payload))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("analyst.providers.google.httpx.AsyncClient", return_value=mock_client):
        text = await provider.complete("prompt")
    assert "SHRINK" in text
    call_kwargs = mock_client.post.call_args.kwargs
    body = call_kwargs["json"]
    assert body["generationConfig"]["response_mime_type"] == "application/json"


def test_providers_require_api_keys() -> None:
    config = AnalystProviderConfig()
    with pytest.raises(ValueError, match="ANTHROPIC"):
        AnthropicProvider(config)
    with pytest.raises(ValueError, match="OPENAI"):
        OpenAIProvider(config)
    with pytest.raises(ValueError, match="GOOGLE"):
        GoogleProvider(config)


@pytest.mark.asyncio
async def test_google_provider_routes_aq_key_to_vertex_express() -> None:
    """An ``AQ.*`` key uses the Vertex Express endpoint and includes role."""
    config = AnalystProviderConfig(
        google_api_key="AQ.testkey", model_google="gemini-3.5-flash"
    )
    provider = GoogleProvider(config)
    assert provider.backend == "vertex"
    payload = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response(payload))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("analyst.providers.google.httpx.AsyncClient", return_value=mock_client):
        await provider.complete("p")

    call_args = mock_client.post.call_args
    url = call_args.args[0]
    body = call_args.kwargs["json"]
    assert "aiplatform.googleapis.com" in url
    assert "publishers/google/models/gemini-3.5-flash" in url
    assert body["contents"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_google_provider_routes_aiza_key_to_generative_language() -> None:
    """An ``AIza*`` key still hits the public Generative Language endpoint."""
    config = AnalystProviderConfig(
        google_api_key="AIzaSyTestKey", model_google="gemini-3.5-flash"
    )
    provider = GoogleProvider(config)
    assert provider.backend == "genlang"
    payload = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response(payload))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("analyst.providers.google.httpx.AsyncClient", return_value=mock_client):
        await provider.complete("p")

    url = mock_client.post.call_args.args[0]
    assert "generativelanguage.googleapis.com" in url
    assert "/models/gemini-3.5-flash" in url
    assert "publishers/google" not in url


def test_google_api_error_message_includes_activation_url() -> None:
    response = MagicMock()
    response.status_code = 403
    response.text = ""
    response.json = MagicMock(
        return_value={
            "error": {
                "code": 403,
                "message": "Gemini API has not been used in project 123 before or it is disabled.",
                "details": [
                    {
                        "metadata": {
                            "activationUrl": "https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=123",
                        }
                    }
                ],
            }
        }
    )
    msg = _google_api_error_message(response)
    assert "Enable the Generative Language API" in msg
    assert "generativelanguage.googleapis.com" in msg
