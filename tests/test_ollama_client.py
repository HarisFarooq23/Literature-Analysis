"""
Unit tests for ollama_client.py's reliability fixes:
  - check_ollama_ready: fails fast and clearly on an unreachable server or a
    missing model, without needing a real Ollama server.
  - OllamaExtractor._call: ConnectionError must propagate immediately (no
    redundant format='json' retry against a server that's simply down);
    ollama.ResponseError (server rejected the schema format) must fall back
    to format='json'.

All ollama.Client interaction is mocked -- these tests never touch a network.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import ollama
import pytest

from llm_extract.config import ModelsConfig
from llm_extract.ollama_client import OllamaExtractor, check_ollama_ready


def _fake_list_response(model_names: list[str]):
    return SimpleNamespace(models=[SimpleNamespace(model=name) for name in model_names])


@pytest.fixture
def models_config() -> ModelsConfig:
    return ModelsConfig(
        primary="granite4.1:8b", cross_check="gemma3:12b", tie_breaker="nemotron-3-nano:30b",
        host="http://localhost:11434",
    )


def test_check_ollama_ready_passes_when_all_models_present(mocker, models_config):
    mock_client = MagicMock()
    mock_client.list.return_value = _fake_list_response(["granite4.1:8b", "gemma3:12b", "nemotron-3-nano:30b"])
    mocker.patch("llm_extract.ollama_client.ollama.Client", return_value=mock_client)

    check_ollama_ready(models_config, ["granite4.1:8b", "gemma3:12b", "nemotron-3-nano:30b"])  # should not raise


def test_check_ollama_ready_raises_clearly_on_unreachable_server(mocker, models_config):
    mock_client = MagicMock()
    mock_client.list.side_effect = ConnectionError("refused")
    mocker.patch("llm_extract.ollama_client.ollama.Client", return_value=mock_client)

    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        check_ollama_ready(models_config, ["granite4.1:8b"])


def test_check_ollama_ready_raises_clearly_on_missing_model(mocker, models_config):
    mock_client = MagicMock()
    mock_client.list.return_value = _fake_list_response(["granite4.1:8b"])  # cross_check/tie_breaker missing
    mocker.patch("llm_extract.ollama_client.ollama.Client", return_value=mock_client)

    with pytest.raises(RuntimeError, match="not found"):
        check_ollama_ready(models_config, ["granite4.1:8b", "gemma3:12b", "nemotron-3-nano:30b"])


class _DummyFields:
    """Minimal stand-in with the one method _call() needs."""
    @staticmethod
    def model_json_schema():
        return {"type": "object", "properties": {}}


def test_connection_error_propagates_without_json_fallback(mocker, models_config):
    """A down/unreachable server must not trigger a second (also doomed)
    request with format='json' -- that's the exact bug being fixed."""
    extractor = OllamaExtractor(host=models_config.host, models_config=models_config)
    mock_generate = MagicMock(side_effect=ConnectionError("refused"))
    extractor.client.generate = mock_generate

    with pytest.raises(ConnectionError):
        extractor._call("granite4.1:8b", "prompt", _DummyFields)

    assert mock_generate.call_count == 1  # no fallback attempt was made


def test_response_error_triggers_json_fallback(mocker, models_config):
    """A server that actively rejects schema-constrained generation (e.g. an
    older Ollama version) should still get the format='json' fallback."""
    extractor = OllamaExtractor(host=models_config.host, models_config=models_config)

    response_error = ollama.ResponseError("bad request", status_code=400)
    mock_generate = MagicMock(side_effect=[response_error, {"response": '{"ok": true}'}])
    extractor.client.generate = mock_generate

    result = extractor._call("granite4.1:8b", "prompt", _DummyFields)

    assert result == '{"ok": true}'
    assert mock_generate.call_count == 2
    second_call_kwargs = mock_generate.call_args_list[1].kwargs
    assert second_call_kwargs["format"] == "json"


def test_unrelated_exception_does_not_trigger_json_fallback(mocker, models_config):
    """Anything other than ConnectionError/ollama.ResponseError should
    propagate too -- the fallback is specifically for 'server rejected the
    schema', not a catch-all."""
    extractor = OllamaExtractor(host=models_config.host, models_config=models_config)
    mock_generate = MagicMock(side_effect=ValueError("something unrelated"))
    extractor.client.generate = mock_generate

    with pytest.raises(ValueError):
        extractor._call("granite4.1:8b", "prompt", _DummyFields)

    assert mock_generate.call_count == 1
