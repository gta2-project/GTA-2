"""Request-shape tests for stateless CaP model calls."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from benchmarks.cap.model import CapModelClient, GEMINI_REQUEST_TIMEOUT_MS


class FakeOpenAIResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="KEYPOINT_QUERIES = {}\ndef policy(ctx):\n    return None\n",
            output=[],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            model="openai-test-served",
            id="response-1",
        )


class FakeGeminiModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text="KEYPOINT_QUERIES = {}\ndef policy(ctx):\n    return None\n",
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=20,
            ),
            model_version="gemini-test-served",
            response_id="response-2",
        )


class CapModelTests(unittest.TestCase):
    def make_image(self, root):
        path = root / "scene.png"
        Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path)
        return path

    def test_openai_matches_gta_reasoning_config_without_conversation_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            responses = FakeOpenAIResponses()
            client = CapModelClient(
                provider="openai",
                model="openai-test",
                api_key="test-key",
                openai_client=SimpleNamespace(responses=responses),
            )
            result = client.generate("system", "task", self.make_image(Path(tmpdir)))

            request = responses.calls[0]
            self.assertEqual(request["reasoning"], {"effort": "high"})
            self.assertNotIn("temperature", request)
            self.assertNotIn("conversation", request)
            self.assertEqual(request["instructions"], "system")
            self.assertEqual(result.metadata["reasoning_effort"], "high")
            self.assertFalse(result.metadata["stateful"])

    def test_gemini_matches_gta_default_decoding_without_chat_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models = FakeGeminiModels()
            client = CapModelClient(
                provider="gemini",
                model="gemini-test",
                api_key="test-key",
                gemini_client=SimpleNamespace(models=models),
            )
            result = client.generate("system", "task", self.make_image(Path(tmpdir)))

            request = models.calls[0]
            self.assertEqual(request["model"], "gemini-test")
            self.assertEqual(request["config"].system_instruction, "system")
            self.assertIsNone(request["config"].temperature)
            self.assertEqual(len(request["contents"]), 2)
            self.assertFalse(result.metadata["stateful"])

    def test_gemini_default_client_has_500_second_request_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models = FakeGeminiModels()
            with patch("google.genai.Client") as client_factory:
                client_factory.return_value = SimpleNamespace(models=models)
                client = CapModelClient(
                    provider="gemini",
                    model="gemini-test",
                    api_key="test-key",
                )
                client.generate("system", "task", self.make_image(Path(tmpdir)))

            options = client_factory.call_args.kwargs["http_options"]
            self.assertEqual(options.timeout, GEMINI_REQUEST_TIMEOUT_MS)


if __name__ == "__main__":
    unittest.main()
