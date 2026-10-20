"""Prompt extraction regression tests; no provider, camera, or robot operations."""

import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from agents.base_agent import BaseAgent
from agents.checkpoint import encode
from communication import MessageBus
from config import Config


ROOT = Path(__file__).resolve().parents[1]
# Captured from the effective strings in config.py BEFORE the structural move.
PROMPTS = [
    ("task_decomposer", "TaskDecomposer", "TaskDecomposerAgent", 11432, "77e41df9fd35c895eaa00aabccad4bfd781ca7c709a89dd189e7275279115945"),
    ("skill_generator", "SkillGenerator", "SkillGeneratorAgent", 9900, "ce87bafb1d872d808c4d4df03456ae3fdd828e326d74c768ce1f99cf988807ab"),
    ("parameter_setter", "ParameterSetter", "ParameterSettingAgent", 3171, "70850c1e8ea123860e9f9f6e37dde6b2ccb9e6bd5b88bd73f659a1bc8c92ed22"),
    ("vision_module", "VisionModule", "VisionModuleAgent", 1200, "a105d4ab5209b4728caf123614218330924078452a8c9ad3cf365a089e3bbe94"),
]


@pytest.fixture(autouse=True)
def forbid_external_operations():
    with (
        patch("agents.base_agent.OpenAI", side_effect=AssertionError("Unexpected OpenAI client")),
        patch("agents.base_agent.genai.Client", side_effect=AssertionError("Unexpected Gemini client")),
        patch.object(BaseAgent, "_init_camera", side_effect=AssertionError("Unexpected camera")),
        patch.object(MessageBus, "publish", side_effect=AssertionError("Unexpected pipeline dispatch")),
    ):
        yield


@pytest.mark.parametrize("stem,name,class_name,byte_count,digest", PROMPTS)
def test_system_prompt_bytes_match_before_extraction(stem, name, class_name, byte_count, digest):
    prompt = importlib.import_module(f"agents.prompts.{stem}").SYSTEM_PROMPT
    payload = prompt.encode("utf-8")
    assert len(payload) == byte_count
    assert hashlib.sha256(payload).hexdigest() == digest


@pytest.mark.parametrize("stem,name,class_name,byte_count,digest", PROMPTS)
def test_fresh_agent_uses_and_exports_its_default_prompt(stem, name, class_name, byte_count, digest):
    agent_class = getattr(importlib.import_module(f"agents.{stem}"), class_name)
    prompt = importlib.import_module(f"agents.prompts.{stem}").SYSTEM_PROMPT
    bus = MessageBus()
    bus.lazy_clients = True
    with patch.object(Config, "get_agent_config", return_value={
        "provider": "openai", "model": "test-model", "api_key": None, "use_vision": True,
    }):
        agent = agent_class(bus)
    assert agent.name == name
    assert agent.system_prompt == prompt
    assert agent.export_checkpoint()["config"]["system_prompt"] == prompt


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("stem,name,class_name,byte_count,digest", PROMPTS)
def test_restored_agent_keeps_saved_prompt_instead_of_current_default(
    stem, name, class_name, byte_count, digest, provider,
):
    agent_class = getattr(importlib.import_module(f"agents.{stem}"), class_name)
    saved_prompt = "Previously approved prompt for " + name + "  \n"
    saved = {
        "config": {"provider": provider, "model": "saved-model", "use_vision": True,
                   "system_prompt": saved_prompt, "generation_settings": {"temperature": 0.2}},
        "history": encode([]),
        "state": encode({"state": {"approved": True}}),
    }
    bus = MessageBus()
    bus.agent_snapshots = {name: saved}
    with patch.object(Config, "get_agent_config", side_effect=AssertionError("Overriding saved config")):
        agent = agent_class(bus)
        agent.restore_checkpoint_state(saved)
    assert agent.system_prompt == saved_prompt
    assert agent.state["approved"] is True
    assert agent.export_checkpoint()["config"] == saved["config"]
    assert not bus.message_history


def test_configuration_no_longer_owns_prompts_or_error_analyzer_settings():
    assert not any(name.endswith("_SYSTEM_PROMPT") for name in vars(Config))
    assert not any(name.startswith("ERROR_ANALYZER_") for name in vars(Config))


def test_config_import_stays_independent_of_agents_and_provider_sdks():
    script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"agents", "openai", "google", "scene"}:
        raise AssertionError("Config imported " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import config
assert config.Config.CAMERA_BACKEND in {"file", "ros1", "ros2"}
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
