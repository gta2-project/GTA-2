"""Four-agent initialization and direct feedback; never call models or hardware."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import pytest

from agents.base_agent import BaseAgent
from api.app import create_app
from api.service import AGENT_NAMES, GtaWebService
from config import Config


ACTIVE_AGENTS = {"TaskDecomposer", "SkillGenerator", "ParameterSetter", "VisionModule"}


@pytest.fixture
def service(tmp_path):
    def agent_config(name):
        assert name in ACTIVE_AGENTS, f"Unexpected agent initialization: {name}"
        return {"provider": "openai", "model": "offline-test", "api_key": None, "use_vision": False}

    with (
        patch.object(Config, "validate", return_value=True),
        patch.object(Config, "get_agent_config", side_effect=agent_config),
        patch("agents.base_agent.OpenAI", side_effect=AssertionError("Provider request")),
        patch("agents.base_agent.genai.Client", side_effect=AssertionError("Provider request")),
        patch.object(BaseAgent, "_init_camera", side_effect=AssertionError("Camera request")),
    ):
        workflow = GtaWebService(tmp_path, scoped=True)
        assert workflow.initialize()
        yield workflow


def test_only_four_agents_are_initialized_and_reported(service):
    assert AGENT_NAMES == ACTIVE_AGENTS
    assert set(service.agents) == ACTIVE_AGENTS
    assert set(service.health()["agent_models"]) == ACTIVE_AGENTS
    assert set(service.message_bus.subscribers) == ACTIVE_AGENTS | {"broadcast"}


@pytest.mark.parametrize("name,method,response", [
    ("TaskDecomposer", "refine_subtasks", {"subtasks": [{"id": 1}]}),
    ("SkillGenerator", "refine_skills", {"skills": [{"subtask_id": 1}]}),
    ("ParameterSetter", "refine_parameters", [{"subtask_id": 1}]),
    ("VisionModule", "refine_keypoints", {"keypoint_locations": {}}),
])
def test_panel_feedback_still_targets_the_selected_agent(service, name, method, response):
    agent = service.agents[name]
    agent.current_task = "approved task"
    agent.current_skills = [{"subtask_id": 1}]
    service._state["grounding_revision"] = service._state["revision"]
    app = create_app(service, initialize_on_startup=False)
    with (
        patch.object(service.message_bus, "run_background") as background,
        patch.object(agent, method, return_value=response) as refine,
        patch.object(agent, "notify_agent") as notify,
        TestClient(app) as client,
    ):
        result = client.post(f"/api/agents/{name}/message", json={"message": "Change only this item"})
        assert result.status_code == 202
        background.assert_called_once_with(service._run_agent_message, name, "Change only this item")
        service._run_agent_message(name, "Change only this item")
        refine.assert_called_once_with("Change only this item")
        if name == "ParameterSetter":
            notify.assert_called_once_with(agent_name="VisionModule", notification_type="parameters_completed",
                                           data={"skills": response, "action": "refined"})


def test_removed_agent_has_no_live_api_or_package_export(service):
    import agents

    assert not hasattr(agents, "ErrorAnalyzerAgent")
    assert not hasattr(service, "_run_error_analyzer")
    app = create_app(service, initialize_on_startup=False)
    with patch.object(service.message_bus, "run_background") as background, TestClient(app) as client:
        result = client.post("/api/agents/ErrorAnalyzer/message", json={"message": "unused"})
        assert result.status_code == 400
        assert "Unknown agent" in result.json()["detail"]
        background.assert_not_called()


def test_workflow_reset_does_not_require_removed_agent(service):
    resets = {}
    for name, agent in service.agents.items():
        agent.reset_conversation = resets[name] = Mock()
    service._reset_workflow_state("new task")
    assert service.state()["task"] == "new task"
    for reset in resets.values():
        reset.assert_called_once_with()
