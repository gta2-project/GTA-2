"""Offline, exact checkpoint round trips and workspace/API boundaries."""

import copy
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agents.base_agent import BaseAgent
from agents.checkpoint import encode, decode
from api.library_app import create_library_app
from api.checkpoints import LibraryManager, read_json, write_json
from api.service import AGENT_NAMES, GtaWebService
from communication import Message, MessageBus
from config import Config


PLAN = [{'subtask_id': 1, 'task': 'Approach the box', 'controllers': [
    {'controller': 'PosAlign(g_eef_TCP, g_Obox_Ptop_Lcenter, a_world_z, offset_height)',
     'offset_parameters': [{'name': 'offset_height', 'value': 0.1}]}],
    'keypoints': [{'name': 'g_Obox_Ptop_Lcenter', 'description': 'Center of the top face'}]}]


def snapshots():
    return {name: {
        'config': {'provider': 'gemini' if name == 'VisionModule' else 'openai',
                   'model': 'saved-model', 'system_prompt': 'exact prompt ' + name,
                   'use_vision': False, 'generation_settings': {'temperature': 0.2}},
        'history': encode([{'role': 'user', 'parts': [{'inline_data': {'mime_type': 'image/png', 'data': b'complete image'},
                                                    'thought_signature': b'opaque signature'}]}]
                          if name == 'VisionModule' else [{'type': 'message', 'id': 'msg_admin', 'role': 'user',
                                                         'status': 'completed', 'content': [{'type': 'input_text', 'text': 'plan this'}]}]),
        'state': encode({'state': {'received_messages': [Message('User', name, 'feedback', {'text': 'keep this'})]},
                         'current_skills': copy.deepcopy(PLAN), 'current_task': 'Approach the box',
                         'last_image_np': np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
                         'last_image_size': (3, 2), 'current_queries': ['box_top_center'],
                         'current_keypoints': ['g_Obox_Ptop_Lcenter'],
                         'keypoint_descriptions': {'g_Obox_Ptop_Lcenter': 'Center of the top face'}}),
    } for name in AGENT_NAMES}


@pytest.fixture
def prepared(tmp_path):
    service = GtaWebService(tmp_path, scoped=True)
    service._create_agents(snapshots())
    service.initialized = True
    service._state.update(task='Approach the box', subtasks=[{'task': 'Approach the box'}],
                          skills=copy.deepcopy(PLAN), parameters=copy.deepcopy(PLAN),
                          keypoints={'g_Obox_Ptop_Lcenter': {'point': [1, 2]}},
                          revision=3, grounding_revision=3, script_revision=3,
                          script='print("reviewed")', script_path=str(tmp_path / 'robot_script.py'))
    (tmp_path / 'skills_output.json').write_text(json.dumps(PLAN))
    (tmp_path / 'keypoint_locations.json').write_text(json.dumps(service._state['keypoints']))
    (tmp_path / 'robot_script.py').write_text(service._state['script'])
    for name in ('environment.jpg', 'resized_img.png', 'annotated_keypoints.png'):
        Image.new('RGB', (3, 2), (21, 45, 77)).save(tmp_path / name)
    service.message_bus.message_history.append(Message('User', 'TaskDecomposer', 'feedback', {'text': 'approved'}))
    service._emit_event({'message_type': 'feedback', 'content': {'text': 'reviewed plan'}})
    return service, LibraryManager(tmp_path, service)


def test_roundtrip_is_exact_independent_and_offline(prepared):
    admin, manager = prepared
    saved_agents = {name: agent.export_checkpoint() for name, agent in admin.agents.items()}
    original = copy.deepcopy(admin.state())
    checkpoint = manager.save('Box library', notes='Place box on table', visible=True)
    directory, manifest, payload = manager.store.load(checkpoint['id'])
    assert manifest['version'] == 1
    assert payload['workflow']['parameters'] == PLAN
    serialized = (directory / 'state.json').read_text()
    assert 'api_key' not in serialized and '.env' not in serialized
    with patch('agents.base_agent.OpenAI', side_effect=AssertionError('Network on load')), \
         patch('agents.base_agent.genai.Client', side_effect=AssertionError('Network on load')), \
         patch.object(MessageBus, 'publish', side_effect=AssertionError('Pipeline replay on load')), \
         patch.object(BaseAgent, '_init_camera', side_effect=AssertionError('Camera on load')):
        first = manager.open(checkpoint['id'])
        a = manager.session
        assert a.state()['parameters'] == original['parameters']
        assert a.state()['keypoints'] == original['keypoints']
        assert a.state()['messages'] == original['messages']
        assert a.state()['events'] == original['events']
        assert not a.state()['grounding_ready'] and not a.state()['script_ready']
        assert a._artifact_file_path('environment.jpg').read_bytes() == admin._artifact_file_path('environment.jpg').read_bytes()
        assert {name: agent.export_checkpoint() for name, agent in a.agents.items()} == saved_agents
        second = manager.open(checkpoint['id'])
        b = manager.session
    assert first['id'] != second['id']
    a.agents['TaskDecomposer'].current_skills[0]['task'] = 'mutated'
    a.agents['VisionModule'].last_image_np[:] = 0
    assert b.agents['TaskDecomposer'].current_skills == PLAN
    np.testing.assert_array_equal(b.agents['VisionModule'].last_image_np, np.arange(18).reshape(2, 3, 3))
    assert admin.state() == original
    assert (a.root_dir / 'session.json').exists()
    assert manager.store.load(checkpoint['id'])[2]['workflow']['parameters'] == PLAN


def replace_checkpoint_agents_for_test(manager, checkpoint_id, mutate):
    """Build older/corrupt fixtures only inside pytest's temporary directory."""
    directory = manager.store.directory(checkpoint_id)
    payload = read_json(directory / 'state.json')
    mutate(payload['agents'])
    write_json(directory / 'state.json', payload)
    manifest = read_json(directory / 'manifest.json')
    manifest['state_sha256'] = hashlib.sha256((directory / 'state.json').read_bytes()).hexdigest()
    write_json(directory / 'manifest.json', manifest)
    return directory


def test_older_checkpoint_restores_four_agents_without_recreating_error_analyzer(prepared):
    admin, manager = prepared
    saved_agents = {name: agent.export_checkpoint() for name, agent in admin.agents.items()}
    checkpoint = manager.save('Older five-agent checkpoint', visible=True)

    def add_retired_agent(agents):
        agents['ErrorAnalyzer'] = {
            'config': {'provider': 'openai', 'model': 'retired-model', 'system_prompt': '',
                       'use_vision': False, 'generation_settings': {}},
            'history': encode([{'role': 'user', 'content': 'historical feedback'}]),
            'state': encode({'state': {}, 'feedback_history': ['original record']}),
        }

    directory = replace_checkpoint_agents_for_test(manager, checkpoint['id'], add_retired_agent)
    original_files = {path.name: path.read_bytes() for path in directory.iterdir()}
    with (
        patch('agents.base_agent.OpenAI', side_effect=AssertionError('Network on load')),
        patch('agents.base_agent.genai.Client', side_effect=AssertionError('Network on load')),
        patch.object(MessageBus, 'publish', side_effect=AssertionError('Pipeline replay on load')),
        patch.object(BaseAgent, '_init_camera', side_effect=AssertionError('Camera on load')),
    ):
        manager.open(checkpoint['id'])
        session = manager.session
        assert set(session.agents) == AGENT_NAMES
        assert 'ErrorAnalyzer' not in session.message_bus.subscribers
        assert {name: agent.export_checkpoint() for name, agent in session.agents.items()} == saved_agents
        revised_id = manager.store.save(session, 'Revised four-agent checkpoint')
    assert set(manager.store.load(revised_id)[2]['agents']) == AGENT_NAMES
    assert 'ErrorAnalyzer' in manager.store.load(checkpoint['id'])[2]['agents']
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original_files


@pytest.mark.parametrize('change', ['missing_active_agent', 'unknown_extra_agent'])
def test_retired_agent_support_does_not_weaken_checkpoint_completeness(prepared, change):
    _, manager = prepared
    checkpoint = manager.save('Strict agent validation')

    def mutate(agents):
        agents['ErrorAnalyzer'] = {}
        if change == 'missing_active_agent':
            del agents['ParameterSetter']
        else:
            agents['UnknownAgent'] = {}

    replace_checkpoint_agents_for_test(manager, checkpoint['id'], mutate)
    with pytest.raises(ValueError, match='Checkpoint is missing agent state'):
        manager.store.load(checkpoint['id'])


def test_hidden_metadata_immutable_output_and_restart(prepared):
    admin, manager = prepared
    checkpoint = manager.save('Hidden')
    assert manager.store.list(True) == []
    with pytest.raises(ValueError, match='not enabled'):
        manager.open(checkpoint['id'])
    state_bytes = (manager.store.directory(checkpoint['id']) / 'state.json').read_bytes()
    manager.store.update(checkpoint['id'], name='Visible', visible=True)
    restarted = LibraryManager(admin.root_dir, GtaWebService(admin.root_dir))
    assert restarted.session is None
    assert restarted.execution.snapshot()['status'] == 'idle'
    assert restarted.store.list(True)[0]['name'] == 'Visible'
    restarted.open(checkpoint['id'])
    assert (manager.store.directory(checkpoint['id']) / 'state.json').read_bytes() == state_bytes
    manager.open(checkpoint['id'], target='admin')
    assert manager.admin is not admin
    assert manager.admin.state()['script_ready']
    assert Path(manager.admin.state()['script_path']).parent == manager.admin.root_dir


def test_busy_handlers_save_switch_and_atomic_failures(prepared):
    admin, manager = prepared
    saved = manager.save('before', visible=True)
    entered, release = threading.Event(), threading.Event()
    worker = admin.message_bus.run_background(lambda: (entered.set(), release.wait(2)))
    entered.wait(1)
    assert admin.state()['busy_stage'] is None
    with pytest.raises(RuntimeError, match='busy'):
        manager.save('during')
    with pytest.raises(RuntimeError, match='busy'):
        manager.open(saved['id'])
    release.set(); worker.join(1)
    before = list(manager.store.root.iterdir())
    with patch.object(admin.agents['VisionModule'], 'export_checkpoint', side_effect=RuntimeError('History unavailable')):
        with pytest.raises(RuntimeError, match='History unavailable'):
            manager.save('invalid')
    with patch('api.checkpoints.shutil.copy2', side_effect=OSError('Disk full')):
        with pytest.raises(OSError, match='Disk full'):
            manager.save('partial')
    assert list(manager.store.root.iterdir()) == before


def test_openai_pages_and_batches_preserve_native_items(prepared):
    admin, _ = prepared
    agent = admin.agents['TaskDecomposer']
    history = [{'type': 'message', 'id': f'msg_{i}', 'role': 'assistant', 'status': 'completed',
                'phase': 'final_answer', 'content': [{'type': 'output_text', 'text': str(i), 'annotations': []}]} for i in range(123)]
    history += [{'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque'}]
    history += [{'type': 'message', 'role': 'user', 'id': 'msg_image', 'status': 'completed',
                 'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,YQ==', 'detail': 'auto'}]}]

    class Pages:
        def __iter__(self):
            for start in range(0, len(history), 100):
                yield from history[start:start + 100]

    old = Mock()
    old.conversations.items.list.return_value = Pages()
    agent.client, agent.conversation_id, agent._pending_history = old, 'original', None
    exported = agent.export_checkpoint()
    assert exported['history'] == history
    old.conversations.items.list.assert_called_once_with('original', order='asc', limit=100,
        include=['message.input_image.image_url', 'reasoning.encrypted_content'])
    bus = MessageBus(); bus.agent_snapshots = {'TaskDecomposer': exported}
    fork = BaseAgent('TaskDecomposer', 'ignored', bus)
    new = Mock(); new.conversations.create.return_value = SimpleNamespace(id='independent')
    with patch.object(Config, 'OPENAI_API_KEY', 'test-only'), patch('agents.base_agent.OpenAI', return_value=new):
        fork._ensure_client()
        fork._ensure_client()
    assert fork.conversation_id == 'independent'
    calls = new.conversations.items.create.call_args_list
    assert [len(call.kwargs['items']) for call in calls] == [20] * 6 + [5]
    inserted = [item for call in calls for item in call.kwargs['items']]
    assert inserted[:124] == history[:124]
    assert inserted[-1]['content'] == history[-1]['content']
    assert all(call.args == ('independent',) for call in calls)
    assert agent.conversation_id == 'original'


def test_gemini_native_bytes_and_failed_fork_remain_retryable(prepared):
    admin, _ = prepared
    agent = admin.agents['VisionModule']
    expected = copy.deepcopy(agent._pending_history)
    client = Mock()
    with patch.object(Config, 'GEMINI_API_KEY', 'test-only'), patch('agents.base_agent.genai.Client', return_value=client):
        agent._ensure_client()
    args = client.chats.create.call_args.kwargs
    assert args['history'] == expected
    assert args['config'] == {'system_instruction': 'exact prompt VisionModule', 'temperature': 0.2}
    client.chats.create.return_value.get_history.return_value = expected
    assert decode(agent.export_checkpoint()['history']) == expected
    client.chats.create.return_value.get_history.assert_called_once_with(curated=False)

    agent = admin.agents['TaskDecomposer']
    before = copy.deepcopy(agent._pending_history)
    client = Mock(); client.conversations.create.return_value.id = 'partial'
    client.conversations.items.create.side_effect = RuntimeError('insert failed')
    with patch.object(Config, 'OPENAI_API_KEY', 'test-only'), patch('agents.base_agent.OpenAI', return_value=client):
        with pytest.raises(RuntimeError, match='insert failed'):
            agent._ensure_client()
    assert agent._pending_history == before and agent.conversation_id is None


def test_session_api_load_reset_scope_and_readiness(prepared):
    admin, _ = prepared
    app = create_library_app(admin, initialize_on_startup=False, root=admin.root_dir)
    with TestClient(app) as client:
        saved = client.post('/api/checkpoints', json={'name': 'API', 'visible': True}).json()
        assert client.get('/api/library/session').json() is None
        session = client.post('/api/library/sessions', json={'checkpoint_id': saved['id']}).json()
        prefix = f"/api/library/sessions/{session['id']}"
        state = client.get(prefix + '/state').json()
        assert not state['grounding_ready']
        assert state['artifacts']['environment.jpg']['url'].startswith(prefix)
        assert client.post(prefix + '/script/generate').status_code == 409
        assert client.post(prefix + '/execution/run').status_code == 409
        assert client.get(prefix + '/reference/environment.jpg').status_code == 200
        assert client.get(prefix + '/reference/.env').status_code == 400
        reset = client.post(prefix + '/reset').json()
        assert reset['id'] != session['id']
        assert client.get(prefix + '/state').status_code == 404
        assert client.get('/api/state').json()['parameters'] == PLAN


def test_library_visibility_filter_and_explicit_run_target(prepared):
    admin, _ = prepared
    app = create_library_app(admin, initialize_on_startup=False, root=admin.root_dir)
    with TestClient(app) as client:
        visible = client.post('/api/checkpoints', json={'name': 'Visible', 'visible': True}).json()
        hidden = client.post('/api/checkpoints', json={'name': 'Hidden', 'visible': False}).json()
        assert {item['id'] for item in client.get('/api/checkpoints').json()} == {visible['id'], hidden['id']}
        assert [item['id'] for item in client.get('/api/checkpoints?visible_only=true').json()] == [visible['id']]
        rejected = client.post('/api/library/sessions', json={'checkpoint_id': hidden['id'], 'target': 'run'})
        assert rejected.status_code == 400
        assert client.get('/api/library/session').json() is None
        assert not (admin.root_dir / 'library_sessions').exists()
        loaded = client.post('/api/library/sessions', json={'checkpoint_id': visible['id'], 'target': 'run'})
        assert loaded.status_code == 200
        session = loaded.json()
        assert app.state.library_manager.session.root_dir == admin.root_dir / 'library_sessions' / session['id']
        assert client.get('/api/library/session').json() == session
        assert client.get('/api/execution').json()['status'] == 'idle'
        prepared_copy = client.post('/api/library/sessions', json={'checkpoint_id': hidden['id'], 'target': 'admin'})
        assert prepared_copy.status_code == 200
        assert client.get('/api/state').json()['parameters'] == PLAN
        assert client.get('/api/library/session').json() == session


def test_failed_changes_invalidate_readiness_and_block_execution(prepared):
    admin, manager = prepared
    saved = manager.save('Run', visible=True)
    manager.open(saved['id'])
    service = manager.session
    service._state['grounding_revision'] = service._state['revision']
    service._state['script_revision'] = service._state['revision']
    with patch.object(service.message_bus, 'publish') as publish:
        service.requery_vision()
        request = publish.call_args.args[0]
        assert request.message_type == 'requery_vision' and request.content['skills'] == PLAN
        assert not service.state()['script_ready'] and not service.state()['grounding_ready']
    service._on_bus_message(Message('VisionModule', 'WebDashboard', 'vision_failed', {'error': 'camera disconnected'}))
    with pytest.raises(RuntimeError, match='Requery'):
        service.generate_script()
    service._on_bus_message(Message('VisionModule', 'VisionModule', 'vision_completed', {'keypoint_locations': {'x': {'point': [1, 2]}}}))
    assert service.state()['grounding_ready'] and not service.state()['script_ready']
    with patch.object(service.message_bus, 'run_background'):
        service.send_agent_message('ParameterSetter', 'Change the offset')
        assert not service.state()['grounding_ready']
    service._state['busy_stage'] = None
    manager.execution.state['status'] = 'stopping'
    for operation in (service.requery_vision, lambda: manager.open(saved['id']), lambda: manager.save('no')):
        with pytest.raises(RuntimeError, match='execution is active'):
            operation()
    manager.execution.state['status'] = 'idle'


def test_requery_uses_new_scene_and_feedback_uses_same_captured_image(prepared):
    admin, manager = prepared
    saved = manager.save('Vision', visible=True)
    manager.open(saved['id'])
    service = manager.session
    agent = service.agents['VisionModule']
    agent.camera_backend = 'ros2'
    current_image = np.full((10, 20, 3), [11, 100, 210], dtype=np.uint8)
    camera = Mock(); camera.get_rgb_image.return_value = current_image
    calls = []
    def detector(prompt, image):
        calls.append((prompt, image.copy()))
        return json.dumps([{'label': 'box_top_center', 'point': [400, 500]}])
    with patch('agents.vision_module.Camera', return_value=camera), \
         patch.object(agent, 'call_keypoint_detector', side_effect=detector), \
         patch.object(agent, 'visualize_keypoints'):
        agent.on_parameters_completed(Message('WebDashboard', 'VisionModule', 'requery_vision', {'skills': PLAN}))
        assert 'newly captured scene' in calls[0][0]
        camera.get_rgb_image.assert_called_once_with(use_new_frame=True)
        approved = copy.deepcopy(agent.current_skills)
        with patch('agents.vision_module.Camera', side_effect=AssertionError('Recapture during feedback')):
            agent.refine_keypoints('Move to the right edge')
        np.testing.assert_array_equal(calls[0][1], calls[1][1])
        assert 'Move to the right edge' in calls[1][0]
        assert agent.current_skills == approved == PLAN
    assert (service.root_dir / 'resized_img.png').is_file()
    assert (service.root_dir / 'reference/environment.jpg').read_bytes() == (admin.root_dir / 'environment.jpg').read_bytes()
    assert (service.root_dir / 'environment.jpg').read_bytes() != (admin.root_dir / 'environment.jpg').read_bytes()


def test_corrupt_checkpoint_and_unembedded_context_are_rejected(prepared):
    admin, manager = prepared
    saved = manager.save('Integrity')
    (manager.store.directory(saved['id']) / 'skills_output.json').write_text('[]')
    with pytest.raises(ValueError, match='integrity'):
        manager.store.load(saved['id'])
    agent = admin.agents['TaskDecomposer']
    agent._pending_history = [{'type': 'message', 'role': 'user', 'content': [
        {'type': 'input_image', 'file_id': 'file_external'}]}]
    with pytest.raises(RuntimeError, match='image context is not embedded'):
        manager.save('Incomplete')


@pytest.mark.parametrize('response', ['not json', '{}', '{"skills": []}'])
def test_failed_parameter_refinement_cannot_reapprove_old_grounding(prepared, response):
    admin, _ = prepared
    agent = admin.agents['ParameterSetter']
    with patch.object(agent, 'call_llm', return_value=response):
        admin.send_agent_message('ParameterSetter', 'Change height')
        deadline = time.monotonic() + 2
        while admin.message_bus.active_handlers and time.monotonic() < deadline:
            time.sleep(0.01)
    assert admin.state()['last_error']
    assert not admin.state()['grounding_ready'] and not admin.state()['script_ready']
    assert admin._state['parameters'] == PLAN


@pytest.mark.parametrize('response,shape', [
    ('not json', None),
    ('{}', 'type=dict, keys=[]'),
    ('[]', 'type=list, items=0'),
    ('{"parameters": []}', "type=dict, keys=['parameters']"),
    ('null', 'type=NoneType, not an object or array'),
])
def test_parameter_refinement_diagnostics_preserve_failures(prepared, capsys, response, shape):
    admin, _ = prepared
    agent = admin.agents['ParameterSetter']
    before = copy.deepcopy(agent.current_skills)
    with patch.object(agent, 'call_llm', return_value=response):
        with pytest.raises(RuntimeError):
            agent.refine_parameters('Lift 4 cm higher')
    output = capsys.readouterr().out
    assert '[ParameterSetter][refinement-debug]' in output
    assert "feedback='Lift 4 cm higher'" in output
    assert f'Raw response (type=str): {response!r}' in output
    if shape is None:
        assert 'Parsed response:' not in output
    else:
        assert 'Parsed response: ' + shape in output
    assert agent.current_skills == before


def test_parameter_refinement_diagnostics_preserve_success(prepared, capsys):
    admin, _ = prepared
    agent = admin.agents['ParameterSetter']
    updated = copy.deepcopy(PLAN)
    updated[0]['controllers'][0]['offset_parameters'][0]['value'] = 0.14
    response = json.dumps({'skills': updated})
    with patch.object(agent, 'call_llm', return_value=response) as call:
        assert agent.refine_parameters('Lift 4 cm higher') == updated
    output = capsys.readouterr().out
    assert f'Raw response (type=str): {response!r}' in output
    assert "Parsed response: type=dict, keys=['skills']" in output
    assert agent.current_skills == updated
    assert call.call_count == 1
