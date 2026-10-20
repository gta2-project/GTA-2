"""Immutable plan checkpoints and independent, explicit library workspaces."""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone
import uuid

from agents.checkpoint import encode, decode, restore_message
from api.service import GtaWebService, ARTIFACT_NAMES, AGENT_NAMES
from gta.plan_validation import validate_plan


def write_json(path, value):
    path = Path(path)
    staging = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        staging.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class CheckpointStore:
    def __init__(self, root):
        self.root = Path(root) / 'library_checkpoints'

    def directory(self, checkpoint_id):
        if not re.fullmatch(r'[0-9a-f]{32}', checkpoint_id):
            raise ValueError('Invalid checkpoint ID')
        directory = self.root / checkpoint_id
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError('Checkpoint not found')
        return directory

    def describe(self, checkpoint_id):
        directory = self.directory(checkpoint_id)
        manifest = read_json(directory / 'manifest.json')
        metadata = read_json(directory / 'metadata.json')
        return {**metadata, 'id': checkpoint_id, 'created_at': manifest['created_at'],
                'task': manifest['task'],
                'thumbnail_url': f'/api/checkpoints/{checkpoint_id}/artifacts/environment.jpg'
                if 'environment.jpg' in manifest['artifacts'] else None}

    def list(self, visible_only=False):
        if not self.root.exists():
            return []
        result = []
        for path in self.root.iterdir():
            if re.fullmatch(r'[0-9a-f]{32}', path.name) and path.is_dir() and not path.is_symlink():
                item = self.describe(path.name)
                if not visible_only or item['visible']:
                    result.append(item)
        return sorted(result, key=lambda item: item['created_at'], reverse=True)

    def save(self, service, name, notes='', visible=False):
        if not name.strip():
            raise ValueError('Checkpoint name cannot be empty')
        checkpoint_name = name.strip()
        with service._lock:
            service.assert_idle()
            validate_plan(service._state['parameters'])
            if service._state.get('last_error'):
                raise RuntimeError('Resolve the current workflow error before saving')
            if set(service.agents) != AGENT_NAMES:
                raise RuntimeError('A full admin workflow is required to save a checkpoint')
            # Export before publishing any directory. An unavailable provider
            # history aborts the save instead of creating an incomplete snapshot.
            agents = {name: agent.export_checkpoint() for name, agent in service.agents.items()}
            payload = {'workflow': copy.deepcopy(service._state), 'agents': agents,
                       'events': copy.deepcopy(service._events),
                       'messages': service._message_history()}
            self.root.mkdir(parents=True, exist_ok=True)
            checkpoint_id = uuid.uuid4().hex
            with tempfile.TemporaryDirectory(prefix='.saving-', dir=self.root) as temp:
                staging = Path(temp) / checkpoint_id
                staging.mkdir()
                artifacts = {}
                for name in sorted(ARTIFACT_NAMES):
                    source = service._artifact_file_path(name)
                    if name == 'robot_script.py' and service._state['script_revision'] != service._state['revision']:
                        continue
                    if source.is_file():
                        destination = staging / name
                        shutil.copy2(source, destination)
                        artifacts[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
                missing = {'skills_output.json', 'environment.jpg'} - artifacts.keys()
                if missing:
                    raise ValueError('Required checkpoint artifacts are missing: ' + ', '.join(sorted(missing)))
                if read_json(staging / 'skills_output.json') != service._state['parameters']:
                    raise ValueError('Skill artifact and workflow state disagree')
                if service._state['grounding_revision'] == service._state['revision']:
                    if 'keypoint_locations.json' not in artifacts or read_json(staging / 'keypoint_locations.json') != service._state['keypoints']:
                        raise ValueError('Keypoint artifact and workflow state disagree')
                write_json(staging / 'state.json', encode(payload))
                write_json(staging / 'manifest.json', {
                    'version': 1, 'id': checkpoint_id,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'task': service._state['task'], 'artifacts': artifacts,
                    'state_sha256': hashlib.sha256((staging / 'state.json').read_bytes()).hexdigest(),
                })
                write_json(staging / 'metadata.json', {'name': checkpoint_name, 'notes': notes, 'visible': bool(visible)})
                staging.rename(self.root / checkpoint_id)
            return checkpoint_id

    def update(self, checkpoint_id, **updates):
        directory = self.directory(checkpoint_id)
        metadata = read_json(directory / 'metadata.json')
        if updates.get('name') is not None:
            if not updates['name'].strip():
                raise ValueError('Checkpoint name cannot be empty')
            metadata['name'] = updates['name'].strip()
        for key in ('notes', 'visible'):
            if updates.get(key) is not None:
                metadata[key] = updates[key]
        write_json(directory / 'metadata.json', metadata)
        return self.describe(checkpoint_id)

    def load(self, checkpoint_id):
        directory = self.directory(checkpoint_id)
        manifest = read_json(directory / 'manifest.json')
        if manifest['version'] != 1:
            raise ValueError('Unsupported checkpoint version')
        for name, digest in {**manifest['artifacts'], 'state.json': manifest['state_sha256']}.items():
            if name not in ARTIFACT_NAMES | {'state.json'}:
                raise ValueError('Invalid checkpoint artifact')
            path = directory / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Checkpoint integrity check failed: {name}')
        payload = decode(read_json(directory / 'state.json'))
        # Four active agents are required. Accept only the known retired entry
        # from earlier checkpoints, without recreating it or rewriting history.
        saved_names = set(payload['agents'])
        if not AGENT_NAMES <= saved_names <= AGENT_NAMES | {'ErrorAnalyzer'}:
            raise ValueError('Checkpoint is missing agent state')
        validate_plan(payload['workflow']['parameters'])
        return directory, manifest, payload


class LibraryManager:
    def __init__(self, root, admin):
        from api.execution import ExecutionSupervisor
        self.root = Path(root)
        self.lock = admin._lock
        self.admin = admin
        self.session = None
        self.session_info = None
        self.store = CheckpointStore(root)
        self.execution = ExecutionSupervisor(self.root, self.lock)
        self._attach(admin)

    def _attach(self, service):
        service._lock = self.lock
        service.execution_guard = self.execution.assert_idle
        if service.message_bus:
            service.message_bus.lifecycle_lock = self.lock

    def save(self, name, notes='', visible=False, source='admin'):
        with self.lock:
            service = self.admin if source == 'admin' else self.get_session(source)
            checkpoint_id = self.store.save(service, name, notes, visible)
            return self.store.describe(checkpoint_id)

    def get_session(self, session_id):
        if self.session is None or self.session_info['id'] != session_id:
            raise FileNotFoundError('This library session is no longer active')
        return self.session

    def open(self, checkpoint_id, target='run'):
        with self.lock:
            if target not in {'run', 'admin'}:
                raise ValueError('Target must be admin or run')
            self.execution.assert_idle()
            self.admin.assert_idle()
            if self.session:
                self.session.assert_idle()
            directory, manifest, payload = self.store.load(checkpoint_id)
            if target == 'run' and not self.store.describe(checkpoint_id)['visible']:
                raise ValueError('This checkpoint is not enabled for Run')
            session_id = uuid.uuid4().hex
            workspace = self.root / 'library_sessions' / session_id
            workspace.mkdir(parents=True)
            for name in manifest['artifacts']:
                shutil.copy2(directory / name, workspace / name)
            reference = workspace / 'reference'
            reference.mkdir()
            for name in ('environment.jpg', 'annotated_keypoints.png'):
                if (workspace / name).exists():
                    shutil.copy2(workspace / name, reference / name)
            service = GtaWebService(workspace, scoped=True)
            self._attach(service)
            service._create_agents(payload['agents'])
            service._state = payload['workflow']
            service._state['busy_stage'] = None
            service._events = payload['events']
            service.message_bus.message_history = [restore_message(m) for m in payload['messages']]
            service.initialized = True
            if target == 'run':
                service.invalidate()
                service._state['image_revision'] = None
                service.api_prefix = f'/api/library/sessions/{session_id}'
                result = {'id': session_id, 'checkpoint_id': checkpoint_id,
                          'checkpoint': self.store.describe(checkpoint_id)}
            elif target == 'admin':
                if service._state.get('script'):
                    service._state['script_path'] = str(workspace / 'robot_script.py')
                result = {'id': session_id, 'checkpoint_id': checkpoint_id}
            else:
                raise ValueError('Target must be admin or run')
            write_json(workspace / 'session.json', result)
            if target == 'run':
                self.session = service
                self.session_info = result
            else:
                self.admin = service
            return result

    def shutdown(self):
        self.execution.shutdown()
