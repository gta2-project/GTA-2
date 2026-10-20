"""Offline tool configuration and generated-geometry tests; no hardware IO."""

import ast
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dotenv import dotenv_values
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from config import Config, validate_tool_geometry
import gta.object_handler as object_handler
from gta.script_generator import generate_robot_script


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFSET = [0, 0, 0.1628]
DEFAULT_AXES = {'x': [0, -1, 0], 'y': [-1, 0, 0], 'z': [0, 0, 1]}
CUSTOM_OFFSET = [0.025, -0.04, 0.2]
# Deliberately not an orthonormal/right-handed basis: these are named directions.
CUSTOM_AXES = {'x': [0, -2, 0], 'y': [-1, 0.5, 0], 'z': [0, 0, -1]}


@pytest.fixture
def read_tool_config():
    def read(**settings):
        # Do not inherit the laptop's .env or contact any external service.
        with patch.dict(os.environ, settings, clear=True), patch('dotenv.load_dotenv'):
            return runpy.run_path(str(ROOT / 'config.py'))['Config']
    return read


@pytest.fixture
def generate(tmp_path, monkeypatch, read_tool_config):
    config = read_tool_config()
    monkeypatch.setattr(Config, 'TOOL_TCP_OFFSET', config.TOOL_TCP_OFFSET)
    monkeypatch.setattr(Config, 'TOOL_AXES', config.TOOL_AXES)
    keypoints = tmp_path / 'keypoints.json'
    skills = tmp_path / 'skills.json'
    keypoints.write_text('{}')
    skills.write_text('[]')

    def generate():
        return generate_robot_script(keypoints, skills, tmp_path / 'script.py')['script']
    return generate


def _assignment_root(node):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _geometry_nodes(script):
    return [
        node for node in ast.parse(script).body
        if isinstance(node, ast.Assign)
        and _assignment_root(node.targets[0]) in {'world', 'eef'}
    ]


def _read_geometry(script):
    # Execute ONLY object/array construction. Never execute generated imports,
    # transport setup, camera reads, previews, controller creation, or motion.
    nodes = _geometry_nodes(script)
    assert len(nodes) == 12
    namespace = {'np': np, 'ObjectHandler': object_handler.ObjectHandler}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<geometry>', 'exec'), namespace)
    return namespace['world'], namespace['eef']


def test_unset_settings_keep_exact_original_defaults(read_tool_config):
    config = read_tool_config()
    assert config.TOOL_TCP_OFFSET == DEFAULT_OFFSET
    assert config.TOOL_AXES == DEFAULT_AXES
    assert repr(config.TOOL_TCP_OFFSET) == '[0, 0, 0.1628]'
    assert repr(config.TOOL_AXES['x']) == '[0, -1, 0]'


def test_env_example_matches_defaults(read_tool_config):
    example = dotenv_values(ROOT / '.env.example')
    config = read_tool_config(**{name: example[name] for name in ('TOOL_TCP_OFFSET', 'TOOL_AXES')})
    assert config.TOOL_TCP_OFFSET == DEFAULT_OFFSET
    assert config.TOOL_AXES == DEFAULT_AXES


def test_custom_settings_preserve_values_and_axis_directions(read_tool_config):
    config = read_tool_config(TOOL_TCP_OFFSET=json.dumps(CUSTOM_OFFSET), TOOL_AXES=json.dumps(CUSTOM_AXES))
    assert config.TOOL_TCP_OFFSET == CUSTOM_OFFSET
    assert config.TOOL_AXES == CUSTOM_AXES


@pytest.mark.parametrize('raw', [
    '', ' ', '[0, 0,', 'null', '{}', '1', '[]', '[0, 1]', '[0, 1, 2, 3]',
    '[[0], 0, 1]', '[true, 0, 1]', '["0", 0, 1]', '[null, 0, 1]',
    '[NaN, 0, 1]', '[Infinity, 0, 1]', '[-Infinity, 0, 1]', '[1e309, 0, 1]',
    '[' + '9' * 400 + ', 0, 1]',
])
def test_bad_offset_is_rejected_without_fallback(read_tool_config, raw):
    with pytest.raises(ValueError, match='TOOL_TCP_OFFSET'):
        read_tool_config(TOOL_TCP_OFFSET=raw)


@pytest.mark.parametrize('raw', [
    '', ' ', '{', 'null', '[]', '{}',
    '{"x":[1,0,0],"y":[0,1,0]}',
    '{"x":[1,0,0],"y":[0,1,0],"z":[0,0,1],"w":[1,1,1]}',
    *[json.dumps(dict(DEFAULT_AXES, z=value)) for value in (
        1, None, [0, 1], [0, 0, 0, 1], [0, 0, 0], [0, False, 1],
        [0, 0, '1'], [0, 0, [1]], [0, 0, float('nan')], [0, 0, float('inf')],
    )],
])
def test_bad_axes_are_rejected_without_fallback(read_tool_config, raw):
    with pytest.raises(ValueError, match='TOOL_AXES'):
        read_tool_config(TOOL_AXES=raw)


def test_zero_offset_is_allowed_and_axes_need_not_form_a_rotation_matrix(read_tool_config):
    config = read_tool_config(TOOL_TCP_OFFSET='[0,0,0]')
    assert config.TOOL_TCP_OFFSET == [0, 0, 0]
    # Existing named directions are left-handed. Do not "repair" their signs.
    assert np.linalg.det(np.array(list(config.TOOL_AXES.values()))) == -1


def test_validation_returns_independent_copies_without_changing_input():
    offset = CUSTOM_OFFSET.copy()
    axes = {key: value.copy() for key, value in CUSTOM_AXES.items()}
    result_offset, result_axes = validate_tool_geometry(offset, axes)
    assert offset == result_offset == CUSTOM_OFFSET
    assert axes == result_axes == CUSTOM_AXES
    offset[0] = 9
    axes['x'][0] = 9
    assert result_offset == CUSTOM_OFFSET
    assert result_axes == CUSTOM_AXES


def test_default_generation_keeps_original_literals_and_world_axes(generate):
    script = generate()
    assert "eef.keypoints['TCP'] = np.array([0, 0, 0.1628])" in script
    for name, axis in DEFAULT_AXES.items():
        assert f"eef.axes['{name}'] = np.array({axis!r})" in script
    world, eef = _read_geometry(script)
    assert world.frame == 'world'
    np.testing.assert_array_equal(world.keypoints['origin'], [0, 0, 0])
    for name, axis in {'x': [0, -1, 0], 'y': [1, 0, 0], 'z': [0, 0, 1]}.items():
        np.testing.assert_array_equal(world.axes[name], axis)
    assert eef.frame == 'robot'
    np.testing.assert_array_equal(eef.keypoints['TCP'], DEFAULT_OFFSET)


def test_custom_config_changes_only_the_four_tool_arrays(generate, monkeypatch, read_tool_config):
    original = generate()
    config = read_tool_config(TOOL_TCP_OFFSET=json.dumps(CUSTOM_OFFSET), TOOL_AXES=json.dumps(CUSTOM_AXES))
    monkeypatch.setattr(Config, 'TOOL_TCP_OFFSET', config.TOOL_TCP_OFFSET)
    monkeypatch.setattr(Config, 'TOOL_AXES', config.TOOL_AXES)
    changed = generate()
    expected = original.replace(
        f"eef.keypoints['TCP'] = np.array({DEFAULT_OFFSET!r})",
        f"eef.keypoints['TCP'] = np.array({CUSTOM_OFFSET!r})",
    )
    for name in DEFAULT_AXES:
        expected = expected.replace(
            f"eef.axes['{name}'] = np.array({DEFAULT_AXES[name]!r})",
            f"eef.axes['{name}'] = np.array({CUSTOM_AXES[name]!r})",
        )
    assert changed == expected


@pytest.mark.parametrize('pose', [
    [0., 0, 0, 0, 0, 0],
    [0.1, -0.5, 0.3, np.pi / 2, 0, 0],
    [-0.2, 0.1, 0.4, 0, np.pi / 2, 0],
    [0.3, -0.4, 0.2, 0.2, -0.3, 0.4],
])
@pytest.mark.parametrize('offset,axes', [(DEFAULT_OFFSET, DEFAULT_AXES), (CUSTOM_OFFSET, CUSTOM_AXES)])
def test_generated_geometry_uses_unchanged_pose_transform(generate, monkeypatch, pose, offset, axes):
    monkeypatch.setattr(Config, 'TOOL_TCP_OFFSET', offset)
    monkeypatch.setattr(Config, 'TOOL_AXES', axes)
    read_pose = Mock(return_value=np.array(pose))
    monkeypatch.setattr(object_handler, 'RobotState', lambda: SimpleNamespace(get_tcp_pose=read_pose))
    world, eef = _read_geometry(generate())
    rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
    np.testing.assert_allclose(eef.get_k('TCP'), np.array(pose[:3]) + rotation @ offset, atol=1e-14)
    for name, axis in axes.items():
        np.testing.assert_allclose(eef.get_a(name), rotation @ axis, atol=1e-14)
        np.testing.assert_array_equal(eef.axes[name], axis)
    np.testing.assert_array_equal(eef.keypoints['TCP'], offset)
    np.testing.assert_array_equal(read_pose.return_value, pose)
    assert read_pose.call_count == 4


def test_existing_script_keeps_geometry_until_regeneration(generate, monkeypatch, tmp_path):
    script = generate()
    # Mutate the in-memory configuration and environment after generation.
    Config.TOOL_TCP_OFFSET[2] = 0.25
    Config.TOOL_AXES['x'][1] = 2
    monkeypatch.setenv('TOOL_TCP_OFFSET', '[1,2,3]')
    monkeypatch.setenv('TOOL_AXES', json.dumps(CUSTOM_AXES))
    assert (tmp_path / 'script.py').read_text() == script
    assert 'Config.TOOL_' not in script
    _, saved_eef = _read_geometry(script)
    np.testing.assert_array_equal(saved_eef.keypoints['TCP'], DEFAULT_OFFSET)
    np.testing.assert_array_equal(saved_eef.axes['x'], DEFAULT_AXES['x'])
    _, new_eef = _read_geometry(generate())
    np.testing.assert_array_equal(new_eef.keypoints['TCP'], [0, 0, 0.25])
    np.testing.assert_array_equal(new_eef.axes['x'], [0, 2, 0])


@pytest.mark.parametrize('name,value', [
    ('TOOL_TCP_OFFSET', [0, 1]),
    ('TOOL_TCP_OFFSET', [0, 0, float('nan')]),
    ('TOOL_AXES', dict(DEFAULT_AXES, z=[0, 0, 0])),
    ('TOOL_AXES', dict(DEFAULT_AXES, z=[0, 0, "__import__('os').system('false')"])),
])
def test_invalid_geometry_cannot_overwrite_reviewed_script(generate, monkeypatch, tmp_path, name, value):
    script = generate()
    monkeypatch.setattr(Config, name, value)
    with pytest.raises(ValueError, match=name):
        generate()
    assert (tmp_path / 'script.py').read_text() == script


@pytest.mark.parametrize('robot_type', ['ur5e', 'robot_x'])
@pytest.mark.parametrize('gta_backend', ['ros1', 'ros2'])
def test_geometry_does_not_depend_on_robot_or_command_backend(generate, monkeypatch, robot_type, gta_backend):
    original = generate()
    monkeypatch.setattr(Config, 'ROBOT_TYPE', robot_type)
    monkeypatch.setattr(Config, 'GTA_BACKEND', gta_backend)
    assert generate() == original
