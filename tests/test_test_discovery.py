"""Audit discovery without importing any of the manual hardware programs."""

from configparser import ConfigParser
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROBOT_SCRIPTS = (
    'close_gripper.py',
    'connection.py',
    'half_open_gripper.py',
    'open_gripper.py',
    'read_force_values.py',
    'reset_joints.py',
    'reset_pouring.py',
    'reset_robot.py',
    'reset_robot_tcp.py',
)


def discovery_settings():
    config = ConfigParser()
    config.read(ROOT / 'pytest.ini')
    return config['pytest']


def test_default_discovery_has_an_explicit_offline_allowlist():
    settings = discovery_settings()
    assert settings['testpaths'].split() == ['tests']
    names = settings['python_files'].split()
    assert names == sorted(set(names))
    assert 'test_test_discovery.py' in names
    for name in names:
        assert Path(name).name == name
        assert name.startswith('test_') and name.endswith('.py')
        assert not any(character in name for character in '*?[]')
        assert (ROOT / 'tests' / name).is_file()
    assert not set(names).intersection(MANUAL_ROBOT_SCRIPTS)
    assert {'scripts', 'generated_plans', 'library_checkpoints', 'library_sessions'}.issubset(
        settings['norecursedirs'].split()
    )


def test_every_test_named_module_has_been_classified():
    # A new file needs an explicit safety review, not silent exclusion or import.
    candidates = {
        str(path.relative_to(ROOT / 'tests'))
        for path in (ROOT / 'tests').rglob('*.py')
        if path.name.startswith('test_') or path.name.endswith('_test.py')
    }
    assert candidates == set(discovery_settings()['python_files'].split())


@pytest.mark.parametrize('name', MANUAL_ROBOT_SCRIPTS)
def test_manual_robot_script_lives_only_under_scripts_ur5e(name):
    path = ROOT / 'scripts' / 'ur5e' / name
    assert path.is_file()
    assert not (ROOT / 'tests' / name).exists()
    # Syntax checking must not import or execute these scripts.
    compile(path.read_text(), str(path), 'exec')


@pytest.mark.parametrize('arguments', [[], ['tests'], ['.']])
def test_collection_does_not_import_excluded_scripts(tmp_path, arguments):
    # Exercise the real config in an isolated synthetic repository. Importing
    # a real manual script to test discovery would itself risk physical motion.
    (tmp_path / 'pytest.ini').write_text((ROOT / 'pytest.ini').read_text())
    tests = tmp_path / 'tests'
    tests.mkdir()
    (tests / 'test_test_discovery.py').write_text('def test_offline():\n    assert True\n')
    forbidden = "raise AssertionError('Unaudited script was imported during collection')\n"
    for name in ('test_unreviewed_hardware.py', 'hardware_test.py'):
        (tests / name).write_text(forbidden)
    nested = tests / 'nested'
    nested.mkdir()
    (nested / 'test_unreviewed.py').write_text(forbidden)

    for directory in ('scripts/ur5e', 'generated_plans', 'library_sessions',
                      'library_checkpoints', 'frontend/node_modules', 'frontend/dist'):
        target = tmp_path / directory
        target.mkdir(parents=True)
        # Even a matching filename in an excluded tree must never be imported.
        (target / 'test_test_discovery.py').write_text(forbidden)
    for name in MANUAL_ROBOT_SCRIPTS:
        (tmp_path / 'scripts' / 'ur5e' / name).write_text(forbidden)

    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='',
               PYTEST_PLUGINS='', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', *arguments],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'tests/test_test_discovery.py::test_offline' in result.stdout
    assert '1 test collected' in result.stdout
