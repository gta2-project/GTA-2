"""Camera/GTA backend independence; no model, ROS, or hardware requests."""

from pathlib import Path
import runpy
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import pytest

from api.service import GtaWebService
from config import Config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def read_config(monkeypatch):
    for name in ('CAMERA_SOURCE', 'CAMERA_BACKEND', 'GTA_BACKEND',
                 'CAMERA_TYPE', 'CAMERA_MODEL', 'LOCAL_IMAGE_PATH', 'CAMERA_IMAGE_PATH'):
        monkeypatch.delenv(name, raising=False)

    def read(**settings):
        for name, value in settings.items():
            monkeypatch.setenv(name, value)
        with patch('dotenv.load_dotenv'):
            return runpy.run_path(str(ROOT / 'config.py'))['Config']

    return read


def test_backend_defaults_are_independent(read_config):
    config = read_config()
    assert config.CAMERA_BACKEND == 'file'
    assert config.GTA_BACKEND == 'ros1'
    assert not hasattr(config, 'CAMERA_SOURCE')
    assert config.CAMERA_MODEL == 'zed2i'
    assert config.CAMERA_IMAGE_PATH == 'environment.jpg'
    assert not hasattr(config, 'CAMERA_TYPE')
    assert not hasattr(config, 'LOCAL_IMAGE_PATH')


def test_ros2_camera_does_not_implicitly_select_ros2_robot(read_config):
    config = read_config(CAMERA_BACKEND='ros2')
    assert config.CAMERA_BACKEND == 'ros2'
    assert config.GTA_BACKEND == 'ros1'


def test_ros2_robot_does_not_implicitly_select_ros2_camera(read_config):
    config = read_config(GTA_BACKEND='ros2')
    assert config.GTA_BACKEND == 'ros2'
    assert config.CAMERA_BACKEND == 'file'


@pytest.mark.parametrize('camera_backend', ['file', 'ros1', 'ros2'])
@pytest.mark.parametrize('gta_backend', ['ros1', 'ros2'])
def test_explicit_backend_combinations_and_normalization(read_config, camera_backend, gta_backend):
    config = read_config(CAMERA_BACKEND=f' {camera_backend.upper()} ', GTA_BACKEND=f' {gta_backend.upper()} ')
    assert config.CAMERA_BACKEND == camera_backend
    assert config.GTA_BACKEND == gta_backend


@pytest.mark.parametrize('name,value', [
    ('GTA_BACKEND', ''), ('GTA_BACKEND', 'ros'), ('GTA_BACKEND', 'local'),
    ('GTA_BACKEND', 'ros3'), ('CAMERA_BACKEND', ''), ('CAMERA_BACKEND', 'zed2i'),
    ('CAMERA_BACKEND', 'local'), ('CAMERA_BACKEND', 'ros'),
])
def test_invalid_backend_fails_without_fallback(read_config, name, value):
    with pytest.raises(ValueError, match=f'{name} must be one of'):
        read_config(**{name: value})


@pytest.mark.parametrize('new_settings', [{}, {'CAMERA_BACKEND': 'ros2'}, {'GTA_BACKEND': 'ros2'}])
def test_unmigrated_legacy_config_fails_with_instructions(read_config, new_settings):
    with pytest.raises(ValueError, match='CAMERA_SOURCE has been replaced'):
        read_config(CAMERA_SOURCE='ros2', **new_settings)


def test_explicit_new_settings_are_not_overridden_by_old_environment(read_config):
    config = read_config(CAMERA_SOURCE='ros2', CAMERA_BACKEND='file', GTA_BACKEND='ros1')
    assert config.CAMERA_BACKEND == 'file'
    assert config.GTA_BACKEND == 'ros1'


@pytest.mark.parametrize('old_name,new_name', [
    ('CAMERA_TYPE', 'CAMERA_MODEL'), ('LOCAL_IMAGE_PATH', 'CAMERA_IMAGE_PATH'),
])
def test_old_camera_settings_fail_instead_of_silently_using_defaults(read_config, old_name, new_name):
    with pytest.raises(ValueError, match=f'{old_name} has been replaced by {new_name}'):
        read_config(**{old_name: 'old-value'})


@pytest.mark.parametrize('old_name,new_name,new_value', [
    ('CAMERA_TYPE', 'CAMERA_MODEL', 'd405'),
    ('LOCAL_IMAGE_PATH', 'CAMERA_IMAGE_PATH', '/tmp/camera images/scene.png'),
])
def test_explicit_camera_settings_do_not_use_old_aliases(read_config, old_name, new_name, new_value):
    config = read_config(**{old_name: 'ignored', new_name: new_value})
    assert getattr(config, new_name) == new_value
    assert not hasattr(config, old_name)


@pytest.mark.parametrize('camera_backend', ['file', 'ros1', 'ros2'])
@pytest.mark.parametrize('gta_backend', ['ros1', 'ros2'])
def test_health_and_capture_use_independent_backends(tmp_path, monkeypatch, camera_backend, gta_backend):
    monkeypatch.setattr(Config, 'CAMERA_BACKEND', camera_backend)
    monkeypatch.setattr(Config, 'GTA_BACKEND', gta_backend)
    service = GtaWebService(tmp_path, scoped=True)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    Image.fromarray(frame).save(tmp_path / 'environment.jpg')
    camera = Mock()
    camera.get_rgb_image.return_value = frame
    with patch('scene.camera.Camera', return_value=camera) as factory:
        result = service.capture_environment_image()
    assert factory.call_args.kwargs['camera_backend'] == camera_backend
    assert factory.call_args.kwargs['camera_model'] == Config.CAMERA_MODEL
    assert factory.call_args.kwargs['camera_image_path'] == str(tmp_path / 'environment.jpg')
    assert result['source'] == camera_backend
    assert result['saved'] is (camera_backend != 'file')
    health = service.health()
    assert health['camera_backend'] == camera_backend
    assert health['gta_backend'] == gta_backend
    assert 'camera_source' not in health
    assert health['camera_image_path'] == Config.CAMERA_IMAGE_PATH
    assert 'local_image_path' not in health


@pytest.mark.parametrize('camera_backend', ['file', 'ros1', 'ros2'])
def test_robot_sensor_utilities_keep_using_camera_backend(monkeypatch, camera_backend):
    from gta import utils
    from gta.utils import sensors

    monkeypatch.setattr(Config, 'CAMERA_BACKEND', camera_backend)
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    camera = Mock()
    with patch.object(sensors, 'Camera', return_value=camera) as factory:
        assert utils.get_rgb_image() is camera.get_rgb_image.return_value
        assert utils.get_point_cloud_data() is camera.get_point_cloud.return_value
    assert len(factory.call_args_list) == 2
    assert all(call.kwargs['camera_backend'] == camera_backend for call in factory.call_args_list)
