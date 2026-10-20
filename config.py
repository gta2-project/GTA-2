"""Configuration management for the multi-agent system."""
import json
import math
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fail clearly for an unmigrated setup instead of silently switching middleware.
if os.getenv('CAMERA_SOURCE') is not None and (
    os.getenv('CAMERA_BACKEND') is None or os.getenv('GTA_BACKEND') is None
):
    raise ValueError(
        'CAMERA_SOURCE has been replaced. Set CAMERA_BACKEND (file, ros1, ros2) '
        'and GTA_BACKEND (ros1, ros2) separately in your environment or .env.'
    )


# Renamed settings are not aliases: fail if an old name would be ignored.
for old_name, new_name in (
    ('CAMERA_TYPE', 'CAMERA_MODEL'),
    ('LOCAL_IMAGE_PATH', 'CAMERA_IMAGE_PATH'),
):
    if os.getenv(old_name) is not None and os.getenv(new_name) is None:
        raise ValueError(f'{old_name} has been replaced by {new_name}. Update your environment or .env.')


def backend_env(name, default, choices):
    """Read an explicit backend choice; never silently fall back on invalid input."""
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {value!r}")
    return value


DEFAULT_T_BASE_CAM = [
    [0.70857228, -0.49764763, 0.50027208, -0.48682316],
    [-0.70322773, -0.43945916, 0.5588796, -0.97660094],
    [-0.05827596, -0.74781179, -0.66134819, 0.5947388],
    [0.0, 0.0, 0.0, 1.0],
]


def parse_matrix_env(raw_value, name):
    """Parse a JSON-encoded 4x4 matrix from an environment variable."""
    if not raw_value:
        return DEFAULT_T_BASE_CAM

    try:
        matrix = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a single-line JSON 4x4 matrix") from exc

    if not isinstance(matrix, list) or len(matrix) != 4:
        raise ValueError(f"{name} must contain 4 rows")

    parsed = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{name} must contain 4 columns in each row")
        try:
            parsed.append([float(value) for value in row])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} values must be numeric") from exc

    return parsed


def _parse_tool_setting(name, default):
    """Read JSON tool settings; an explicit invalid value must not use a default."""
    try:
        return json.loads(os.getenv(name, default))
    except json.JSONDecodeError as exc:
        raise ValueError(f'{name} must be valid single-line JSON') from exc


def _tool_vector(value, name):
    """Validate without normalizing, flipping signs, or changing numeric values."""
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f'{name} must be a JSON array of 3 finite numbers')
    for item in value:
        if type(item) not in (int, float):
            raise ValueError(f'{name} must contain only finite numbers (not strings or booleans)')
        try:
            finite = math.isfinite(item)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f'{name} must contain only finite numbers')
    return list(value)


def validate_tool_geometry(tcp_offset, axes):
    """Return independent, validated copies for config and script generation.

    Axes are named directions, not a rotation matrix: preserve their handedness
    and magnitude. The offset is local to the pose returned by RobotState, not a
    command to change the physical robot's configured TCP.
    """
    offset = _tool_vector(tcp_offset, 'TOOL_TCP_OFFSET')
    if not isinstance(axes, dict) or set(axes) != {'x', 'y', 'z'}:
        raise ValueError('TOOL_AXES must be a JSON object with exactly x, y, and z directions')
    directions = {}
    for name in ('x', 'y', 'z'):
        direction = _tool_vector(axes[name], f'TOOL_AXES.{name}')
        if not any(direction):
            raise ValueError(f'TOOL_AXES.{name} must be a nonzero direction')
        directions[name] = direction
    return offset, directions


class Config:
    """Configuration class for the multi-agent system."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Per-Agent Configuration (each agent must specify its provider and model)
    # TaskDecomposer
    TASK_DECOMPOSER_PROVIDER = os.getenv('TASK_DECOMPOSER_PROVIDER')
    TASK_DECOMPOSER_MODEL = os.getenv('TASK_DECOMPOSER_MODEL')
    
    # SkillGenerator
    SKILL_GENERATOR_PROVIDER = os.getenv('SKILL_GENERATOR_PROVIDER')
    SKILL_GENERATOR_MODEL = os.getenv('SKILL_GENERATOR_MODEL')
    
    # ParameterSetter
    PARAMETER_SETTER_PROVIDER = os.getenv('PARAMETER_SETTER_PROVIDER')
    PARAMETER_SETTER_MODEL = os.getenv('PARAMETER_SETTER_MODEL')
    
    # VisionModule
    VISION_MODULE_PROVIDER = os.getenv('VISION_MODULE_PROVIDER', 'gemini')
    VISION_MODULE_MODEL = os.getenv('VISION_MODULE_MODEL', 'gemini-3.1-pro-preview')
    
    # Camera Configuration
    CAMERA_BACKEND = backend_env('CAMERA_BACKEND', 'file', ('file', 'ros1', 'ros2'))
    CAMERA_IMAGE_PATH = os.getenv('CAMERA_IMAGE_PATH', 'environment.jpg')
    CAMERA_NODE = os.getenv('CAMERA_NODE', '/zed_node')
    CAMERA_MODEL = os.getenv('CAMERA_MODEL', 'zed2i')
    # Optional full ROS 2 topic overrides. When unset, scene/camera/ros2.py derives the
    # same camera-model-specific topics used by the ROS 1 camera implementation.
    CAMERA_COLOR_TOPIC = os.getenv('CAMERA_COLOR_TOPIC')
    CAMERA_POINT_CLOUD_TOPIC = os.getenv('CAMERA_POINT_CLOUD_TOPIC')
    ROSMASTER_IP = os.getenv('ROSMASTER_IP', 'localhost')
    ROSMASTER_PORT = int(os.getenv('ROSMASTER_PORT', '11311'))
    # Independent device selection; ROS implementations follow GTA_BACKEND.
    ROBOT_TYPE = os.getenv('ROBOT_TYPE', 'ur5e').strip().lower()
    GRIPPER_TYPE = os.getenv('GRIPPER_TYPE', 'robotiq').strip().lower()
    GTA_BACKEND = backend_env('GTA_BACKEND', 'ros1', ('ros1', 'ros2'))
    ROBOT_IP = os.getenv('ROBOT_IP', '192.168.1.10')
    # Tool geometry is embedded into newly generated scripts, not read at execution.
    # Defaults preserve the existing setup; vectors are local to RobotState's pose.
    TOOL_TCP_OFFSET, TOOL_AXES = validate_tool_geometry(
        _parse_tool_setting('TOOL_TCP_OFFSET', '[0, 0, 0.1628]'),
        _parse_tool_setting('TOOL_AXES', '{"x":[0,-1,0],"y":[-1,0,0],"z":[0,0,1]}'),
    )
    ROS2_GRIPPER_ACTION_NAME = os.getenv(
        'ROS2_GRIPPER_ACTION_NAME',
        '/robotiq_gripper_controller/gripper_cmd',
    )
    ROS2_GRIPPER_WAIT_TIMEOUT_S = float(
        os.getenv('ROS2_GRIPPER_WAIT_TIMEOUT_S', '10.0')
    )
    ROS2_PUBLISHER_MATCH_TIMEOUT_S = float(
        os.getenv('ROS2_PUBLISHER_MATCH_TIMEOUT_S', '5.0')
    )
    T_BASE_CAM = parse_matrix_env(os.getenv('T_BASE_CAM'), 'T_BASE_CAM')
    
    @classmethod
    def get_agent_config(cls, agent_name: str):
        """Get provider and model configuration for a specific agent."""
        # Convert agent name to config variable format (e.g., TaskDecomposer -> TASK_DECOMPOSER)
        import re
        config_name = re.sub(r'(?<!^)(?=[A-Z])', '_', agent_name).upper()
        
        provider = getattr(cls, f'{config_name}_PROVIDER', None)
        model = getattr(cls, f'{config_name}_MODEL', None)
        if agent_name == 'VisionModule':
            # VisionModule always passes its prepared environment image explicitly.
            # It should not auto-fetch from the shared camera via BaseAgent.
            use_vision = False
        else:
            use_vision_str = os.getenv(f'{config_name}_USE_VISION', 'false')
            use_vision = use_vision_str.lower() in ['true', '1', 'yes']
        
        # Validate that agent has configuration
        if not provider:
            raise ValueError(f"Provider not configured for {agent_name}. Set {config_name}_PROVIDER in .env")
        if not model:
            raise ValueError(f"Model not configured for {agent_name}. Set {config_name}_MODEL in .env")
        
        # Get the appropriate API key
        if provider == 'openai':
            api_key = cls.OPENAI_API_KEY
            if not api_key:
                raise ValueError(f"OPENAI_API_KEY not set but {agent_name} requires OpenAI")
        elif provider == 'gemini':
            api_key = cls.GEMINI_API_KEY
            if not api_key:
                raise ValueError(f"GEMINI_API_KEY not set but {agent_name} requires Gemini")
        else:
            raise ValueError(f"Invalid provider '{provider}' for {agent_name}. Must be 'openai' or 'gemini'")
        
        return {
            'provider': provider,
            'model': model,
            'api_key': api_key,
            'use_vision': use_vision
        }

    @classmethod
    def validate(cls):
        """Validate that all required configuration is present."""
        if not cls.OPENAI_API_KEY and not cls.GEMINI_API_KEY:
            raise ValueError("Either OPENAI_API_KEY or GEMINI_API_KEY is not set. Please create a .env file with your API key.")
        return True
