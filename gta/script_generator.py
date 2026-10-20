"""Generate executable robot scripts from GTA skill and keypoint artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple, Union


def parse_keypoint_name(keypoint_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse keypoint names like g_Obox_Pside_Lcenter."""
    parts = keypoint_name.split("_")
    object_name = None
    part_name = None
    location_name = None

    for part in parts:
        if part.startswith("O"):
            object_name = part[1:]
        elif part.startswith("P"):
            part_name = part[1:]
        elif part.startswith("L"):
            location_name = part[1:]

    if object_name and part_name and location_name:
        return object_name, part_name, location_name
    return None, None, None


def object_to_variable_name(object_name: str) -> str:
    """Convert an object name into a safe Python variable name."""
    variable_name = re.sub(r"[^0-9a-zA-Z_]+", "_", object_name.strip()).strip("_").lower()
    if not variable_name:
        variable_name = "object"
    if variable_name[0].isdigit():
        variable_name = f"obj_{variable_name}"
    return variable_name


def _split_controller_args(args_str: str) -> List[str]:
    args = []
    depth = 0
    current_arg = ""

    for char in args_str:
        if char == "," and depth == 0:
            args.append(current_arg.strip())
            current_arg = ""
            continue

        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        current_arg += char

    if current_arg.strip():
        args.append(current_arg.strip())
    return args


def apply_dynamic_parameters(controller_name: str, translated_args: List[str]) -> List[str]:
    """Wrap controller arguments that need runtime evaluation."""
    dynamic_params_config = {
        "PosAlign": [0],
        "AxisAlign": [0],
        "AxisWaypoint": [0],
        "PosWaypoint": [0],
        "PosDMPSpiral": [0],
    }
    dynamic_indices = dynamic_params_config.get(controller_name, [])

    return [
        f"lambda: {arg}" if index in dynamic_indices else arg
        for index, arg in enumerate(translated_args)
    ]


def translate_controller_arg(
    arg: str,
    keypoint_map: Dict[str, Tuple[str, str]],
    surface_normals: Optional[MutableMapping[str, List[Dict[str, str]]]] = None,
) -> str:
    """Translate a controller argument token into generated script code."""
    arg = arg.strip()

    if arg.startswith("offset_"):
        param_name = arg[7:]
        return f"offsets['{param_name}']"

    if "(" in arg and arg.endswith(")"):
        func_name = arg[: arg.index("(")]
        args_str = arg[arg.index("(") + 1 : arg.rindex(")")]

        if func_name == "a_surface_normal":
            keypoint_arg = args_str.strip()
            if keypoint_arg.startswith("g_"):
                object_name, part_name, location_name = parse_keypoint_name(keypoint_arg)
                if object_name and part_name and location_name:
                    object_var = object_to_variable_name(object_name)
                    axis_name = f"{part_name}_{location_name}_surface_normal"
                    if surface_normals is not None:
                        surface_normals.setdefault(object_var, []).append(
                            {
                                "keypoint_name": keypoint_arg,
                                "part_name": part_name,
                                "location_name": location_name,
                                "axis_name": axis_name,
                            }
                        )
                    return f"{object_var}.get_a('{axis_name}')"

        parsed_args = _split_controller_args(args_str)
        translated_args = [
            translate_controller_arg(item, keypoint_map, surface_normals)
            for item in parsed_args
        ]
        return f"{func_name}({', '.join(translated_args)})"

    if arg.startswith("g_"):
        if arg == "g_eef_TCP":
            return "eef.get_k('TCP')"
        if arg in keypoint_map:
            object_var, keypoint_key = keypoint_map[arg]
            return f"{object_var}.get_k('{keypoint_key}')"

    if arg.startswith("a_"):
        if arg.startswith("a_global_"):
            axis = arg.split("_")[-1]
            return f"world.get_a('{axis}')"
        if arg.startswith("a_eef_"):
            axis = arg.split("_")[-1]
            return f"eef.get_a('{axis}')"

    if arg.startswith("[") and arg.endswith("]"):
        items_str = arg[1:-1]
        translated_items = [
            translate_controller_arg(item, keypoint_map, surface_normals)
            for item in _split_controller_args(items_str)
        ]
        return f"[{', '.join(translated_items)}]"

    return arg


def translate_controller(
    controller_str: str,
    keypoint_map: Dict[str, Tuple[str, str]],
    surface_normals: Optional[MutableMapping[str, List[Dict[str, str]]]] = None,
) -> str:
    """Translate one GTA controller string into generated script code."""
    controller_name = controller_str[: controller_str.index("(")]
    args_str = controller_str[controller_str.index("(") + 1 : controller_str.rindex(")")]
    parsed_args = _split_controller_args(args_str)
    translated_args = [
        translate_controller_arg(arg, keypoint_map, surface_normals)
        for arg in parsed_args
    ]
    wrapped_args = apply_dynamic_parameters(controller_name, translated_args)
    return f"{controller_name}({', '.join(wrapped_args)})"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_offsets(skills_data: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    offsets: Dict[str, Any] = {}
    for subtask in skills_data:
        for controller in subtask.get("controllers", []):
            for param in controller.get("offset_parameters", []):
                param_name = param.get("name", "")
                if param_name.startswith("offset_"):
                    offsets[param_name[7:]] = param.get("value", 0.0)
    return offsets


def generate_robot_script(
    keypoint_locations_path: Union[str, Path] = "keypoint_locations.json",
    skills_output_path: Union[str, Path] = "skills_output.json",
    output_path: Union[str, Path] = "robot_script.py",
    strict: bool = False,
) -> Dict[str, Any]:
    """Generate ``robot_script.py`` from current keypoint and skill artifacts."""
    from config import Config, validate_tool_geometry

    # Freeze this deployment's tool geometry into the script being reviewed.
    tool_tcp_offset, tool_axes = validate_tool_geometry(Config.TOOL_TCP_OFFSET, Config.TOOL_AXES)
    keypoint_locations_path = Path(keypoint_locations_path)
    skills_output_path = Path(skills_output_path)
    output_path = Path(output_path)

    if not keypoint_locations_path.exists():
        raise FileNotFoundError(f"{keypoint_locations_path} not found")
    if not skills_output_path.exists():
        raise FileNotFoundError(f"{skills_output_path} not found")

    keypoint_locations = _load_json(keypoint_locations_path)
    skills_data = _load_json(skills_output_path)
    if strict:
        from .plan_validation import validate_plan
        validate_plan(skills_data)
    offsets = _extract_offsets(skills_data)

    objects: Dict[str, Dict[str, Any]] = {}
    keypoint_map: Dict[str, Tuple[str, str]] = {}
    surface_normals: Dict[str, List[Dict[str, str]]] = {}

    for keypoint_name in keypoint_locations.keys():
        object_name, part_name, location_name = parse_keypoint_name(keypoint_name)
        if not object_name or not part_name or not location_name:
            continue

        object_var_name = object_to_variable_name(object_name)
        keypoint_var_name = f"{part_name}_{location_name}"
        objects.setdefault(
            object_var_name,
            {
                "name": object_name,
                "keypoints": [],
            },
        )
        objects[object_var_name]["keypoints"].append((keypoint_name, keypoint_var_name))
        keypoint_map[keypoint_name] = (object_var_name, keypoint_var_name)

    for subtask in skills_data:
        for controller in subtask.get("controllers", []):
            controller_str = controller.get("controller", "")
            if "(" not in controller_str:
                continue
            try:
                translate_controller(controller_str, keypoint_map, surface_normals)
            except Exception:
                continue

    script_content = f"""#!/usr/bin/env python
# Generated Robot Script
import time
import numpy as np
import json
from config import Config

from gta.utils import get_3d_position, get_point_cloud_data, a_surface_normal, a_crossprod_axes, a_along_keypoints, visualize_point_cloud, get_rgb_image
from gta.object_handler import ObjectHandler
from gta.skill_execution import execute_subtask, initialize_ros_publishers
from gta.managed_run import managed_preview, managed_subtask
from controllers import (
    PosAlign, AxisAlign, PosWaypoint, AxisWaypoint,
    GripperControl, ForceConstant, PosDMPSpiral
)

# Initialize the transport selected by GTA_BACKEND.
if Config.GTA_BACKEND == 'ros2':
    ROS_AVAILABLE = initialize_ros_publishers()
    if not ROS_AVAILABLE:
        raise RuntimeError(
            "ROS 2 robot transport is unavailable. Start the UR5e server, wait "
            "for its readiness marker, and verify all five /robot/* subscriptions."
        )
else:
    try:
        import rospy
        rospy.init_node('skill_executor', anonymous=True)
        ROS_AVAILABLE = initialize_ros_publishers()
    except ImportError:
        ROS_AVAILABLE = False

if ROS_AVAILABLE:
    print("[robot_script] ROS integration enabled")
else:
    print("[robot_script] Running in standalone mode (no ROS transport)")

# Read keypoint pixel locations from JSON file
with open('keypoint_locations.json', 'r') as f:
    keypoint_data = json.load(f)

# Get point cloud data for surface normal computation
pcd = get_point_cloud_data()
rgb = get_rgb_image()

T_BASE_CAM = np.array(Config.T_BASE_CAM, dtype=float)

# Create world object and assign its keypoints and axes
world = ObjectHandler(name='world')
world.keypoints['origin'] = np.array([0, 0, 0])
world.axes['x'] = np.array([0, -1, 0])
world.axes['y'] = np.array([1, 0, 0])
world.axes['z'] = np.array([0, 0, 1])
world.frame = 'world'

# Create eef object and assign its keypoints and axes
eef = ObjectHandler(name='eef')
eef.keypoints['TCP'] = np.array({tool_tcp_offset!r})
eef.axes['x'] = np.array({tool_axes['x']!r})
eef.axes['y'] = np.array({tool_axes['y']!r})
eef.axes['z'] = np.array({tool_axes['z']!r})
eef.frame = 'robot'

# Create objects with keypoints
"""

    sorted_objects = sorted(objects.items(), key=lambda item: item[1]["name"].lower())
    object_variable_names = ["world", "eef"] + [
        object_var_name for object_var_name, _ in sorted_objects
    ]

    for object_var_name, object_info in sorted_objects:
        script_content += f"\n# {object_info['name'].capitalize()} object\n"
        script_content += f"{object_var_name} = ObjectHandler(name='{object_info['name']}')\n"
        for original_name, var_name in object_info["keypoints"]:
            script_content += (
                f"{object_var_name}.keypoints['{var_name}'] = "
                f"get_3d_position(pcd, keypoint_data['{original_name}']['point'])\n"
            )

    if surface_normals:
        script_content += "\n# Compute surface normals for required keypoints\n"
        for object_var_name, normals_list in surface_normals.items():
            seen_axes = set()
            for normal_info in normals_list:
                axis_name = normal_info["axis_name"]
                if axis_name in seen_axes:
                    continue
                seen_axes.add(axis_name)
                keypoint_key = f"{normal_info['part_name']}_{normal_info['location_name']}"
                script_content += (
                    f"{object_var_name}.axes['{axis_name}'] = "
                    f"a_surface_normal({object_var_name}.keypoints['{keypoint_key}'], pcd)\n"
                )

    script_content += "\n# Offset parameters for skills\n"
    script_content += "offsets = {\n"
    for key, value in offsets.items():
        script_content += f"    '{key}': {value},\n"
    script_content += "}\n\n"

    script_content += "\nwith managed_preview():\n    visualize_point_cloud(\n"
    script_content += "        pcd_np=pcd,\n"
    script_content += "        colors=rgb,\n"
    script_content += "        camera_pose=T_BASE_CAM,\n"
    script_content += f"        objects=[{', '.join(object_variable_names)}]\n"
    script_content += "    )\n"

    script_content += "# Subtasks with translated controllers\n"
    script_content += "# Each subtask is an array containing controllers to execute\n\n"

    for subtask in skills_data:
        subtask_id = subtask.get("subtask_id")
        task = subtask.get("task", "Unknown task")
        script_content += f"\n# Subtask {subtask_id}: {task}\n"
        script_content += f"subtask_{subtask_id} = [\n"

        for controller in subtask.get("controllers", []):
            controller_str = controller.get("controller", "")
            try:
                translated = translate_controller(controller_str, keypoint_map, surface_normals)
                script_content += f"        {translated},\n"
            except Exception as exc:
                if strict:
                    raise ValueError(f'Cannot translate controller in subtask {subtask_id}: {controller_str}') from exc
                script_content += f"        # Error translating: {controller_str}\n"
                script_content += f"        # {exc}\n"

        script_content += "]\n"
        script_content += f"\n# Execute Subtask {subtask_id}\n"
        script_content += f"with managed_subtask({subtask_id!r}, {task!r}):\n"
        script_content += f"    execute_subtask(subtask_{subtask_id})\n"

    script_content += """
# To run with ROS:
#   1. Start roscore
#   2. Run this script: python robot_script.py
#   3. Subscribe to topics in your robot hardware interface node

"""

    compile(script_content, str(output_path), 'exec')
    output_path.write_text(script_content, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "script": script_content,
        "keypoint_count": len(keypoint_locations),
        "skill_count": len(skills_data),
    }
