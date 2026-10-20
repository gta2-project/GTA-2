"""GTA execution runtime and reusable helpers for generated robot scripts."""

from importlib import import_module

# Importing the managed control channel must not first load Open3D/ROS/RTDE.
# Resolve public helpers on first access; state backends load on construction.
def __getattr__(name):
    modules = {'get_3d_position': 'utils', 'project_errors': 'error_projection',
               'get_nullspace_matrix': 'error_projection'}
    if name in __all__:
        return getattr(import_module(f'.{modules.get(name, "skill_execution")}', __name__), name)
    raise AttributeError(name)

__all__ = [
    'get_3d_position',
    'project_errors',
    'get_nullspace_matrix',
    'start_skill',
    'update_control_commands',
    'end_skill',
    'execute_subtask',
    'initialize_ros_publishers',
]
