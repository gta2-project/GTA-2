"""Lazy robot selection. State, helpers, and server are separate capabilities."""

from importlib import import_module
from inspect import isabstract
import re

from config import Config
from .base import BaseRobotHelpers, BaseRobotState, RobotStateError


def _backend_class(capability, name, contract):
    robot_type = Config.ROBOT_TYPE
    if not isinstance(robot_type, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", robot_type):
        raise ValueError('ROBOT_TYPE must be a lowercase robot directory name, such as ur5e')
    package = f'{__name__}.{robot_type}'
    module_name = f'{package}.{capability}'
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {package, module_name}:
            raise RuntimeError(
                f'No robot {capability} backend for ROBOT_TYPE={robot_type!r}. '
                f'Expected scene/robot/{robot_type}/{capability}.py; no fallback will be used.'
            ) from exc
        raise ImportError(f'Robot backend {module_name} requires missing dependency {exc.name!r}') from exc
    backend = getattr(module, name, None)
    if not isinstance(backend, type) or not issubclass(backend, contract) or isabstract(backend):
        raise RuntimeError(f'{module_name} must define a concrete {name} subclass of {contract.__name__}')
    return backend


class RobotState:
    """Read the selected robot's TCP pose without changing its tool geometry."""

    def __new__(cls) -> BaseRobotState:
        return _backend_class('state', 'RobotState', BaseRobotState)()


class RobotHelpers:
    """Create only the selected robot's helper implementation, not its server."""

    def __new__(cls) -> BaseRobotHelpers:
        return _backend_class('helpers', 'Helpers', BaseRobotHelpers)()


__all__ = ['RobotState', 'RobotHelpers', 'BaseRobotState', 'BaseRobotHelpers', 'RobotStateError']
