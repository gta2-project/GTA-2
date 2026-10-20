"""Select a gripper independently of the robot, without eager hardware imports."""

from importlib import import_module
from inspect import isabstract
import re

from config import Config
from .base import BaseGripper


class Gripper:
    """Construct the selected client in an already-initialized ROS environment.

    ROS 2 callers supply node and spin_until_complete(future, timeout_s).
    check_cancel is optional. Client-specific wait policies are explicit kwargs.
    """

    def __new__(cls, *, gta_backend=None, **kwargs) -> BaseGripper:
        # A directly launched ROS server supplies its own protocol explicitly.
        kind = Config.GRIPPER_TYPE
        backend = Config.GTA_BACKEND if gta_backend is None else gta_backend
        if not isinstance(kind, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', kind):
            raise ValueError('GRIPPER_TYPE must be a lowercase gripper directory name, such as robotiq')
        if backend not in {'ros1', 'ros2'}:
            raise ValueError('GTA_BACKEND must be ros1 or ros2')
        package = f'{__name__}.{kind}'
        name = f'{package}.{backend}'
        try:
            module = import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name in {package, name}:
                raise RuntimeError(
                    f'No gripper backend for GRIPPER_TYPE={kind!r}, GTA_BACKEND={backend!r}. '
                    f'Expected scene/gripper/{kind}/{backend}.py; no fallback will be used.'
                ) from exc
            raise ImportError(f'Gripper backend {name} requires missing dependency {exc.name!r}') from exc
        impl = getattr(module, 'Gripper', None)
        if not isinstance(impl, type) or not issubclass(impl, BaseGripper) or isabstract(impl):
            raise RuntimeError(f'{name} must define a concrete Gripper subclass of BaseGripper')
        return impl(**kwargs)


__all__ = ['Gripper', 'BaseGripper']
