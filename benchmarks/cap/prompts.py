"""Prompt construction for zero-shot and few-shot CaP conditions."""

from __future__ import annotations

from textwrap import dedent

from .conditions import ApiKind, ConditionSpec


POLICY_CONTRACT = dedent(
    """
    Return only one complete Python program, without Markdown fences.

    The program must contain:

    KEYPOINT_QUERIES = {
        "alias": {
            "object": "visible object name",
            "description": "precise semantic description of the physical point"
        }
    }

    def policy(ctx):
        ...

    Choose the semantic keypoints required by the task. Each alias must be a
    valid Python identifier and each query must have a non-empty object and
    description. The frozen observation context provides:

    ctx.point("alias") -> base-frame 3D point at the alias's localized RGB-image pixel.
    ctx.normal("alias") -> base-frame local surface normal at that point.
    ctx.current_pose() -> live [x, y, z, rx, ry, rz] TCP pose
    ctx.tcp_position() -> live base-frame TCP tip position
    ctx.tcp_axis("x" | "y" | "z") -> live base-frame TCP axis; z is the
    approach axis, x is the grasp axis, and y is the orthogonal axis.

    Perception points and normals remain fixed during policy execution. Use only
    the documented API, local variables, arithmetic, indexing, conditionals,
    helper functions, and statically bounded for-loops. Do not import modules,
    access files or networks, invoke ROS/RTDE directly, use while-loops, or use
    reflection. The numerical aliases np, math, and
    scipy.spatial.transform.Rotation as R are available without imports. All
    physical dimensions are SI units.
    """
).strip()


CARTESIAN_API = dedent(
    """
    Available manipulation API:

    move_to(x, y, z, rx, ry, rz): Move the TCP to an absolute base-frame UR pose using Cartesian linear motion.

    gripper(state): Open or close the gripper. state must be "open" or "close".

    move_force(x, y, z, rx, ry, rz, fx, fy, fz, fn): Move toward an absolute base-frame UR pose while RTDE force mode applies fn Newtons along the base-frame axis [fx, fy, fz].

    Pose rotation values are UR rotation-vector components in radians. Force-mode
    compliance is handled by RTDE.
    """
).strip()


CONTROLLER_API = dedent(
    """
    Available GTA controller API:

    PosAlign(g1, g2, a1, o1): Align live tool point g1 with point g2 along axis a1 with offset o1 meters.
    AxisAlign(a1, a2, a3, o1): Align live tool axis a1 with a2 by rotating about a3 with offset o1 degrees.
    PosWaypoint(g1, g2, a1, offsets): Follow sequential positional offsets in meters along a1 from g2.
    AxisWaypoint(a1, a2, a3, offsets): Follow sequential angular offsets in degrees about a3.
    ForceConstant(a1, o1): Apply signed force o1 Newtons along a1.
    PosDMPSpiral(g1, a1, o1): Execute the learned planar spiral of radius o1 meters around the starting point of g1, normal to a1.
    GripperControl(state, time, object=None): Open or close the gripper at normalized subtask time in [0, 1]. When object is provided, closing attaches that object rigidly to the current TCP frame and opening detaches it back into the world frame.
    execute_subtask([controllers]): Execute all listed controllers concurrently for one 10-second GTA phase.

    Controller attachment helpers:
    ctx.object_for("alias") -> the attachable object named by that keypoint query's object field.
    object.get_k("alias") -> the queried point in the base frame; it follows the live TCP pose while the object is attached.
    object.get_a("alias") -> the queried surface normal in the base frame; it follows the live TCP orientation while the object is attached.

    The first g1 argument of PosAlign, PosWaypoint, and PosDMPSpiral must be a
    zero-argument callable returning the current controlled tool point. Use
    ctx.tcp_position when that point is the TCP. For a queried point on an
    attached tool, use lambda: object.get_k("alias"). The first a1 argument of
    AxisAlign and AxisWaypoint must likewise be a live callable; for a TCP axis,
    use lambda: ctx.tcp_axis("x"|"y"|"z"). Other point and axis arguments are
    fixed vectors evaluated when the controller is created.
    Controllers are ordered by priority in each execute_subtask list. 
    Available vector helpers are a_along_keypoints(g1, g2) and a_crossprod_axes(a1, a2).
    """
).strip()


CARTESIAN_FEW_SHOT = dedent(
    r'''
    Three independent examples follow. Note that these are just examples and there may be many other ways to implement the same tasks.

    Example: pick and place
    KEYPOINT_QUERIES = {
        "object_grasp": {"object": "block", "description": "top point on the block"},
        "destination": {"object": "tray", "description": "central point in the tray"},
        "orientation_tail": {"object": "tray", "description": "tail of the visible orientation arrow"},
        "orientation_tip": {"object": "tray", "description": "tip of the visible orientation arrow"}
    }
    def policy(ctx):
        pose = ctx.current_pose()
        grasp = ctx.point("object_grasp")
        move_to(grasp[0], grasp[1], grasp[2], pose[3], pose[4], pose[5])
        gripper("close")
        destination = ctx.point("destination")
        orientation_tail = ctx.point("orientation_tail")
        orientation_tip = ctx.point("orientation_tip")
        desired_direction = normalize(orientation_tip - orientation_tail)
        current_rotation = R.from_rotvec(pose[3:6])
        alignment = R.align_vectors([desired_direction], [ctx.tcp_axis("x")])[0]
        target_rotation = (alignment * current_rotation).as_rotvec()
        move_to(destination[0], destination[1], destination[2], target_rotation[0], target_rotation[1], target_rotation[2])
        gripper("open")

    Example: force contact on a surface
    KEYPOINT_QUERIES = {
        "contact": {"object": "surface", "description": "center of the marked contact area"}
    }
    def policy(ctx):
        pose = ctx.current_pose()
        target = ctx.point("contact")
        normal = ctx.normal("contact")
        move_force(target[0], target[1], target[2], pose[3], pose[4], pose[5], normal[0], normal[1], normal[2], -5.0)

    Example: point traverse
    KEYPOINT_QUERIES = {
        "center": {"object": "point", "description": "center of the point to traverse"}
    }
    def policy(ctx):
        pose = ctx.current_pose()
        center = ctx.point("center")
        normal = ctx.normal("center")
        tangent = ctx.tcp_axis("x")
        for offset in [-0.04, 0.04]:
            target = center + tangent * offset
            move_to(target[0], target[1], target[2], pose[3], pose[4], pose[5])
    '''
).strip()


CONTROLLER_FEW_SHOT = dedent(
    r'''
    Three independent examples follow. Note that these are just examples and there may be many other ways to implement the same tasks.

    Example: pick and place
    KEYPOINT_QUERIES = {
        "object_grasp": {"object": "block", "description": "top point on the block"},
        "destination": {"object": "tray", "description": "central point in the tray"},
        "orientation_tail": {"object": "tray", "description": "tail of the visible orientation arrow"},
        "orientation_tip": {"object": "tray", "description": "tip of the visible orientation arrow"}
    }
    def policy(ctx):
        grasp = ctx.point("object_grasp")
        block = ctx.object_for("object_grasp")
        execute_subtask([PosAlign(ctx.tcp_position, grasp, normalize(grasp - ctx.tcp_position()), 0.0)])
        execute_subtask([GripperControl("close", 0.0, block)])
        destination = ctx.point("destination")
        orientation_tail = ctx.point("orientation_tail")
        orientation_tip = ctx.point("orientation_tip")
        desired_direction = normalize(orientation_tip - orientation_tail)
        rotation_axis = a_crossprod_axes(ctx.tcp_axis("x"), desired_direction)
        live_grasp = lambda: block.get_k("object_grasp")
        execute_subtask([
            AxisAlign(lambda: ctx.tcp_axis("x"), desired_direction, rotation_axis, 0.0),
            PosAlign(live_grasp, destination, normalize(destination - live_grasp()), 0.0)
        ])
        execute_subtask([GripperControl("open", 0.0, block)])

    Example: force contact on a surface
    KEYPOINT_QUERIES = {
        "contact": {"object": "surface", "description": "center of the marked contact area"}
    }
    def policy(ctx):
        target = ctx.point("contact")
        normal = ctx.normal("contact")
        execute_subtask([
            PosAlign(ctx.tcp_position, target, np.array([1.0, 0.0, 0.0]), 0.0),
            PosAlign(ctx.tcp_position, target, np.array([0.0, 1.0, 0.0]), 0.0),
            PosAlign(ctx.tcp_position, target, np.array([0.0, 0.0, -1.0]), -0.01)
        ])
        execute_subtask([
            PosAlign(ctx.tcp_position, target, np.array([1.0, 0.0, 0.0]), 0.0),
            PosAlign(ctx.tcp_position, target, np.array([0.0, 1.0, 0.0]), 0.0),
            ForceConstant(normal, -5.0)
        ])

    Example: point traverse along eef x axis
    KEYPOINT_QUERIES = {
        "center": {"object": "point", "description": "center of the point to traverse"}
    }
    def policy(ctx):
        center = ctx.point("center")
        normal = ctx.normal("center")
        tangent = ctx.tcp_axis("x")
        execute_subtask([PosWaypoint(ctx.tcp_position, center, tangent, [-0.04, 0.04])])
    '''
).strip()


def build_system_prompt(condition: ConditionSpec) -> str:
    api = CARTESIAN_API if condition.api_kind == ApiKind.CARTESIAN else CONTROLLER_API
    sections = [
        "You are a Code-as-Policies robot-program generator. Produce one safe, direct policy for the requested manipulation task.",
        POLICY_CONTRACT,
        api,
    ]
    if condition.few_shot:
        if condition.api_kind == ApiKind.CARTESIAN:
            examples = CARTESIAN_FEW_SHOT
        else:
            examples = CONTROLLER_FEW_SHOT
        sections.append(examples)
    return "\n\n".join(sections)


def build_user_prompt(task: str, success_criteria=None) -> str:
    prompt = f"Task: {task.strip()}"
    if success_criteria:
        if isinstance(success_criteria, (list, tuple)):
            criteria = "\n".join(f"- {item}" for item in success_criteria)
        else:
            criteria = str(success_criteria)
        prompt += f"\n\nEvaluation criteria:\n{criteria}"
    return prompt
