# Configuration

GTA reads environment variables and the repository's `.env` through
`python-dotenv`. Existing process environment variables take precedence.
Start applications from the repository root and restart them after configuration
changes. See [sensitive data](artifacts.md#sensitive-data) for credential handling.

This reference covers the GTA application, not optional benchmark settings.
The source of truth is `config.py`, the application entry points, and
`.env.example`; example values are not a calibrated workcell configuration.

## Agents and model access

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Credential for agents with provider `openai` |
| `GEMINI_API_KEY` | Credential for agents with provider `gemini` |
| `TASK_DECOMPOSER_PROVIDER`, `TASK_DECOMPOSER_MODEL` | Required task agent provider/model |
| `SKILL_GENERATOR_PROVIDER`, `SKILL_GENERATOR_MODEL` | Required recipe agent provider/model |
| `PARAMETER_SETTER_PROVIDER`, `PARAMETER_SETTER_MODEL` | Required parameter agent provider/model |
| `VISION_MODULE_PROVIDER`, `VISION_MODULE_MODEL` | Vision agent; configure explicitly for an available image-capable model |
| `TASK_DECOMPOSER_USE_VISION` | Attach scene images to task requests; default false |
| `SKILL_GENERATOR_USE_VISION` | Attach scene images to recipe requests; default false |
| `PARAMETER_SETTER_USE_VISION` | Attach scene images to parameter requests; default false |

Providers are `openai` or `gemini`; there is no generic provider URL or
Claude backend. A provider name does not verify model access. Choose models
available to your account that support the requests the agent sends. If
`USE_VISION` is true, that agent needs image support too.

VisionModule sends its prepared image explicitly; it does not use a
`VISION_MODULE_USE_VISION` switch. Its code default is Gemini with
`gemini-3.1-pro-preview`; availability is not guaranteed, so do not rely on
the default as a recommendation. The other three provider/model pairs have
no usable code default.

For new OpenAI agents the current client sets reasoning effort `medium`
and text verbosity `low`. These are effective generation settings in
`agents/base_agent.py`, not general `.env` switches. Models must accept
the settings and API features used by the client. Gemini uses its native chat
configuration. Checkpoints preserve effective model, prompt, and generation
settings for continuation; changing the current default prompt does not
rewrite saved contexts.

## Camera

| Variable | Code default | Meaning |
|---|---|---|
| `CAMERA_BACKEND` | `file` | `file`, `ros1`, or `ros2` |
| `CAMERA_IMAGE_PATH` | `environment.jpg` | File input path; library working copies manage their own images |
| `CAMERA_NODE` | `/zed_node` | Namespace used to derive topic names |
| `CAMERA_MODEL` | `zed2i` | Topic-layout choice, not an installed driver |
| `CAMERA_COLOR_TOPIC` | unset | Full ROS 2 RGB topic override |
| `CAMERA_POINT_CLOUD_TOPIC` | unset | Full ROS 2 point-cloud topic override |
| `ROSMASTER_IP` | `localhost` | ROS 1 master host |
| `ROSMASTER_PORT` | `11311` | ROS 1 master port |
| `T_BASE_CAM` | Workcell-specific matrix | JSON 4×4 camera-to-base calibration; replace with your calibration |

ROS 2 discovery uses the sourced environment, including `ROS_DOMAIN_ID`,
not the ROS 1 master settings. GTA does not force a universal domain ID.
Full topic overrides apply to ROS 2. See [camera geometry](../hardware/camera.md)
before assuming a compatible topic type is sufficient.

## Robot, gripper, and tool

| Variable | Code default | Meaning |
|---|---|---|
| `ROBOT_TYPE` | `ur5e` | Robot state/helper/server package |
| `GRIPPER_TYPE` | `robotiq` | Independent gripper package |
| `GTA_BACKEND` | `ros1` | Robot-command and gripper middleware, `ros1` or `ros2` |
| `ROBOT_IP` | `192.168.1.10` | Shared UR5e address for application interfaces |
| `TOOL_TCP_OFFSET` | `[0,0,0.1628]` | Additional point offset in metres, local to the reported robot TCP |
| `TOOL_AXES` | `{"x":[0,-1,0],"y":[-1,0,0],"z":[0,0,1]}` | Named local tool directions |
| `ROS2_GRIPPER_ACTION_NAME` | `/robotiq_gripper_controller/gripper_cmd` | Selected ROS 2 action endpoint |
| `ROS2_GRIPPER_WAIT_TIMEOUT_S` | `10.0` | Action availability/operation wait policy |
| `ROS2_PUBLISHER_MATCH_TIMEOUT_S` | `5.0` | Wait for all five GTA command subscriptions |

Camera and GTA backends are independent. UR5e and Robotiq implementations are bundled.
Unknown device names fail instead of selecting a different device.

Tool values are validated for shape and finite values; named axes must be
nonzero. They are not automatically normalized or made right-handed by
configuration. Changing tool values requires restarting and generating a new
script. See [TCP and tool geometry](../hardware/tool-geometry.md).
Standalone UR5e utilities may have separate code-defined addresses and poses.

## Web applications

| Variable | Default |
|---|---|
| `GTA_WEB_HOST` | `127.0.0.1` |
| `GTA_WEB_PORT` | `8000` |
| `GTA_WEB_RELOAD` | `false` |
| `GTA_LIBRARY_HOST` | `127.0.0.1` |
| `GTA_LIBRARY_PORT` | `8001` |

The applications are local operator tools without authentication. Do not
expose them to an untrusted network. Disable development reload during
hardware operation.
