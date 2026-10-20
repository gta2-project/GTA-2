# Install GTA

**Outcome:** open the planning dashboard with an image file, without connecting
to ROS or a robot.

**Prerequisites:** Git, a Linux terminal, Python 3.10
with `venv`, Node.js 22 with npm, and access to a supported model provider.
These instructions use the Ubuntu 22.04 / Python 3.10 environment associated
with the ROS 2 Humble reference workcell. Other platforms require their own
dependency validation.

## 1. Create the application environment

Get the source and enter the project folder:

```bash
git clone https://github.com/gta2-project/GTA-2.git
cd GTA-2
```

If you already have the source, use its root folder—the directory containing
`main.py`, `config.py`, and `requirements.txt`. Create the Python environment there:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the same activated Python environment each time you start GTA. Do not
install ROS packages with pip. Hardware setup later uses a Python environment
compatible with the sourced ROS installation; see [ROS 2 setup](../hardware/ros2.md).

## 2. Build the web interface

```bash
cd frontend
npm ci
npm run build
cd ..
```

This builds both web entry points. The Python applications serve these built
files; you do not need a separate frontend development server for normal use.

## 3. Create your private configuration

If you do not already have a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` locally. Configure all four agents.
Supported provider values are `openai` and `gemini`; each configured provider
needs its corresponding key. You can use one provider for all agents or mix
them.

| Agent | Required settings |
|---|---|
| Task Decomposer | `TASK_DECOMPOSER_PROVIDER`, `TASK_DECOMPOSER_MODEL` |
| Skill Generator | `SKILL_GENERATOR_PROVIDER`, `SKILL_GENERATOR_MODEL` |
| Parameter Setter | `PARAMETER_SETTER_PROVIDER`, `PARAMETER_SETTER_MODEL` |
| Vision Module | `VISION_MODULE_PROVIDER`, `VISION_MODULE_MODEL` |

For example, the following **template** selects Gemini for every agent.
Replace the key and model placeholders with your own settings:

```text
GEMINI_API_KEY=YOUR_PRIVATE_API_KEY
TASK_DECOMPOSER_PROVIDER=gemini
TASK_DECOMPOSER_MODEL=YOUR_VISION_MODEL_ID
SKILL_GENERATOR_PROVIDER=gemini
SKILL_GENERATOR_MODEL=YOUR_VISION_MODEL_ID
PARAMETER_SETTER_PROVIDER=gemini
PARAMETER_SETTER_MODEL=YOUR_VISION_MODEL_ID
VISION_MODULE_PROVIDER=gemini
VISION_MODULE_MODEL=YOUR_VISION_MODEL_ID
TASK_DECOMPOSER_USE_VISION=true
SKILL_GENERATOR_USE_VISION=true
PARAMETER_SETTER_USE_VISION=true
CAMERA_BACKEND=file
CAMERA_IMAGE_PATH=environment.jpg
GTA_BACKEND=ros2
```

Use model IDs available to your account. Every agent with vision enabled,
and always the Vision Module, needs an image-capable model. For OpenAI, use
`OPENAI_API_KEY`, provider `openai`, and models compatible with GTA's Responses/
Conversations use and current generation settings. See the
[configuration reference](../reference/configuration.md).

Create keys using the provider's official instructions:
[OpenAI API setup](https://developers.openai.com/api/docs/quickstart) or
[Gemini API keys](https://ai.google.dev/gemini-api/docs/api-key).
See [sensitive data](../reference/artifacts.md#sensitive-data) for handling
credentials and saved outputs.

`GTA_BACKEND` selects the command transport; `CAMERA_BACKEND=file` selects
image-file input independently.

## 4. Prepare the example image

Copy the bundled example image to the runtime image location:

```bash
cp docs/_static/images/cube-scene.jpg environment.jpg
```

If your root `environment.jpg` already contains an image you want to keep,
back it up or use a different `CAMERA_IMAGE_PATH` instead. Use a separate
runtime path for live camera captures.

## 5. Start and verify

```bash
python main.py
```

Open `http://127.0.0.1:8000`. The page should show the cube scene and no
configuration error. Missing ROS is expected for this image-only workflow.

Optionally inspect the local health response:

```bash
curl http://127.0.0.1:8000/api/health
```

The service should report `ready: true`. This confirms local configuration,
not successful provider authentication; the first generation checks that path.

**Next:** [your first task](first-task.md).
If setup fails, use [installation troubleshooting](../troubleshooting/installation.md).
