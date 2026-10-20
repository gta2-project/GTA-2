# GTA-2

GTA-2 (Grounded Task-Axes) turns a manipulation task into a reviewable plan,
controller recipes, parameters, grounded keypoints, and a robot script.

Start with the [documentation website](https://gta2-project.github.io/documentation/), especially
[installation](docs/start/installation.md) and
[your first image-only task](docs/start/first-task.md).
The first tutorial uses the supplied cube image and model access, without
ROS or a robot. It is not a simulator.

## Applications

- `python main.py`: planning dashboard, normally at `http://127.0.0.1:8000`.
- `python library.py`: Prepare / Library / Run, normally at
  `http://127.0.0.1:8001`, for saving and reusing independent task checkpoints.

Build the frontend and configure all four agents before launching; follow the
installation guide for the required settings.

## Documentation website

Build and preview the site in a separate documentation environment:

```bash
python3.12 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements.txt
.venv-docs/bin/python -I -m sphinx -M html docs docs/_build -n -W --keep-going
.venv-docs/bin/python -m http.server 8765 --bind 127.0.0.1 --directory docs/_build/html
```

Open `http://127.0.0.1:8765`. See
[documentation development](docs/development/documentation.md) for checks and
asset maintenance.

## Hardware and extension guides

The reference integration is ROS 2 Humble, UR5e, Robotiq, and ZED.
Other devices require compatible interfaces or implementations.
Start with the [hardware readiness checklist](docs/hardware/checklist.md), then
[add a controller](docs/extend/controller.md), [robot](docs/extend/robot.md),
[camera](docs/extend/camera.md), or [gripper](docs/extend/gripper.md).

Managed execution requires ROS 2; follow the
[execution guide](docs/using/execution.md) for Run, Skip, Cancel, and robot controls.
ROS 1 support is described separately. Standalone
[UR5e utilities](scripts/ur5e/README.md) contain workcell-specific settings and
bypass dashboard supervision.

See [offline tests](tests/README.md) for verification without hardware.
For saved outputs and sensitive data, see [artifacts and storage](docs/reference/artifacts.md).

## License

GTA-2 is available under the [PolyForm Noncommercial License 1.0.0](LICENSE).
Use, modification, and redistribution are permitted under its terms.
Commercial uses outside the license's permitted purposes require a separate
agreement with the rights holders. For licensing inquiries, contact the project
at [gta2.project.email@gmail.com](mailto:gta2.project.email@gmail.com).
Third-party dependencies remain subject to their own licenses.
