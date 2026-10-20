# GTA-2 documentation

:::{container} eyebrow
GTA-2 · Grounded Task Axes
:::

:::{container} hero
This documentation covers GTA-2 installation and configuration, planning and
refinement workflows, hardware setup, and controller and device interfaces.
:::

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Your first task
:link: start/first-task
:link-type: doc
Use the cube scene to generate, refine, and inspect a plan. No robot required.
:::

:::{grid-item-card} Connect your workcell
:link: hardware/checklist
:link-type: doc
Check ROS, camera geometry, robot state, and gripper readiness before motion.
:::

:::{grid-item-card} Extend GTA
:link: extend/controller
:link-type: doc
Build a small controller, then explore robot, camera, and gripper adapters.
:::

:::{grid-item-card} Reuse a task
:link: using/library
:link-type: doc
Save an approved plan, load a working copy, and ground it in a new scene.
:::
::::

```{figure} _static/diagrams/pipeline.svg
:target: _static/diagrams/pipeline.svg
:alt: GTA-2 pipeline from task decomposition through controller selection, parameterization, visual grounding, script generation, and robot execution.
:class: diagram

Task decomposition, controller selection, parameterization, visual grounding,
script generation, and robot execution.
```

Select any diagram or screenshot to open the full-size asset.

## Choose your starting point

**New to GTA?** Read [what GTA does](start/introduction.md), complete
[installation](start/installation.md), and follow the [image-only tutorial](start/first-task.md).

**Already running a robot?** Start with the [hardware checklist](hardware/checklist.md).
The reference workcell uses ROS 2 Humble, UR5e, Robotiq, and ZED; other devices
need compatible interfaces or an adapter.

**Looking up a detail?** Use the [configuration reference](reference/configuration.md),
[controller catalog](reference/controllers.md), or [glossary](glossary.md).

For robot operation, follow the [execution guide](using/execution.md).

```{toctree}
:caption: Start here
:maxdepth: 1
:hidden:

start/introduction
start/installation
start/first-task
```

```{toctree}
:caption: Using GTA
:maxdepth: 1
:hidden:

using/dashboard
using/refinement
using/vision
using/library
using/execution
```

```{toctree}
:caption: Concepts
:maxdepth: 1
:hidden:

concepts/pipeline
concepts/controllers
concepts/frames
concepts/runtime
```

```{toctree}
:caption: Hardware setup
:maxdepth: 1
:hidden:

hardware/checklist
hardware/ros2
hardware/camera
hardware/robot
hardware/tool-geometry
hardware/ros1
```

```{toctree}
:caption: Extend GTA
:maxdepth: 1
:hidden:

extend/controller
extend/robot
extend/camera
extend/gripper
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

reference/configuration
reference/controllers
reference/devices
reference/ros
reference/artifacts
```

```{toctree}
:caption: Troubleshooting
:maxdepth: 1
:hidden:

troubleshooting/installation
troubleshooting/vision
troubleshooting/execution
```

```{toctree}
:caption: Development
:maxdepth: 1
:hidden:

development/testing
development/documentation
glossary
```
