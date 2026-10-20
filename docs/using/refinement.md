# Refine a plan

**Prerequisites:** an initialized dashboard and a completed planning stage.
**Outcome:** change the intended part of a task, then review all affected outputs.

## Choose the right stage

| Change you want | Send feedback to | Downstream work |
|---|---|---|
| Reorder or add a subtask | Task Decomposer | Recipes, parameters, and vision are updated |
| Change controllers or their priority | Skill Generator | Parameters and vision are updated |
| Change a distance, angle, force, or timing value | Parameter Setter | Updated parameters continue to vision |
| Correct a point on the displayed captured image | Vision Module | Annotations are refined; regenerate the script |

Refinement updates the selected stage and its downstream outputs.

## Request a change

1. Wait until the workspace is idle.
2. Describe the desired change in natural language.
3. Optionally include a subtask, parameter, or exact value if you have one in mind.
4. Submit to that stage's feedback control.
5. Review the updated stage and every affected downstream stage.
6. Generate a new script only when grounding is current.

Feedback can express a high-level intent without naming a controller or giving
an exact value. For example:

> Can you lift the cube a bit higher?

If you want a specific adjustment, you can also write:

> In subtask 3, increase the lift offset by 0.04 m. Keep the grasp timing
> and the approach subtask unchanged.

Both forms are supported. Subtask numbers and numeric values are optional;
the agents interpret the request using the current plan and conversation context.
Review the updated plan to check that the resulting adjustment matches your intent.

## When a refinement fails

Read the error, inspect the current stage output, and resolve any reported
configuration or response-format issues before retrying. If an otherwise valid
update does not match your intent, give further feedback.

A complete parameter response uses a JSON object containing `skills`.
The application validates that field and checks the parameterized plan before
accepting it.

**Verify:** the requested change appears in the current output, downstream
grounding has completed successfully, and a newly generated script reflects it.
See [model troubleshooting](../troubleshooting/installation.md).
