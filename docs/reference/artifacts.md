# Artifacts and storage

GTA writes image, plan, and script files into the active workspace. The normal
dashboard uses the repository workspace; library working copies have isolated
workspace directories. Switching a session does not change the web process's
working directory.

| Artifact | Meaning |
|---|---|
| `environment.jpg` | Captured/working RGB scene image |
| `resized_img.png` | Padded/resized image used for model coordinates |
| `annotated_keypoints.png` | Display annotation image |
| `skills_output.json` | Parameterized controller recipes |
| `keypoint_locations.json` | Named pixel coordinates and descriptions |
| `robot_script.py` | Generated Python robot script |
| `generated_plans/` | Save Workflow exports |
| `library_checkpoints/<id>/` | Versioned saved checkpoint, copied artifacts, agent/workflow state |
| `library_sessions/<id>/` | Independent mutable working copy |
| `execution_logs/<run-id>/` | Frozen script, structured events, server/script or helper logs |

Library execution logs are nested under the session's workspace. Execution
records remain on disk until explicitly removed; there is no automatic
retention policy. An application restart returns to checkpoint selection.
Browser refresh reconnects to the running application's
current session and status.

## Checkpoint versus working copy

A checkpoint preserves the approved task, subtasks, controller recipes,
parameters, keypoint definitions and grounding, images, available script,
messages/events, and agent context. It also stores each agent's provider,
model, system prompt, and effective generation settings.

Saving requires a valid parameterized plan and no active background handlers.
Required state and artifacts are gathered together; the checkpoint becomes
visible only after a complete save. A provider-context export failure aborts
the save.

Loading copies the snapshot without camera capture, model requests, or
pipeline-event replay. OpenAI context is recreated in a new conversation on
the next model operation; Gemini chat content is reconstructed independently.
Feedback does not append to the preparation conversation. Loading does not
make saved coordinates valid for a new physical scene.

Current-machine credentials, camera/calibration, ROS environment, robot
address, and device settings apply to live operations. Saved model/prompt
settings apply to continuation. Requery and regenerate the script for the
current scene and tool configuration.

## Workspace revisions and readiness

Image, grounding, and script readiness are tied to workspace revisions.
Requery/refinement invalidates affected readiness before work begins.
Readiness is restored when the new operation completes successfully;
previous artifact files remain on disk.

Reset to Checkpoint makes a new working copy, retaining previous session
artifacts and execution records on disk. Deleting a session does not delete its
source checkpoint, but it does remove any unsaved feedback and run records in
that session. Stop the application/operations before managing live workspaces.

## Sensitive data

Checkpoints, exports, and logs can contain camera images, task text, model
responses, feedback, conversation content, and local artifact paths. OpenAI
snapshots may include available encrypted reasoning content. Configuration
serialization excludes API keys and `.env`. User messages and logs are
retained as supplied.

Keep credentials out of feedback, screenshots, and shared reports. Review
images, messages, and local paths before sharing checkpoints, exports, or logs.
Exclude `.env` and private runtime artifacts from published source.
