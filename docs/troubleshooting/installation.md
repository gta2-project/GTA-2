# Installation and model access

Start with the first error in the application terminal.

## The dashboard is missing or blank

Build both frontend entry points with `npm ci` and `npm run build` in
`frontend/`. Run `main.py` from the repository root and open the configured
port. `library.py` is a separate application, normally on port 8001.
Frontend development uses a separate Python service process.

Inspect `/api/health` on the same application. If it reports an initialization
error, fix that before submitting a task. Check browser console errors and
failed local requests.

## An agent is not configured

All four agents need a provider and model. Provider values are `openai` or `gemini`.
The selected provider's key must be present. Configure VisionModule explicitly
even though the code has defaults. See [configuration](../reference/configuration.md).

Restart the application after editing `.env`. An already-exported shell
variable can override the file; check the relevant provider and model settings.

## “Connection error,” authentication, or model rejection

These describe different failure classes:

| Symptom | Check |
|---|---|
| Connection/timeout | DNS, internet access, proxy/VPN, TLS certificates, provider status |
| Authentication/permission | Key validity, project/account access, billing and model entitlement |
| Model not found | Exact model ID and availability for that account/API |
| Unsupported parameter | Compatibility with the client's generation settings and API |
| Image input rejected | Image support for every vision-enabled agent |
| Rate/quota limit | Provider quota and request limits; avoid repeated automatic retries |

Use the provider's official [OpenAI setup](https://developers.openai.com/api/docs/quickstart)
or [Gemini key guide](https://ai.google.dev/gemini-api/docs/api-key).
The health endpoint checks local configuration. Provider authentication and
model access are checked when a task request is sent.

## Parameter feedback reports a missing `skills` field

The refinement parser expects the response schema, including a top-level
`skills` field. A bare array, prose, or a differently wrapped object can fail
even if it looks like a sensible plan. Inspect the response and the effective
prompt for that agent.
Checkpoint continuation preserves saved prompts and generation settings.
See [refinement](../using/refinement.md) and [checkpoint context](../reference/artifacts.md).

## Python can import packages in one terminal but not another

Check `python --version` and `python -m pip --version` in the terminal
starting GTA. Use the intended virtual environment. ROS 2 Humble binary Python
extensions need compatible Python and shared libraries; arbitrary Conda
environments can mix incompatible components. Follow
[ROS 2 environment checks](../hardware/ros2.md).
