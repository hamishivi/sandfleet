# TMAX training with Sandfleet

Sandfleet serves the v1 API used by the Sandfleet backend in
[TMAX's training recipes](https://github.com/hamishivi/tmax/pull/11).
TMAX includes its own HTTP adapter, so training environments do not need to
install the Sandfleet Python package. Install Sandfleet on the controller and
worker hosts as described in the [human guide](human-guide.md).

The training GPU allocation and the sandbox worker allocations are independent.
Run one controller that outlives the training job. Controller, agents, and
prepared workers must run matching Sandfleet sources; do not update an active
fleet in place.

Every path, hostname, scheduler setting, and resource size below is an example.
Supply real deployment configuration privately at launch time.

## Controller and worker capacity

The worker interpreter must expose Linux namespace and pidfd APIs. Check the
exact Python executable configured for the workers, on a worker host:

```bash
WORKER_PYTHON=/path/to/worker-python
"$WORKER_PYTHON" - <<'PY'
import os
import signal

required = ((os, "pidfd_open"), (os, "setns"), (signal, "pidfd_send_signal"))
missing = [name for module, name in required if not callable(getattr(module, name, None))]
if missing:
    raise RuntimeError(f"Worker Python lacks required Linux APIs: {', '.join(missing)}")
pidfd = os.pidfd_open(os.getpid())
try:
    signal.pidfd_send_signal(pidfd, 0)
finally:
    os.close(pidfd)
print("Worker Python capability check passed")
PY
```

Python version alone does not prove these capabilities: some portable builds
omit them. Select a suitable Linux interpreter instead of bypassing namespace
or process-identity checks.

The training adapter has two acquisition modes:

| Training configuration | Controller requirement |
| --- | --- |
| `SANDFLEET_POOL` names a pool | Create that pool with the intended CPU/RAM profile. Its profile determines sandbox resources. |
| `SANDFLEET_POOL` is unset | Enable declarative resources with `--enable-dynamic-resources` and explicit dynamic worker CPU/RAM ceilings. |

In resource mode, the adapter requests **two CPUs** and converts `mem_limit` to
MiB; its default `4g` requests **4096 MiB**. In named-pool mode, the client memory
setting does not resize the pool. Each worker allocation also needs Sandfleet's
512 MiB agent headroom in addition to its sandbox slots.

For an operator-owned named pool, use the service/admin credential:

```bash
export SANDFLEET_URL=http://controller.example:8765
export SANDFLEET_TOKEN_FILE=/path/to/shared/secrets/service.token

sandfleet pool create training-smoke \
  --min-capacity 0 --warm-capacity 0 --max-capacity 16 \
  --slots-per-job 2 --cpus 2 --memory-mb 4096 --idle-timeout 600
```

The single-GPU TMAX recipe configures a pool of 16 environment actors. Provide
enough lease capacity for those actors; the eight rollouts in a learner batch
are not the whole environment pool. Additional training jobs share controller
limits, so use distinct named pools when capacity should be reserved per job.

Leave disk-overlay mode disabled for ordinary training. The Harbor adapter has
separate disk and verification requirements.

## Images and connectivity

Prepare the complete task images as nonempty, immutable SIF files. The training
environment resolves each task image against `SWERL_APPTAINER_SIF_DIR` before
sending its absolute path to Sandfleet. The same files must be visible at the
same absolute paths on the worker hosts. A filesystem visible only inside the
training container is insufficient.

The SIF directory must cover the selected dataset. TMAX matches sanitized image
reference filenames or supported trailing-hash filenames and rejects missing
images by default. A base image from a task Dockerfile's `FROM` line does not
contain the complete task environment. Keep remote image fallback disabled for
a prepared deployment.

Training actors must reach both the controller and the node-agent addresses
returned by lease acquisition. The controller does not proxy commands or file
content. The agents must also reach the controller for registration, renewal,
and step-start permits.

If the job environment sets HTTP proxies, configure `NO_PROXY` and `no_proxy`
for the actual controller and agent hostnames or addresses, as well as
`localhost` and `127.0.0.1`. Configure this on the controller, workers, and
training actors. A bypass for only the controller leaves direct agent requests
subject to the proxy. Ensure the launcher passes these settings into the job;
local shell exports alone do not configure a remote container.

For example, replace these reserved example domains with the deployment's
actual hostnames or worker DNS suffix:

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}controller.example,.workers.example,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
```

Sandboxes are offline by default. The training adapter does not request
sandbox internet access. Bake task dependencies into the images; host-side
model downloads and sandbox-side networking are separate concerns.

## Training client configuration

Run these commands in the TMAX training environment. Give training actors only
the client credential, not service/admin or metrics credentials:

```bash
export SANDFLEET_URL=http://controller.example:8765
export SANDFLEET_POOL=training-smoke
export SANDFLEET_REGISTRY=/path/to/shared/runtime/controller.json
export SWERL_APPTAINER_SIF_DIR=/path/to/shared/sif-pool

# The training adapter reads the direct client-token variable.
CLIENT_TOKEN_PATH=/path/to/shared/secrets/client.token
export SANDFLEET_CLIENT_TOKEN="$(< "$CLIENT_TOKEN_PATH")"
```

`SANDFLEET_REGISTRY` is optional; set it when the controller uses `--registry`
and the file is visible to all actors. Keep `SANDFLEET_URL` as a bootstrap
address. Controller requests can discover a replacement address; commands stay
on their original agent and are never replayed after worker loss.

The Sandfleet CLI accepts `SANDFLEET_CLIENT_TOKEN_FILE`; this TMAX adapter reads
`SANDFLEET_CLIENT_TOKEN` directly. Do not print it or put it in command-line
arguments. When the controller's startup environment retains the file form,
use a separate client environment so the CLI is not given both token forms.

The full RL recipes forward `NO_PROXY` and `no_proxy` alongside the sandbox
configuration. Both full and minimal Ray runtime environments preserve the
Sandfleet client connection settings, `SWERL_*` configuration, and proxy
variables. Minimal mode controls Ray's explicit environment overrides; it does
not remove variables already present in a node's environment. Keep service/admin
credentials out of training-node environments.

## Check the backend before GPU training

Select one complete task SIF containing Bash and the standard command utilities
required by Sandfleet. Run this from TMAX's `training/open-instruct/` directory:

```bash
export SANDFLEET_SMOKE_IMAGE=/path/to/shared/sif-pool/task.sif

uv run python - <<'PY'
import os
import uuid

from open_instruct.environments.sandfleet_backend import SandfleetBackend

backend = SandfleetBackend(image=os.environ["SANDFLEET_SMOKE_IMAGE"], pwd="/")
marker = f"/tmp/sandfleet-smoke-{uuid.uuid4().hex}"
try:
    backend.start()
    result = backend.run_command("printf 'ready'", timeout=30)
    assert result.exit_code == 0 and result.stdout == "ready", result
    backend.write_file(marker, b"example contents")
    assert backend.read_file(marker, binary=True) == b"example contents"
    backend.restart()
    result = backend.run_command(f"test ! -e {marker}", timeout=30)
    assert result.exit_code == 0, result
finally:
    backend.close()
print("Sandfleet backend smoke passed")
PY
```

Then use the existing single-GPU recipe with its Sandfleet backend selected:

```bash
SANDBOX_BACKEND=sandfleet \
  bash scripts/tmax/RL/qwen35_2b_1gpu.sh
```

If `TOOL_CONFIGS` is set, it replaces the generated tool configuration. Include
`"backend": "sandfleet"` and the intended initial working directory in that
JSON; setting `SANDBOX_BACKEND` alone does not rewrite an existing override.
Keep the dataset and its task data consistent when overriding either setting.

Confirm successful sandbox rollouts, finite learner metrics, all requested
learner steps, and a clean exit. Keep job identifiers, paths, tokens, and raw
run evidence outside the public repository. Delete only the dedicated test
pool when it is no longer needed, before stopping its controller.
