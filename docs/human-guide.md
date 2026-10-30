# Sandfleet human guide

This guide explains how to deploy and operate Sandfleet for client jobs on a
private Slurm cluster. It assumes one trusted team, shared storage, Apptainer,
and Slurm cgroup enforcement.

For Harbor evaluation, use the [separate backend guide](../harbor/README.md).

Protocol features such as `command-env-v1`, `command-cancel-v1`, and disk
overlays need matching controller, agent, worker, and client sources. Deploy
them as a fresh fleet; never update a running training fleet in place.

Disk overlays (`--max-storage-mb`) are opt-in; leave the setting at its default
zero for training. Use a separate, bounded evaluation controller and source
directory with independently scoped tokens. Disk size is nominal image MiB, not
usable file bytes; inspect the reported available bytes and agree on benchmark
disk accounting before a full campaign. Disk workers need util-linux
`fallocate`; every backing block is reserved and verified before Apptainer
starts. Retain artifacts and retry failed cleanup; remove test pools before
stopping the test controller.

Read-only mounts after start (`--sandbox-readonly-mount-root DIR`, 0.13) are
opt-in too and need disk overlays. Allowlist only directories whose whole
contents any lease of that controller may read, such as a verifier-only model
pool, never a shared home or a directory with credentials. Adding mounts
restarts the sandbox and stops its processes. See
[sandbox confinement](production.md#sandbox-confinement).

Read-only mounts at start (`--sandbox-agent-readonly-mount-root DIR`, 0.14) are
for inputs the agent may read, such as public model weights or data. Keep
agent-phase and verifier-phase roots apart (for example `/pool/agent` and
`/pool/verifier`); the controller refuses an agent root that contains a verifier
root.

## Mental model

Sandfleet has four relevant objects:

- A **controller** is one long-running service that owns pool state and is the
  only process that calls `sbatch`, `squeue`, and `scancel`.
- A **worker job** is a single-node Slurm allocation submitted by the
  controller. One worker can hold several sandbox slots.
- A **step** is a persistent `srun` execution inside a worker allocation. Each
  ready sandbox slot has exactly one step, which gives it a Slurm-managed CPU
  set and memory cgroup.
- A **lease** temporarily assigns one ready step to one caller. Apptainer starts
  inside that step. Releasing the lease stops Apptainer but retains the step for
  reuse.

For example, 32 ready sandboxes means 32 persistent steps. With
`slots-per-job=8`, those steps are packed into four worker jobs. Commands and Apptainer restarts reuse the
existing steps.

```text
Client applications
    │ acquire/release
    ▼
Sandfleet controller ── sbatch ──► worker allocation
                                      ├─ srun step ─► Apptainer sandbox
                                      ├─ srun step ─► Apptainer sandbox
                                      └─ srun step ─► Apptainer sandbox
```

## Before deployment

Confirm the following:

- Caller jobs and worker nodes can reach the controller hostname and dynamic
  agent ports over a private network.
- The controller can run `sbatch`, `squeue`, and `scancel`; workers can run
  `srun` and unprivileged Apptainer instances.
- All nodes can read the same immutable Sandfleet release, Python environment,
  and SIF images.
- `uv` is available on the login or controller host used to install a release.
  Workers only need the resulting shared `.venv`; they do not run `uv`.
- Slurm constrains job-step CPU sets and RAM through cgroup v2.
- The selected CPU partition, account, QOS, and time limits are valid.

Do not update a release directory while its controller or workers are running.
Controller, agent, and step-worker processes may start at different times, so a
mutable source tree can produce mixed versions. Use a versioned location such
as `/path/to/shared/sandfleet/releases/example-release` and deploy a new controller for an update.

## Install a versioned release

Clone the repository onto storage visible to controller and worker
nodes, then detach at a reviewed tag or commit. Do not run production from a
moving branch checkout.

```bash
git clone https://github.com/hamishivi/sandfleet.git \
  /path/to/shared/sandfleet/releases/example-release
cd /path/to/shared/sandfleet/releases/example-release
git checkout --detach <reviewed-tag-or-commit>
uv sync --frozen --no-dev --python /usr/bin/python3.12
```

The step-worker interpreter must also expose Linux `os.pidfd_open`, `os.setns`,
and `signal.pidfd_send_signal`. Some portable builds omit these APIs even when
their version is recent enough. Select a suitable Linux interpreter for
`SANDFLEET_WORKER_PYTHON`; namespace and process-identity checks must remain
enabled.

`uv` creates the release-local `.venv` and installs the locked Sandfleet
package into it. The source files, lockfile, and `.venv` in this directory must
remain unchanged for the deployment. The supplied job template does write
state and logs under its `runtime/` subdirectory, so the controller service
account needs write access there.

## Create credentials

Create three independent random credentials on shared storage. The files must
be readable by the controller; the service token file must also be readable by
worker jobs.

```bash
umask 077
mkdir -p /path/to/shared/sandfleet/secrets
python3 -c 'import secrets; print(secrets.token_hex(32))' \
  > /path/to/shared/sandfleet/secrets/service.token
python3 -c 'import secrets; print(secrets.token_hex(32))' \
  > /path/to/shared/sandfleet/secrets/client.token
python3 -c 'import secrets; print(secrets.token_hex(32))' \
  > /path/to/shared/sandfleet/secrets/metrics.token
```

The roles are exact and do not fall back to one another:

| Credential | Use |
| --- | --- |
| Service/admin | Worker registration and pool administration |
| Client | Health, lease acquisition, execution, renewal, and release |
| Metrics | Dashboard data, JSON metrics, and Prometheus scraping |

Never put token values in a command line, registry file, Git repository, job
name, or Slurm log. Prefer the corresponding `_FILE` environment variables.

## Launch one controller

Run one controller independently of the client jobs that use it. If three
training jobs share a controller, the controller must outlive all three; do not
attach its lifecycle to only one training job.

The included template accepts site-specific settings through environment
variables:

```bash
export SANDFLEET_ROOT=/path/to/shared/sandfleet/releases/example-release
export SANDFLEET_TOKEN_FILE=/path/to/shared/sandfleet/secrets/service.token
export SANDFLEET_CLIENT_TOKEN_FILE=/path/to/shared/sandfleet/secrets/client.token
export SANDFLEET_METRICS_TOKEN_FILE=/path/to/shared/sandfleet/secrets/metrics.token
export SANDFLEET_WORKER_PARTITION=compute
export SANDFLEET_WORKER_ACCOUNT=my-cpu-account
export SANDFLEET_WORKER_QOS=my-cpu-qos       # optional
export SANDFLEET_WORKER_TIME_LIMIT=04:00:00     # longer than your longest trial
export SANDFLEET_AGENT_ADVERTISE_MODE=fqdn   # fqdn, hostname, or route-ip
export SANDFLEET_CPU_ALLOCATION_GRANULARITY=1
export SANDFLEET_ENABLE_DYNAMIC_RESOURCES=1
export SANDFLEET_DYNAMIC_WORKER_MAX_CPUS=128
export SANDFLEET_DYNAMIC_WORKER_MAX_MEMORY_MB=524288
export SANDFLEET_DYNAMIC_WORKER_MAX_GPUS=0
export SANDFLEET_DYNAMIC_POOL_SLOTS_PER_JOB=16
export SANDFLEET_DYNAMIC_POOL_MAX_CAPACITY=512
export SANDFLEET_DYNAMIC_POOL_WARM_CAPACITY=0
export SANDFLEET_DYNAMIC_POOL_IDLE_TIMEOUT_SECONDS=600
export SANDFLEET_MAX_DYNAMIC_POOLS=8
export SANDFLEET_AGENT_PYTHON="$SANDFLEET_ROOT/.venv/bin/python"
export SANDFLEET_APPTAINER_BINARY=/usr/bin/apptainer
export PATH="$SANDFLEET_ROOT/.venv/bin:$PATH"

sbatch --partition=compute --account=my-cpu-account \
  "$SANDFLEET_ROOT/examples/controller.sbatch"
```

For a GPU-specific controller deployment, replace the zero GPU worker limit
and add the explicit catalog and isolation gate only after a live step-level
device-cgroup verification:

```bash
export SANDFLEET_DYNAMIC_WORKER_MAX_GPUS=8
export SANDFLEET_MAX_GPUS_PER_SANDBOX=2
export SANDFLEET_GPU_TYPE_CATALOG=/path/to/shared/sandfleet/gpus.json
export SANDFLEET_GPU_DEVICE_ISOLATION_VERIFIED=1
export SANDFLEET_GPU_CAPABILITY_REFRESH_SECONDS=300
```

The catalog maps each exact, memory-qualified name to one Slurm GRES type:

```json
{
  "types": {
    "h100_80gb": {"slurm_type": "h100-80gb", "memory_mb": 81920},
    "a100_40gb": {"slurm_type": "a100-40gb", "memory_mb": 40960}
  }
}
```

Keep one controller bound to one trusted account/QOS. Deploy a separate
controller when another caller has a different placement policy; clients never
choose an account, partition, or QOS.

The controller writes its private URL to
`$SANDFLEET_ROOT/runtime/controller.json`. Export the URL and least-privileged
credential in operator and caller shells:

```bash
export SANDFLEET_URL="$($SANDFLEET_AGENT_PYTHON -c \
  'import json, os; print(json.load(open(os.environ["SANDFLEET_ROOT"] + "/runtime/controller.json"))["url"])')"
export SANDFLEET_CLIENT_TOKEN_FILE=/path/to/shared/sandfleet/secrets/client.token

sandfleet health
```

A hard controller-host failure cannot run cleanup logic. The controller state
file and its `.bak` generation let a restart recover desired and lost-worker
cleanup state. If recovery uses the fallback, health and metrics identify the
quarantined primary. With `--registry`, recovery-enabled agents and clients
follow a restarted controller to a new address; see
[controller restart recovery](commands.md#controller-restart-recovery).
Sandfleet does not provide active/active controllers.

## Create pools

A pool has one immutable per-sandbox CPU/memory profile. Use an elastic pool
for training:

```bash
export SANDFLEET_TOKEN_FILE=/path/to/shared/sandfleet/secrets/service.token

sandfleet pool create train-a \
  --min-capacity 0 \
  --warm-capacity 16 \
  --max-capacity 512 \
  --slots-per-job 16 \
  --cpus 1 \
  --memory-mb 4096 \
  --idle-timeout 600
```

Wait for the warm slots, then prove the complete lease, step, SIF, and cleanup
path before starting training:

```bash
sandfleet pool wait train-a --capacity 16 --timeout 900
sandfleet exec train-a /path/to/shared/images/sandbox.sif \
  'printf "sandfleet ready\\n"' --timeout 120
```

When declarative resources are enabled, callers may instead select CPU and RAM
directly without pool-administration credentials:

```bash
sandfleet exec-resources /path/to/shared/images/sandbox.sif \
  'printf "dynamic sandfleet ready\\n"' \
  --cpus 4 --memory-mb 8192 --timeout 120
```

The first request creates a controller-managed resource shape; later matching
requests reuse it. The controller's per-sandbox, aggregate-capacity, pool-count,
pending-request, and dynamic-shape limits all apply. Dynamic pools scale to zero
but remain in the catalog for reuse. Operators can inspect or delete them with
the normal pool commands. Set the worker CPU and memory ceilings to schedulable
single-node budgets for the selected partition; Sandfleet automatically packs
fewer large sandboxes into each worker.

A client that requests a resource shape instead of a named pool requires
`SANDFLEET_ENABLE_DYNAMIC_RESOURCES=1` plus explicit dynamic worker CPU and
memory ceilings on the controller. If they are absent, acquisition fails with a
configuration error naming `--enable-dynamic-resources`; it is not a capacity
wait and should not be retried indefinitely.

`--cpus` is passed directly to Slurm as the step's CPU allocation request.
Sandfleet accepts site-specific core granularity—for example, one Slurm CPU may
produce a cpuset containing two sibling hyperthreads. It still verifies that
every step has a distinct cgroup, stays within its worker allocation, and has a
cpuset disjoint from the worker's other sandbox steps.

On sites where each `srun` step rounds up to a whole physical core, set
`SANDFLEET_CPU_ALLOCATION_GRANULARITY` to the scheduler's indivisible CPU
allocation size. A two-thread whole-core site uses `2`. This changes only the
parent worker reservation: the sandbox still receives the `--cpus` value from
its resource profile. Leave the default `1` on CPU-granularity clusters.

The target number of ready steps is:

```text
clamp(active leases + pending requests + warm capacity,
      min capacity,
      max capacity)
```

Set `max capacity` at or above the caller's peak simultaneous lease demand,
including resampling and retry headroom. Demand above the ceiling remains in
the FIFO acquisition queue and can exceed the caller's acquisition timeout;
elasticity never overrides this operator-owned safety limit.

For three independent training jobs that require reserved capacity, prefer
`train-a`, `train-b`, and `train-c`. Declarative requests share resource-shaped
pools, one client identity, global FIFO assignment, and no per-client quota; an
early burst from one job can consume matching capacity first.

For a 256-rollout job, start with 16 slots per worker and 16 worker array tasks.
`slots-per-job` remains configurable: smaller workers reduce the number of
rollouts lost with a preempted allocation, while larger workers reduce agent
and Slurm job-record overhead. The number of persistent sandbox steps remains
256 either way.

If three jobs may each use 32 sandboxes, configure aggregate capacity for
96 slots. At the default eight step starts per second, creating those steps
takes at least 12 seconds after workers begin running. Slurm
queue time is additional. With 16 slots per worker, the fleet uses roughly 96
worker array tasks. Confirm acceptable persistent-step counts and start rates
with the cluster administrators before a large run.

## Connect a Python client

Client applications share the controller URL and a client credential. Keep the
credential in a permission-restricted file:

```bash
export SANDFLEET_URL=http://controller.example:8765
export SANDFLEET_CLIENT_TOKEN_FILE=/path/to/shared/sandfleet/secrets/client.token
```

The Python client reads the protected file through the same credential helper
as the CLI. Request a resource shape or choose a named pool:

```python
import os

from sandfleet.client import SandfleetClient
from sandfleet.http import secret_from_env

client = SandfleetClient(
    os.environ["SANDFLEET_URL"],
    secret_from_env("SANDFLEET_CLIENT_TOKEN"),
)
with client.acquire(cpus=1, memory_mb=1024) as lease:
    lease.start(image="/path/to/shared/images/sandbox.sif")
    result = lease.exec("printf 'hello\n'")
    print(result)
```

Use `client.acquire(pool="example")` for a named pool. GPU requests pass `gpu`
and an ordered list of exact `gpu_type` alternatives;
`sandfleet gpus` shows the operator-enabled types and cached availability.

Controller traffic covers acquisition, polling, renewal, and release. Commands
and file operations go directly to the selected node agent. A lost sandbox
raises an error; callers must handle the lost work explicitly rather than
replay a possibly non-idempotent command.

## Operate and monitor

Useful commands:

```bash
sandfleet health
sandfleet pool status train-a
sandfleet pool wait train-a --capacity 16 --timeout 900
SANDFLEET_METRICS_TOKEN_FILE=/path/to/shared/sandfleet/secrets/metrics.token \
  sandfleet metrics
```

Observability endpoints are:

- Dashboard: `http://<controller>:8765/` (enter the metrics credential)
- JSON: `GET /v1/metrics`
- Prometheus: `GET /metrics`
- Health: `GET /v1/health`

Watch ready versus target capacity, pending leases, scheduler health,
`last_scale_error`, `scaling_suspended`, stale/lost agents, and agent
`slot_errors`. Pool status is cached controller state and does not call Slurm.

If repeated worker boot failures suspend scaling, fix the underlying account,
partition, executable, filesystem, or cgroup problem, then run:

```bash
sandfleet pool resume train-a
```

## Shut down safely

Pool deletion serializes with worker submission and lease assignment. It may
wait for a bounded in-flight operation so that every new job and lease is
included in cleanup. If cancellation fails, retain the pool/state and retry;
do not remove state to make the error disappear. Restart recovery reclaims
submission records left without a pool by older deletion races.

Stop callers first, then delete their pools using the service/admin credential:

```bash
sandfleet pool delete train-a
sandfleet pool delete train-b
sandfleet pool delete train-c
```

Pool deletion releases known leases and cancels its worker allocations. After
all pools are gone, stop the controller job and verify that no Sandfleet jobs
remain in `squeue`. Remove obsolete test credentials; retain production
credentials only in their permission-restricted secret location.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Controller job is pending | Slurm `Reason`, account/QOS, partition capacity |
| Worker is pending | Requested `slots × CPU`, `slots × memory + 512 MiB`, node fit |
| `ready_capacity` stays zero | Agent log, `slot_errors`, `srun`, cgroup v2 files |
| Scaling is suspended | `last_scale_error`; fix it and use `pool resume` |
| Caller cannot acquire | Correct pool, client token, target/max capacity, routing |
| Caller reaches controller but not agent | Compute-node DNS, dynamic-port firewall |
| Lost worker has `cleanup_pending=true` | Inspect `last_cleanup_error`; Sandfleet retries with bounded backoff |
| Release or pool deletion fails | Restore agent/Slurm reachability, then retry; cleanup state is retained |
| `state_recoveries` increased | Inspect `persistence.last_recovery`; preserve the quarantined `.corrupt-*` file for diagnosis |
| Controller reports no valid state generation | Preserve the state, `.bak`, and `.tmp` files; do not delete them or restart with an empty path |
| Memory allocation exceeds requested RAM | Inspect `memory.swap.max`; swap may be unlimited |
| Container start fails | SIF visibility, Apptainer path, fakeroot, writable tmpfs |
| Commands fail before running | Image must contain `bash`; in the default PID namespace, commands with a command ID need util-linux `setsid -w`; others need GNU `timeout` |

Sandfleet validates a finite `memory.max` and the effective CPU set before a
slot becomes ready. It cannot impose a root-owned swap constraint. A strict
RAM-plus-swap limit requires Slurm administrator configuration such as
`ConstrainSwapSpace=yes` with the site's chosen `AllowedSwapSpace`.

## Security boundary

Each sandbox has private localhost ports. Sandboxes are offline by default:
only loopback, no DNS, no route to the cluster or the internet. With
`--sandbox-internet` on the controller, leases that request internet reach
allowlisted hosts through the worker's authenticated egress forwarder, and
nothing else. See the [networking prerequisites and live
test](production.md#sandbox-networking) before upgrading workers. A network
setup failure stops sandbox startup rather than sharing the host network.
Prepared-slot readiness does not check network prerequisites; these are
verified at instance startup.

Sandfleet is for a trusted team on a private cluster network. It uses plain
HTTP and shared Unix identities, and does not provide per-client authorization
or hostile multi-tenant isolation; disk limits apply only to opt-in disk
overlays. Sandboxes do run under a seccomp filter and a per-sandbox task limit,
and every command is written to an audit log (see
[sandbox confinement](production.md#sandbox-confinement)), but these reduce
kernel attack surface rather than make the shared UID a security boundary. Do not expose controller or agent ports publicly. For
adversarial workloads, add stronger filesystem and network isolation,
short-lived worker credentials, quotas, audit logs, TLS, and usually a VM or
microVM boundary.

For every command-line option, continue with the
[command reference](commands.md). For alerting, recovery, and token rotation,
see the [production runbook](production.md).
