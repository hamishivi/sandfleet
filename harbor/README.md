# Harbor integration

`sandfleet_harbor.py` implements a Harbor 0.23.0 environment using Sandfleet's
durable lease queue and opt-in disk-backed writable storage. Run evaluation
workers independently of training workers. All host paths and hostnames in
this guide are placeholders; supply operator-owned configuration at deployment.

## Backend usage

Use an isolated Sandfleet controller (0.7.2 or newer) configured with `--max-storage-mb`
and bounded CPU/RAM dynamic resource capacity. Zero (the default) disables disk
mode. On the Harbor driver, install **Harbor 0.23.0**, then configure:

Disk mode uses the same private networking as ordinary leases: loopback only and
no routes, plus a `127.0.0.1` proxy port relayed to the egress forwarder for
internet leases. Both the network and writable mounts must pass startup checks.
Failed proxy-relay cleanup retains the owned disk for retry; no host-network
fallback is introduced.
See the [cluster networking prerequisites](../docs/production.md#sandbox-networking).
Stacked local XFS/ext3/ext4 mounts are supported; empty mount reports or any
reported RAM/network filesystem are rejected.
Automatic bind suppression also disables Apptainer's resolver/hosts generation.
Disk mode therefore binds owned, mode-0600 files read-only at `/etc/resolv.conf`
and `/etc/hosts`, verifying their contents before becoming ready. The resolver file is a
snapshot of the host's, which sandboxes cannot reach (they have no route); hosts
contains only IPv4/IPv6 localhost entries. No host directory is exposed by these binds.

```bash
export PYTHONPATH="$SANDFLEET_DIR:$SANDFLEET_DIR/harbor${PYTHONPATH:+:$PYTHONPATH}"
export SANDFLEET_URL=http://evaluation-controller.example:8765
export SANDFLEET_CLIENT_TOKEN_FILE=/path/to/evaluation/client.token
export SANDFLEET_HARBOR_IMAGE_MANIFEST=/path/to/evaluation/images.json
# The raw SANDFLEET_CLIENT_TOKEN also works; do not set both forms.
harbor run --environment-import-path sandfleet_harbor:SandfleetEnvironment \
  --path /path/to/task --agent oracle
```

Sandbox internet (Sandfleet 0.8.0, a controller started with `--sandbox-internet`) is **on by default** whenever
the controller offers it: every task whose network mode is public (Harbor's default) gets per-lease internet.
Pass the environment kwarg `internet=false` (`--ek internet=false` or `environment.kwargs.internet: false`) to
opt out; `internet=true` requires it and fails if the controller does not offer internet. Internet sandboxes get
proxy env and can reach nothing but the forwarder, which refuses cluster-internal addresses: an agent inside the
sandbox cannot call an LLM endpoint on the cluster network. Offline sandboxes have only loopback.

apt is blocked inside sandboxes by default (Sandfleet `SANDFLEET_SANDBOX_BLOCK_APT`, a temporary measure). Harbor
agents that apt-install system dependencies at setup therefore fail unless the task image already contains them.
If an evaluation needs that, set `SANDFLEET_SANDBOX_BLOCK_APT=0` in the environment of that controller's workers.

Commands Harbor sends without a timeout (e.g. agent install steps) get the task's largest agent/verifier
timeout. Trial-level timeout overrides are not visible to the environment; pass `command_timeout_sec` to match
them (Harbor's own phase timeouts still cancel commands).

The backend requires `command-cancel-v1` (Sandfleet 0.7.2). On async exec
cancellation it stops that command through the independent control channel,
waits for the result, and retains the original lease/files for verification.
If interruption fails, it reports the failure and aborts the lease. Matching
controller, agent, worker and client versions are required; do not hot-update
a running fleet.

It also requires `command-env-v1` (Sandfleet 0.7.1) so image/per-command
environment values do not appear in the timeout supervisor's command line.
The backend never falls back to service/admin credentials. It verifies protocol
features, exact requested CPU/RAM, effective memory limit, pinned SIF hash and
writable mounts. It records `sandfleet-environment.json` inside each trial
(a separate verifier: `sandfleet-verifier-environment.json`).
The canonical lease client also checks the disk echo on start/restart: an older
agent cannot silently substitute writable-tmpfs for a requested disk volume.
Missing required protocol/configuration fields are errors, not default values.
Commands preserve output and exit status (including GNU timeout's 124); output
over the 1,000,000-byte transport limit is returned with an explicit truncation
notice in stderr, preserving the command's exit status. File-transfer metadata
commands still reject incomplete output rather than accepting corrupt inventories. Transfers
stream 256 KiB binary chunks, including verifier logs when the sandbox disk is
full. Task/image ENV, WORKDIR and per-command timeouts are respected.

`await env.kill_sandbox_processes()` (Sandfleet 0.12 workers) SIGKILLs every
process in the sandbox except its init and returns how many it killed. Harbor
runs the verifier in the sandbox the agent used, and commands keep their
background processes, so call it after the agent phase and before verification:
otherwise a process the agent left behind (say, a loop rewriting
`/logs/verifier/reward.json`) could tamper with the result. Then reset the
verifier directories. It keeps the lease and filesystem, and raises if any
process survives; do not verify then. The worker does the kill host-side: it
joins the instance's pinned user and PID namespaces and runs the kernel's atomic
`kill(-1)` (never the mount namespace, so no image code runs), then verifies with
a host `/proc` scan, so a sandbox root cannot fake it and a process forked during
the kill is still caught. This closes the leftover-process vector only: a root
policy can still persist through the filesystem (shell/profile env scripts, a
shadowed binary, `ld.so.preload`), so a tampered filesystem is not made safe.
Trusting the sandbox filesystem against its own root is out of scope. Do not run
other commands or file transfers on the environment concurrently with the kill.

**Verifier-only mounts** (Sandfleet 0.13 controller and workers). A task can
declare host directories that only its verifier may see, such as a hidden
target model, in `task.toml`:

```toml
[metadata]
verifier_mounts = [
  { source = "/path/to/shared/verifier-models/verifier-model", target = "/verifier/models/verifier-model" },
]
```

The agent phase never sees them. After the agent phase, the trial calls
`await env.add_verifier_mounts()` before resetting `/logs/verifier` and `/tests`; it
returns `{"added_mounts": [...], "stopped_processes": N}`, or `None` for a task
without declarations, and repeated calls return the first result. Sandfleet
restarts the sandbox from its disk overlay with the directories bound
read-only, so the agent's files persist but every process it left running is
stopped (a verifier that expects an agent-started server will not find it). Use
a dedicated target prefix such as `/verifier`: a symlink or file the agent left
on the way to a target is replaced by a directory. Targets may not cover or sit
inside `/logs/*` or `/tests`, or be under `/etc`, `/proc`, `/sys` or `/dev`.
Each source must be under one of the controller's `readonly_mount_roots`
(`sandfleetd --sandbox-readonly-mount-root`); the environment checks this and
the controller feature `lease-readonly-mounts-v1` before acquiring a lease, and
that the lease's worker supports the operation right after start, so an
unsupported task fails before its agent runs. If adding the mounts raises, do
not run the verifier; release the environment. `harbor run` does not call
`add_verifier_mounts()` by itself.

**Agent-visible mounts** (Sandfleet 0.14). Inputs the agent may read, such as
model weights or data, are declared as `agent_mounts` and mounted read-only when
the sandbox starts:

```toml
[metadata]
agent_mounts = [
  { source = "/path/to/shared/agent-assets/agent-model", target = "/assets/models/agent-model" },
]
```

The same target rules apply, and agent and verifier targets may not overlap.
Sources must be under the controller's `agent_readonly_mount_roots`
(`sandfleetd --sandbox-agent-readonly-mount-root`), which never contain a
verifier-phase root; the environment checks this and the feature
`lease-start-readonly-mounts-v1` before acquiring a lease, and the client checks
that the worker reported each mount after start. The class attributes
`supports_agent_mounts` and `supports_verifier_mounts` say which phases an
environment implements.

**Separate verifier environments** (`[verifier] environment_mode = "separate"`).
Harbor runs the verifier in a fresh environment (session `<trial>__verifier__<key>`)
and copies the task's `[[artifacts]]` into it. There the environment adds the
`verifier_mounts` right after it starts (same add-mounts mechanism and
`readonly_mount_roots`, so verifier-only data never needs an agent root), and a
verifier target may be `/tests` itself (Harbor uploads no tests to a separate
verifier). That lets one **shared verifier image** serve every task: set
`[verifier.environment] docker_image = "<name>"` and register the SIF once under
the manifest's `"named"` map (same fields as an `images` entry); the task's
`tests/` (and any answers) arrive as read-only mounts, so no per-task verifier
image is built. The agent sandbox then has only its `agent_mounts`, plus
`produce_mounts`, which the trial adds with `await env.add_produce_mounts()`
after the agent phase (add-mounts, verifier roots) so a pipeline-owned producer
can run the agent's deliverable on hidden inputs before artifacts are collected.
The trial driver owns this producer phase.
The environment's role comes from its build context (`tests/` for the verifier,
`environment/` for the agent), not from the session id. A task whose agent or
producer mount source is, contains or lies inside a verifier mount source (after
resolving host symlinks) is refused. Harbor collects artifacts best-effort, so
`env.transfer_infra_failures` lists downloads that failed for an infrastructure
reason (lease, transport, host disk; not a missing path or a refused symlink),
for a caller to mask the trial rather than score incomplete artifacts.

**Worker end time** (Sandfleet 0.16). After `start()`, `env.worker_end_time`
is the Unix time at which the lease's worker job ends (Slurm's
`SLURM_JOB_END_TIME`), or None for an older worker. Leases are not placed by
remaining time, so a trial that runs past it is lost with the worker; callers
can compare it with the trial's budget and warn.

**GPU sandboxes** (Sandfleet 0.15 adapter; any controller with GPU sandboxes
enabled). A task asks for GPUs with Harbor's own fields, and may require that
nothing else uses them:

```toml
[environment]
cpus = 32
memory_mb = 196608
storage_mb = 65536
gpus = 1
gpu_types = ["gb300_288gb"]   # optional: exact Sandfleet types or families ("GB300")

[metadata]
exclusive_gpu = true          # optional
```

Before queueing for a lease the environment reads `GET /v1/capabilities/gpus`.
A controller without an operator GPU type catalog, or whose per-sandbox limit is
below `gpus`, or that enables none of the task's `gpu_types`, fails the start
with a clear error before any lease or agent runs. `gpu_types` entries that are
enabled exact types (`gb300_288gb`) select that type; other entries select every
enabled type of that family (`GB300` matches `gb300_<memory>gb`); order is kept
as preference; omitted means any enabled type. The lease is acquired with
`gpu`/`type` resources, so it comes from a dynamic GPU pool whose Slurm step owns
that many whole devices. After start the environment checks that the lease
reports exactly `gpus` devices and identities of an accepted type, and runs
`nvidia-smi` once inside the fresh sandbox (before the agent) to check that the
sandbox sees exactly those GPU UUIDs and no IMEX channel (the worker's Slurm step
can open the host's `/dev/nvidia-caps-imex-channels`; only the contained `/dev`
keeps them out). The UUIDs are recorded under `gpus` in
`sandfleet-environment.json`. `add_verifier_mounts()` restarts the instance in
the same Slurm step and re-checks that the GPUs are unchanged.

`exclusive_gpu` always holds on Sandfleet: a GPU belongs to one slot's Slurm
step (whole-device GRES, device cgroup), prepared slots on a worker must have
disjoint devices, there is no MPS or sharing, and CPU slots fail closed if they
can open a GPU device. The flag is validated (`true`/`false`, needs `gpus >= 1`)
and advertised as `supports_exclusive_gpus`, so a caller can refuse it on an
environment that cannot honour it. GPU sandboxes keep every other isolation
property: network mode, seccomp, capability drops, read-only mounts, the
process kill and disk overlays. Apptainer's GPU mode is the worker's
(`--nvccli` is selected automatically on IMEX hosts such as GB200/GB300).

A worker reserves `slots × gpus` devices, bounded by the controller's
`--dynamic-worker-max-gpus`; the CPU and RAM limits also bound slots. For
example, with a 4-GPU node budget, `--dynamic-worker-max-gpus 4` and the task
shape above (32 CPUs, 192 GiB), one worker holds up to 4 single-GPU sandboxes.

**Network modes.** Offline leases have no network at all (loopback only, no
routes, since Sandfleet 0.9), so the environment advertises
`disable_internet` and runs Harbor's `no-network` tasks (including the legacy
`allow_internet = false`) on offline leases; `internet=true` is refused for them.
Each such lease must also be reported offline by its worker (`network.mode` is
`none`, `internet` false; Sandfleet >= 0.11 workers), at start and after the
`add-mounts` restart, or the start fails. A lease's network is fixed, so phase
policies that differ from the environment's are refused. `allowlist` network
policies remain unsupported.

Cancellation drains the threaded operation, interrupts its own active step if
necessary, and releases the lease. Failed cleanup retains its identity for
retry. Commands are never replayed. Stop always releases the ephemeral sandbox,
including when Harbor passes `delete=False`; persistent stopped environments
are not provided. Normal successive trials reuse the prepared step, not its
old writable filesystem.

### Image manifest and supported scope

Build each **complete** task environment first, convert to a read-only SIF, and
record the build's OCI digest/context and image metadata. This adapter is not a
Dockerfile builder and never substitutes a FROM image for missing task layers.
`image_manifest.context_digest(environment_dir)` hashes every context file's
path, permission bits and contents; it rejects symlinks/special files. A manifest
maps that digest to a reviewed image:

```json
{
  "schema_version": 1,
  "images": {
    "<complete-environment-context-sha256>": {
      "sif_path": "/path/to/shared/eval-images/task.sif",
      "sif_sha256": "<64-lowercase-hex-digest>",
      "build_provenance": "Full-context build recipe and pinned OCI digest",
      "user": "root",
      "workdir": "/workspace",
      "env": {"LANG": "C.UTF-8"}
    }
  },
  "named": {"<shared-image-name>": {"sif_path": "...", "...": "same fields"}}
}
```

The operator is responsible for truthful build provenance and immutable SIFs;
context hashing cannot prove a Docker build was performed correctly. Workers
re-hash the SIF before starting it. Root-user Linux images with Bash and GNU
coreutils/find are currently supported. Non-root user switching, TPUs,
Compose, Windows, `allowlist` `network_mode`, forced rebuilds, resource overrides,
omitted CPU/RAM/storage and CPU/memory enforcement policies other than `auto` or
`guarantee` fail explicitly. Symlinks/special files in transferred directories are rejected, not
silently followed. Validate each task image against these requirements.

### Disk policy

`storage_mb` is the **nominal ext3 image size in MiB**, including filesystem
metadata. The shared immutable SIF is not charged to this writable allowance.
Available bytes are reported separately and can be lower than nominal size
because filesystem metadata and reserved blocks consume space. Establish the
required accounting before comparing task-resource limits across backends.
No resource is silently inflated to hide the difference.

Each image is fully allocated on node-local XFS/ext3/ext4, not sparse, tmpfs or
shared network storage. Disk mode requires util-linux `fallocate`: after
Apptainer formats a sparse temporary image (avoiding full-image zero writes),
the worker explicitly reserves the full nominal extent without changing its
contents, fsyncs it, and verifies allocated blocks.
An incorrect image size is rejected before allocation; missing/failed allocation
or remaining holes fail startup, with no sparse fallback. This does not reserve
space for unrelated files or guarantee against device failure. The worker checks
local free space plus 256 MiB headroom, but there is no scheduler-level disk reservation
or cross-node orphan scavenger. Operators must bound enabled slots/storage,
monitor free disk and inspect retained overlays after node loss. Memory still
includes reclaimable disk page cache; swap is a separate cluster policy.

## Resource validation

Use the pinned Harbor 0.23.0 parser to resolve each task's resources and phase
timeouts. The backend rejects omitted CPU, memory, or storage values instead
of guessing. Build each complete task environment into a pinned SIF; an image
from only a Dockerfile's `FROM` line does not contain its task-specific layers.
Keep model selection, task inventories, infrastructure configuration, and run
records in operator-owned storage outside this repository.

## Validation tools

The standalone `storage_probe.py` checks disk-backed overlays inside an
operator-created, isolated Slurm step. It needs a Python-capable SIF and a new
report path:

```bash
uv run --no-project --python /path/to/python storage_probe.py \
  --image /path/to/python.sif --scratch /path/to/disk-backed/scratch \
  --output /path/to/reports/storage-probe.json
```

The probe creates two fresh overlays sequentially, checks filesystem backing
and available capacity, exercises durable writes and disk-full behavior, and
checks cleanup. Disk I/O can use reclaimable page cache; RAM and swap limits
remain separate. It retains an overlay when instance shutdown cannot be
verified. Run it only with capacity and cleanup procedures appropriate to your
cluster; it does not certify a deployment on a different host.

`live_smoke.py` exercises the backend and an Oracle trial with the synthetic
fixture under `tests/fixtures/basic`. Supply a reviewed image manifest for that
fixture and a separately configured controller:

```bash
python "$SANDFLEET_DIR/harbor/live_smoke.py" \
  --isolated-controller --output /path/to/reports/smoke-results
```

Run the adapter's unit tests in the same Harbor environment:

```bash
PYTHONPATH="$SANDFLEET_DIR:$SANDFLEET_DIR/harbor" \
  "$HARBOR_PYTHON" -m unittest discover -s "$SANDFLEET_DIR/harbor/tests"
```

Before a larger run, verify effective CPU, RAM, GPU, disk, networking, timeout,
and cleanup behavior on the target cluster. Record requested resources, image
identity, incomplete transfers, lost leases, and verifier startup failures so
infrastructure errors remain distinguishable from task failures. Keep those
records and all real deployment identifiers outside the public repository.
