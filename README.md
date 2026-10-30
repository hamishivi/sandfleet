# Sandfleet

Sandfleet provides lease-based, resource-isolated sandboxes backed by Slurm
allocations and Apptainer. A small control plane launches CPU worker jobs;
agents inside those jobs own persistent `srun` steps, so Slurm applies the CPU
and memory cgroups while callers use a stable HTTP API.

Sandfleet has two planes:

- The controller creates pools, launches worker allocations, and issues leases.
- Node agents serve the lease data path and run Apptainer inside persistent
  Slurm steps.

The controller never proxies command output or file content. An acquired lease
contains a node-agent URL and a scoped random token, avoiding a central data
plane bottleneck.

## Status

Sandfleet is intended for a single trusted team on a private cluster network.
It supports elastic pools, job-array scale-up, idle scale-down, automatic worker
replacement, durable desired state, expiring/renewable leases, globally paced
Slurm step creation, prepared-step reuse, command-scoped cancellation and
environment overrides, optional GPU sandboxes, role-separated tokens, and
built-in metrics. It is not a public or hostile multi-tenant security boundary.

An opt-in [Harbor 0.23.0 backend](harbor/README.md) uses pinned SIFs and a fresh,
disk-backed ext3 overlay per trial. Disk mode defaults **off**; existing training
leases keep their writable-tmpfs behavior. The backend rejects unsupported
resources and records effective RAM, disk capacity and image identity per trial.
Disk workers require util-linux `fallocate`; every backing block is reserved and
verified before the sandbox starts.

Each sandbox has its own network namespace and localhost ports. Sandboxes are
offline by default: they get only loopback, with no DNS and no route anywhere
else. An operator can enable
[opt-in sandbox internet](docs/production.md#opt-in-sandbox-internet): leases
that request `internet: true` get credentials for an allowlisted, per-worker
egress forwarder, reached through a `127.0.0.1` proxy port inside the sandbox
that the worker relays from outside. Internet sandboxes have no routes either,
so direct connections and DNS still fail. Sandfleet verifies the namespace and
that it has no routes before allowing commands, and never falls back to the host
network.
See [sandbox networking](docs/production.md#sandbox-networking) for cluster
prerequisites and a live acceptance test. Sandboxes also run under a seccomp
filter and a per-sandbox task limit, and workers keep a JSONL audit log of every
command ([sandbox confinement](docs/production.md#sandbox-confinement)). A
disk-backed lease can add operator-allowlisted host directories as read-only
mounts after it starts, for example verifier-only inputs after an agent phase;
this restarts the sandbox from its overlay, so files persist and processes stop.
A lease can also mount host directories from separate, agent-phase roots
read-only when it starts (0.14), for inputs the agent may read.

Controller state is stored as two fsynced generations. On restart, Sandfleet
loads the newest valid generation, quarantines a corrupt primary, records the
recovery in health and metrics, and refuses to start if no valid generation
exists.

Start with the [human guide](docs/human-guide.md) to deploy a controller,
create pools, and connect client applications. See
[the design notes](docs/design.md) for the resource model and
[the production runbook](docs/production.md) for deployment and monitoring.
The [command reference](docs/commands.md) documents the CLI, controller, and
agent entry points, controller restart recovery, and additional API features. Repository automation and coding agents should read
[`AGENTS.md`](AGENTS.md) before changing the implementation. A generic
controller job template is available in
[`examples/controller.sbatch`](examples/controller.sbatch).

## Development

All Sandfleet processes require Python 3.11 or newer, including deployments
that run directly from the source tree. Older interpreters fail at import.

Sandfleet uses `uv` for Python installation, dependency locking, and command
execution. The checked-in Python version and lockfile make a fresh checkout
reproducible:

```bash
uv sync --frozen
uv run --frozen pytest -q
```

Use `uv run --frozen sandfleet --help` and
`uv run --frozen sandfleetd --help` to run the local checkout without
activating its `.venv`. Production releases use the same lockfile without
development tools; the human guide has the complete shared filesystem
deployment procedure.

## Controller

All paths, hostnames, scheduler settings, and resource sizes below are examples.
Replace `/path/to/...` and `.example` hosts with your own operator configuration.


The controller must run somewhere that can invoke Slurm commands and is
reachable from both worker and caller jobs.

```bash
umask 077
python -c 'import secrets; print(secrets.token_hex(32))' > /path/to/shared/sandfleet/service.token
python -c 'import secrets; print(secrets.token_hex(32))' > /path/to/shared/sandfleet/client.token
python -c 'import secrets; print(secrets.token_hex(32))' > /path/to/shared/sandfleet/metrics.token
export SANDFLEET_TOKEN_FILE=/path/to/shared/sandfleet/service.token
export SANDFLEET_CLIENT_TOKEN_FILE=/path/to/shared/sandfleet/client.token
export SANDFLEET_METRICS_TOKEN_FILE=/path/to/shared/sandfleet/metrics.token

sandfleetd \
  --host 0.0.0.0 \
  --port 8765 \
  --advertise-host "$(hostname -f)" \
  --registry /path/to/shared/sandfleet/controller.json \
  --state-file /path/to/shared/sandfleet/controller-state.json \
  --partition compute \
  --account research \
  --qos normal \
  --agent-advertise-mode fqdn \
  --cpu-allocation-granularity 1 \
  --enable-dynamic-resources \
  --dynamic-worker-max-cpus 128 \
  --dynamic-worker-max-memory-mb 524288 \
  --dynamic-pool-slots-per-job 16 \
  --dynamic-pool-max-capacity 512 \
  --max-dynamic-pools 8 \
  --agent-python /path/to/python \
  --source-root /path/to/shared/sandfleet \
  --log-dir /path/to/shared/sandfleet/logs \
  --apptainer-binary /path/to/apptainer \
  --step-start-rate 8 --step-start-burst 8 \
  --lease-ttl-seconds 3600
```

GPU sandboxes are off by default. Enabling them requires an operator GPU type
catalog and verified Slurm step device isolation; see the
[human guide](docs/human-guide.md).

Create an elastic pool and run a command:

```bash
sandfleet --url http://controller.example:8765 pool create smoke \
  --min-capacity 0 --warm-capacity 2 --max-capacity 32 \
  --slots-per-job 8 --cpus 1 --memory-mb 4096 --idle-timeout 600

sandfleet --url http://controller.example:8765 pool wait smoke --timeout 600

sandfleet --url http://controller.example:8765 exec smoke /path/to/shared/python312.sif \
  'python3 -c "print(6 * 7)"'

sandfleet --url http://controller.example:8765 exec-resources /path/to/shared/python312.sif \
  'python3 -c "print(6 * 7)"' --cpus 4 --memory-mb 8192
```

`exec-resources` asks for CPU and RAM directly. The controller validates the
shape against its per-sandbox, aggregate-capacity, and dynamic-pool limits,
then reuses one homogeneous pool for all matching requests. Clients never
submit raw Slurm options. Named pools remain available for operator-managed
capacity and compatibility. Set the dynamic worker CPU/RAM ceilings to
schedulable per-node budgets for the selected worker partition; Sandfleet
reduces slots per worker for larger sandbox shapes.

Clients can inspect the enabled exact types and cached scheduler availability,
then request a count and an ordered list of acceptable types:

```bash
sandfleet --url "$SANDFLEET_URL" gpus

sandfleet --url "$SANDFLEET_URL" exec-resources /path/to/shared/cuda.sif \
  'nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader' \
  --cpus 4 --memory-mb 32768 --gpu 1 \
  --type h100_80gb --type a100_40gb
```

Availability is advisory and reserves nothing; each prepared lease verifies its
exact GPU count and identities.

`SANDFLEET_TOKEN` is the operator/service credential used by agents and pool
administration. `SANDFLEET_CLIENT_TOKEN` can acquire, renew, use, and release
leases without changing pools. `SANDFLEET_METRICS_TOKEN` can only read
observability endpoints. All three roles are required and never substitute for
one another. The controller requires `SANDFLEET_TOKEN_FILE` because that path is
passed to workers; the other roles accept direct values or permission-restricted
files. The registry contains no token.

Lease acquisition waits in Sandfleet when no slot is ready. Pending requests
drive scale-up and receive an agent lease as workers register. Commands and
files then travel directly between the caller and the selected agent.

## Slurm load protection

- The first demand burst is coalesced for two seconds, then its worker deficit
  is submitted as one homogeneous Slurm job array.
- A single background loop queries all active submissions with one expanded
  `squeue` call, no more than once every 15 seconds.
- Pool status and lease polling read controller memory and never query Slurm.
- Submissions are rate-limited, failed worker boots use exponential backoff,
  and five consecutive boot failures suspend scaling until `pool resume`.
- The controller grants step-start permits from one global token bucket
  (eight/second with burst eight by default). Worker-local concurrency and
  jitter provide a second bound, without producing extra Slurm RPCs.
- Agents prepare resource-limited `srun` steps before advertising capacity.
  Releasing a lease stops Apptainer and recycles the existing step.
- Pending jobs count toward desired capacity, preventing duplicate submission
  while the cluster is busy.

When an agent heartbeat is stale, it stops receiving leases. After the loss
grace period, Sandfleet marks its active leases lost and submits bounded
replacement capacity. If the latest scheduler snapshot already confirms that
the worker vanished, no redundant `scancel` is issued. Otherwise cancellation
is retried with durable bounded backoff while the controller remains available.
The lost sandbox filesystem cannot be recovered; callers receive
`SandboxLostError` and should resample that rollout.

To validate those behaviors on a real cluster with a dedicated test pool:

```bash
python examples/elastic_smoke.py \
  --url "$SANDFLEET_URL" --token "$SANDFLEET_TOKEN" \
  --client-token "$SANDFLEET_CLIENT_TOKEN" \
  --image /path/to/shared/python312.sif --pool elastic-smoke
```

The smoke test intentionally cancels one worker. It verifies batched array
scale-up, Slurm memory enforcement, lost-lease reporting, survivor continuity,
replacement, and idle scale-to-zero, then deletes its pool.

## API overview

- `POST /v1/pools` creates a fixed-profile elastic pool.
- `GET /v1/pools/<name>` reports requested, ready, and free capacity.
- `GET /v1/metrics` returns a machine-readable fleet snapshot for agents.
- `GET /v1/capabilities/gpus` returns operator-enabled exact GPU types plus the
  last reconciler-owned scheduler inventory and request-validity probe.
- `POST /v1/capabilities/gpus/refresh` asks the reconciler to refresh that cache
  and requires the admin token.
- `GET /metrics` exposes Prometheus text format.
- `GET /` serves an auto-refreshing dashboard after a metrics token is entered.
- `POST /v1/lease-requests` queues an acquisition by named pool or declarative
  `{cpus, memory_mb}` resources and drives elasticity. GPU requests add a `gpu`
  count and ordered exact, memory-qualified `type` alternatives, such as
  `{"gpu": 1, "type": ["h100_80gb", "a100_40gb"]}`. Raw Slurm GRES strings,
  physical device IDs, and an ambiguous `any` selector are not accepted.
- `GET /v1/lease-requests/<id>` returns pending or assigned state.
- `POST /v1/pools/<name>/resume` resets a scaling circuit breaker.
- `POST /v1/leases/<id>/renew` extends the lease TTL. The Python client renews
  automatically; dead callers are reaped by the controller.
- `DELETE /v1/leases/<id>` closes the sandbox and returns the slot.
- Agent endpoints under `/v1/leases/<id>/...` start/restart the sandbox,
  execute commands, and transfer files.

Every lease has a distinct token. Controller-to-agent operations use the
service token. Requests are plain HTTP and must remain on a trusted private
cluster network; TLS termination is required before exposing Sandfleet beyond
that network. Bind to a private interface or use firewall rules; do not route
controller or agent ports from the public Internet.

For an agent-readable snapshot:

```bash
curl -H "Authorization: Bearer $SANDFLEET_METRICS_TOKEN" \
  "$SANDFLEET_URL/v1/metrics"
sandfleet --token "$SANDFLEET_METRICS_TOKEN" metrics
```
