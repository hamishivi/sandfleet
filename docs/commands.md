# Sandfleet command reference

Sandfleet installs three executables:

| Command | Audience | Purpose |
| --- | --- | --- |
| `sandfleet` | users and operators | Inspect pools, acquire a sandbox, and run commands |
| `sandfleetd` | operators | Run the controller and submit elastic Slurm workers |
| `sandfleet-agent` | internal | Serve prepared sandbox slots inside a worker allocation |

Run `<command> --help` for the complete option list for the installed version.

## Client configuration

Global options must appear before the subcommand:

```bash
sandfleet --url http://controller.example:8765 --token "$TOKEN" health
```

`--url` defaults to `SANDFLEET_URL`. An explicit `--token` always wins. Without
it, the CLI requires the exact role credential for the operation and supports
either a direct value or its `_FILE` counterpart:

| Operation | Credential preference |
| --- | --- |
| Pool create, delete, resume | `SANDFLEET_TOKEN[_FILE]` |
| Metrics | `SANDFLEET_METRICS_TOKEN[_FILE]` |
| Health, GPU capabilities, status, wait, exec, exec-resources | `SANDFLEET_CLIENT_TOKEN[_FILE]` |

The client token cannot change pools, and the metrics token can only read
observability endpoints. Roles do not silently fall back to a more privileged
credential.

## Health and metrics

```bash
sandfleet health
sandfleet metrics
```

`health` verifies that the controller is reachable and reports its cached
Slurm scheduler health. It does not run `squeue` for the request. It also
advertises `features` (`disk-overlay-v1`, `file-chunks-v1`, `lease-abort-v1`,
`command-env-v1`, `command-cancel-v1`, `lease-internet-v1`,
`lease-kill-processes-v1`, `lease-readonly-mounts-v1`,
`lease-start-readonly-mounts-v1`), the operator's `max_storage_mb` policy (zero
by default), `readonly_mount_roots` (verifier-phase roots for `add-mounts`) and
`agent_readonly_mount_roots` (agent-phase roots for start-time mounts), both
empty by default.

`command-cancel-v1`: pass a fresh unique `command_id` to
`lease.run_command(..., command_id=...)`, then call `lease.cancel_command(id)`
concurrently to stop that command without discarding its filesystem. Check the
cancellation response for errors and `interrupted`; a request is not proof of
successful interruption. Do not reuse command IDs. Existing calls without IDs
retain their previous behavior.

`command-env-v1`: Python `lease.run_command(..., env={...})`
uses `/exec-env` and a distinct worker operation. Older agents/workers reject
the operation rather than silently ignoring the environment. Overrides must
have shell-variable names and literal NUL-free string values; they travel via
the process environment with Apptainer `--no-eval`, not supervisor argv.

For a separate disk-enabled controller, `sandfleetd --max-storage-mb N` enables
nominal ext3 image sizes up to N MiB. The controller propagates the same policy
to agents and prepared steps. The operator must bound slots and local disk
capacity; this is not a Slurm disk reservation or a new pool resource dimension.
Disk workers also require util-linux `fallocate`; formatted overlays are explicitly
allocated and fsynced, then checked for exact size and full backing allocation.
The temporary format uses `apptainer overlay create --sparse` to skip zero-filling;
the final overlay must still be fully allocated before use.
Dynamic worker RAM caps include the 512 MiB agent headroom (for example, a
1024 MiB task needs `--dynamic-worker-max-memory-mb` of at least 1536).

The canonical Python lease `start(..., storage_mb=N, image_sha256=SHA256)`
requests this mode for an absolute prebuilt `.sif`; `restart()` preserves those
fields but creates a fresh writable volume. Omitting storage preserves the
legacy start payload. Disk mode forbids caller-supplied Apptainer start flags.

`read_file_chunk(path, offset=N, size=262144)` and
`write_file_chunk(path, content, offset=N, truncate=False)` add bounded binary
transfers to the existing file routes. Chunk sizes are at most 1 MiB; only the
first write (offset zero) may truncate. Legacy whole-file requests are unchanged.
`lease.abort()` interrupts that lease's step without waiting for its active exec
lock. It does not release the lease; callers must drain their operation and call
`close()`. Never replay an interrupted command. A failed close retains the lease
so cleanup can be retried. See [the Harbor adapter](../harbor/README.md).

`lease-kill-processes-v1`: `lease.kill_processes()` (`POST
/v1/leases/<id>/kill-processes`) sends SIGKILL to every process in the sandbox
except its init (PID 1), including work a command left running in the
background, which commands otherwise keep. Call it before running a verifier
in the same sandbox, so nothing an agent started can still write its results.
It returns `killed`, the number of processes found and killed, and
`cancelled_commands`, the IDs of commands that were running; those report
`cancelled`. Processes still present after 10 seconds are an error. The
filesystem, the instance and the lease are kept, and later commands work as
before. It does not wait for a running command and is recorded in the audit
log. Older agents answer 404.

`lease-readonly-mounts-v1`: `lease.add_readonly_mounts([{"source": host_dir,
"target": sandbox_dir}, ...])` (`POST /v1/leases/<id>/add-mounts`) adds host
directories to a started disk-overlay sandbox as read-only mounts, for inputs
that only a verifier may see (for example a hidden model). Each source must
resolve, on the worker and after symlinks, inside one of the operator's
`sandfleetd --sandbox-readonly-mount-root DIR` roots (repeatable; none disables
the operation), and may not have another mount below it: Apptainer binds
recursively but makes only the top mount read-only. Targets are absolute, normalized and unique, may not nest, may
not be `/` or under `/proc`, `/sys`, `/dev`, `/etc` or `/.singularity.d`, and
may not cover `/tmp`, `/var/tmp`, `/root` or the working directory. Paths may
not contain `:`, `,` or control characters; at most 16 mounts per call.

Apptainer cannot add a mount to a running instance (`apptainer exec --bind`
into an instance ignores the bind), so the worker restarts the sandbox: it
SIGKILLs every sandbox process (host-side, as `kill-processes` does), makes each
target a real directory (removing a symlink or non-directory on the way to it;
an existing directory's contents are hidden by the mount, not deleted), stops
the instance, and starts it again from the same disk overlay with `--bind
<source>:<target>:ro` for every mount added so far. **Files persist; processes do
not**, including servers a verifier might expect to find running. The new
instance passes the same network, seccomp, PID-namespace and storage checks as
a fresh start. The worker then confirms that each target is the source
directory (same device and inode) on a read-only mount, using its own Python
helper that joins only the instance's user namespace (the overlay is a FUSE
mount that refuses callers outside it) and opens the target beneath the
instance init's `/proc/<pid>/root` without following any symlink, and it reads
the init's mount table from the host: each target is mounted read-only with no
mount below it, and no other network or FUSE mount is visible. Instance start
runs the overlay's `/.singularity.d/startscript`, which the agent could rewrite,
so the worker kills every sandbox process once more after the restart. The response is the new lease status
plus `added_mounts` and `stopped_processes`; lease status reports
`readonly_mounts` and `readonly_mount_roots`. If the restart fails the sandbox
is closed and must be released. Writable-tmpfs sandboxes are refused (their
files would not survive the restart), as are workers whose seccomp profile does
not deny every mount syscall: fakeroot root keeps `CAP_SYS_ADMIN` in its user
namespace and could otherwise remount the bind read-write. The operation holds
the lease's exec lock like `restart`, is audited, and is answered 404 by agents
older than 0.13.

`lease-start-readonly-mounts-v1`: `lease.start(image, ...,
readonly_mounts=[{"source": host_dir, "target": sandbox_dir}, ...])` (the
`readonly_mounts` field of `start`/`restart`; `restart()` reuses it) mounts host
directories read-only from the moment the sandbox starts, for inputs the agent
may read (for example model weights or data). It works in writable-tmpfs and
disk-overlay mode and needs no restart: in disk mode the targets are created in
the fresh overlay (`apptainer overlay create --create-dir`) before the first
start. Each source must resolve inside one of the operator's agent-phase roots,
`sandfleetd --sandbox-agent-readonly-mount-root DIR` (repeatable; none disables
start-time mounts), never a verifier-phase root: no agent root may equal or
contain a verifier root (checked when configured and again after resolving
symlinks on the worker), so a wrong task mount list cannot show a verifier-only
model to the agent. Source, target, nested-mount, seccomp and verification rules
are those of `add-mounts` above, and the mounts count towards its 16-mount limit
and overlap check. The start fails (and the lease is broken) on any refusal;
`start`/`restart` audit records carry `readonly_mounts`, including refused ones.
Lease status reports every mount with its `phase` (`agent` or `verifier`) and
`agent_readonly_mount_roots`. Agents older than 0.14 ignore the field, so the
Python client raises unless the worker reports each requested mount.

This closes the leftover-process vector only: it stops work an agent left
running (background loops, daemons) from acting during a later verifier in the
same sandbox. It does not undo filesystem changes. A sandbox root can still
arrange to run code on the next command through the filesystem (shell/profile
env scripts, a shadowed `bash` or other binary, `ld.so.preload`); defending the
sandbox filesystem against its own root is out of scope. The kill itself does
not trust the image: the worker spawns an isolated Python helper that joins the
instance's user and PID namespaces (pinned by held file descriptors from start,
so the namespace cannot be reused by another sandbox mid-kill) — not the mount
namespace, so only the worker's own interpreter runs, never image code — and runs
the kernel's atomic `kill(-1)`, then verifies with a host-side `/proc` scan and
repeats until nothing remains. `kill(-1)` is
scoped by the joined PID namespace: it reaches processes forking concurrently
but skips the namespace's init and the worker, and cannot reach another sandbox
of the shared host UID. Do not run other commands or file transfers on the lease
concurrently with a kill; their in-sandbox helper processes are killed too, so
they fail.

`metrics` returns the same JSON snapshot as `GET /v1/metrics`: fleet totals,
pool and agent state, rate-limiter configuration, controller limits, and
counters. Prometheus uses `GET /metrics`; the browser dashboard is at the
controller root URL.

`sandfleet gpus` returns the exact memory-qualified types enabled by the
operator and the last cached scheduler inventory/request-validity check. It
never contacts Slurm from the HTTP handler. Counts such as `free_estimate` and
`largest_node_free_estimate` are advisory and do not reserve capacity.

## Pool commands

A pool defines one immutable sandbox resource profile plus its elastic policy.
All pools managed by one controller use that controller's worker partition and
account.

### Create an elastic pool

```bash
sandfleet pool create rollouts \
  --min-capacity 0 \
  --warm-capacity 8 \
  --max-capacity 512 \
  --slots-per-job 16 \
  --cpus 1 \
  --memory-mb 4096 \
  --idle-timeout 600
```

Important options:

- `--min-capacity`: capacity retained even with no demand.
- `--warm-capacity`: spare slots added above active and queued leases.
- `--max-capacity`: hard ceiling for this pool.
- `--slots-per-job`: sandboxes packed into each single-node Slurm worker.
- `--cpus` and `--memory-mb`: resources for each sandbox step, not the whole
  worker. Worker requests are approximately `slots × resources`, plus 512 MiB
  of agent memory headroom. CPU counts use Slurm's allocation units; a
  core-granularity site may expose multiple sibling threads for one requested
  CPU.
- `--idle-timeout`: seconds a completely unused surplus worker remains before
  Sandfleet cancels it.

The elastic target is:

```text
clamp(active leases + pending requests + warm capacity,
      min capacity,
      max capacity)
```

Creation is idempotent when the existing specification is identical. Resource
profiles are immutable; delete and recreate a pool to change one.

### Create a fixed-size pool

```bash
sandfleet pool create fixed \
  --capacity 32 --slots-per-job 8 --cpus 1 --memory-mb 4096
```

`--capacity` is shorthand for equal minimum, warm, and maximum capacity. It is
mutually exclusive with `--max-capacity`.

### Inspect or wait for capacity

```bash
sandfleet pool status rollouts
sandfleet pool wait rollouts --capacity 8 --timeout 900
```

`status` reads controller memory and never contacts Slurm directly. It includes
prepared/free capacity, lease demand, worker jobs, cached scheduler states, and
scaling errors.

`wait` polls until the requested number of prepared Slurm steps is ready. If
`--capacity` is omitted, it waits for the pool's current elastic target. Client
polling uses bounded exponential backoff and jitter.

### Resume suspended scaling

```bash
sandfleet pool resume rollouts
```

Repeated worker boot failures open a circuit breaker to avoid continually
submitting bad jobs. After correcting the partition, account, environment, or
filesystem problem, `resume` clears that failure count and wakes reconciliation.

### Delete a pool

```bash
sandfleet pool delete rollouts
```

Deletion releases known leases, cancels the pool's Slurm worker submissions,
and removes its desired state. It is intentionally destructive. Agent cleanup
or `scancel` failure leaves the affected controller state intact; correct the
underlying failure and retry the command.

## Execute one command in a sandbox

```bash
sandfleet exec rollouts /path/to/shared/images/python312.sif \
  'python3 -c "print(6 * 7)"' \
  --acquire-timeout 900 --timeout 120
```

The command performs the complete lease lifecycle:

1. Queue a lease request with the controller.
2. Wait while Sandfleet starts elastic capacity if necessary.
3. Receive a prepared agent slot and a lease-scoped token.
4. Start the SIF as an Apptainer instance inside its existing `srun` step.
5. Execute the shell command and print a JSON result.
6. Stop Apptainer and return the still-running step to the pool.

The CLI exits with the sandbox command's exit code. `--timeout` governs both
the sandbox command and image start allowance; `--acquire-timeout` governs only
capacity acquisition.

To select resources without a pre-created pool:

```bash
sandfleet exec-resources /path/to/shared/images/python312.sif \
  'python3 -c "print(6 * 7)"' \
  --cpus 4 --memory-mb 8192 \
  --acquire-timeout 900 --timeout 120
```

The controller must enable declarative resources. It validates CPU and RAM,
then maps the request to a reusable homogeneous dynamic pool. GPU syntax is
`--gpu 1 --type h100_80gb --type a100_40gb`; repeated `--type` values are
ordered exact alternatives. Query `sandfleet gpus` first. The controller
prefers an alternative with enough advisory per-node free capacity, then Slurm
makes the authoritative allocation decision. An ambiguous `any` selector and
raw GRES values are rejected.

## Controller: `sandfleetd`

The controller may run beside a training process on a GPU node while submitting
independent workers to a CPU-only partition:

```bash
sandfleetd \
  --host 0.0.0.0 --port 8765 \
  --advertise-host "$(hostname -f)" \
  --registry "$RUN_DIR/controller.json" \
  --state-file "$RUN_DIR/controller-state.json" \
  --partition compute \
  --account "$CPU_ACCOUNT" \
  --qos "$CPU_QOS" \
  --agent-advertise-mode fqdn \
  --cpu-allocation-granularity 1 \
  --enable-dynamic-resources \
  --dynamic-worker-max-cpus 128 \
  --dynamic-worker-max-memory-mb 524288 \
  --dynamic-pool-slots-per-job 16 \
  --dynamic-pool-max-capacity 512 \
  --dynamic-pool-warm-capacity 0 \
  --dynamic-pool-idle-timeout-seconds 600 \
  --max-dynamic-pools 8 \
  --source-root /path/to/shared/sandfleet \
  --agent-python /path/to/shared/sandfleet/.venv/bin/python \
  --worker-python /path/to/shared/sandfleet/.venv/bin/python \
  --log-dir "$RUN_DIR/logs" \
  --apptainer-binary /usr/bin/apptainer \
  --cleanup-on-exit
```

Required operational groups:

- GPU resources additionally require `--gpu-type-catalog`,
  `--gpu-device-isolation-verified`, positive `--max-gpus-per-sandbox` and
  `--dynamic-worker-max-gpus` limits. The catalog maps each public exact type
  to a distinct trusted Slurm GRES type and memory size. The verification flag
  is an operator assertion; prepared steps still fail closed unless the device
  cgroup exposes exactly their allocated GPU count.
- `--gpu-container-mode {auto,nv,nvccli}` selects Apptainer's GPU integration
  (default `auto`). Each worker picks `nvccli` where `--nv` cannot build the
  contained `/dev`, i.e. GB200/GB300 hosts whose
  `/dev/nvidia-caps-imex-channels` is a directory (this requires
  `nvidia-container-cli` there), and `nv` everywhere else.
- `--gpu-capability-refresh-seconds` controls the reconciler-owned scheduler
  cache (default 300; minimum 30 when GPUs are enabled). Admins can request an
  immediate refresh with `POST /v1/capabilities/gpus/refresh`; ordinary client
  reads never call Slurm.
- Worker placement: `--partition`, `--account`, optional `--qos`, and
  `--time-limit` (health `worker_time_limit_seconds`; null when unlimited).
  Leases are not placed by a worker's remaining time, and a lease ends with its
  worker job; each worker reports that job's end as `worker_end_time` (see
  0.16.0) so callers can warn about long trials.
- Shared runtime: `--source-root`, Python paths, Apptainer path, cache/temp
  paths, log directory, state file, and optional registry.
- Fleet limits: maximum aggregate capacity, per-sandbox CPU/memory bounds,
  pool count, pending request count, and maximum workers per scale action.
- Dynamic-resource policy: explicit enablement, slots per worker, per-shape
  maximum/warm capacity, idle timeout, maximum number of shapes, and the CPU
  and RAM budgets that one worker job may request.
- Slurm protection: scheduler poll interval, submit interval, scale debounce,
  step-start rate/burst, worker-local start concurrency, and launch jitter.
- Failure policy: heartbeat stale/lost intervals, submission grace, boot-failure
  circuit threshold, lease TTL, and request retention.

The controller is the only process that invokes `sbatch`, `squeue`, `scancel`,
and `scontrol`. Agents receive in-memory step-start permits over HTTP before invoking
`srun` within their own allocations.

`--agent-advertise-mode` controls the address agents publish to callers:
`fqdn` preserves the default, `hostname` publishes the short node name, and
`route-ip` publishes the local IPv4 address selected for the route to the
controller. Choose an address reachable from every caller node.

`--cpu-allocation-granularity` controls worker-job sizing when Slurm allocates
CPUs in units larger than a hardware thread. The default `1` preserves ordinary
CPU-granularity behavior. On a whole-core site with two threads per core, use
`2`; Sandfleet rounds each sandbox's CPU request up to that multiple when
sizing the parent worker, while the sandbox step still requests the original
profile value. For example, 16 one-CPU slots reserve 32 parent CPUs, while 16
two-CPU slots also reserve 32.

`--enable-dynamic-resources` lets a client token create demand for validated
CPU/RAM shapes without pool-administration authority. The controller creates a
deterministically named homogeneous pool for the first request and reuses it
for matching requests. Every shape is still subject to `--max-capacity`,
`--max-pools`, `--max-cpus-per-sandbox`, `--max-memory-mb-per-sandbox`, and the
dynamic policy flags. Clients cannot choose Slurm placement or submission
options. `--dynamic-pool-slots-per-job` is a maximum: Sandfleet lowers the
actual slots for a shape when necessary to fit
`--dynamic-worker-max-cpus`/`--dynamic-worker-max-memory-mb`, including agent
memory headroom and CPU-allocation granularity.

`--cleanup-on-exit` cancels pools when the controller receives a normal
termination signal. A hard controller-host failure cannot run that handler;
agents then exit after their controller-loss timeout (see below).
An unexpected reconciler failure preserves durable desired and cleanup state
for a supervised restart instead of starting a second destructive cleanup pass.

`--state-file` owns two generations: the requested path and a `.bak` sibling.
Saves fsync the new snapshot and its parent directory before and after rotation.
Startup restores the newest valid generation and quarantines a corrupt primary
as `.corrupt-<timestamp>`. If state artifacts exist but none validates, startup
fails and preserves them; it never silently creates empty state.

## Agent: `sandfleet-agent`

Operators normally do not launch this command. `sandfleetd` constructs worker
`sbatch` submissions that run it with the pool name, slot count, per-slot CPU
and memory, controller URL, and executable paths.

An agent:

- registers its allocated and prepared capacity;
- obtains globally paced step permits;
- creates one persistent, resource-limited `srun` step per slot;
- reserves prepared steps for leases;
- starts, restarts, and stops Apptainer inside those steps;
- verifies a private namespace with no routes before accepting commands;
- reports heartbeats and preparation metrics; and
- exits when its Slurm worker allocation is cancelled.

The service token is supplied through required `SANDFLEET_TOKEN_FILE`; Sandfleet
does not support embedding it in Slurm's recorded submit command.

An agent exits when it has had no successful controller heartbeat for
`--controller-loss-timeout-seconds`, so workers of a lost controller release
their allocation. The controller sets it to its `--lease-ttl-seconds`, after
which no lease on that worker can still be valid; `0` disables it.

Controller and agent networking options:

- `--network-mode loopback` is the only mode (the default). Every sandbox gets
  only loopback and no routes; internet leases also get a `127.0.0.1:3128`
  proxy listener relayed to the egress forwarder. There is no host-network mode
  and no sandbox DNS. `--network-mode fakeroot` and `--network-dns` were removed
  in 0.9.0, and `--network-mode slirp4netns` in 0.11.0.
- `--sandbox-internet off|allowlist|any` (controller) runs a per-worker egress
  forwarder for leases that request internet. The agent equivalent is
  `--internet`. `--sandbox-internet-allowed-hosts` overrides the allowlist,
  `--sandbox-internet-link-local-dns` lets the forwarder accept fake
  `169.254.x` resolutions of the worker's own DNS, and
  `--sandbox-internet-max-connections`, `-max-tunnels-per-lease` and
  `-idle-timeout-seconds` bound forwarder resources.

Controller and agent confinement options (see
[sandbox confinement](production.md#sandbox-confinement)):

- `--sandbox-audit-log on|off` (controller, on) and
  `--sandbox-audit-log-max-mb N` (256) write per-worker JSONL records of every
  sandbox command and file write under `<log-dir>/audit`. The agent equivalents
  are `--audit-log-dir DIR` (unset disables) and `--audit-log-max-mb`.
- Step workers read `SANDFLEET_SANDBOX_SECCOMP` (on; `0` or a profile path) and
  `SANDFLEET_SANDBOX_PIDS_LIMIT` (4096; `0` disables) from their environment.

These operator settings are carried to each persistent worker. Lease callers
cannot override networking through `extra_start_flags`. Start/restart status
includes `network.mode`, `namespace`, `host_namespace`, `internet`, and
`helper_pid` (always null since 0.11.0). The client request schema remains unchanged. See
[production networking](production.md#sandbox-networking) for prerequisites.

## Controller restart recovery

Run the controller with `--registry /path/to/shared/service/endpoint.json` and the same
persisted `--state-file` and role credentials on every restart. Set
`SANDFLEET_REGISTRY=/path/to/shared/service/endpoint.json` in each client's environment
(or pass `registry=` to `SandfleetClient`). The controller passes this absolute
path to newly launched workers. It must be readable at the same path on every
client/worker node and writable only by the trusted service operator. Both
controller and existing workers/clients must run the recovery-enabled code;
upgrading the controller alone does not retrofit older worker processes.

The registry is published atomically after state loading and socket binding.
Clients and worker heartbeats reread it for controller requests. Renewals retry
transient transport/408/429/502/503/504 errors with jittered backoff for at most
300 seconds (also bounded by half the lease TTL); authentication, malformed
registry, and other protocol errors surface immediately. Leases retain their
persisted expiry: an outage longer than the retry window or lease lifetime can
still lose work. Agent commands and ambiguous lease acquisitions are never
replayed. Reads/releases also use bounded retries. This is single-controller
recovery, not leader election or high availability; do not run two controllers
against one state file.

Existing dynamic pools retain their persisted capacity and worker packing when
controller defaults change. Defaults apply only to new resource shapes; a
restart does **not** resize existing pools. In particular, changing a default
from 640 to 2048 does not prove that 2048 slots are available. Drain/delete and
recreate the pool for a deliberate configuration change, between training runs.

Deploy between runs. Validate on an isolated controller before a production
restart, including address movement, active renewal, and fresh resource requests.

## Additional API features

### Worker end time

A worker reads `SLURM_JOB_END_TIME` when its job starts and reports the optional
`worker_end_time` Unix timestamp in lease `start`, `restart`, and `status`
replies. Controller health reports `worker_time_limit_seconds`, or null when
unlimited or not parseable. The Harbor environment exposes the lease value as
`SandfleetEnvironment.worker_end_time`.

These values are informational and do not reserve sufficient remaining job
time for a lease. Callers should check their own deadlines. A later scheduler
time-limit update is not reflected in the worker's captured value.

### GPU resources in Harbor

The Harbor adapter accepts task `[environment] gpus`, optional `gpu_types`, and
`[metadata] exclusive_gpu`. It checks the controller's capabilities, requests
exact GPU types, and verifies device count, identities, and visibility inside
the sandbox. It repeats those checks after a mount-triggered restart. Tasks
requesting no network use offline leases.

### Read-only mounts

`lease.start(..., readonly_mounts=[...])` and `restart` accept read-only host
mounts from operator-allowlisted agent roots. Configure those roots with
`--sandbox-agent-readonly-mount-root DIR`; controller health reports them as
`agent_readonly_mount_roots`. Harbor exposes this through task
`[metadata] agent_mounts`. Agent roots must not contain verifier roots.

`lease.add_readonly_mounts(...)` adds mounts to a running disk-backed sandbox
from `--sandbox-readonly-mount-root DIR` roots, reported as
`readonly_mount_roots`. The sandbox restarts from its overlay, retaining files
and stopping processes. Harbor's `add_verifier_mounts()` and
`add_produce_mounts()` use this operation for their corresponding task mounts.
Writable-tmpfs instances cannot use it.

Mounts are checked for allowed sources, protected targets, nested mounts,
read-only status, and the required seccomp policy. Startup and restart verify
resource limits, namespaces, and mount policy before allowing commands. Match
controller, agent, worker, and client protocol features before relying on them.

### Stop sandbox processes

`lease.kill_processes()` and Harbor's `kill_sandbox_processes()` stop sandbox
processes except its init while retaining the filesystem and lease. The worker
uses pinned user/PID namespaces and verifies completion from the host. Failed
cleanup remains an error and retains state for retry. This does not make a
filesystem written by an untrusted root safe for a verifier; use a separate
verifier environment when filesystem independence is required.

### Command deadlines and cancellation

Commands with an explicit command ID use a worker deadline and command-scoped
TERM/KILL cleanup. Results expose `timed_out` where supported. Exit status alone
is not sufficient to identify timeout, and explicit cancellation takes
precedence over a racing deadline. Foreground completion disarms the deadline
before output pipes are drained, preserving intended background work. Cleanup
errors remain visible.

### Archive uploads and cleanup

Archive uploads to absolute destinations run from `/`, independently of the
caller's working directory. Extraction failures remain errors.

Lease release waits for or interrupts in-flight work and requires a bounded
drain. Expected cleanup contention returns `503 LeaseCleanupPending`; expired
leases remain assigned and reconciliation retries cleanup. An Apptainer
instance reported absent during stop is accepted only when a successful
listing confirms that exact instance is gone. Other errors retain sandbox
identity and storage until cleanup succeeds.

See [sandbox networking and confinement](production.md#sandbox-networking) for
current isolation, egress, audit, seccomp, and task-limit configuration.
