# Design

Sandfleet separates scheduling from the sandbox data path.

```text
caller job
  ├─ controller API: create pool, acquire/release lease
  └─ agent API: start/restart, exec, files

controller job or service host
  └─ sbatch worker allocation(s)

worker allocation
  └─ node agent
      ├─ persistent srun step → Apptainer instance
      ├─ persistent srun step → Apptainer instance
      └─ ...
```

## Resource model

A pool has one fixed sandbox profile: CPU count and memory in MiB. Its elastic
policy contains minimum, warm-spare, and maximum capacities, slots per worker,
and an idle timeout. Sandfleet packs the current target into single-node worker
allocations. Agents first create an empty persistent `srun` step with the pool
profile. CPU counts are Slurm allocation units, so Sandfleet does not assume
that one requested CPU maps to one Linux CPU ID. Instead, it verifies that each
step has a distinct cgroup, its non-empty cpuset stays inside the worker
allocation, and sibling sandbox cpusets do not overlap. It also requires a
finite inherited memory limit no greater than the profile. Only then is that
slot reported ready. Missing or weaker boundaries fail preparation.
A lease starts Apptainer inside the prepared step. Restarts replace Apptainer,
and release stops it while retaining the step for the next lease.

The verified memory limit is `memory.max`. Swap is a separate cluster policy;
an unlimited `memory.swap.max` permits allocations beyond the RAM limit to page
out. Sandfleet reports preparation failures through each agent's `slot_errors`,
but cannot tighten a root-owned Slurm cgroup itself.

The worker allocation includes 512 MiB of memory headroom for its agent. CPU
and memory must fit on a node or Slurm will leave the worker job pending.
Worker CPU sizing rounds each sandbox request to the configured allocation
granularity before multiplying by slots. This deployment policy is independent
of the sandbox profile presented to clients.

### Client-selected resources

Clients may acquire from an operator-created named pool or submit a declarative
`{cpus, memory_mb}` resource request. Exactly one mode is required. When
dynamic resources are enabled, the controller validates the request and maps
it to a deterministic homogeneous pool named for the effective shape. Matching
requests from any caller reuse that pool, so prepared-step reuse, array
submission, and predictable packing remain unchanged; Sandfleet does not issue
one `sbatch` call per lease.

Dynamic pool policy sets slots per worker, maximum and warm capacity per shape,
idle timeout, the maximum number of shapes, and schedulable per-worker CPU/RAM
budgets. Slots per worker is an upper bound: for each shape the controller
reduces packing until the worker fits those budgets after CPU allocation
granularity and agent memory headroom. Existing controller limits still bound
per-sandbox CPU/RAM, total configured capacity, total pool count, and pending
requests. Client input never contains partition, QOS, account, worker count, or
raw Slurm flags.

GPU requests extend the same object with an exact count and ordered type
alternatives, for example `{"gpu": 1, "type": ["h100_80gb", "a100_40gb"]}`.
Model names include memory because `a100_40gb` and `a100_80gb` are different
resource types. Sandfleet may only choose from the caller's list. An
operator-owned catalog maps each logical name to one distinct trusted Slurm
GRES type; clients cannot send GRES strings or `any`. The controller caches
partition-scoped GPU inventory and one `sbatch --test-only` result per enabled
type. Typed `Gres`/`GresUsed` data remains authoritative when a cluster's TRES
fields contain only aggregate `gres/gpu` counts. Sandfleet never interprets an
aggregate TRES count as a GPU type. It prefers an alternative with enough
estimated free capacity on one node, but only a prepared step proves allocation.
When the controller is QOS-bound, the same refresh also records its aggregate
`GrpTRES` GPU headroom. This is a placement-policy ceiling rather than a typed
inventory, and remains advisory. A controller has one trusted account/QOS;
different caller placement policies use separate controller deployments rather
than client-provided QOS strings.

A GPU worker reserves `slots × GPUs-per-slot`; each persistent `srun` step
requests its exact subset. Slurm device-cgroup enforcement is a mandatory
operator gate. Before a slot becomes ready, its worker opens the GPU data
devices visible through the step cgroup and requires exactly the requested
count. A step-scoped `nvidia-smi` probe must report the same number of
model/memory/UUID identities, which are exposed in lease status; sibling slots
must have disjoint device sets. GPU containers use
worker-selected Apptainer `--nv` or `--nvccli` (`--gpu-container-mode`) and inherit the step's
`CUDA_VISIBLE_DEVICES`. GPU memory is type metadata, not a process memory quota;
this release supports whole GPUs, not MPS/shard sharing.

## Control plane

The controller is the only process that calls `sbatch`, `squeue`, `scontrol`,
or `scancel`. GPU discovery runs on a background controller thread and its
failures are recorded, not fatal; API capability reads use the cached snapshot.
Agents make outbound authenticated registration and heartbeat requests, so the
controller does not need to discover their dynamic ports through Slurm.

Pool creation is idempotent for an identical specification. Deleting a pool
releases known leases and cancels its worker jobs.

Cleanup state changes are transactional. An assigned request remains assigned
until its agent confirms release, so a transient agent failure can be retried
without orphaning the lease. Scale-down temporarily removes a worker from
placement before `scancel`; if cancellation fails, the worker returns to active
placement instead of remaining stuck in a draining state.

## Elastic reconciliation

The desired slot target is:

```text
clamp(active leases + pending requests + warm spare slots,
      min capacity,
      max capacity)
```

The target worker count is the ceiling of target slots divided by slots per
worker. Running agents and unregistered active array tasks both count toward
that target, so Slurm queueing never causes duplicate submissions.

One rate-limited reconciliation thread owns scheduler interaction. It submits
worker deficits as job arrays, maintains a cached expanded `squeue` snapshot,
and drains only completely idle workers after their idle timeout. API handlers
never invoke scheduler commands. Expected scheduler failures become explicit
health state and backoff. A lost worker omitted by the latest successful
snapshot is already gone and is not cancelled again. If a still-present lost
worker cannot be cancelled, its agent record retains durable cleanup-pending
state and the controller retries with bounded backoff. An unexpected
reconciler exception still terminates the controller instead of leaving a live
but inert API.

`srun` step creation is globally admitted by the controller's token bucket.
This is separate from scheduler polling/submission limits: permits are ordinary
in-memory HTTP operations, while each granted permit authorizes one agent to
ask `slurmstepd` for a new step. The default is eight starts per second with a
burst of eight across all workers.

Failed submissions back off exponentially. If five consecutive submissions
end before any agent registers, the pool opens a circuit breaker. An operator
can inspect `last_scale_error` and explicitly run `sandfleet pool resume` after
fixing the underlying account, partition, image, or environment problem.

## Failure model

Agents heartbeat every five seconds. A stale agent is removed from placement;
after the loss grace period its leases transition to `lost` and missing
capacity is replaced. The cached scheduler snapshot determines cleanup: an
absent worker is an absorbed preemption, while a still-present allocation is
cancelled with durable retry state. Sandfleet never replays the last command
because commands are not necessarily idempotent, and it never pretends an
ephemeral filesystem survived a node failure.

Leases have a controller TTL and clients renew them in the background. A caller
that dies without releasing its lease therefore cannot hold a slot forever.

The durable state file records pools, submissions, agents, lease routing,
pending requests, lost-worker cleanup attempts, and the preemption/cleanup/
termination counters. A controller process restarted at the same reachable URL
can reload desired state and resume cleanup, while live agents re-register
after a failed heartbeat.

Each save fsyncs a new snapshot before rotating the current snapshot to a
previous generation, then fsyncs the parent directory after each rename. On
restart the controller selects the newest valid generation, quarantines an
invalid primary, restores the fallback, and reports the recovery through health
and metrics. If state artifacts exist but none validates, startup fails without
deleting them or inventing empty state.

## Data plane

Lease acquisition first creates a durable pending request. It is assigned FIFO
when an agent has an idle slot and returns:

- an opaque lease ID;
- a random, lease-scoped bearer token; and
- the agent's direct URL.

The caller then communicates directly with that agent. The controller is not
in the exec or file-transfer path.

## Security boundary

Sandfleet agents intentionally expose arbitrary code execution. Sandfleet
separates service/admin, client, metrics, and per-lease bearer tokens, but uses
plain HTTP and is suitable only on a trusted private cluster network. Do not
expose it publicly. A hostile multi-user deployment still requires TLS, caller
identity, per-principal authorization and quotas, audit logging, and typically
firewall policy between sandbox payloads and other routable services.

Every instance has a private network namespace and never shares host
networking. Offline instances (the default) have only loopback. Internet leases
have no routes either. Their only way out is a `127.0.0.1` proxy listener that
the worker creates inside the namespace from outside (`nsenter` plus an
`SCM_RIGHTS` descriptor handoff) and relays to the worker's authenticated egress
forwarder on the host's loopback. Sandbox commands lack `CAP_NET_ADMIN`,
`CAP_NET_RAW` and `CAP_SYS_PTRACE` to change that. The empty route set is checked
before start succeeds, and failed cleanup retains ownership for
retry. A restart replaces the network inside the same Slurm step and keeps the
lease's internet choice. See the
[networking contract](production.md#sandbox-networking) for configuration,
lifecycle checks, and the precise isolation boundary.

Instances also start under a seccomp profile that denies user-namespace
creation, mounts, `bpf` and other exploit-oriented syscalls. Commands run with
`RLIMIT_NPROC` lowered inside the sandbox's own user namespace, which caps that
sandbox's tasks rather than the shared host UID. Startup refuses a sandbox that
lacks the filter or shows host network filesystems. Each worker writes a JSONL
audit record of every command and file write. See
[sandbox confinement](production.md#sandbox-confinement).

A disk-overlay lease can add operator-allowlisted host directories as read-only
mounts after it starts (`add-mounts`), typically after an agent phase and
before verification. Apptainer cannot add a mount to a running instance, so the
worker kills the sandbox's processes and restarts the instance from the same
overlay with the extra `:ro` binds, inside the same Slurm step and network
policy. The restart repeats every startup check; the mounts are then verified
from the host. Read-only depends on the seccomp filter on mount syscalls, so the
operation is refused without it.

A lease can also ask for read-only mounts when its sandbox starts, for inputs the
agent may read. They come from separate operator roots (agent phase) than
`add-mounts` (verifier phase), and no agent root may contain a verifier root, so
a start request cannot expose verifier-only files. Start-time mounts need no
restart; they pass the same checks and host-side verification.

## Opt-in writable disks

With `--max-storage-mb` nonzero, a lease may start a pinned SIF with a nominal
`storage_mb` ext3 image on node-local XFS/ext3/ext4. It is fully allocated, not
sparse. `/`, `/tmp`, `/var/tmp`, `/root`, and the task working directory must
resolve onto its root overlay; auto-bind paths are disabled. RAM and CPU remain
enforced by the existing persistent Slurm step. Disk size includes filesystem
metadata; the immutable SIF is shared and uncharged. It is not a guarantee of
that many usable file bytes or a scheduler-level disk reservation.

Storage is ephemeral per Apptainer start/restart. Normal release verifies
instance stop and removes the owned image, reusing the same prepared step.
Uncertain stop retains the instance/lease/storage identity for cleanup retry.
After interrupted step loss, disk removal requires a stopped srun and an empty
or removed recorded task cgroup. If those checks fail, retain the directory.
Node loss can leave local remnants; this version does not run a cross-node
orphan scavenger. Use a separate, bounded evaluation controller and operator
inspection for such remnants; do not change a healthy training fleet.

Wire additions are optional for existing clients and do not change persisted
controller fields or resource-shaped pool IDs. Harbor requires explicit feature
advertisement and matching worker policy; it will not silently use an older
worker or writable-tmpfs. The training adapter's legacy payload remains valid.

## Limitations

- Following a controller to a new hostname requires the shared `--registry`;
  the durable state file alone does not redirect live agents.
- A lost active sandbox is reported, not migrated. The caller ends and resamples the
  affected rollout on its next reset.
- Agent jobs are one node each; the controller can shard a pool across many
  jobs but does not yet use multi-node worker allocations.
- GPU support requires whole-device GRES and verified Slurm device cgroups.
  MIG slices share a parent device node, so their slots fail closed; MPS and
  shard sharing are not supported. Cached availability can change before Slurm
  places a worker, and slot GPU memory is not checked against the catalog.
- CPU slots also fail closed if their step can open a GPU device, so CPU pools
  on GPU nodes need Slurm device constraints too.
- A saved GPU pool is not scaled up or accepted for new requests while the
  current catalog does not enable its type, maps it to another Slurm type, or
  caps GPUs below it. Workers it already has keep serving until they exit;
  delete the pool to stop them and reuse its name.
- Named-pool profiles are immutable; declarative requests select another
  resource-shaped pool instead of mutating a live profile.
- Declarative callers share one identity and global FIFO queue. Policy provides
  controller-wide bounds, not per-caller quotas or fairness.
- Dynamic resource pools remain in the catalog at zero workers so later
  matching requests can reuse them; the configured shape limit bounds catalog
  growth, and operators may delete unused shapes.
- Image conversion/cache warming is an operator responsibility.
- High-volume runtime counters reset after controller restart; preemption,
  cleanup, controller-termination, and state-recovery counters are durable. Use
  a Prometheus scraper for longer retention and alert history.
