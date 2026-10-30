# Sandfleet instructions for coding agents

Read this file before editing Sandfleet. The service creates and controls
Slurm allocations, so an apparently small fallback or retry can leak compute,
hide broken isolation, or increase load on the Slurm controller.

## Product boundary

Sandfleet provides lease-based Apptainer sandboxes for one trusted team on
a private Slurm network. It is not an Internet-facing or hostile multi-tenant
runtime. Do not describe plain HTTP, a shared Unix UID, or Apptainer fakeroot as
a complete security boundary.

The controller may serve many client jobs. Declarative resource requests share
one client identity, global FIFO assignment, and controller-wide limits; they
do not add per-client quotas or fair scheduling. Use named pools when jobs need
reserved capacity. Exactly one controller may own a state file at a time.

## Architectural invariants

Preserve these invariants unless the change explicitly replaces them and adds
tests for the new model:

1. A pool is either fixed (`desired_capacity`) or elastic (`max_capacity`),
   never an ambiguous mixture. Resource profiles are immutable after creation.
2. Lease acquisition always uses the durable `/v1/lease-requests` queue. Do not
   add an immediate side path that bypasses elasticity, ordering, or limits.
3. Only the reconciler calls `sbatch`, `squeue`, and `scancel`. API handlers
   read cached state and wake reconciliation.
4. Scheduler polling is batched, worker deficits are submitted as arrays, and
   pending array tasks count toward capacity. Do not poll Slurm per request.
5. One ready sandbox slot owns one persistent `srun` step. Only `prepare()` may
   launch that step. `AgentState.prepare_one()` requires a global step-start
   permit callback; do not make it optional or add a bypass. `start()` and
   `restart()` replace Apptainer inside the existing step.
6. A slot is not ready until it verifies an effective cgroup v2 CPU set and a
   finite `memory.max` no weaker than its configured profile.
7. Cleanup is transactional. If agent cleanup fails, retain the controller
   lease and assigned request so release can be retried. If `scancel` fails
   during scale-down, restore worker placement state; do not report successful
   cleanup or strand a worker in `draining`.
8. Expected capacity failures become explicit state and bounded backoff.
   Lost-worker cleanup is expected: a successful scheduler snapshot that omits
   the worker is authoritative, while cancellation failures for a still-present
   worker remain durable cleanup-pending state and are retried with backoff.
   Unexpected reconciler, heartbeat, or prewarmer failures terminate their
   service instead of leaving a live but inert process.
9. Service/admin, client, metrics, and per-lease credentials are exact roles.
   Do not silently substitute a more privileged token.
10. Never embed the service token value in an `sbatch` command. Workers receive
    only the path from `SANDFLEET_TOKEN_FILE`.
11. Lost commands are never replayed automatically. Callers must resample a
    rollout because commands may be non-idempotent.
12. Persisted and wire-protocol required fields are accessed strictly. State
    snapshots use two fsynced generations and parent-directory fsyncs. Load the
    newest valid generation, quarantine a corrupt primary, and fail rather than
    silently starting empty when no valid generation remains. Do not add
    defaults that turn malformed state or version skew into plausible data.
13. Declarative clients provide only `cpus`, `memory_mb`, and the optional
    `gpu`/`type` pair; never accept raw Slurm flags. GPU `type` is an ordered
    list of exact, memory-qualified types (no `any`). Resource-shaped pools are
    deterministic, homogeneous, and policy-bounded. GPU requests require an
    operator catalog and verified device isolation, and prepared slots must
    fail closed on any device mismatch.

## Code map

| File | Responsibility |
| --- | --- |
| `sandfleet/controller.py` | Durable desired state, lease broker, elastic reconciliation, controller HTTP routes |
| `sandfleet/resources.py` | Declarative shape validation and dynamic worker-packing policy |
| `sandfleet/slurm.py` | Bounded `sbatch`, batched `squeue`, and checked `scancel` adapter |
| `sandfleet/agent.py` | Prepared-slot lifecycle, lease routing, heartbeat, agent HTTP routes |
| `sandfleet/step_backend.py` | Client for one persistent `srun` step |
| `sandfleet/step_worker.py` | Process inside the step that owns the replaceable Apptainer instance |
| `sandfleet/network.py` | Operator-owned network policy and rootless helper lifecycle |
| `sandfleet/storage.py` | Opt-in disk overlays for one prepared step |
| `sandfleet/state_file.py` | Two-generation fsynced controller state |
| `sandfleet/recovery.py` | Registry discovery and bounded controller retries |
| `sandfleet/client.py` | Canonical Python controller/agent client and lease renewal |
| `sandfleet/http.py` | Authenticated standard-library JSON transport |
| `sandfleet/cli.py` | Thin command dispatch over the canonical client |
| `docs/design.md` | Resource, reconciliation, failure, and security model |
| `docs/human-guide.md` | Human deployment and operating workflow |

The largest remaining code boundary is `ControllerState`. If splitting it,
preserve atomic state transitions and use explicit components such as a pool
catalog, lease broker, and Slurm reconciler. Do not replace clear control flow
with a generic event framework or route registry solely to reduce file length.

Client integrations should delegate to the installable `sandfleet.client`
package. When changing the protocol, update and test the included Harbor
adapter and document any required client changes.

## Error-handling rules

- Catch only failures the current layer can classify or recover from.
- Preserve the primary exception and attach cleanup/diagnostic failures with
  `add_note`; do not mask the original failure.
- Use `discard()` for a step already known to be dead or unusable. Use
  `close()` when graceful shutdown is expected and its failure matters.
- Do not perform network, RPC, subprocess, or step shutdown while holding the
  agent state lock.
- Do not use broad exception suppression in active request, renewal,
  reconciliation, or preparation paths. Best-effort suppression is acceptable
  only during final process teardown when all resources are already being
  abandoned.
- Malformed `squeue` output and failed `scancel` are errors, not empty state or
  successful cleanup.
- A transport failure may be retried when bounded by an existing lifecycle;
  authentication, configuration, and protocol failures must surface.

## Resource and Slurm semantics

A Slurm job is an allocation created by `sbatch`. A Slurm step is an `srun`
execution inside that allocation. Sandfleet creates one persistent step per
ready sandbox slot because the step owns the sandbox CPU and memory cgroup.

Do not equate sandbox count with `sbatch` job count. For 32 ready sandboxes
and eight slots per worker, expect four worker array tasks containing 32
steps. Steps are created once, globally paced, and reused across commands and
Apptainer restarts.

`memory.max` constrains cgroup-accounted RAM. Swap is separate. A host with
`memory.swap.max=max` may page allocations beyond the RAM limit; Sandfleet
cannot tighten that root-owned policy. Do not claim a total RAM-plus-swap bound
unless the live cluster proves one.

The cgroup v2 root may omit controller files even when a finite Slurm ancestor
enforces the step. Accept that layout only after discovering a finite ancestor.
Older kernels may expose `oom` and `oom_kill` without `oom_group_kill`.

## Publication and privacy

Keep operator identities, organization details, private paths, hostnames, model
inventories, job records, credentials, and internal issue references out of
source, examples, tests, commit messages, and pull requests. Use synthetic
fixtures and explicit placeholders instead. Do not import private Git history,
tags, or branches. Obtain the repository owner's approval for the proposed
changes before making a commit or pushing to a public remote.

## Local workflow

Preserve unrelated work and inspect the tree before editing:

```bash
git status --short
git diff --check
```

Use `uv` for the Python environment, dependency installation, and command
execution. The repository pins its development Python and commits `uv.lock`:

```bash
uv sync --frozen
```

The development dependency group pins Ruff because newer releases enable
additional rules against this unconfigured repository. Upgrade the pin only in
a dedicated lint migration that fixes the newly enabled rules and records the
chosen configuration. Update dependencies intentionally with `uv lock`, and
commit the resulting lockfile.

Run all of the following before handing off a change:

```bash
uv run --frozen ruff format --check sandfleet tests
uv run --frozen ruff check sandfleet tests
uv run --frozen ruff check --select C90,PLR0911,PLR0912,PLR0915 sandfleet tests
uv run --frozen pytest -q
git diff --check
```

Add a focused regression test for every fixed failure mode. Unit tests must
prove that strict errors remain visible, failed cleanup retains state, and
rate-limited paths cannot be bypassed.

If a change modifies required persisted fields, bump `STATE_VERSION`; do not
pretend an older schema loaded correctly. If it modifies package or API
behavior, update `sandfleet/version.py`, `pyproject.toml`, the README, command
reference, human guide, and included adapters as applicable.

## Live Slurm testing

Run a live test for changes to Slurm commands, cgroup discovery, agent
supervision, Apptainer lifecycle, networking, or cleanup. Local mocks are not a
substitute for the target cluster's cgroup hierarchy.

Keep site-specific SSH hosts, partitions, accounts, QOS names, executable
paths, storage paths, quotas, node names, and job identifiers out of the
repository and its public review trail. Discover them from private operator
configuration at test time, and re-check access, policy, and free capacity
before every live run.

For every live test:

1. Use a uniquely named controller job, pool, runtime directory, and log.
2. Verify no existing Sandfleet job or state file belongs to another run.
3. Generate mode-0600 temporary role-token files and never print their values.
4. Use the smallest pool and image that exercise the relevant behavior; begin
   with one slot, one CPU, and a bounded memory request.
5. Verify the reported `cpuset.cpus.effective`, finite
   `memory.max.effective`, command execution, Apptainer restart, and identical
   step-worker PID across restart.
6. Delete the test pool before stopping the controller.
7. Cancel only the exact test job IDs, remove temporary token files, and verify
   with `squeue` that both controller and worker allocations are gone.

Do not use recursive deletion, broad `scancel -u`, or an existing production
runtime directory for cleanup.

## Multiple callers

One controller is intentionally shared by many caller processes and training
jobs. Controller requests cover acquisition, polling, renewal, release, and
cached status. Exec and file data go directly to node agents.

For independent training jobs:

- use one durable controller that outlives every caller;
- use a distinct pool per job when fairness or capacity separation matters;
- size controller aggregate capacity for the sum of pool maxima;
- keep pending demand below `max_pending_requests`; and
- agree with Slurm administrators on persistent-step count and the global
  step-start rate before a large cold start.

Do not add per-client behavior without defining identity, quotas, fairness,
metrics, and backward-compatible configuration together.

## Known limitations and follow-up work

- Service and pool-administration authority still share one credential.
- Plain HTTP and shared Unix identities require a trusted private network.
- The per-sandbox task limit is `RLIMIT_NPROC` in the sandbox's user namespace
  (Linux 5.14+), applied to commands, not to `%startscript`; there is no cgroup
  `pids` limit. Disk limits exist only for opt-in disk overlays.
- The seccomp profile's `clone3` handling relies on `SCMP_ACT_TRACE` returning
  `ENOSYS`, because Apptainer forces `EPERM` for errno rules.
- Swap may be unlimited even when RAM is constrained.
- The controller is a single active process without leader election.
- Address recovery requires a shared registry and recovery-enabled agents/clients.
- High-volume runtime counters reset with the controller process; preemption,
  cleanup, and controller-termination counters are durable.
- File/archive transfer uses base64 JSON with a 64 MiB request limit.
- Read-only mounts (`add-mounts`) are added by restarting the instance from its
  disk overlay, which stops every sandbox process; their read-only guarantee
  rests on the seccomp filter on mount syscalls. Writable-tmpfs sandboxes
  cannot use them. Start-time mounts (0.14) need no restart and use separate
  agent-phase roots; keeping verifier-only files out of them depends on the
  operator not exposing verifier content inside an agent root by other means
  (for example a host bind mount at a source path).

Do not hide these limitations with a fallback. Either implement and test the
missing guarantee or document it explicitly.
