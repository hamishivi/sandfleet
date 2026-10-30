# Production operations

Sandfleet targets a single trusted team on a private Slurm network. It is
not an Internet-facing or hostile multi-tenant service.

## Required infrastructure

- Python 3.11 or newer for the controller, agents, and persistent step workers.
  This applies to source/PYTHONPATH launches as well as installed packages.
  Sandfleet rejects older interpreters on import, before service startup.
- The persistent step-worker's Linux Python build must provide `os.pidfd_open`,
  `os.setns`, and `signal.pidfd_send_signal`. Verify the actual interpreter on
  worker hosts; a version number alone does not prove these capabilities.
- A stable controller hostname reachable from caller and worker nodes. Run the
  controller on a supervised service host, or put a stable private proxy/DNS
  name in front of a requeueable controller job.
- A shared filesystem for the Sandfleet source, SIF images, state file, registry
  file, and logs. The registry contains only controller location metadata.
- Private routing between caller jobs and agent ports. Firewall these ports
  from the public Internet.
- `sbatch`, `squeue`, `scancel`, and `srun` available to the service account,
  plus Apptainer with unprivileged instance support.
- Slurm CPU and memory enforcement enabled for job steps. Sandfleet does not
  depend on Apptainer's own cgroup manager.
- For GPU pools, typed whole-device GRES and Slurm step device enforcement.
  Confirm `ConstrainDevices=yes` (or the site's equivalent verified behavior)
  and correct `gres.conf` device enumeration before setting Sandfleet's
  explicit isolation-verification flag. GB200/GB300 hosts with
  `/dev/nvidia-caps-imex-channels` also need `nvidia-container-cli`.

Set `--agent-advertise-mode` to an address form reachable from callers. Use
`fqdn` when cluster DNS is shared, `hostname` when short node names resolve, or
`route-ip` when routed private IPs are the stable cross-partition interface.

Sandfleet treats `--cpus` as Slurm allocation units, which may be cores or
hardware threads depending on site configuration. It verifies that step
cpusets are non-empty, contained by the worker allocation, and pairwise
disjoint instead of assuming one requested CPU always maps to one Linux CPU ID.
If the site allocates CPUs to steps only in larger indivisible units, set
`--cpu-allocation-granularity` explicitly so the parent worker reserves one
rounded unit per sandbox. Do not infer it from the controller host: controller
and worker partitions may have different CPU topology.

Verify both `memory.max` and `memory.swap.max` on a real worker. Sandfleet
rejects a step without a finite RAM limit, but Slurm may still leave swap
unbounded. In that configuration `--mem` caps charged RAM while a sandbox can
spill additional pages to swap. If total RAM-plus-swap must be bounded, the
cluster must enable Slurm's swap constraint (normally `ConstrainSwapSpace=yes`
with the site's chosen `AllowedSwapSpace`); this is an administrator setting,
not something Sandfleet or `srun` can apply per sandbox.

## Sandbox networking

Every sandbox gets a private Apptainer network namespace (`--net
--network=none`) and never shares the worker's network:

- **Offline leases** (the default) have only loopback, which Apptainer brings
  up. There is no other interface, no route, and no DNS: name lookups and
  outbound connections fail immediately, while services on `127.0.0.1` work.
  No helper process runs.
- **Internet leases** (see [opt-in sandbox internet](#opt-in-sandbox-internet))
  have the same loopback-only namespace with no routes. In addition, the worker
  creates a listening socket on `127.0.0.1:3128` inside the sandbox's namespace
  from outside. As the instance's user-namespace owner it runs `nsenter --user
  --net` with a short Python helper that binds the socket and passes the
  descriptor back over `SCM_RIGHTS`. The worker checks that the socket belongs
  to the sandbox's namespace (`SIOCGSKNS`). The step worker then relays each
  accepted connection to the egress forwarder on the host's loopback. The
  sandbox's proxy variables point at `127.0.0.1:3128`. Every other destination,
  including the worker's own address, fails with "network unreachable".
- Sandbox commands run without `CAP_NET_ADMIN`, `CAP_NET_RAW` and
  `CAP_SYS_PTRACE` (see `SANDFLEET_SANDBOX_DROP_NET_CAPS` below), so they
  cannot reconfigure their namespace, send raw packets, or ptrace `appinit`
  to regain those capabilities. Apptainer drops them from the bounding set too.

| Option | Behavior |
| --- | --- |
| `--network-mode loopback` (default, only mode) | As above. Requires unprivileged user/network namespaces on each worker. Internet leases also require util-linux `nsenter` on the worker's PATH. It does not require `slirp4netns`, iproute2, privileged CNI or subordinate UID/GID mappings. |

`--network-mode fakeroot` and `--network-dns` were removed in 0.9.0: the CNI
bridge gave every sandbox unfiltered egress, and sandboxes no longer have DNS.
`--network-mode slirp4netns` was removed in 0.11.0, when internet sandboxes
stopped using slirp4netns. All three are rejected at startup.

Sandbox commands and the image's `%startscript` run without `CAP_NET_ADMIN`,
`CAP_NET_RAW` or `CAP_SYS_PTRACE` whenever `SANDFLEET_SANDBOX_DROP_NET_CAPS` is
on. Apptainer's PID 1 shim (`appinit`) keeps every capability regardless of
`--drop-caps`; it runs no sandbox code, and without `CAP_SYS_PTRACE` no sandbox
process can ptrace it to regain them. `nsenter` runs with
`--preserve-credentials`, so a root-only fakeroot mapping without subordinate
IDs (where `setgroups` is denied) suffices.

Agents report their version at registration and the controller rejects agents
older than 0.9.0. On upgrade, agents recovered from pre-0.9 controller state are
marked lost (their leases too) and their worker jobs cancelled, because they may
run workers whose offline sandboxes had DNS and NAT egress. Offline instances
report network `mode` `none`.

Startup reads `/proc/net/route` through an actual instance exec and requires
it to be empty, for offline and internet leases alike. Any route fails the
start. The sandbox's
`/etc/resolv.conf` is the worker's (Apptainer's default, or a snapshot in disk
mode); the sandbox has no route to those resolvers.

Each instance has a separate localhost and port space. Two slots on the same
worker can bind the same port, including `0.0.0.0` listeners. No inbound
host-port forwarding is created. Port 3128 on an internet sandbox's loopback is
taken by its proxy listener.

Starting with `--network=none` creates the empty namespace; for internet leases
Sandfleet then creates the proxy listener before returning a successful start
(a `%startscript` that already bound port 3128 fails the start). It fails
startup when networking cannot be established. A failed cleanup retains
ownership for retry. The relay runs as threads of the sandbox's step worker,
inside its Slurm resource limits, caps itself at 64 concurrent connections,
reports a stopped relay on command execution, and dies with its worker.
Restart replaces the namespace and relay while reusing the persistent Slurm
step, and keeps the lease's internet choice.

Network prerequisites are checked when an instance starts, after its lease
is assigned. Agent registration and prepared-slot readiness verify service
and cgroup readiness; they do not certify working Apptainer networking. A
missing `nsenter` or unavailable user/network namespaces can therefore cause
repeated start failures on an otherwise registered worker. Boot-time network
preflight is deferred. Validate worker images and partitions with the smoke
test before relying on their capacity; the controller host's capabilities
are not evidence about compute nodes.

Clients cannot override managed network, namespace-sharing, fakeroot, or DNS
flags through `extra_start_flags`. Inherited Apptainer/Singularity settings for
those options are removed. This remains a trusted-team runtime, not an
adversarial network-policy boundary.

On upgrade from versions older than 0.6.0, drain and replace existing agents/prepared workers so all
slots use the new network contract. Mixed old/new step-worker status is
rejected; the controller state format and client start/exec request shapes are
unchanged. Clients receive the `network` status metadata without a new
request field. Use the same operator configuration when
restarting the controller and agents.

The example controller batch script accepts `SANDFLEET_NETWORK_MODE` and fails
if the removed `SANDFLEET_NETWORK_DNS` is set. Step workers read
`SANDFLEET_SANDBOX_DROP_NET_CAPS` (on): `0` keeps `CAP_NET_ADMIN` and
`CAP_NET_RAW` in sandbox commands, and internet leases then refuse to start.
`CAP_SYS_PTRACE` is kept only when both it and
`SANDFLEET_SANDBOX_DROP_SETUID_CAPS` are `0`. Apptainer also needs writable
space for instance metadata and logs. If home storage is unsuitable or over quota, operators can
export `APPTAINER_CONFIGDIR` to a writable directory before submitting the
controller. That directory also controls Apptainer registry credentials,
remote configuration, and keys; configure those explicitly if needed.

### Opt-in sandbox internet

Sandboxes are offline unless an operator lets individual leases opt into
internet access. Upstream connections are made by the forwarder on the host:


| Option | Behavior |
| --- | --- |
| `--sandbox-internet off` (default) | No forwarder runs; leases requesting internet are rejected with 400. |
| `--sandbox-internet allowlist` (recommended) | Opted-in leases may reach only the allowlisted hosts on ports 80 and 443. |
| `--sandbox-internet any` | Opted-in leases may reach any public host on ports 80 and 443. Prefer `allowlist`: policy models invent download URLs, and `any` leaves the cluster's own egress filter as the only stop. |
| `--sandbox-internet-allowed-hosts H[,H...]` | Replaces the default allowlist (PyPI, npm, GitHub, crates.io, Go proxy, Maven Central, conda). `*.example.com` matches subdomains only. |
| `--sandbox-internet-link-local-dns` | Lets the forwarder, which resolves names on the host, accept names that resolve into `169.254.0.0/16`. Off by default. Sandboxes never resolve names themselves. Enable it only when the operator explicitly approves link-local DNS answers. Cloud metadata addresses such as `169.254.169.254` are refused even then. |
| `--sandbox-internet-max-connections N` (256) | Caps the forwarder's open connections per worker. Excess connections get 503. |
| `--sandbox-internet-max-tunnels-per-lease N` (32) | Caps one lease's concurrent tunnels. Excess tunnels get 429. |
| `--sandbox-internet-idle-timeout-seconds N` (300) | Closes a tunnel after this long with no traffic in either direction. |

`examples/controller.sbatch` reads `SANDFLEET_SANDBOX_INTERNET`,
`SANDFLEET_SANDBOX_INTERNET_ALLOWED_HOSTS`,
`SANDFLEET_SANDBOX_INTERNET_LINK_LOCAL_DNS=1`, and the
`SANDFLEET_SANDBOX_INTERNET_MAX_CONNECTIONS`, `_MAX_TUNNELS_PER_LEASE` and
`_IDLE_TIMEOUT_SECONDS` limits.

When enabled, each worker runs an HTTP CONNECT and plain-HTTP forwarder in its
host network namespace for the worker's lifetime. Its upstream connections are
ordinary host connections, and it never chains to another proxy. It resolves
`CONNECT` and plain-HTTP hostnames itself, so sandboxes need no DNS.

The forwarder binds `127.0.0.1` in the worker's host namespace on an ephemeral
port, so other machines cannot reach it. Sandboxes cannot reach it directly
either, because their loopback is their own. It accepts connections only from
`127.0.0.1`, and only callers with lease credentials can use it.

The forwarder admits 32 unauthenticated connections per worker, shared by all
of the worker's sandboxes. Each sandbox's relay therefore keeps at most
`max(1, 32 // slots)` connections pending, and stops accepting while at that
limit, so excess connections wait in the sandbox's own listen backlog. A
connection stops being pending at the first of these: the forwarder's first
byte (its answer after authenticating and connecting, or its refusal), the end
of the sandbox's request head reaching the forwarder, or 12 seconds after
connecting (the forwarder's 10-second authentication timeout plus slack). A
silent or slow forwarder therefore cannot hold a slot indefinitely. Workers
with more than 32 slots still give each sandbox one pending connection.

Resource limits keep it from starving the agent:

- At most 32 connections may wait unauthenticated, and each gets 10 seconds to
  authenticate.
- Once one side of a tunnel closes, the other side gets 30 seconds before both
  are closed.
- The worker's descriptor limit is raised to fit the connection cap. If it
  cannot be raised enough, the worker fails at startup.

Agents advertise the capability when they register. The controller places
internet leases only on capable agents, so agents that predate the setting are
never chosen. If an agent rejects a reservation, only that request fails.

A lease opts in with `"internet": true` in its lease request. In the Python
client this is `acquire(..., internet=True)`, and in the CLI it is
`exec --internet`. Only that lease's instance gets `HTTP_PROXY`, `HTTPS_PROXY`,
`ALL_PROXY` (plus lowercase forms) and `NO_PROXY=localhost,127.0.0.1,::1`.
These point at `http://<lease_id>:<secret>@127.0.0.1:3128`, the sandbox's own
relayed listener.

The secret is per lease and is distinct from the lease control token. Release
revokes it and closes the lease's open tunnels. Connections with missing,
unknown or revoked credentials get 407. Clients still cannot set network flags.

Policy is checked against the requested hostname, literal IPs, and every
resolved address:

- Private, loopback, link-local, metadata and other non-public addresses are
  always refused. This includes IPv4 embedded in IPv6 forms such as NAT64
  `64:ff9b::/96`, IPv4-mapped, 6to4 and Teredo.
- Single-label, numeric and internal-suffix names (`.local`, `.internal`, ...)
  are refused.
- The forwarder connects only to addresses it checked, trying each in turn.
- For plain HTTP, the `Host` header is rewritten to the checked destination.

Each connection is logged to the agent log with its lease, target, decision
and byte counts. Agent metrics report `egress_connections_allowed`,
`egress_connections_denied`, `egress_connections_rejected` (limits),
`egress_auth_failures`, `egress_idle_timeouts`, `egress_bytes_sent` and
`egress_bytes_received`. The controller counts `internet_leases_created`.

Security note: sandboxes stay offline unless both the operator and the lease
opt in.

- A policy with internet can fetch reference solutions, leak data, or download
  untrusted code. Treat internet-enabled RL rollouts as a different
  experimental condition, and prefer the allowlist.
- The allowlist is a destination filter, not an exfiltration boundary. CONNECT
  tunnels carry opaque TLS, so a client can present a different SNI or `Host`
  inside the tunnel and reach other virtual hosts on an allowlisted provider's
  servers (domain fronting). GitHub and package registries also host arbitrary
  user content.
- The proxy secret is visible to every process in its sandbox, and to other
  same-node processes that can read that sandbox's environment. It cannot be
  used from other machines and does not outlive the lease.

Validate on the actual cluster with a uniquely named, small dedicated
controller and the portable smoke test:

```bash
PYTHONPATH=. python examples/network_smoke.py --help
```

The test uses two 512 MiB slots on one worker. It verifies finite resource
limits, distinct namespaces, simultaneous reuse of a localhost port, the
network policy (no DNS, direct egress or default route; with `--internet`,
HTTPS through the forwarder), restart isolation, unchanged step-worker PID, and cleanup. It refuses
to reuse an existing pool and records partial evidence when a check fails.
Choose partition/account/QOS and executable paths at deployment; the runtime
and smoke test use operator-supplied scheduling configuration.

Keep the service/admin, client, and metrics tokens in a secret store or
permission-restricted files, and configure `SANDFLEET_TOKEN_FILE`,
`SANDFLEET_CLIENT_TOKEN_FILE`, and `SANDFLEET_METRICS_TOKEN_FILE`. This prevents
the service token itself from appearing in Slurm's recorded `sbatch` command.
All roles are required; none falls back to another role's token.
Never put tokens in the registry JSON.
Rotate a role token by restarting the controller; rotating the service token
also requires worker replacement because agents inherit it at submission.

## Sandbox confinement

Policy models run arbitrary commands, and every sandbox maps its fakeroot root
to the service account's host UID. Beyond networking, each sandbox is confined
as follows. Each control is on by default, read by the step workers from the
environment they inherit from the controller, and checked when a sandbox
starts.

**Seccomp filter** (`SANDFLEET_SANDBOX_SECCOMP`, on). Instances start with
Apptainer `--security seccomp:<profile>`; `apptainer exec` into the instance
inherits the filter. The built-in profile (`sandfleet/seccomp.py`) allows
everything except syscalls that sandboxed workloads do not need and that mainly
serve kernel exploits:

- `unshare` and `clone` with `CLONE_NEWUSER` (other namespace flags still work);
- `mount`, `umount2`, `pivot_root` and the new mount API (`fsopen`, `fsmount`,
  `move_mount`, `open_tree`, ...);
- `bpf`, `perf_event_open`, `userfaultfd`, `io_uring_*`;
- `keyctl`, `add_key`, `request_key`;
- `kexec_*`, `init_module`, `finit_module`, `delete_module`;
- `open_by_handle_at`.

These fail with `EPERM`. Apptainer 1.5 applies `EPERM` to every errno rule, but
`clone3` cannot be filtered on its flags and glibc falls back from `clone3` to
`clone` only on `ENOSYS`. `clone3` therefore uses `SCMP_ACT_TRACE`, which the
kernel answers with `ENOSYS` when no tracer has asked for seccomp events. The
profile refuses `PTRACE_O_TRACESECCOMP` (through `PTRACE_SETOPTIONS` and
`PTRACE_SEIZE`), so no sandbox process can become such a tracer. Ordinary
`gdb` and `strace` tracing of a sandbox's own children still works. The
profile's `architectures` lists only the native one (x86_64 or aarch64; other
hosts must set a profile path or disable seccomp). Apptainer adds rules only for
listed non-native architectures, so libseccomp's default bad-architecture
action kills any thread that makes a syscall through another ABI (for example
i386 or x32 on x86_64, or AArch32 on arm64). Set the variable to an absolute
path to use your own OCI-format profile instead, or to `0` to disable seccomp.
Startup refuses a sandbox whose commands do not report `Seccomp: 2`, and clients
cannot pass `--security` in `extra_start_flags`.

**Per-sandbox task limit** (`SANDFLEET_SANDBOX_PIDS_LIMIT`, 4096; `0`
disables). Every command runs after `ulimit -u <limit>` inside the instance,
which lowers both the soft and hard `RLIMIT_NPROC`. Since Linux 5.14 this limit
is checked against the tasks of the caller's UID in its own user namespace, and
each fakeroot instance has its own. It therefore caps one sandbox, not every
process of the shared host UID. Threads count, and so do roughly 25 Apptainer
helper threads per instance. Raising the hard limit again needs
`CAP_SYS_RESOURCE` in the initial user namespace, which no sandbox has.
Processes started by the image's `%startscript` are not limited. Apptainer's
`--pids-limit` is not used: it needs a systemd cgroup manager, which Slurm
steps do not provide, and many sites do not enable the cgroup v2 `pids`
controller at all. Kernels older than 5.14 count `RLIMIT_NPROC` per host UID;
there, set this to `0`. 4096 tasks is ample for typical builds and test suites,
but heavily parallel builds that spawn thousands of threads (large `make -j`,
Bazel, JVM- or browser-based test suites) may need more; raise it for those
workloads.

**No host network filesystems.** `--containall` does not disable the `bind path`
entries in the site's `apptainer.conf`. Startup reads the sandbox's
`/proc/self/mountinfo` and refuses any NFS, CIFS, Lustre, GPFS, Ceph,
GlusterFS, BeeGFS, 9p or AFS mount. It also refuses any FUSE mount other
than the local image filesystems Apptainer uses (a lease's own read-only mounts
at their requested targets are exempt): `fuse-overlayfs`,
`squashfuse`, `squashfuse_ll`, and `ext4` (the type e2fsprogs' `fuse2fs` uses
for ext3 overlay images), because FUSE can front remote storage (sshfs, s3fs,
gcsfuse, and S3-backed site filesystems). If that happens, ask the site to
remove the bind path, or use disk mode, which starts with
`--no-mount bind-paths`.

**Read-only mounts after start** (`sandfleetd --sandbox-readonly-mount-root
DIR`, repeatable; none by default). A lease can add subdirectories of these
roots to its started disk-overlay sandbox as read-only binds (`add-mounts`, see
the [command reference](commands.md#health-and-metrics)), for example a model
only a verifier may see. Choose roots that hold nothing any sandbox may write
or that the agent phase of any task must not read: a lease may mount anything
under them. Each source is resolved with symlinks on the worker and must stay
inside a root. The operation restarts the instance from its overlay, so the
agent's files persist and every process stops. Read-only relies on the seccomp
profile: fakeroot root keeps `CAP_SYS_ADMIN` in its user namespace, and only
the filter on `mount`, `mount_setattr` and the rest of the mount API stops it
from remounting a `:ro` bind read-write and writing to the host directory as
the service UID. Workers therefore refuse the operation when seccomp is off or
a custom profile does not deny every mount syscall outright. The restarted
instance passes every startup check above (no routes, seccomp, PID namespace,
host network filesystems), except that the requested targets may be network
storage. The worker then checks that each target is the source directory (same
device and inode) on a read-only mount, with its own helper that joins only the
instance's user namespace and opens the target beneath the instance init's
`/proc/<pid>/root` without following symlinks; no image code runs.
Capabilities, the task limit and the network namespace are set up exactly as at
a fresh start.

**Read-only mounts at start** (`sandfleetd --sandbox-agent-readonly-mount-root
DIR`, repeatable; none by default). A lease's `start` may mount subdirectories
of these agent-phase roots read-only for the whole sandbox lifetime, in
writable-tmpfs or disk mode. Every rule above applies (symlink resolution on
the worker, no mount below a source, seccomp precondition, host-side check of
each target's identity and read-only flag and of the init's mount table). They
are deliberately separate from the verifier-phase roots: a start request is
checked against agent roots only, and an agent root that equals or contains a
verifier root is refused at configuration time and again on the worker after
resolving symlinks. Anything under an agent root is readable by every agent of
that controller. In disk mode the targets are created in the new overlay image
before the first start, so no agent-written file is involved.

**Audit log** (`sandfleetd --sandbox-audit-log on`, the default). Each worker
appends one JSON line per lease operation to
`<log-dir>/audit/audit-<agent-id>.jsonl`:

```json
{"ts": "2000-01-01T00:00:00Z", "op": "exec", "pool": "example", "agent_id": "1000-example",
 "job_id": "1000", "node": "node.example", "slot": 0, "lease_id": "example-lease", "internet": false,
 "command": "curl -fsSL https://isolation-test.invalid/...", "command_truncated": false, "command_id": null,
 "timeout": 120, "exit_code": 6, "cancelled": null, "timed_out": null, "duration_s": 0.412}
```

`exec` and `exec-env` records carry the command (truncated to 4 KiB, with
`command_truncated`), command ID, timeout, exit code, `cancelled`, `timed_out`
and duration. `exec-env` adds `env_keys`, the override names without their
values. A command started in the background (`nohup ... &`) is recorded when its
launching command returns. `write-file` and `put-archive` record the path and
byte count, not the content; `start` and `restart` record the image;
`kill-processes` records `killed` and `cancelled_commands`; `add-mounts`
records `mounts` (source and target) and `stopped_processes`; `start` and
`restart` add `readonly_mounts` when the request has any. A failed
operation records `error` instead of an exit code. Reads and status calls are
not recorded.

A record is written after its operation finishes. Commands that are still
running when their worker dies have no record. When a file would exceed
`--sandbox-audit-log-max-mb` (256), it is renamed to `.1`, replacing any older
`.1`, so each worker keeps at most twice that much. If a write fails (for
example the shared filesystem is unavailable), the operation's result is still
returned, because the command has already run and may not be idempotent. The
failure is printed to the agent log and counted in the agent metric
`audit_write_failures`, and the next record is attempted normally. Alert on
that metric if the audit trail must be complete. Files are mode 0600 and hold
whatever secrets commands put on their command line.

## Operational defaults

The defaults cap scheduler interaction to one cached `squeue` snapshot per 15
seconds, one worker-array submission per 10 seconds, and 64 workers per array.
Step creation is separately limited to eight `srun` starts per second with an
eight-start burst. Tune downward if site administrators request it.

Declarative resources are disabled unless the controller starts with
`--enable-dynamic-resources`. Before enabling them, set the per-sandbox CPU/RAM
limits, aggregate capacity, maximum pool count, and dynamic shape policy. The
defaults use 16 slots per worker, at most 512 slots per shape, zero warm slots,
an idle timeout of 600 seconds, and at most eight dynamic shapes. These are
ceilings, not reservations: demand still controls worker submission. Each
shape's maximum counts toward aggregate configured capacity, so adding a new
shape fails before it can exceed the controller-wide limit.

Enabling declarative resources also requires explicit per-worker CPU and memory
budgets. Set them to values schedulable on one node in the worker partition.
Sandfleet treats configured slots per worker as a maximum and reduces packing
for a large shape so the parent job fits both budgets after CPU granularity and
the 512 MiB agent headroom are included.

All callers share one client credential and FIFO lease queue. Dynamic policy
prevents unbounded Slurm growth but does not provide per-caller identity,
quotas, or fairness. Use separate named pools when a training job needs reserved
capacity.

GPU types come from an operator catalog (see [the design notes](design.md)).
Enable them only after verifying step device cgroups, and bind each controller
deployment to one trusted account/QOS.

Leases expire after one hour by default and official clients renew every
`min(30 s, TTL / 3)`. Set `--lease-ttl-seconds 0` only when leaked leases are
handled by another mechanism.

Back up the state file and keep its parent directory mode restricted. Use one
active controller per state file; this release does not implement leader
election for active/active replicas.

## Monitoring

- Dashboard: `http://<private-controller>:8765/`
- JSON: `GET /v1/metrics` with the metrics token
- Prometheus: `GET /metrics` with the metrics token
- Liveness: `GET /v1/health` with a client token

Alert on scheduler health becoming false, nonzero agent
`audit_write_failures`, scaling suspension, nonzero recent
submission/step-prepare failures, ready capacity staying below target, stale or
lost agents, nonempty agent `slot_errors`, and pending leases growing without
ready capacity. Alert immediately when `controller_terminations` increases.
Track `worker_preemptions_absorbed` as expected preemptible-capacity churn,
`lease_release_transport_failures` as unreachable-agent cleanup retried by the
lease/heartbeat lifecycle, and
`worker_cleanup_failures` as a degraded cleanup path. Alert when
`state_recoveries` increases and inspect `persistence.last_recovery` in
`/v1/metrics` or `state_recovery` in `/v1/health`. Controller termination,
worker cleanup/preemption, and state-recovery counters are durable; release
transport failures reset with the process, so Prometheus should retain their
history.

## Recovery

The controller persists two fsynced desired-state generations and fsyncs their
parent directory after renames. Restart it at the same stable URL and live
agents will re-register after their next failed heartbeat. If the primary is
invalid, Sandfleet quarantines it, restores the newest valid fallback, and
reports the recovery. If state artifacts exist but no generation validates,
startup fails without deleting evidence or starting empty. With `--registry`,
agents and clients reread the controller address on each request, so they follow
a controller restarted at a new URL; see
[controller restart recovery](commands.md#controller-restart-recovery). A lost
sandbox is reported to the caller and is never replayed.

After a worker disappears, the latest successful scheduler snapshot is
authoritative. Sandfleet does not issue `scancel` for an already-absent job. If
the allocation remains visible but cancellation fails, its agent reports
`cleanup_pending`, `cleanup_attempts`, `next_cleanup_at`, and
`last_cleanup_error`; the controller retries without terminating.

After fixing a repeated worker-boot problem, inspect `last_scale_error` and run
`sandfleet pool resume <pool>`. Pool deletion is destructive: it releases
leases and cancels that pool's worker allocations.
