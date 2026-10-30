#!/bin/bash
# Submit with site-specific controller resources, for example:
#   sbatch --partition=compute --account=research examples/controller.sbatch
#
# Required environment variables:
#   SANDFLEET_ROOT, SANDFLEET_TOKEN_FILE, SANDFLEET_CLIENT_TOKEN_FILE,
#   SANDFLEET_METRICS_TOKEN_FILE, SANDFLEET_WORKER_PARTITION,
#   SANDFLEET_WORKER_ACCOUNT, SANDFLEET_AGENT_PYTHON,
#   SANDFLEET_APPTAINER_BINARY
#
# Optional (script default in parentheses; unset means the controller default):
#   SANDFLEET_PORT (8765)
#   SANDFLEET_WORKER_PYTHON (SANDFLEET_AGENT_PYTHON)
#   SANDFLEET_WORKER_QOS, SANDFLEET_WORKER_TIME_LIMIT (04:00:00)
#   SANDFLEET_AGENT_ADVERTISE_MODE (fqdn)
#   SANDFLEET_CPU_ALLOCATION_GRANULARITY (1)
#   SANDFLEET_LEASE_TTL_SECONDS (3600)
#   SANDFLEET_CLEANUP_ON_EXIT=1
#   SANDFLEET_MAX_STORAGE_MB: enables disk-backed sandboxes
#   SANDFLEET_CACHE_DIR, SANDFLEET_TMP_DIR: worker-node Apptainer cache/tmp
#     (not used by disk-backed sandboxes)
#   SANDFLEET_NETWORK_MODE (loopback; the only mode; slirp4netns was removed in 0.11.0)
#   SANDFLEET_GPU_CONTAINER_MODE (auto)
#   SANDFLEET_SANDBOX_INTERNET (off): allowlist or any lets leases opt into
#     internet; SANDFLEET_SANDBOX_INTERNET_ALLOWED_HOSTS overrides the allowlist;
#     SANDFLEET_SANDBOX_INTERNET_LINK_LOCAL_DNS=1 to accept
#     operator-approved link-local DNS answers;
#     SANDFLEET_SANDBOX_INTERNET_MAX_CONNECTIONS (256),
#     _MAX_TUNNELS_PER_LEASE (32), _IDLE_TIMEOUT_SECONDS (300)
#   SANDFLEET_SANDBOX_BLOCK_APT (on; temporary): 0 lets sandboxes run apt. Read by
#     the step workers, which inherit this environment.
#   SANDFLEET_SANDBOX_BLOCK_USER_SWITCH (on; temporary): 0 lets sandboxes run su,
#     sudo, runuser, setpriv and chroot. Read by the step workers, like the above.
#   SANDFLEET_SANDBOX_DROP_SETUID_CAPS (on): 0 keeps CAP_SETUID/CAP_SETGID in sandbox
#     commands, letting them switch to non-root users (host subuids). CAP_SYS_PTRACE stays
#     dropped unless SANDFLEET_SANDBOX_DROP_NET_CAPS=0 too. Read by the step workers.
#   SANDFLEET_SANDBOX_DROP_NET_CAPS (on): 0 keeps CAP_NET_ADMIN/CAP_NET_RAW (and, with the
#     above also 0, CAP_SYS_PTRACE) in sandbox commands and %startscript; internet leases
#     then fail to start, since a sandbox could reroute around the forwarder. Read by the
#     step workers.
#   SANDFLEET_SANDBOX_SECCOMP (on): 0 disables the default seccomp profile, or set an
#     absolute path to an OCI seccomp profile. Read by the step workers.
#   SANDFLEET_SANDBOX_PIDS_LIMIT (4096): tasks per sandbox (RLIMIT_NPROC in its user
#     namespace); 0 disables. Read by the step workers.
#   SANDFLEET_SANDBOX_READONLY_MOUNT_ROOTS: colon-separated host directories whose
#     subdirectories leases may add as read-only mounts after starting (for example
#     verifier-only inputs; needs disk-backed sandboxes and the seccomp profile)
#   SANDFLEET_SANDBOX_AGENT_READONLY_MOUNT_ROOTS: colon-separated host directories whose
#     subdirectories leases may mount read-only when they start (inputs the agent may
#     read; needs the seccomp profile). None may contain a verifier-phase root above.
#   SANDFLEET_SANDBOX_AUDIT_LOG (on): off disables the per-worker command audit log under
#     runtime/logs/audit; SANDFLEET_SANDBOX_AUDIT_LOG_MAX_MB (256) rotates it.
#   SANDFLEET_STEP_START_RATE (8), SANDFLEET_STEP_START_BURST (8)
#   SANDFLEET_ENABLE_DYNAMIC_RESOURCES=1 with SANDFLEET_DYNAMIC_*,
#     SANDFLEET_MAX_DYNAMIC_POOLS (see the dynamic-resources block)
#   SANDFLEET_GPU_TYPE_CATALOG with SANDFLEET_GPU_DEVICE_ISOLATION_VERIFIED=1,
#     SANDFLEET_MAX_GPUS_PER_SANDBOX, SANDFLEET_GPU_CAPABILITY_REFRESH_SECONDS
#     (see the GPU block)

#SBATCH --job-name=sandfleet-controller
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00

set -euo pipefail

: "${SANDFLEET_ROOT:?}"
: "${SANDFLEET_TOKEN_FILE:?}"
: "${SANDFLEET_CLIENT_TOKEN_FILE:?}"
: "${SANDFLEET_METRICS_TOKEN_FILE:?}"
: "${SANDFLEET_WORKER_PARTITION:?}"
: "${SANDFLEET_WORKER_ACCOUNT:?}"
: "${SANDFLEET_AGENT_PYTHON:?}"
: "${SANDFLEET_APPTAINER_BINARY:?}"

export PYTHONPATH="${SANDFLEET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${SANDFLEET_ROOT}/runtime/logs"

if [[ -n "${SANDFLEET_NETWORK_DNS:-}" ]]; then
  echo "SANDFLEET_NETWORK_DNS was removed: sandboxes have no DNS; internet leases" \
    "resolve names through the egress forwarder. Unset it." >&2
  exit 2
fi

controller_extra_args=()
if [[ -n "${SANDFLEET_WORKER_QOS:-}" ]]; then
  controller_extra_args+=(--qos "${SANDFLEET_WORKER_QOS}")
fi
if [[ -n "${SANDFLEET_MAX_STORAGE_MB:-}" ]]; then
  controller_extra_args+=(--max-storage-mb "${SANDFLEET_MAX_STORAGE_MB}")
fi
if [[ -n "${SANDFLEET_CACHE_DIR:-}" ]]; then
  controller_extra_args+=(--cache-dir "${SANDFLEET_CACHE_DIR}")
fi
if [[ -n "${SANDFLEET_TMP_DIR:-}" ]]; then
  controller_extra_args+=(--tmp-dir "${SANDFLEET_TMP_DIR}")
fi
if [[ -n "${SANDFLEET_SANDBOX_INTERNET_ALLOWED_HOSTS:-}" ]]; then
  controller_extra_args+=(
    --sandbox-internet-allowed-hosts "${SANDFLEET_SANDBOX_INTERNET_ALLOWED_HOSTS}"
  )
fi
if [[ "${SANDFLEET_SANDBOX_INTERNET_LINK_LOCAL_DNS:-0}" == "1" ]]; then
  controller_extra_args+=(--sandbox-internet-link-local-dns)
fi
if [[ -n "${SANDFLEET_SANDBOX_READONLY_MOUNT_ROOTS:-}" ]]; then
  IFS=: read -r -a readonly_mount_roots <<< "${SANDFLEET_SANDBOX_READONLY_MOUNT_ROOTS}"
  for root in "${readonly_mount_roots[@]}"; do
    controller_extra_args+=(--sandbox-readonly-mount-root "${root}")
  done
fi
if [[ -n "${SANDFLEET_SANDBOX_AGENT_READONLY_MOUNT_ROOTS:-}" ]]; then
  IFS=: read -r -a agent_readonly_mount_roots <<< "${SANDFLEET_SANDBOX_AGENT_READONLY_MOUNT_ROOTS}"
  for root in "${agent_readonly_mount_roots[@]}"; do
    controller_extra_args+=(--sandbox-agent-readonly-mount-root "${root}")
  done
fi
if [[ "${SANDFLEET_CLEANUP_ON_EXIT:-0}" == "1" ]]; then
  controller_extra_args+=(--cleanup-on-exit)
fi
if [[ "${SANDFLEET_ENABLE_DYNAMIC_RESOURCES:-0}" == "1" ]]; then
  : "${SANDFLEET_DYNAMIC_WORKER_MAX_CPUS:?required for dynamic resources}"
  : "${SANDFLEET_DYNAMIC_WORKER_MAX_MEMORY_MB:?required for dynamic resources}"
  controller_extra_args+=(
    --enable-dynamic-resources
    --dynamic-worker-max-cpus "${SANDFLEET_DYNAMIC_WORKER_MAX_CPUS}"
    --dynamic-worker-max-memory-mb "${SANDFLEET_DYNAMIC_WORKER_MAX_MEMORY_MB}"
    --dynamic-worker-max-gpus "${SANDFLEET_DYNAMIC_WORKER_MAX_GPUS:-0}"
    --dynamic-pool-slots-per-job "${SANDFLEET_DYNAMIC_POOL_SLOTS_PER_JOB:-16}"
    --dynamic-pool-max-capacity "${SANDFLEET_DYNAMIC_POOL_MAX_CAPACITY:-512}"
    --dynamic-pool-warm-capacity "${SANDFLEET_DYNAMIC_POOL_WARM_CAPACITY:-0}"
    --dynamic-pool-idle-timeout-seconds "${SANDFLEET_DYNAMIC_POOL_IDLE_TIMEOUT_SECONDS:-600}"
    --max-dynamic-pools "${SANDFLEET_MAX_DYNAMIC_POOLS:-8}"
  )
fi
if [[ -n "${SANDFLEET_GPU_TYPE_CATALOG:-}" ]]; then
  if [[ "${SANDFLEET_GPU_DEVICE_ISOLATION_VERIFIED:-0}" != "1" ]]; then
    echo "GPU catalog requires verified Slurm device-cgroup isolation" >&2
    exit 2
  fi
  : "${SANDFLEET_MAX_GPUS_PER_SANDBOX:?required for GPU resources}"
  controller_extra_args+=(
    --gpu-type-catalog "${SANDFLEET_GPU_TYPE_CATALOG}"
    --gpu-device-isolation-verified
    --max-gpus-per-sandbox "${SANDFLEET_MAX_GPUS_PER_SANDBOX}"
    --gpu-capability-refresh-seconds "${SANDFLEET_GPU_CAPABILITY_REFRESH_SECONDS:-300}"
  )
fi

exec "${SANDFLEET_AGENT_PYTHON}" -m sandfleet.controller \
  --host 0.0.0.0 \
  --port "${SANDFLEET_PORT:-8765}" \
  --registry "${SANDFLEET_ROOT}/runtime/controller.json" \
  --state-file "${SANDFLEET_ROOT}/runtime/controller-state.json" \
  --partition "${SANDFLEET_WORKER_PARTITION}" \
  --account "${SANDFLEET_WORKER_ACCOUNT}" \
  --agent-advertise-mode "${SANDFLEET_AGENT_ADVERTISE_MODE:-fqdn}" \
  --cpu-allocation-granularity "${SANDFLEET_CPU_ALLOCATION_GRANULARITY:-1}" \
  --time-limit "${SANDFLEET_WORKER_TIME_LIMIT:-04:00:00}" \
  --source-root "${SANDFLEET_ROOT}" \
  --agent-python "${SANDFLEET_AGENT_PYTHON}" \
  --worker-python "${SANDFLEET_WORKER_PYTHON:-${SANDFLEET_AGENT_PYTHON}}" \
  --log-dir "${SANDFLEET_ROOT}/runtime/logs" \
  --apptainer-binary "${SANDFLEET_APPTAINER_BINARY}" \
  --network-mode "${SANDFLEET_NETWORK_MODE:-loopback}" \
  --gpu-container-mode "${SANDFLEET_GPU_CONTAINER_MODE:-auto}" \
  --sandbox-internet "${SANDFLEET_SANDBOX_INTERNET:-off}" \
  --sandbox-internet-max-connections "${SANDFLEET_SANDBOX_INTERNET_MAX_CONNECTIONS:-256}" \
  --sandbox-internet-max-tunnels-per-lease "${SANDFLEET_SANDBOX_INTERNET_MAX_TUNNELS_PER_LEASE:-32}" \
  --sandbox-internet-idle-timeout-seconds "${SANDFLEET_SANDBOX_INTERNET_IDLE_TIMEOUT_SECONDS:-300}" \
  --step-start-rate "${SANDFLEET_STEP_START_RATE:-8}" \
  --step-start-burst "${SANDFLEET_STEP_START_BURST:-8}" \
  --lease-ttl-seconds "${SANDFLEET_LEASE_TTL_SECONDS:-3600}" \
  --sandbox-audit-log "${SANDFLEET_SANDBOX_AUDIT_LOG:-on}" \
  --sandbox-audit-log-max-mb "${SANDFLEET_SANDBOX_AUDIT_LOG_MAX_MB:-256}" \
  "${controller_extra_args[@]}"
