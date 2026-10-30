"""Sandfleet exception types."""


class SandfleetError(RuntimeError):
    """Base error returned by a Sandfleet service or agent."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "SandfleetError",
        status: int | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.status = status


class SandboxOOMError(SandfleetError):
    """The sandbox exceeded its Slurm step memory limit."""


class SandboxLostError(SandfleetError):
    """The worker allocation hosting a sandbox disappeared."""


class NoCapacityError(SandfleetError):
    """A pool has no immediately available slots."""


class LeaseCleanupPending(SandfleetError):
    """Lease cleanup is waiting for an in-flight operation to drain."""

    def __init__(self, message: str):
        super().__init__(message, error_type="LeaseCleanupPending", status=503)
