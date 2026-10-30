"""Slurm-backed sandbox fleet service."""

# Check the interpreter before importing runtime modules.
from sandfleet.version import VERSION

from sandfleet.client import SandfleetClient, SandboxLease
from sandfleet.errors import SandfleetError, SandboxLostError, SandboxOOMError
from sandfleet.resources import GPURequest, SandboxResources

__all__ = [
    "SandfleetClient",
    "SandfleetError",
    "GPURequest",
    "SandboxLease",
    "SandboxLostError",
    "SandboxOOMError",
    "SandboxResources",
]
__version__ = VERSION
