"""Platform adapters: the simulators a policy is evaluated on.

A platform turns its own observation format into the canonical observation the
execution kernel understands.  Platforms know nothing about which policy will
be run, and the kernel knows nothing about which simulator produced an
observation.  MAIL-Bench evaluates on RoboCasa; the adapter is imported
from :mod:`mail_bench.platforms.robocasa` so that importing the package does not
require the simulator.
"""

__all__ = ["robocasa"]
