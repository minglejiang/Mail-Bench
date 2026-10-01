"""Canonical observation schema and adapter interfaces for execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence


def _dict(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class CanonicalObservation:
    """One platform-neutral observation at an environment control step."""

    step: int
    cameras: Mapping[str, Any]
    availability: Mapping[str, bool]
    robot_state: Any = None
    language: str = ""
    capture_time_ms: Mapping[str, Optional[float]] = field(default_factory=dict)
    arrival_time_ms: Mapping[str, Optional[float]] = field(default_factory=dict)
    sequence_id: Mapping[str, Optional[int]] = field(default_factory=dict)
    new_frame: Mapping[str, bool] = field(default_factory=dict)
    source_step: Mapping[str, Optional[int]] = field(default_factory=dict)
    source_age_steps: Mapping[str, Optional[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("observation step must be >= 0")
        cameras = _dict(self.cameras)
        availability = {str(k): bool(v) for k, v in self.availability.items()}
        camera_ids = set(cameras)
        if camera_ids != set(availability):
            raise ValueError("cameras and availability must contain identical camera ids")
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "availability", availability)
        for name in (
            "capture_time_ms",
            "arrival_time_ms",
            "sequence_id",
            "new_frame",
            "source_step",
            "source_age_steps",
        ):
            values = _dict(getattr(self, name))
            unknown = set(values) - camera_ids
            if unknown:
                raise ValueError(f"{name} references unknown cameras: {sorted(unknown)}")
            object.__setattr__(self, name, values)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.cameras))

    def with_fault(
        self,
        cameras: Mapping[str, Any],
        availability: Mapping[str, bool],
        *,
        source_step: Mapping[str, Optional[int]],
        source_age_steps: Mapping[str, Optional[int]],
    ) -> "CanonicalObservation":
        """Return the policy-visible observation after fault injection.

        The fields that describe *the frame* follow the frame. A frozen or
        delayed camera delivers something captured earlier, so it is not a new
        frame, and the sequence number and capture time of the current step are
        not its own: the injector keeps frames by step and not their capture
        metadata, so those are reported as unknown rather than as the current
        step's, which would be false. ``arrival_time_ms`` is left alone because
        it is when the delivery happened, and the delivery is happening now --
        that split between capture and arrival is what the two fields are for.

        Reporting the current step's values here would publish a
        self-contradicting message (``source_age_steps`` ten steps old beside
        ``new_frame`` True); an availability-aware policy reads exactly these
        fields to decide whether it is looking at something new, which is the
        kind of policy this benchmark exists to evaluate.
        """
        ages = dict(source_age_steps)
        fresh = {
            camera: (cameras.get(camera) is not None and ages.get(camera) == 0)
            for camera in cameras
        }
        return replace(
            self,
            cameras=dict(cameras),
            availability=dict(availability),
            source_step=dict(source_step),
            source_age_steps=ages,
            new_frame=fresh,
            sequence_id={c: (self.sequence_id.get(c) if fresh[c] else None) for c in cameras},
            capture_time_ms={
                c: (self.capture_time_ms.get(c) if fresh[c] else None) for c in cameras
            },
        )


@dataclass(frozen=True)
class EnvironmentStep:
    """Result of executing one action in an environment adapter."""

    observation: CanonicalObservation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyOutput:
    """A non-empty action chunk returned by a policy query."""

    actions: Sequence[Any]

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("policy output must contain at least one action")
        object.__setattr__(self, "actions", actions)


class EnvironmentAdapter(ABC):
    """Boundary between a simulator/dataset and the execution kernel."""

    @abstractmethod
    def reset(self, environment_seed: int) -> CanonicalObservation:
        """Reset to a deterministic frozen scene and return step-zero observation."""

    @abstractmethod
    def step(self, action: Any) -> EnvironmentStep:
        """Execute exactly one control action."""

    @abstractmethod
    def camera_inventory(self) -> Sequence[str]:
        """Return the platform's canonical camera ids."""

    @abstractmethod
    def official_metrics(self) -> Mapping[str, Any]:
        """Return official platform metrics for the current episode."""

    @abstractmethod
    def success(self) -> bool:
        """Return the official episode success decision."""

    def dataset_identity(self) -> Optional[tuple[str, str]]:
        """Return ``(dataset_id, revision)`` for real execution authorization."""
        return None

    def manifest_identity(self) -> Optional[tuple[str, Any, Any]]:
        """Return ``(adapter_version, scene_or_task, episode_or_sequence)``."""
        return None

    def close(self) -> None:
        """Release simulator resources when the adapter owns any."""


@dataclass(frozen=True)
class EpisodeContext:
    """What a deployed policy knows before an episode starts.

    Task identity, which scene of it this is, and the official horizon are
    public in the frozen suite and known to any policy at deployment; a
    policy that budgets its own behaviour against the horizon (a lease, a
    fallback duration) needs them and cannot recover them from the stream.
    Nothing here touches a frame: the fault schedule, onset and the hidden
    healthy frames are deliberately absent.
    """

    task: Optional[str] = None
    episode_index: Optional[int] = None
    official_horizon: Optional[int] = None
    instruction: Optional[str] = None

    def as_message_fields(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "episode_index": None if self.episode_index is None else int(self.episode_index),
            "official_horizon": (
                None if self.official_horizon is None else int(self.official_horizon)
            ),
            "instruction": self.instruction,
        }


class PolicyAdapter(ABC):
    """Boundary between canonical observations and one embodied policy."""

    def announce_episode(self, context: EpisodeContext) -> None:
        """Tell the policy which episode is about to start; precedes ``reset``."""

    @abstractmethod
    def reset(self, policy_seed: int) -> None:
        """Reset policy RNG, recurrent state and route state."""

    @abstractmethod
    def act(self, observation: CanonicalObservation) -> PolicyOutput:
        """Query the policy once and return one action chunk."""

    def invalidate_action_chunk(self, reason: str, step: int) -> None:
        """Notify the adapter that queued actions were invalidated."""

    def reset_visual_memory(self, reason: str, step: int) -> None:
        """Optional hook for a preregistered recurrent/world-state reset."""

    def close(self) -> None:
        """Release policy runtime resources such as a server connection."""
