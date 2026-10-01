"""The three functional visual roles the ranking asks about.

Enumerating a platform's physical camera subsets instead would make a
three-camera platform answer seven questions and a two-camera platform three,
and -- worse -- no condition would mean the same thing on two platforms. A
two-camera platform has no counterpart to "one of RoboCasa's two agentviews is
gone", so a cross-platform number could only ever average quantities that are
not comparable.

Three roles are comparable everywhere: the wrist view, the agentview, and all of
it. Where a platform serves a role with two cameras they move together, because
losing one of two redundant onboard views is a materially milder condition than
losing the role.

MAIL-Bench evaluates RoboCasa Atomic-Seen; a later release that adds a platform
adds its role map here together with its adapter and profile.

A partial combination (one RoboCasa agentview alone) is a legitimate diagnostic
and still executes; it is simply not part of a ranking whose conditions have to
mean the same thing on every platform.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

#: The three missing states, in ranking order. ``healthy`` is the fourth
#: availability state but is not a *missing* one, so it is kept separate.
MISSING_STATES = ("wrist_missing", "agentview_missing", "all_vision_missing")
AVAILABILITY_STATES = ("healthy",) + MISSING_STATES

#: The onset fractions the ranking sweeps, as task phases rather than steps.
RANKING_ONSETS = (0.30, 0.45, 0.60)

#: Which cameras each platform uses for each role. Keyed by the platform id the
#: profiles use, not by the dataset registry id.
PLATFORM_ROLES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "robocasa365": {
        "wrist": ("robot0_eye_in_hand",),
        "third_person": ("robot0_agentview_left", "robot0_agentview_right"),
    },
}


class SemanticStateError(ValueError):
    """A platform's cameras do not map onto the three roles."""


def roles_for(platform: str) -> Mapping[str, tuple[str, ...]]:
    if platform not in PLATFORM_ROLES:
        raise SemanticStateError(
            f"no semantic camera roles are defined for platform {platform!r}; "
            f"known platforms are {sorted(PLATFORM_ROLES)}"
        )
    return PLATFORM_ROLES[platform]


def state_cameras(platform: str, state: str) -> tuple[str, ...]:
    """The cameras a missing state removes on this platform."""
    roles = roles_for(platform)
    if state == "wrist_missing":
        return tuple(sorted(roles["wrist"]))
    if state == "agentview_missing":
        return tuple(sorted(roles["third_person"]))
    if state == "all_vision_missing":
        return tuple(sorted(set(roles["wrist"]) | set(roles["third_person"])))
    if state == "healthy":
        return ()
    raise SemanticStateError(
        f"unknown availability state {state!r}; expected one of {list(AVAILABILITY_STATES)}"
    )


def validate_platform(platform: str, platform_camera_ids: Sequence[str]) -> None:
    """Every camera in the ranking must exist, and cover the platform's grid.

    A role naming a camera the platform does not have would silently produce a
    condition that removes nothing. A platform camera in no role would sit
    outside the ranking entirely, which is a design decision and not something
    that should happen by omission.
    """
    provided = set(str(camera) for camera in platform_camera_ids)
    roles = roles_for(platform)
    named = set(roles["wrist"]) | set(roles["third_person"])
    unknown = named - provided
    if unknown:
        raise SemanticStateError(
            f"{platform} roles name cameras the platform does not provide: {sorted(unknown)}"
        )
    unassigned = provided - named
    if unassigned:
        raise SemanticStateError(
            f"{platform} cameras belong to no semantic role: {sorted(unassigned)}; "
            "every ranked camera must sit in the wrist or third-person role"
        )
    overlap = set(roles["wrist"]) & set(roles["third_person"])
    if overlap:
        raise SemanticStateError(
            f"{platform} assigns {sorted(overlap)} to both roles"
        )


def state_of(platform: str, faulted_cameras: Sequence[str]) -> Optional[str]:
    """Which semantic state a set of faulted cameras is, or ``None``.

    ``None`` is a partial combination -- a single agentview, say. It executes
    and is reported, and sits outside the ranking because it has no counterpart
    on a platform whose role is served by one camera.
    """
    faulted = tuple(sorted(str(camera) for camera in faulted_cameras))
    if not faulted:
        return "healthy"
    for state in MISSING_STATES:
        if faulted == state_cameras(platform, state):
            return state
    return None
