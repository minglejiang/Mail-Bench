"""Camera mountings of the evaluated platform, and one vocabulary for them.

A condition is defined by what the policy loses -- a wrist view, every
third-person view, all vision -- so the profile records what each camera
physically is, read from the simulator's own model, and the three places that
name cameras (the profile, the semantic role map and the adapter) must agree.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs" / "platform_profiles.yaml"
PROTOCOL = ROOT / "configs" / "perturbation_protocol.yaml"


@pytest.fixture(scope="module")
def profiles():
    return yaml.safe_load(PROFILES.read_text(encoding="utf-8"))


def test_every_camera_records_the_body_it_is_mounted_on(profiles):
    for platform, profile in profiles["platforms"].items():
        assert profile["verified_from"], f"{platform} does not say how this was checked"
        for camera, entry in profile["cameras"].items():
            assert entry.get("parent_body"), f"{platform}/{camera}"
            if entry.get("in_policy_grid"):
                assert entry.get("canonical_id"), f"{platform}/{camera} has no canonical_id"
            assert isinstance(entry["moves_with_robot"], bool), f"{platform}/{camera}"
            # A camera on the world frame is the only kind that cannot move.
            assert entry["moves_with_robot"] == (entry["parent_body"] != "world"), (
                f"{platform}/{camera} contradicts its own mounting"
            )


def test_grid_cameras_are_canonical_ids_of_the_cameras_marked_for_the_grid(profiles):
    """The grid speaks the benchmark's names, not the simulator's."""
    for platform, profile in profiles["platforms"].items():
        marked = {entry["canonical_id"] for entry in profile["cameras"].values()
                  if entry.get("in_policy_grid")}
        assert marked == set(profile["grid_cameras"]), platform
        for grid_camera in profile["grid_cameras"]:
            sources = [name for name, entry in profile["cameras"].items()
                       if entry.get("canonical_id") == grid_camera]
            assert len(sources) == 1, f"{platform}/{grid_camera} -> {sources}"


def test_the_evaluated_platform_is_the_only_one_and_has_no_world_fixed_view(profiles):
    assert list(profiles["platforms"]) == ["robocasa365"]
    profile = profiles["platforms"]["robocasa365"]
    assert profile["control_hz"] == 20
    by_canonical = {entry["canonical_id"]: entry for entry in profile["cameras"].values()
                    if entry.get("in_policy_grid")}
    assert all(by_canonical[camera]["moves_with_robot"] for camera in profile["grid_cameras"])


def test_three_files_name_the_same_cameras(profiles):
    """The profile, the role map, the adapter and the protocol must name the
    same cameras; a second vocabulary would make a preflight match nothing.
    Each of these is asserted on its own elsewhere; this is the one place they
    are held to each other.
    """
    from mail_bench.platforms.robocasa import ROBOCASA_CAMERA_KEYS
    from mail_bench.semantic_states import MISSING_STATES, PLATFORM_ROLES, state_cameras

    profile = profiles["platforms"]["robocasa365"]
    grid = sorted(profile["grid_cameras"])
    assert grid == sorted(ROBOCASA_CAMERA_KEYS)
    assert set(PLATFORM_ROLES) == {"robocasa365"}
    roles = PLATFORM_ROLES["robocasa365"]
    assert sorted(roles["wrist"] + roles["third_person"]) == grid
    for state in MISSING_STATES:
        assert sorted(profile["missing_states"][state]) == sorted(state_cameras("robocasa365", state))
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert sorted(protocol["platform_profiles"]["robocasa"]["cameras"]) == grid
