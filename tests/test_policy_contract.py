import pytest

from mail_bench.policy_api import (
    ContractError,
    cameras_outside_declared_scope,
    validate_declaration,
)


def declaration(**overrides):
    base = dict(
        protocol="mcr-custom-v1",
        model_id="somebody/their-own-model",
        checkpoint_sha256="a" * 64,
        policy_visible_cameras=["agentview", "eye_in_hand"],
        availability_consumed_by_policy=True,
        predicted_action_chunk=16,
        native_execution_horizon=8,
        action_dimension=7,
        stateful_policy=True,
        reset_semantics="clears_all_state",
        training_regime="fault_aware",
        category="hybrid_router_policy",
    )
    base.update(overrides)
    return base


def test_a_custom_architecture_is_admissible():
    # The benchmark must accept a submission it has never heard of, as long as
    # the declaration is complete: architecture is not an admission condition.
    parsed = validate_declaration(
        declaration(), environment_cameras=["agentview", "eye_in_hand"]
    )
    assert parsed.category == "hybrid_router_policy"
    assert parsed.training_regime == "fault_aware"


def test_every_required_field_is_enforced():
    from mail_bench.policy_api.contract import REQUIRED_IDENTITY_FIELDS

    for field in REQUIRED_IDENTITY_FIELDS:
        if field == "protocol":
            continue                      # checked separately, with its own message
        with pytest.raises(ContractError, match="missing required fields"):
            validate_declaration(declaration(**{field: None}))


def test_a_policy_must_identify_its_weights_exactly_once():
    """One file has a hash; a snapshot of shards and remote code has a digest.

    Both identify what ran, and a server that declares neither -- or both, which
    could disagree -- has not said which weights produced its actions.
    """
    with pytest.raises(ContractError, match="exactly one of"):
        validate_declaration(declaration(checkpoint_sha256=None))

    with pytest.raises(ContractError, match="exactly one of"):
        validate_declaration(declaration(checkpoint_sha256="a" * 64,
                                         inventory_sha256="b" * 64))

    parsed = validate_declaration(declaration(checkpoint_sha256=None,
                                              inventory_sha256="c" * 64))
    assert parsed.weight_digest_field == "inventory_sha256"
    assert parsed.checkpoint_sha256 == "c" * 64

    parsed = validate_declaration(declaration())
    assert parsed.weight_digest_field == "checkpoint_sha256"


def test_declaration_must_be_internally_consistent():
    with pytest.raises(ContractError, match="cannot exceed predicted_action_chunk"):
        validate_declaration(declaration(native_execution_horizon=32))
    with pytest.raises(ContractError, match="contradicts stateful_policy"):
        validate_declaration(declaration(reset_semantics="stateless"))
    with pytest.raises(ContractError, match="unknown reset_semantics"):
        validate_declaration(declaration(reset_semantics="maybe"))
    with pytest.raises(ContractError, match="positive integer"):
        validate_declaration(declaration(action_dimension=0))
    with pytest.raises(ContractError, match="must be boolean"):
        validate_declaration(declaration(stateful_policy="yes"))
    with pytest.raises(ContractError, match="policy_determinism"):
        validate_declaration(declaration(policy_determinism="sometimes"))


def test_policy_cameras_must_exist_on_the_platform():
    with pytest.raises(ContractError, match="does not provide"):
        validate_declaration(
            declaration(policy_visible_cameras=["agentview", "lidar"]),
            environment_cameras=["agentview", "eye_in_hand"],
        )
    with pytest.raises(ContractError, match="duplicates"):
        validate_declaration(declaration(policy_visible_cameras=["agentview", "agentview"]))


def test_a_single_view_policy_is_admissible_and_its_blind_camera_is_noted():
    parsed = validate_declaration(
        declaration(policy_visible_cameras=["agentview"]),
        environment_cameras=["agentview", "eye_in_hand"],
    )
    # Removing a camera the policy never reads is a no-op for its design; the
    # cells are still ranked, and the declaration is published beside the score.
    assert cameras_outside_declared_scope(parsed, ["agentview", "eye_in_hand"]) == ("eye_in_hand",)
    assert cameras_outside_declared_scope(parsed, ["agentview"]) == ()


def test_the_execution_kernel_never_imports_a_reference_model():
    # The benchmark is model-agnostic by construction: a submission is a policy
    # server plus a declaration, so no kernel module may reach for a named model.
    import pathlib

    core = pathlib.Path(__file__).resolve().parents[1] / "src" / "mail_bench"
    kernel = [
        "runner.py", "experiment.py", "manifest.py", "operators.py", "onset.py",
        "io.py", "registry.py", "seeds.py", "cohort.py", "driver.py", "scoring.py",
        "aggregate.py", "interfaces.py", "scenes.py", "suite.py", "semantic_states.py",
    ]
    for name in kernel:
        source = (core / name).read_text(encoding="utf-8")
        assert "reference_models" not in source, f"{name} imports a reference model"
        for model in ("pi05", "Pi05", "openvla", "OpenVla", "groot", "univla"):
            assert model not in source, f"{name} mentions the model {model!r}"
    # The policy contract itself must also stay free of model names.
    contract = (core / "policy_api" / "contract.py").read_text(encoding="utf-8")
    for model in ("pi05", "openvla", "groot", "univla"):
        assert model not in contract


def test_any_declared_training_regime_is_accepted():
    # There are no fixed training tracks. The benchmark records how a submission says
    # it was trained and takes no position on the answer: forbidding training on
    # the fault conditions would forbid fault-aware training, routers, imputation
    # and memory adaptation, which are among the things it exists to measure.
    for regime in ("clean_trained", "camera dropout p=0.15 during finetuning",
                   "pretrained on our own fault manifests", "whatever"):
        parsed = validate_declaration(declaration(training_regime=regime))
        assert parsed.training_regime == regime
    # Silence is still not an answer.
    with pytest.raises(ContractError, match="non-empty declaration"):
        validate_declaration(declaration(training_regime="   "))
    with pytest.raises(ContractError, match="missing required fields"):
        validate_declaration(declaration(training_regime=None))


def test_adaptation_track_is_read_as_training_regime():
    # ``adaptation_track`` is accepted as an alias of ``training_regime``; an
    # explicit ``training_regime`` wins.
    legacy = declaration()
    legacy["adaptation_track"] = legacy.pop("training_regime")
    assert validate_declaration(legacy).training_regime == "fault_aware"
    both = declaration(training_regime="camera dropout")
    both["adaptation_track"] = "clean_trained"
    assert validate_declaration(both).training_regime == "camera dropout"


def test_training_disclosure_is_carried_but_never_required():
    # Enough to reproduce and interpret, not every field that could exist: a
    # field left out is more honest than one invented.
    assert validate_declaration(declaration()).training_disclosure == {}
    parsed = validate_declaration(declaration(
        training_data_scope="demonstration data only",
        fault_augmentation_used=True,
        fault_types_seen_during_training=["hard_missing"],
    ))
    assert parsed.training_disclosure == {
        "training_data_scope": "demonstration data only",
        "fault_augmentation_used": True,
        "fault_types_seen_during_training": ["hard_missing"],
    }
