"""The benchmark standardizes evaluation, not model design.

Each test here is a restriction the protocol deliberately does not have. They
exist because such rules are easy to reintroduce by accident, in an enum, a
roster check or a validation message, and because their absence is a design
decision rather than an oversight.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from mail_bench.policy_api.contract import ContractError, validate_declaration

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def protocol():
    return yaml.safe_load(
        (ROOT / "configs" / "perturbation_protocol.yaml").read_text(encoding="utf-8")
    )


def declaration(**overrides):
    base = {
        "protocol": "mcr-policy-v1", "model_id": "somebody/router",
        "checkpoint_sha256": "a" * 64, "policy_visible_cameras": ["cam0"],
        "availability_consumed_by_policy": True, "predicted_action_chunk": 4,
        "native_execution_horizon": 4, "action_dimension": 7,
        "stateful_policy": True, "reset_semantics": "clears_all_state",
        "training_regime": "clean_trained",
    }
    base.update(overrides)
    return base


def test_training_on_the_fault_conditions_is_neither_forbidden_nor_blessed(protocol):
    openness = protocol["model_and_submission_openness"]["training_regime"]
    assert openness["value"] == "declared_by_submitter"
    assert openness["ranked"] is False
    assert openness["admission_condition"] is False
    # A method whose contribution is fault-aware training runs in the ordinary
    # cohort: forbidding it would rule out routers, imputation and memory
    # adaptation, which are among the things the benchmark exists to measure.
    parsed = validate_declaration(declaration(
        training_regime="finetuned on hard_missing episodes",
        fault_types_seen_during_training=["hard_missing", "freeze"],
    ))
    assert parsed.training_regime == "finetuned on hard_missing episodes"
    assert parsed.training_disclosure["fault_types_seen_during_training"] == [
        "hard_missing", "freeze",
    ]


def test_the_protocol_never_enumerates_permitted_training_regimes(protocol):
    assert protocol["model_and_submission_openness"]["training_regime"]["ranked"] is False


def test_architecture_is_never_an_admission_condition(protocol):
    unconstrained = set(protocol["model_and_submission_openness"]["unconstrained"])
    for freedom in ("architecture", "adaptation_strategy", "memory", "router",
                    "imputation", "world_model", "test_time_adaptation",
                    "action_representation"):
        assert freedom in unconstrained
    # The benchmark evaluates one suite, so which platforms a submission enters is not a
    # freedom the protocol grants -- there is nothing to choose.
    assert "which_platforms_are_entered" not in unconstrained
    parsed = validate_declaration(declaration(category="hybrid_router_policy"))
    assert parsed.category == "hybrid_router_policy"


def test_what_is_still_closed_stays_closed(protocol):
    # Openness is about model design. Everything that decides what a number
    # means is still fail-closed.
    assert protocol["causal_rules"]["unavailable_camera_post_onset_frames_visible_to"] == []
    assert protocol["onset_definition"]["fallback"] == "none"
    assert protocol["pairing"]["healthy_arm"]["official_replicates_per_seed"] == 1
    # A declaration that says nothing about how it was trained is still refused:
    # the benchmark does not judge the answer, but it does require one.
    with pytest.raises(ContractError, match="non-empty declaration"):
        validate_declaration(declaration(training_regime=""))


def test_the_main_ranking_alone_is_a_complete_submission(protocol):
    # Named for what they study rather than numbered: a reader should not have
    # to learn an index before understanding an experiment.
    dimensions = protocol["evaluation_dimensions"]["what_how"]
    assert dimensions["required_for_a_valid_submission"] == ["main_ranking"]
    assert dimensions["optional_mechanism_extensions"] == [
        "failure_form", "missing_duration", "visual_recovery",
    ]


def test_a_partial_task_set_is_an_experiment_not_a_submission(protocol):
    coverage = protocol["model_and_submission_openness"]["task_coverage"]
    assert coverage["official_submission_requires"] == "all_eighteen_atomic_seen_tasks"
    # Openness is about model design, not about what a number may be called.
    assert "never as state of the art" in coverage["partial_task_coverage"]
    assert "eighteen specialists" in coverage["one_policy_configuration"]


def test_the_four_dimensions_are_the_concept_layer(protocol):
    dimensions = protocol["evaluation_dimensions"]
    assert list(dimensions) == ["capability", "when", "what_how", "outcome"]
    # Named for what they study rather than numbered, so a reader need not
    # learn an index before understanding an experiment.
    assert dimensions["what_how"]["measured_by"] == [
        "main_ranking", "failure_form", "missing_duration", "visual_recovery",
    ]
    assert dimensions["when"]["estimand"] == (
        "phase_normalized_robustness_not_identical_absolute_time"
    )
    assert dimensions["when"]["fractions"] == [0.30, 0.45, 0.60]


def test_the_protocol_describes_one_platform(protocol):
    """MAIL-Bench evaluates RoboCasa Atomic-Seen, and says only that.

    The implementation evaluates exactly the platform the protocol names.
    """
    assert list(protocol["semantic_camera_states"]["platform_mapping"]) == ["robocasa365"]
    assert list(protocol["platform_profiles"]) == ["robocasa"]


def test_the_library_evaluates_exactly_the_platform_the_protocol_names():
    """A platform is present only with its adapter, profile and role map
    together."""
    import yaml

    profiles = yaml.safe_load(
        (ROOT / "configs" / "platform_profiles.yaml").read_text(encoding="utf-8")
    )
    assert list(profiles["platforms"]) == ["robocasa365"]
    assert "MAIL-Bench evaluates robocasa365 only" in profiles["scope"]

    from mail_bench.semantic_states import PLATFORM_ROLES

    assert set(PLATFORM_ROLES) == {"robocasa365"}


def test_the_mail_roster_is_one_configuration_per_baseline():
    import yaml

    roster = yaml.safe_load(
        (ROOT / "configs" / "mail_bench_roster.yaml").read_text(encoding="utf-8")
    )
    assert roster["platform"] == "robocasa365" and roster["suite"] == "atomic_seen"
    assert roster["scene_bank_id"] == "mail_robocasa_atomic_seen_v1"
    for model in roster["models"]:
        # One identity per baseline: a row that could mean eighteen checkpoints
        # would let a submission be eighteen specialists wearing one name.
        assert "inventory_sha256" in model or "checkpoint_sha256" in model
        assert "suite_checkpoints" not in model
    assert roster["admission"]["clean_success_is"] == "reported_not_gated"


def test_the_suite_is_frozen_in_the_repository():
    """A benchmark whose task identity comes from an installed package is
    immutable only until somebody upgrades that package.

    The protocol says the task set is immutable. Reading it from RoboCasa's
    TARGET_TASKS at run time would make that a statement about whichever
    RoboCasa happens to be installed.
    """
    from mail_bench.suite import SuiteMismatch, horizons, scenes_per_task, task_ids
    from mail_bench.suite import verify_against_platform

    assert len(task_ids()) == 18
    assert scenes_per_task() == 50
    assert len(task_ids()) * scenes_per_task() == 900
    assert sorted(task_ids()) == list(task_ids())      # a stable published order

    frozen = horizons()
    # A platform that renamed a task, dropped one, or changed a horizon is
    # refused rather than evaluated silently as if it were the suite.
    verify_against_platform(list(frozen), lambda task: frozen[task])
    with pytest.raises(SuiteMismatch, match="missing"):
        verify_against_platform([t for t in frozen if t != "OpenDrawer"],
                                lambda task: frozen[task])
    with pytest.raises(SuiteMismatch, match="horizons differ"):
        verify_against_platform(list(frozen),
                                lambda task: 999 if task == "OpenDrawer" else frozen[task])


def test_a_downloaded_checkpoint_is_not_a_baseline():
    """Bytes pinned is not capability shown.

    Upstream results are reported against upstream's own evaluation layer, and
    a checkpoint that carries a RoboCasa embodiment has not thereby been shown
    to work on the frozen eighteen scenes through this benchmark's observation
    and action interface.
    """
    import yaml

    roster = yaml.safe_load(
        (ROOT / "configs" / "mail_bench_roster.yaml").read_text(encoding="utf-8")
    )
    accepted = {m["id"] for m in roster["models"]}
    assert accepted
    # Nothing is listed as a baseline on the strength of a download alone: every
    # entry pins its bytes and points at the server that produced its cells.
    for entry in roster["models"]:
        assert entry["pin_status"] == "pinned"
        assert len(entry["inventory_sha256"]) == 64
        assert entry["source_revision"] and entry["source_revision"] != "main"
        assert (ROOT / entry["verified_receipt"]).is_file()
        if entry.get("server"):
            assert (ROOT / entry["server"]).is_file()
