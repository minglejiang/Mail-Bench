"""The declaration every policy must make to be evaluated, whoever wrote it.

MAIL-Bench evaluates *any* embodied policy under time-structured visual
availability failures.  The execution kernel therefore never learns what a
particular model is: it only sees a :class:`~mail_bench.interfaces.PolicyAdapter`
and the declaration below.  A VLA, a world-action model, a router in front of two
policies, a reliability estimator feeding a fusion module, or something nobody has
built yet are all admissible as long as they declare these fields honestly and
run under the same observations, tasks and fault schedule.

What is *not* part of the contract is just as important.  ``family`` and
``category`` are analysis metadata that group results in a table; they are
never admission conditions, so ``category: custom`` is a perfectly valid entry.
What the benchmark does insist on is the information needed to compare fairly:
which cameras the policy actually reads, whether it consumes the availability
signal, how many actions it predicts and how many it executes before replanning,
whether it carries state across steps, what a reset is supposed to clear, which
weights ran, and how the submission was trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


#: How the submission was trained, in the submitter's own words.
#:
#: There are no fixed training tracks and no enum that would make a training
#: strategy something the benchmark accepts or rejects.  The benchmark takes no
#: position on whether a policy may be trained on its fault conditions: ruling
#: that out would rule out fault-aware training, routers, imputation and memory
#: adaptation, which are among the contributions this benchmark exists to
#: measure, and explicitly blessing it would steer submissions just as hard.  So
#: the field is free text, never ranked and never an admission condition; the
#: disclosure fields below are what let a reader tell one regime from another.
TRAINING_REGIME_DISCLOSURE_FIELDS = (
    "training_data_scope",
    "adaptation_recipe",
    "fault_augmentation_used",
    "fault_types_seen_during_training",
)

#: A server may announce ``adaptation_track`` instead; it is read as the
#: training regime rather than rejected.
LEGACY_TRAINING_REGIME_FIELD = "adaptation_track"

#: Fields every policy server must announce in its identity response.
#: ``policy_visible_cameras`` is the static upper bound of cameras that may
#: influence the policy's output, not the set a particular rollout happened to
#: use: a router that can switch between two cameras declares both, because
#: losing the other is not a no-op for it.  Nothing is inferred from an
#: observed routing trace; the declaration is what counts.
REQUIRED_IDENTITY_FIELDS = (
    "protocol",
    "model_id",
    "policy_visible_cameras",
    "availability_consumed_by_policy",
    "predicted_action_chunk",
    "native_execution_horizon",
    "action_dimension",
    "stateful_policy",
    "reset_semantics",
    "training_regime",
)

#: A policy must identify its weights with exactly one of these. ``checkpoint_sha256``
#: is the hash of a single checkpoint file; ``inventory_sha256`` is the digest of a
#: whole snapshot, for weights that are several shards plus auxiliary heads plus code
#: loaded through trust_remote_code, where no single file identifies what ran.
WEIGHT_DIGEST_FIELDS = ("checkpoint_sha256", "inventory_sha256")

#: What a ``reset`` is required to clear, declared so that a stateful policy
#: cannot silently carry an episode's memory into the next one, and so that the
#: benchmark never has to guess whether a reset also wiped visual memory.
RESET_SEMANTICS = {
    # Reset re-seeds sampling only; the policy holds no cross-step state.
    "stateless",
    # Reset clears recurrent/world state and any cached observations.
    "clears_all_state",
    # Reset clears recurrent state but a declared cache survives; the cache
    # contents must be described in the method configuration.
    "clears_policy_state_only",
}


class ContractError(ValueError):
    """Raised when a policy declaration is missing or self-inconsistent."""


@dataclass(frozen=True)
class PolicyDeclaration:
    """One policy's validated declaration, independent of its architecture."""

    protocol: str
    model_id: str
    #: The digest that identifies the weights, whether it hashes one file or a
    #: whole snapshot. ``weight_digest_field`` records which was declared.
    checkpoint_sha256: str
    policy_visible_cameras: tuple[str, ...]
    availability_consumed_by_policy: bool
    predicted_action_chunk: int
    native_execution_horizon: int
    action_dimension: int
    stateful_policy: bool
    reset_semantics: str
    #: Free text, declared by the submitter. Never ranked, never gated on.
    training_regime: str
    #: Whichever of TRAINING_REGIME_DISCLOSURE_FIELDS the server chose to send.
    #: Optional, because a field left out is more honest than one invented, and
    #: the protocol asks for enough provenance to reproduce and interpret a
    #: result rather than for every field that could exist.
    training_disclosure: Mapping[str, Any] = field(default_factory=dict)
    #: Grouping label for analysis only; never an admission condition.
    category: Optional[str] = None
    #: 'deterministic' or 'stochastic'; may be left unset and detected at run time.
    policy_determinism: Optional[str] = None
    #: Which of WEIGHT_DIGEST_FIELDS the server identifies its weights with.
    weight_digest_field: str = "checkpoint_sha256"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_declaration(
    identity: Mapping[str, Any],
    *,
    environment_cameras: Optional[Sequence[str]] = None,
) -> PolicyDeclaration:
    """Validate a policy server's identity against the contract.

    ``environment_cameras``, when given, is the platform's camera inventory; the
    policy's visible cameras must be a subset of it.  A camera the policy never
    reads is not a defect -- a single-view policy is admissible -- but removing
    such a camera is a no-op for that policy, so the subset is declared and
    published beside the score as provenance; every cell is ranked regardless.
    Declare the upper bound: for a router, every camera it may ever select.
    """
    # ``adaptation_track`` is read as ``training_regime``; an explicit
    # ``training_regime`` wins.
    if identity.get("training_regime") is None and identity.get(
        LEGACY_TRAINING_REGIME_FIELD
    ) is not None:
        identity = {
            **identity,
            "training_regime": identity[LEGACY_TRAINING_REGIME_FIELD],
        }
    missing = [field for field in REQUIRED_IDENTITY_FIELDS if identity.get(field) is None]
    _require(not missing, f"policy declaration is missing required fields: {missing}")

    # Contract integrity, not scientific semantics: a declaration whose fields are
    # not even well formed cannot identify what ran.
    for name in ("protocol", "model_id"):
        value = identity[name]
        _require(
            isinstance(value, str) and value.strip() != "",
            f"{name} must be a non-empty string, got {value!r}",
        )
    declared_digests = [field for field in WEIGHT_DIGEST_FIELDS
                        if identity.get(field) is not None]
    _require(
        len(declared_digests) == 1,
        f"a policy must identify its weights with exactly one of "
        f"{list(WEIGHT_DIGEST_FIELDS)}, got {declared_digests}",
    )
    digest_field = declared_digests[0]
    digest = identity[digest_field]
    _require(isinstance(digest, str), f"{digest_field} must be a string")
    digest = digest.strip()
    _require(
        len(digest) == 64 and all(character in "0123456789abcdefABCDEF" for character in digest),
        f"{digest_field} must be 64 hexadecimal characters, got {identity[digest_field]!r}",
    )
    digest = digest.lower()

    cameras = tuple(str(camera) for camera in identity["policy_visible_cameras"])
    _require(bool(cameras), "a policy must declare at least one visible camera")
    _require(len(set(cameras)) == len(cameras), "policy_visible_cameras contains duplicates")
    if environment_cameras is not None:
        unknown = set(cameras) - {str(camera) for camera in environment_cameras}
        _require(
            not unknown,
            f"policy declares cameras the platform does not provide: {sorted(unknown)}",
        )

    regime = str(identity["training_regime"]).strip()
    # Non-empty is all that is checked. The benchmark records how a submission
    # says it was trained; it does not decide which answers are allowed.
    _require(regime != "", "training_regime must be a non-empty declaration")
    reset = str(identity["reset_semantics"])
    _require(
        reset in RESET_SEMANTICS,
        f"unknown reset_semantics {reset!r}; expected one of {sorted(RESET_SEMANTICS)}",
    )

    for name in ("predicted_action_chunk", "native_execution_horizon", "action_dimension"):
        value = identity[name]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"{name} must be a positive integer, got {value!r}",
        )
    for name in ("availability_consumed_by_policy", "stateful_policy"):
        _require(isinstance(identity[name], bool), f"{name} must be boolean")

    _require(
        int(identity["native_execution_horizon"]) <= int(identity["predicted_action_chunk"]),
        "native_execution_horizon cannot exceed predicted_action_chunk: a policy cannot "
        "execute more actions than it predicted",
    )
    stateless_but_stateful = (
        reset == "stateless" and bool(identity["stateful_policy"]) is True
    )
    _require(
        not stateless_but_stateful,
        "reset_semantics='stateless' contradicts stateful_policy=true",
    )

    determinism = identity.get("policy_determinism")
    _require(
        determinism in (None, "deterministic", "stochastic"),
        f"policy_determinism must be 'deterministic', 'stochastic' or unset, got {determinism!r}",
    )

    return PolicyDeclaration(
        protocol=str(identity["protocol"]).strip(),
        model_id=str(identity["model_id"]).strip(),
        checkpoint_sha256=digest,
        weight_digest_field=digest_field,
        policy_visible_cameras=cameras,
        availability_consumed_by_policy=bool(identity["availability_consumed_by_policy"]),
        predicted_action_chunk=int(identity["predicted_action_chunk"]),
        native_execution_horizon=int(identity["native_execution_horizon"]),
        action_dimension=int(identity["action_dimension"]),
        stateful_policy=bool(identity["stateful_policy"]),
        reset_semantics=reset,
        training_regime=regime,
        training_disclosure={
            field: identity[field]
            for field in TRAINING_REGIME_DISCLOSURE_FIELDS
            if identity.get(field) is not None
        },
        category=identity.get("category"),
        policy_determinism=determinism,
    )


def cameras_outside_declared_scope(
    declaration: PolicyDeclaration,
    environment_cameras: Sequence[str],
) -> tuple[str, ...]:
    """Platform cameras the policy declares it never reads.

    Provenance, not a ranking rule. MAIL-Bench ranks every policy on all ten
    conditions: a policy that keeps working without a camera it never read is
    telling us how much it was using vision, and that belongs in the score
    exactly as it does for total visual loss. The declaration is published
    beside the score so a reader can see which conditions were no-ops for the
    design under test.
    """
    visible = set(declaration.policy_visible_cameras)
    return tuple(sorted(str(camera) for camera in environment_cameras if camera not in visible))
