"""The MAIL-Bench scoring hierarchy: rollouts to condition, task, benchmark.

Each step is a plain average, and each is an average over a different thing, so
the order matters and is fixed here rather than left to whoever builds a table.
The eighteen tasks are averaged equally, so a task with more scenes does not
weigh more and the benchmark score is not a rollout average wearing a task
average's name.

The missing-condition score is conditional on the policy being able to do the
task: it is measured on the scenes whose canonical healthy rollout succeeded,
because a scene the policy never solved has no task phase to place a fault in.
That makes it answer "of the scenes it can solve, how many survive this fault",
which is not the same quantity as the share of all scenes solved under the
fault. MAIL-Bench ranks on the conditional one and reports no second reading of
it: two numbers for one condition invite a reader to pick whichever supports the
claim, and the healthy score already stands beside them saying how much of the
suite the policy can do at all.

The conditional denominator is the number of healthy-successful scenes, not the
number of fault rollouts that were written. A fault cell that was owed and never
produced therefore lowers the score, instead of disappearing from its own
denominator and leaving the gap invisible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from .semantic_states import MISSING_STATES, RANKING_ONSETS


def benchmark_task_count() -> int:
    """How many tasks an official score averages: the frozen suite's, read once.

    Not a literal: the suite file is the one place the task list lives, and a
    second copy of its length here would be a number that could drift from it.
    """
    from .suite import task_ids

    return len(task_ids())

#: The nine missing conditions, in reporting order.
MISSING_CONDITIONS = tuple(
    (state, onset) for state in MISSING_STATES for onset in RANKING_ONSETS
)
#: Ten equally weighted components: one healthy and the nine above.
RANKING_COMPONENTS = 1 + len(MISSING_CONDITIONS)


class ScoringError(ValueError):
    """A score was asked for from evidence that cannot support it."""


@dataclass(frozen=True)
class ConditionScore:
    """One missing condition of one task.

    The counts answer different questions and collapsing them loses the
    difference between a cell that was never run and one that ran without the
    intervention ever reaching the policy. Every count is over the scenes the
    condition *owes* -- the ones whose healthy rollout succeeded -- because a
    fault on any other scene has no healthy pairing and no place in ``M_c``.
    """

    state: str
    onset_fraction: float
    #: Successes among the *valid* owed rollouts. A success on a cell the fault
    #: never reached is not a success under the fault.
    successes: int
    #: Owed fault rollouts actually written, whatever their validity.
    written: int
    #: Written owed rollouts in which the policy actually consumed a faulted
    #: observation.
    valid: int
    #: Healthy-successful scenes of the task: the score's denominator.
    healthy_successes: int
    #: Owed scenes with no fault cell, and fault cells on scenes that are not
    #: owed. Either is a run that does not match its own healthy phase; both are
    #: named so the gap is locatable, not merely counted.
    missing_episodes: tuple[int, ...] = ()
    extra_episodes: tuple[int, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.state}@{self.onset_fraction:.2f}"

    @property
    def invalid(self) -> int:
        """Rollouts that ran but in which the fault never reached the policy."""
        return self.written - self.valid

    @property
    def episodes_match_healthy(self) -> bool:
        """Whether the scenes this condition covered are exactly the owed ones.

        Counting alone cannot tell {1,3,5,7} from {1,2,4,7}, and the protocol
        pairs a fault with the scene the policy solved, not with some other
        scene of the same task.
        """
        return not self.missing_episodes and not self.extra_episodes

    @property
    def score(self) -> Optional[float]:
        """``M_c``: valid successes over the scenes the healthy rollout solved.

        The denominator is the healthy-success count and never shrinks. A cell
        that was owed and never written lowers the score; so does one that ran
        without the fault ever reaching the policy, because a task completed
        before the intervention arrived says nothing about robustness. Only
        owed scenes are in the numerator, so the ratio cannot exceed one.
        """
        if self.healthy_successes == 0:
            return None
        return self.successes / self.healthy_successes

    @property
    def complete(self) -> bool:
        """Whether every owed fault rollout was executed, and nothing else was.

        Coverage of execution, not of exposure. A scene whose healthy rollout
        finishes in one step leaves a 60% onset with no episode left to act on,
        so its cell is structurally invalid however many times it is run;
        requiring valid == owed would make such a benchmark permanently
        incomplete for a reason no submission can fix. The invalid count is
        reported instead.
        """
        return self.written == self.healthy_successes and self.episodes_match_healthy

    @property
    def ranked_value(self) -> float:
        """What enters the task score.

        ``None`` becomes zero here and only here. A task with no healthy success
        has no scene on which any fault could be evaluated, and the protocol
        scores that as zero rather than dropping the task -- dropping it would
        let a policy improve its platform score by failing a task completely.
        """
        value = self.score
        return 0.0 if value is None else value


@dataclass(frozen=True)
class TaskScore:
    task: str
    scenes: int
    healthy_successes: int
    conditions: tuple[ConditionScore, ...]

    @property
    def healthy_score(self) -> float:
        """``H``: unconditional, over every scene of the task."""
        if self.scenes == 0:
            raise ScoringError(f"task {self.task!r} has no scenes")
        return self.healthy_successes / self.scenes

    @property
    def score(self) -> float:
        if len(self.conditions) != len(MISSING_CONDITIONS):
            raise ScoringError(
                f"task {self.task!r} has {len(self.conditions)} missing conditions, "
                f"the ranking needs {len(MISSING_CONDITIONS)}"
            )
        total = self.healthy_score + sum(c.ranked_value for c in self.conditions)
        return total / RANKING_COMPONENTS

    def as_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "scenes": self.scenes,
            "healthy_successes": self.healthy_successes,
            "healthy_score": self.healthy_score,
            "conditions": {
                c.key: {"score": c.score, "successes": c.successes,
                        "written": c.written, "valid": c.valid,
                        "invalid": c.invalid, "owed": c.healthy_successes,
                        "episodes_match_healthy": c.episodes_match_healthy,
                        "missing_episodes": list(c.missing_episodes),
                        "extra_episodes": list(c.extra_episodes),
                        "complete": c.complete}
                for c in self.conditions
            },
            "task_score": self.score,
        }


def condition_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    healthy_successes: int,
    healthy_episodes: Optional[Sequence[int]] = None,
) -> tuple[ConditionScore, ...]:
    """Build the nine condition scores from ``{state, onset_fraction, episode_index,
    success, valid}`` rows.

    Every condition appears, including ones with nothing to evaluate: a missing
    row would silently shorten the task score's denominator.

    ``healthy_episodes`` names the scenes the condition owes. When it is given,
    only rows on those scenes count -- a fault success on a scene the policy
    never solved has no pairing and would push ``M_c`` past one -- and the rows
    are keyed by scene, so a second row for one scene is refused rather than
    counted twice. ``healthy_successes`` must then equal its length. Without it
    (unit tests of the arithmetic alone) rows are counted as given.
    """
    owed: Optional[set[int]] = None
    if healthy_episodes is not None:
        owed = {int(e) for e in healthy_episodes}
        if len(owed) != len(tuple(healthy_episodes)):
            raise ScoringError("healthy_episodes lists a scene twice")
        if len(owed) != int(healthy_successes):
            raise ScoringError(
                f"healthy_successes={healthy_successes} but {len(owed)} healthy "
                "episodes were named; the denominator is the owed scene set"
            )
    tally: dict[tuple[str, float], list[int]] = {
        (state, onset): [0, 0, 0] for state, onset in MISSING_CONDITIONS
    }
    covered: dict[tuple[str, float], set[int]] = {key: set() for key in tally}
    extra: dict[tuple[str, float], set[int]] = {key: set() for key in tally}
    for row in rows:
        key = (str(row["state"]), round(float(row["onset_fraction"]), 2))
        if key not in tally:
            # A partial combination or an unranked onset: executed and reported
            # elsewhere, but not part of the ranking.
            continue
        episode = row.get("episode_index")
        if owed is not None:
            if episode is None:
                raise ScoringError(f"a fault row of {key} carries no episode_index")
            episode = int(episode)
            if episode not in owed:
                extra[key].add(episode)
                continue
            if episode in covered[key]:
                raise ScoringError(
                    f"scene {episode} has two fault cells for {key[0]}@{key[1]:.2f}; "
                    "one cell per owed scene is the protocol"
                )
            covered[key].add(episode)
        elif episode is not None:
            covered[key].add(int(episode))
        successes, written, valid = tally[key]
        written += 1
        # A rollout in which the fault never reached the policy is written but
        # not valid, and cannot contribute a success however it ended.
        if row.get("valid", True):
            valid += 1
            if row["success"]:
                successes += 1
        tally[key] = [successes, written, valid]
    return tuple(
        ConditionScore(state=state, onset_fraction=onset,
                       successes=tally[(state, onset)][0],
                       written=tally[(state, onset)][1],
                       valid=tally[(state, onset)][2],
                       healthy_successes=healthy_successes,
                       missing_episodes=(
                           tuple(sorted(owed - covered[(state, onset)]))
                           if owed is not None else ()),
                       extra_episodes=tuple(sorted(extra[(state, onset)])))
        for state, onset in MISSING_CONDITIONS
    )


@dataclass(frozen=True)
class BenchmarkScore:
    """The MAIL-Bench score: the equal average of the eighteen task scores."""

    tasks: tuple[TaskScore, ...]
    expected_tasks: int = field(default_factory=benchmark_task_count)

    @property
    def official(self) -> bool:
        """Whether this is an official-ranking submission rather than an experiment."""
        return len(self.tasks) == self.expected_tasks

    @property
    def score(self) -> float:
        if not self.tasks:
            raise ScoringError("a benchmark score needs task scores")
        if not self.official:
            raise ScoringError(
                f"{len(self.tasks)} of {self.expected_tasks} tasks were evaluated. "
                "A partial result is reportable as an experiment, but an average "
                "over a subset of tasks is a different quantity and may not carry "
                "the benchmark's name or be claimed as state of the art."
            )
        # Equal weight per task, so a task with more scenes does not weigh more.
        return sum(task.score for task in self.tasks) / len(self.tasks)

    def as_record(self) -> dict[str, Any]:
        return {
            "tasks": [task.as_record() for task in self.tasks],
            "tasks_evaluated": len(self.tasks),
            "tasks_expected": self.expected_tasks,
            "official": self.official,
            "mail_bench_score": self.score if self.official else None,
        }


def one_policy_configuration(checkpoint_identities: Iterable[str]) -> str:
    """The single configuration a submission ran, or a refusal.

    The official ranking evaluates one fixed multi-task policy across all
    eighteen tasks. A submission that switched checkpoints per task is a
    benchmark of eighteen specialists, which measures something other than a
    policy, so the identities are checked rather than assumed.
    """
    identities = {str(identity) for identity in checkpoint_identities}
    if not identities:
        raise ScoringError("no checkpoint identity was recorded")
    if len(identities) > 1:
        raise ScoringError(
            "the official ranking evaluates one fixed policy configuration across "
            f"all tasks, but {len(identities)} were used: {sorted(identities)}. "
            "Per-task checkpoint switching is not an official submission."
        )
    return identities.pop()
