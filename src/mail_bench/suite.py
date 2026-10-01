"""The frozen MAIL-Bench evaluation suite.

The task identity of a benchmark cannot be "whatever the installed RoboCasa
currently calls Atomic-Seen". The protocol says the task set is immutable,
and reading it from ``TARGET_TASKS`` at run time would make it immutable only
until somebody upgrades a package. The eighteen names, the fifty scenes and the
per-task horizons live in ``configs/mail_bench_suite.yaml``; this module
reads them and, when RoboCasa is present, refuses to proceed if the two
disagree.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "mail_bench_suite.yaml"


class SuiteMismatch(RuntimeError):
    """The installed platform does not offer the suite this benchmark froze."""


@lru_cache(maxsize=4)
def _load(path: str) -> Mapping[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def suite(config: Optional[Path] = None) -> Mapping[str, Any]:
    return _load(str(config or CONFIG))


def task_ids(config: Optional[Path] = None) -> tuple[str, ...]:
    return tuple(entry["id"] for entry in suite(config)["tasks"])


def horizons(config: Optional[Path] = None) -> dict[str, int]:
    return {entry["id"]: int(entry["horizon"]) for entry in suite(config)["tasks"]}


def scenes_per_task(config: Optional[Path] = None) -> int:
    return int(suite(config)["scenes_per_task"])


def verify_against_platform(
    available_tasks: Any,
    available_horizon: Any,
    config: Optional[Path] = None,
) -> None:
    """Refuse to run if the installed platform is not the frozen suite.

    A RoboCasa that renamed a task, dropped one, or changed a horizon would
    otherwise be evaluated silently: the run would produce eighteen task scores
    that are not this benchmark's eighteen, and nothing in the output would say
    so.
    """
    frozen = horizons(config)
    theirs = {str(task) for task in available_tasks}
    missing = sorted(set(frozen) - theirs)
    extra = sorted(theirs - set(frozen))
    if missing or extra:
        raise SuiteMismatch(
            f"the installed platform does not offer the frozen suite: "
            f"missing {missing}, unexpected {extra}"
        )
    differing = {
        task: (expected, int(available_horizon(task)))
        for task, expected in frozen.items()
        if int(available_horizon(task)) != expected
    }
    if differing:
        raise SuiteMismatch(
            f"horizons differ from the frozen suite: {differing}. A horizon "
            "decides when a rollout ends, so a changed one changes every score "
            "without changing a line of this repository."
        )
