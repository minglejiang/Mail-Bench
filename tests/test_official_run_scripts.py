"""The scripts that produced the official data are part of the frozen identity.

A scene bank that only exists because of a script in somebody's home directory
cannot be reproduced from a tag, however carefully the tag pins everything else.
These do not execute the scripts -- that needs RoboCasa, MuJoCo and a GPU -- but
they check the constants a reader would otherwise have to take on trust, and
that each script earns its completion marker rather than printing it on the way
past.
"""

import ast
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
BANK = SCRIPTS / "build_robocasa_scene_bank.py"
ONSET = SCRIPTS / "build_onset_manifest.py"


def constants(path):
    """Top-level literal assignments, without importing (these need RoboCasa)."""
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        targets = node.targets[0]
        names = targets.elts if isinstance(targets, ast.Tuple) else [targets]
        values = value if isinstance(targets, ast.Tuple) else [value]
        for name, item in zip(names, values):
            if isinstance(name, ast.Name):
                found[name.id] = item
    return found


def test_the_bank_is_built_at_the_official_scale():
    from mail_bench.manifest import PROTOCOL_VERSION

    found = constants(BANK)
    # 18 Atomic-Seen tasks x 50 episodes, at RoboCasa's own native scale.
    assert found["EPISODES"] == 50
    assert found["SPLIT"] == "target"
    assert found["FRESH_EVERY"] == 20
    assert len(found["CAMERAS"]) == 3
    source = BANK.read_text(encoding="utf-8")
    assert 'namespace="official"' in source
    # The script records the protocol it was written for and refuses any
    # other; the protocol does not enter a scene seed.
    assert found["EXPECTED_PROTOCOL"] == PROTOCOL_VERSION
    assert "assert PROTOCOL_VERSION == EXPECTED_PROTOCOL" in source


def test_every_official_script_earns_its_completion_marker():
    # A marker printed on the way past the last line says a run succeeded when
    # all it did was reach the end.
    source = BANK.read_text(encoding="utf-8")
    marker = "OFFICIAL_BANK_DONE"
    assert marker in source
    printed = [line for line in source.splitlines() if f'print("{marker}")' in line]
    assert printed
    # Indented under a conditional, never at column zero.
    assert all(line.startswith("    ") for line in printed)
    assert "sys.exit(1)" in source


def test_the_frozen_suite_reaches_the_execution_runners():
    """A rule the scorer enforces and the runner ignores is not an invariant.

    Both scripts that produce official data call verify_against_platform, and
    an official cohort takes its task list from the frozen suite.
    """
    cohort = (SCRIPTS / "run_robocasa_cohort.py").read_text(encoding="utf-8")
    bank = BANK.read_text(encoding="utf-8")
    for source in (cohort, bank):
        assert "verify_against_platform(" in source
        assert "from mail_bench.suite import" in source
    # Official runs are pinned and checked; a custom experiment is not making
    # the claim, so it keeps the platform's own list.
    # An official run is the suite by construction: a task subset is refused
    # rather than recorded, and the profile's tasks and scene count come from
    # the frozen suite, not from the installed platform or a literal.
    assert 'raise SystemExit("--official evaluates the frozen suite' in cohort
    assert "tasks=tuple(task_ids())," in cohort
    assert "episode_indices=tuple(range(scenes_per_task()))," in cohort
    # ... on weights that are pinned, not reported by the server.
    assert "--official requires --model-revision and --inventory-sha256" in cohort
    # ... with the protocol's replicate count, so a CLI override is a deviation.
    assert 'healthy_replicates=int(healthy_arm.get("official_replicates_per_seed")' in cohort
    # ... and a healthy-phase certificate that carries the run's identity.
    assert "runner.healthy_certificate(my_units, outcomes)" in cohort
    assert "tasks = task_ids()" in cohort
    assert 'verify_against_platform(TARGET_TASKS["atomic_seen"], get_task_horizon)' in cohort
    assert "TASKS = list(task_ids())" in bank


def test_the_onset_manifest_is_built_per_model_from_its_own_healthy_run():
    source = ONSET.read_text(encoding="utf-8")
    assert "onset_manifest_row" in source
    assert "--checkpoint-identity" in source
    # The check reads the run certificate, because the cells carry no
    # protocol_version of their own.
    assert "the run declares protocol" in source
    assert 'result.get("protocol_version"' not in source
    # And the checkpoint identity is derived from the cells rather than accepted
    # from the command line, so a published manifest cannot name weights the
    # cells never saw.
    assert "were produced by" in source
    # And the published identity is the cells', not the abbreviation typed on
    # the command line.
    assert "checkpoint_identity = observed" in source
    # A certificate carrying no protocol at all is not one carrying the right
    # one, so an empty declaration is refused rather than passed over.
    assert "if declared != {PROTOCOL_VERSION}:" in source
    # H is reported beside the onsets.
    assert '"clean_success"' in source


def test_the_onset_manifest_derives_from_the_standard_cells():
    """One account of the healthy phase, not two.

    A manifest built from a separate record of the same run could disagree with
    the tables about which scenes were solved and how long they took, with no
    way to tell which was right. It reads the cells, with the aggregator's own
    reader.
    """
    source = ONSET.read_text(encoding="utf-8")
    assert "from mail_bench.aggregate import load_cells" in source
    assert "--run-root" in source
    # And it refuses to derive onsets from a healthy phase that is not complete:
    # a missing scene would silently shrink H's denominator, and a duplicated one
    # would make that scene's onset ambiguous.
    assert "may not be built" in source
    assert "more than one healthy cell" in source


def test_the_healthy_report_checks_completeness_before_capability():
    source = (SCRIPTS / "report_healthy_phase.py").read_text(encoding="utf-8")
    for number in ("N_expected", "N_unique", "N_missing", "N_duplicate"):
        assert number in source
    # Every task is reported; none is filtered out for a low score, because a
    # task the policy cannot do is a result about the policy and dropping it
    # would make H a different quantity.
    assert "load_cells" in source
    # The marker is earned: the incomplete branch exits non-zero above it, so a
    # run with a missing or duplicated scene cannot announce that it is done.
    lines = source.splitlines()
    marker = next(i for i, line in enumerate(lines) if "HEALTHY_PHASE_COMPLETE" in line)
    guard = next(i for i, line in enumerate(lines) if "raise SystemExit(1)" in line)
    assert guard < marker
    assert "INCOMPLETE" in source


def test_the_scene_bank_is_frozen_under_one_identity():
    """One hash stands for the canonical starting scenes.

    Without it a run can only say which directory it read, and a scene
    regenerated or an episode quietly added would surface as an unexplained
    change in a score rather than as a different bank.
    """
    source = (SCRIPTS / "freeze_scene_bank.py").read_text(encoding="utf-8")
    assert "root_manifest_hash" in source
    # Integrity gates the freeze: a bank with 899 of 900 scenes would otherwise
    # shrink every denominator downstream with nothing to notice it.
    for number in ("expected", "unique", "missing", "duplicate"):
        assert number in source
    lines = source.splitlines()
    marker = next(i for i, line in enumerate(lines) if "SCENE_BANK_FROZEN" in line)
    guard = next(i for i, line in enumerate(lines) if "raise SystemExit(1)" in line)
    assert guard < marker
    # And every scene must agree on which bank it belongs to.
    assert "set(bank_ids) == {SCENE_BANK_ID}" in source
    assert "set(schemas) == {SCENE_SCHEMA_VERSION}" in source


def test_every_reference_server_rekeys_its_own_sampling_on_reset():
    """Seeding random and numpy is not seeding the model. The openpi flow head
    draws its noise from Policy._rng, a JAX key, and GR00T's from torch; a
    reset that leaves those alone lets every healthy/fault pair diverge before
    onset."""
    source = (SCRIPTS / "pi05_robocasa_policy_server.py").read_text(encoding="utf-8")
    # The benchmark's seeds are 64-bit; jax.random.key takes a C long, so
    # the seed is reduced to 32 bits.
    assert "policy._rng = jax.random.key(seed % (1 << 32))" in source
    assert 'hasattr(policy, "_rng")' in source
    assert '"policy_sampling_seeded": True' in source
    groot = (SCRIPTS / "groot15_robocasa_policy_server.py").read_text(encoding="utf-8")
    assert "torch.manual_seed(seed)" in groot and "torch.cuda.manual_seed_all(seed)" in groot
    assert '"policy_sampling_seeded": True' in groot


def test_the_groot15_server_reads_its_interface_off_its_checkpoint():
    """The server takes every interface fact from the checkpoint it loads
    (experiment_cfg/metadata.json and the fork's data config) and refuses a
    checkpoint that says otherwise; it has a port of its own and seeds torch on reset."""
    groot = (SCRIPTS / "groot15_robocasa_policy_server.py").read_text(encoding="utf-8")
    assert 'metadata = json.loads((checkpoint / "experiment_cfg" / "metadata.json").read_text())' in groot
    assert 'DATA_CONFIG_MAP[DATA_CONFIG]' in groot and 'EMBODIMENT = "new_embodiment"' in groot
    assert "predicted_chunk = len(data_config.action_indices)" in groot
    assert "torch.manual_seed(seed)" in groot and "default=7614" in groot
    cohort = (SCRIPTS / "run_robocasa_cohort.py").read_text(encoding="utf-8")
    assert '"groot_n1_5_robocasa": 7614' in cohort



def test_the_mechanism_analyses_have_a_command_line_and_are_never_official():
    """Failure Form, Missing Duration and Visual Recovery are run by the same
    cohort script as declared experiments: the mode, the end of the loss and the
    protocol's millisecond settings are arguments, resolved at the platform's
    control rate, and an --official run refuses every one of them."""
    cohort = (SCRIPTS / "run_robocasa_cohort.py").read_text(encoding="utf-8")
    for flag in ("--fault-mode", "--duration-fraction", "--recovery-fraction", "--stale-ms", "--burst-ms", "--burst-available-rate"):
        assert f'"{flag}"' in cohort
    assert 'choices=("hard_missing", "blackout", "freeze", "stale_k", "burst_dropout")' in cohort
    assert "control_steps_for(args.stale_ms, control_hz)" in cohort
    assert "the mechanism analyses run as declared experiments without --official" in cohort
    assert "fault_duration_fraction=args.duration_fraction" in cohort
    driver = (ROOT / "src" / "mail_bench" / "driver.py").read_text(encoding="utf-8")
    assert "recovery_fraction=self.fault_recovery_fraction" in driver
