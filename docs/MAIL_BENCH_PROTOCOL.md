# MAIL-Bench — Official Evaluation Protocol

**MAIL-Bench: Benchmarking Robotic Manipulation under Visual Availability Interruptions and Loss**

> `protocol_id: mail_bench_perturbation_v1` · `scene_bank_id: mail_robocasa_atomic_seen_v1`
>
> Machine-readable spec: [`configs/perturbation_protocol.yaml`](../configs/perturbation_protocol.yaml).

---

## 0. Terms

- **Cell**: one rollout and its record: a task, a frozen scene, a camera state, an onset. A healthy cell has no fault; a fault cell is paired with the healthy cell of the same scene.
- **Unit**: one (task, scene) pair; the ten cells of a unit replay the same frozen scene.
- **Cohort**: one policy evaluated on a set of units in one configuration; a run is a cohort executed by one or more workers.
- **Certificate**: the record a run writes beside its cells, naming the protocol, the policy identity, the execution profile and every deviation from the official profile; `leaderboard_eligible` is its flag for "comparable with the official ranking" (there is no hosted leaderboard).
- **Onset manifest**: the per-scene table of a policy's own healthy completion steps and the onsets derived from them.
- **Measurement identity**: the tuple (model, weights digest, kernel configuration, protocol) every cell of a run carries; a score covers exactly one.
- **Official vs experiment**: an official run meets every condition in §6 and may carry the benchmark's name; anything else is an experiment, scored under the neutral name `score`.

---

## 1. What the benchmark measures

MAIL-Bench evaluates how robust an embodied manipulation policy remains when
visual information becomes partially or completely unavailable *during* task
execution. Four questions:

1. **Capability** — can the policy solve the task with every visual stream available?
2. **When** — at what stage of its own successful execution is vision lost?
3. **What / How** — which visual information becomes unavailable, and in what form?
4. **Outcome** — does the policy still finish, and how does completion efficiency change?

MAIL-Bench standardizes the **evaluation**, not the model:

> **Any model design and training method are allowed.**

### Suite

| Suite | Tasks | Frozen scenes per task | Total |
| --- | ---: | ---: | ---: |
| RoboCasa Atomic-Seen | 18 | 50 | **900** |

Task semantics, simulator, initial states, success criteria, per-task horizons,
and the observation and action interfaces are RoboCasa's. MAIL-Bench adds a
standardized visual-availability layer on top of them. The same frozen scene
bank serves the healthy rollout and every fault rollout paired with it.

### Evaluated policy

The official ranking evaluates **one fixed multi-task policy configuration
across all 18 tasks**. A checkpoint, a LoRA or an adapter is all fine; the
evaluated parameters may not change between tasks. Task instructions change
naturally, model parameters do not.

Per-task checkpoint switching is not an official submission — a benchmark of
eighteen specialists measures something other than a policy. Partial task sets
are reportable as experiments, but may not carry the benchmark's name or be
claimed as state of the art.

---

## 2. Healthy reference and fault timing

Each policy first performs **one canonical healthy rollout** on every frozen
scene. Its cell records `success`, the completion step
(`action_execution_count`) and `policy_query_count`; the onset manifest
derived from it carries them as `healthy_success` and
`healthy_completion_step`. A rollout succeeds when RoboCasa's
own success criterion is first satisfied, and terminates at that point or when
the task's horizon is exhausted.

For task *t*:

```
H_t = N(healthy success) / 50
```

### Phase-normalized onset

Fault timing is relative to the evaluated policy's **own** successful healthy
trajectory. With `T_healthy` the healthy completion step:

```
t_target(f) = half_up(f · T_healthy),   f ∈ {0.30, 0.45, 0.60}
```

corresponding to an early, middle and late task phase. Two policies may
therefore be faulted at different absolute steps on the same scene. That is the
estimand: a fixed absolute step is early in the task for a slow policy and late
for a fast one, so comparing on it would measure speed rather than robustness.

For policies that execute several actions between observation queries, both
`target_onset_step` and `realized_onset_step` are recorded; the fault becomes
visible at the first valid observation-query boundary at or after the target.

### When healthy fails

```
healthy_success = false      fault_evaluable = false
healthy_completion_step = null      skip_reason = healthy_unsolved
```

No onset is defined, no fault rollout is run for that scene, and **there is no
horizon-based substitute**. Thirty percent of a completion that never happened
is not thirty percent of anything.

Low healthy performance does not block the benchmark. Scenes the policy did
solve proceed to fault evaluation; only `N(healthy success) = 0` across the
whole cohort stops it, and then because there is nothing to run.

---

## 3. Visual availability states

RoboCasa's three streams group into two functional roles — **wrist vision**
(`robot0_eye_in_hand`) and **agentview vision** (`robot0_agentview_left` +
`robot0_agentview_right`) — giving four states:

| State | Unavailable |
| --- | --- |
| Healthy | none |
| Wrist Missing | wrist camera |
| Agentview Missing | both agentview cameras |
| All Vision Missing | all three |

The two agentviews move together: losing one of two redundant onboard views is
a materially milder condition than losing the role. A single agentview is a
legitimate diagnostic and still executes, but it is not part of the ranking,
because it has no counterpart on a platform whose role is served by one camera.

---

## 4. Main ranking

`fault_mode = hard_missing`; once active at the realized onset the stream stays
unavailable (`availability = false`, `frame = None`) until the rollout ends.
There is no recovery in the ranking.

**Ten equally weighted conditions:** 1 healthy + 3 wrist + 3 agentview + 3 all
vision, at 30 / 45 / 60 % — that is 10 % healthy and 30 % each missing role.

### Condition score

Fault conditions are evaluated only on scenes the same policy solved healthily:

```
M_c,t = N(successful fault rollouts under c) / N(healthy success in t)
```

The denominator is the count of healthy-successful scenes, **not** the count of
fault rollouts that were written: a cell that was owed and never produced lowers
the score rather than disappearing from its own denominator, where the gap would
be invisible. If a task has no healthy success, its nine scores are `N/A` and
contribute zero.

### Task and benchmark score

```
S_t  = (H_t + M_W30 + M_W45 + M_W60 + M_A30 + M_A45 + M_A60
             + M_V30 + M_V45 + M_V60) / 10

MAIL = (1/18) Σ S_t
```

Each task carries equal weight, so a task with more scenes does not weigh more.

### Completion efficiency

Secondary; it does not affect the ranking. Where both the healthy and the
corresponding fault rollout succeed:

```
ΔT = T_fault − T_healthy       S_slow = (T_fault − T_healthy) / T_healthy
```

Failed rollouts have no completion time and are never assigned the horizon as a
substitute.

---

## 5. Mechanism analyses

Optional, and none affects the MAIL-Bench score. All use the same three visual
roles.

**Failure Form** — how the stream becomes unreliable, at a 60 % onset:
`hard_missing`, `blackout` (a black frame, still marked available), `freeze`
(the last valid pre-fault frame repeated), `stale_k` (live but delayed by
**500 ms**), `burst_dropout` (**500 ms** blocks; the first is always missing,
each later one drawn deterministically 50/50 from the fault seed). Durations are
in milliseconds rather than control steps so that one setting means the same
thing at any control rate; RoboCasa's 20 Hz makes 500 ms ten steps. Hard Missing
at 60 % already exists in the ranking and is referenced, not rerun.

**Missing Duration** — how long the loss persists, at a 45 % onset, for 0.25,
0.50 and 1.00 of the remaining episode. *Remaining episode* is
`T_healthy − onset_step` (the end step clamped to `[onset_step + 1, horizon]`), what is left of the policy's own successful
trajectory, not what is left of the horizon: the onset is a phase of that
trajectory, and a duration measured against anything else makes one fraction a
different condition for every policy, and one measured against the horizon's
remainder would interrupt a policy finishing in a quarter of the horizon for
longer, relative to its own task, than a slow one. The full-duration cell is the ranking condition at 45 % and is not rerun.

**Visual Recovery** — whether the policy continues after vision returns. Both
ends are fixed: lost at 30 % of `T_healthy`, returned at 60 %, so the
interruption is 30 % of the task. Both ends being phases of one trajectory is
why this is not a duration: no fixed fraction of anything else expresses it, so
`freeze_fault_manifest` takes `recovery_fraction` and resolves the return with
the same arithmetic as the onset. Sweeping the recovery point would answer how
long an interruption can be, which is what Missing Duration studies; how a
policy resets memory or invalidates an action chunk when vision returns is model
design and is measured, not imposed.

The cohort script runs all three as declared experiments (never `--official`;
the healthy phase and the onset manifest are shared with the ranking run):

```bash
python scripts/run_robocasa_cohort.py --scene-bank <bank> --phase fault --onsets 0.60 --fault-mode blackout --output-dir <out>/blackout ...
python scripts/run_robocasa_cohort.py --scene-bank <bank> --phase fault --onsets 0.45 --duration-fraction 0.25 --output-dir <out>/dur025 ...
python scripts/run_robocasa_cohort.py --scene-bank <bank> --phase fault --onsets 0.30 --recovery-fraction 0.60 --output-dir <out>/recovery ...
```

Durations are given in milliseconds (`--stale-ms`, `--burst-ms`) and resolved
at the platform's control rate; `scripts/report_mail_bench_score.py
--allow-experiment` prints each record under the neutral name `score`.

---

## 6. Validity and pipeline

**Pairing.** Healthy and fault rollouts share task, frozen scene, initial
simulator state, instruction, policy configuration and policy-initialization
semantics. The visual availability intervention is the intended difference.
The policy seed a pair shares must drive every source of randomness in the
policy, its own action sampling included: a flow or diffusion head re-keyed
from the reset seed draws the same noise in the healthy rollout and in every
fault arm, so the two trajectories coincide until onset and any divergence
after it is the fault's. A policy whose sampling ignores the seed diverges
before onset, and the report shows that as a non-zero pre-fault divergence
rate; a fault cell that fails before the fault reaches the policy is invalid
and never counts as a success, whatever its cause.

Bit-identical pairing is a property of the whole stack, and the platform's
renderer does not fully deliver it: on the hardware the reference results were
produced on, physics and two of three cameras are identical across processes,
but the wrist camera differs by rasterisation noise in a handful of pixels per
frame (`scripts/probe_render_determinism.py`), which a continuous policy can
amplify into a diverging trajectory. The seeding rules remove every source of
divergence the benchmark controls; the residual is measured and reported as the
pre-fault divergence rate, not hidden.

**Episode independence.** Every rollout starts from the frozen state and a fresh
episode-level policy state; one evaluation episode may not influence another.

**Causal boundary.** While a stream is unavailable, its hidden healthy frames
must not enter the policy's accessible state — not as input, and not through
memory, router, gate or imputer updates. Information genuinely observed before
onset may remain. For `stale` the delayed frames are valid observations; for
`freeze` only the last valid pre-fault frame may be repeated.

> The policy may only use information available under the evaluated condition.

**Healthy completeness.** Before the onset manifest is derived:
`expected = unique = 900`, `missing = 0`, `duplicate = 0`. An incomplete healthy
phase does not proceed.

**Single source of truth.** Standard result cells, and nothing else:

```
900 healthy cells → integrity check → healthy report → onset manifest
                  → fault evaluation → condition scores → 18 task scores → MAIL
```

**Scene identity.** A scene's seed derives from `scene_bank_id`, namespace,
platform, task and episode index — and deliberately not from the protocol
version. Changing a scoring rule or a name must not redraw 900 frozen scenes.
The bank is frozen under a single `root_manifest_hash`; a run states which bank
it evaluated on, and any later divergence shows up as a different hash rather
than as an unexplained change in a score. The three run-time random streams
(environment, policy, fault) do include the protocol version.

**What makes a run official.** Each condition is checked by the tooling, never
declared:

| Condition | Checked by |
|---|---|
| The bank is the published 900 scenes, scene for scene | `--official` recomputes every scene's identity record and hashes its payload files against `configs/scene_bank_identity.json` before any rollout |
| The suite is the frozen 18 tasks with the platform's horizons | `--official` compares the frozen suite against the installed RoboCasa |
| The weights are pinned, not reported by the server | `--official` requires `--model-revision` and `--inventory-sha256` (or `--checkpoint-sha256` for a single file), and the server re-verifies the checkpoint before and after loading |
| One fixed policy configuration across all 18 tasks | the scorer refuses more than one measurement identity under a run root |
| Every scene has one healthy cell and every solved scene its nine fault cells | the scorer counts them and names what is missing |
| The run's own certificates declare no deviation | the scorer refuses a score if any certificate is not `leaderboard_eligible` |

---

## 7. Reporting

Per task: healthy score; wrist, agentview and all-vision missing at each of
30 / 45 / 60 %; the task score. Benchmark level: all 18 task scores, the
MAIL-Bench score, and completion-efficiency statistics. The fixed policy
configuration used across all 18 tasks must be identified well enough to
determine exactly which model was evaluated.

---

## 8. Version

MAIL-Bench is 18 RoboCasa Atomic-Seen tasks, 50 frozen scenes each, one
shared multi-task policy configuration, and ten equally weighted ranking
conditions. Once released, the task set, scene bank, scoring rule, fault
semantics and timing definitions are immutable. Later releases may add suites,
embodiments or robustness dimensions; they may not change what the benchmark means.
A released protocol version is never modified in place. The wire protocol name
`mcr-policy-v1`, the seed namespaces `mcr.env`, `mcr.policy`, `mcr.fault` and the
hash prefixes `mcr_bench.prefix.v1` and `mcr_bench.burst` are frozen identifiers:
changing them would change every seed and digest of every run.

---

## 9. What a delivered observation says about itself

Every camera arrives with metadata describing the frame beside it, and the
metadata describes *that frame*, not the step it was delivered at. A policy
that wants to know whether it is looking at something new reads these rather
than guessing from pixels.

| Mode | `available` | `source_age_steps` | `new_frame` |
|---|---|---|---|
| healthy | true | 0 | true |
| `hard_missing` | false | -- | false |
| `burst_dropout` | alternates | 0 while available | true while available |
| `stale_k` | true | k, constant | false |
| `freeze` | true | grows without bound | false |
| `blackout` | true | 0 | true |

Two consequences worth stating, because they decide what a method can act on.

Absence is not the only self-announcing failure. A frozen stream reports a
frame whose age grows every step, and a delayed one reports a constant lag, so
both are detectable from the metadata alone: a policy does not have to judge
whether an image looks wrong. `sequence_id` and `capture_time_ms` are reported
as unknown for such a frame rather than as the current step's, because the
injector keeps frames by step and not their capture metadata, and reporting the
current step's would be false. `arrival_time_ms` stays the delivery time, which
is what separates it from capture time.

`blackout` is the one mode a policy cannot detect from metadata. It is online,
it delivers a frame captured now, and every field matches a healthy camera's;
only the pixels are wrong. Recognising it requires judging content, which is a
different capability from consuming availability, and the benchmark does not
require it. A method that handles availability should be expected to leave
`blackout` untouched, and reporting that is a statement about the method's
scope rather than a result.

## 10. Scope of the implementation

- **The three mechanism analyses are driven by the submitter.** The operators,
  the duration rule and the recovery rule are implemented and tested, and the
  cohort script runs each study (§5); no command tabulates them into one table,
  because each is a separate study with its own reading. A run that omits them
  is complete.
- **The published reference results cover the main ranking only**, every cell
  `hard_missing` to the end of the episode; the mechanism operators are
  verified by unit tests against the protocol.
- **The cohort specification carries one horizon for the whole suite**, the
  maximum over the frozen suite's per-task horizons, so a task-subset
  experiment hashes exactly as the full run does; each adapter still stops at
  its own task's horizon, and a changed horizon is refused before an official
  run starts.
- **The reference servers are held to their rules by source-level tests.**
  Seeding on reset, the XLA flags, the digest before and after the load, and
  the absent-camera substitution are asserted by reading the scripts;
  `scripts/check_submission.py` exercises a running server.
- **The published onset manifest is not what the fault phase reads.** The
  driver computes every onset from the healthy cells; the scorer recomputes
  it from the paired healthy cell and refuses an official score if any
  executed onset differs. The manifest is therefore checked against execution
  at scoring time, not consumed by it.

## 11. What the scorer checks that cells alone cannot show

`mail_bench_report` reads more than the cells. It reads the run's own
certificates and is official only if every fault-phase certificate is
leaderboard-eligible and every healthy-phase certificate complete, since a
custom-profile run writes cells of exactly the same shape. It pairs each fault
cell's pre-onset action digest with the healthy cell's prefix chain *at that
onset* -- a healthy cell has no onset of its own and carries no pre-fault
digest -- and reports the divergence rate as provenance. It counts a
condition's successes over the owed scenes only, names any scene missing or
extra, and refuses a second cell for one scene. And it reports the protocol's
secondary reading, completion efficiency, as the median over scenes of the
paired per-scene slowdown where both rollouts succeeded, never over failures.
