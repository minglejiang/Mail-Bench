# The frozen scene bank

The 900 starting scenes MAIL-Bench evaluates: 18 tasks x 50 scenes, frozen
once as bytes so that every rollout of every policy, on any machine, begins from
the same state. This is what makes a healthy rollout and its nine fault arms the
same experiment.

```bash
zstd -d -c mail_bench_scene_bank_official.tar.zst | tar -x -C <where you keep data>
sha256sum -c SHA256SUMS          # before extracting, if you prefer
```

That gives you `<where you keep data>/official/robocasa365/<task>/<0000..0049>/`,
which is the `--scene-bank` argument. An `--official` run verifies it against
`configs/scene_bank_identity.json` scene by scene before the first rollout, so
a download that lost or altered anything is refused rather than scored.

## What a scene is

| File | What it holds |
| --- | --- |
| `ep_meta.json` | RoboCasa's recipe: kitchen layout and style, fixtures, object placements, the robot's initial base pose, camera configuration, the instruction |
| `model.xml` | the MuJoCo scene that recipe generates |
| `state0.npy` | the simulator state **after the scene has settled**; freezing before objects come to rest would give a starting point physics does not reproduce |
| `episode.json` | the identity record: a SHA256 of each of the three, the scene seed, RoboCasa's commit, the asset digest, and the scene identity derived from them |

Rebuilding a bank locally is possible (`scripts/build_robocasa_scene_bank.py`),
but it is not the way to obtain *this* bank: the
settled state depends on the simulator's version, its assets and its sampler, so
a rebuild is a different bank unless every one of those matches. Download this
one and the identity check passes by construction.

## Provenance

Built with RoboCasa at `a07e365c958c4216cd6bbd5f30b47f09a65c6f00`, rendered
under EGL. The bank's root manifest hash is in
`configs/scene_bank_identity.json` together with all 900 scene identities;
the archive is reproducible (sorted entries, fixed ownership and timestamps).

RoboCasa is MIT licensed. This archive carries scene descriptions and simulator
states, not RoboCasa's mesh and texture assets, which you install with RoboCasa
itself.
