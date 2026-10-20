# Live-session fixtures — all synthetic

Every file here is **synthetic**. None of it was captured from the OnePlus 9: no
live logcat has been run through this checker yet. The numbers are internally
consistent but invented, and every file says so in its own header.

That is enforced, not merely requested. `scripts/evaluate-live-session.py` fails its
`EVIDENCE` requirement on any capture containing a synthetic marker, so **no file in
this directory can ever exit 0** — which is what makes them negative controls rather
than results. Removing a marker is the only way to make one pass, and the self-test
(`test_cli_exit_codes_and_no_synthetic_pass`) catches that.

The log format each file follows is the real one emitted by `LiveDepthPipeline.kt` and
`MainActivity.kt` under the `DepthQNN` tag, including the logcat line prefix. That
format is documented in the module docstring of `scripts/evaluate-live-session.py`
(`--help` prints it).

| Fixture | Shape | What it must trigger |
|---|---|---|
| `pass-cpu-live/` | ~0.58 depth updates/s over 102.8 s, 97% of offered frames dropped as busy, camera `Streaming`, every reported `fps` matching `inferences/elapsedS` | every measurement requirement **PASSES** — the positive control, without which the rest prove nothing. `EVIDENCE` **FAILs** on the synthetic marker, so `OVERALL: FAIL`, exit 1 |
| `no-rate-lines/` | camera streams, model loads, then nothing: zero `depth rate` lines | `EVIDENCE` **FAIL** — and all seven rate-derived requirements **FAIL** rather than passing vacuously |
| `frozen-counter/` | `inferences` stuck at 15 while `elapsedS` and `offered` keep climbing (a capture padded with copied lines) | `PROGRESS` **FAIL** *and nothing else* — `UPDATES` passes on purpose, because the count of 15 clears the floor. That the two disagree is the proof they are judged independently |
| `no-drops-queueing/` | `droppedBusy=0` on every line at ~0.58 Hz | `DROPS` **FAIL** *and nothing else* — frames would be queueing unboundedly, the stale mask in a new shape |
| `too-small-frames/` | `tooSmall` climbs to 137 | `TOO_SMALL` **FAIL** *and nothing else* |
| `inference-errors/` | `errors` climbs to 6 | `ERRORS` **FAIL** *and nothing else* |
| `fps-mismatch/` | honest lines, then a final line claiming `fps=24.00` when its own counts say 0.58/s | `SELF_CONSISTENCY` **FAIL** *and nothing else* |
| `rate-burst-no-elapsed/` | a healthy-looking `12.50/s` printed by six lines inside ~2 s of wall clock, backed by 4.8 s elapsed | `SPAN` **FAIL** *and nothing else*. This is the fixture that matters most: every other check is fooled by it, so a checker without `SPAN` reports a fabricated burst as a fast healthy session |
| `camera-never-streamed/` | healthy depth pipeline, camera goes `Starting` → `Idle` → `Failed(NO_BACK_CAMERA)` | `CAMERA` **FAIL** *and nothing else* |
| `camera-status-absent/` | healthy depth lines, **no** `camera status=` line at all (capture filtered too tightly) | `CAMERA` **BLOCKED** and no other requirement but `EVIDENCE` fails. Deliberately *not* the same as a camera that failed: the capture says nothing about the camera, so the verdict is UNKNOWN rather than bad. As a committed file it therefore exits **1**, because the synthetic `EVIDENCE` failure outranks the block; strip the `#` header and it exits **2**, which is what a real capture missing that line would do |

Run them all:

```
python3 scripts/evaluate-live-session.py --self-test
python3 scripts/evaluate-live-session.py --log testdata/live-session-fixtures/pass-cpu-live/live-session.logcat
python3 scripts/evaluate-live-session.py --log testdata/live-session-fixtures/rate-burst-no-elapsed/live-session.logcat
```

`test_each_negative_fixture_fails_exactly_its_own_check` enforces the isolation the
table claims, so a fixture cannot quietly start failing for a second reason and stop
proving what it was written to prove.
