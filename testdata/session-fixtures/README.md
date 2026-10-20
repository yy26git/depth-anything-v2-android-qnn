# Session fixtures — all synthetic

Every file here is **synthetic**. No device session data exists yet, so these were
hand-generated to exercise `scripts/evaluate-depth-session.py`. They carry
`"synthetic": true` and a `provenance` string **inside `session.json`**, which makes
the checker's `EVIDENCE` requirement fail on all of them. That is deliberate: a
fixture can prove a *check* fires, and can never stand in for a device record.

Manifest schema and the exact semantics of every field are documented in the module
docstring of `scripts/evaluate-depth-session.py` (`--help` prints it). That schema is
a **proposed contract** for a live-capture stage that does not exist yet, and must be
reconciled with that stage's real output before any manifest it produces is trusted.

| Fixture | Shape | What it must trigger |
|---|---|---|
| `pass-fast-qnn/` | 12.5 Hz, 3 steady minutes, verified clock mapping, per-result QNN placement evidence | `THROUGHPUT`/`FRESHNESS`/`INTEGRITY`/`BACKEND` **PASS** — the positive control, without which the rest prove nothing |
| `slow-session/` | 12.5 Hz except one steady minute at 5.0 Hz; whole-session average still ~11 Hz | `THROUGHPUT` **FAIL** — intervals are never averaged |
| `missing-intervals/` | declared 360 s, minutes 3–4 have no interval at all | `THROUGHPUT` + `INTEGRITY` **FAIL** — a missing minute is not inferred |
| `truncated-session/` | declared 660 s, bucketed 150 s, last interval flagged `complete: false` | `INTEGRITY` + `THROUGHPUT` **FAIL** — a short run does not pass for being fast |
| `mislabelled-backend/` | configured `qnn_htp`, every result labelled `cpu` | `BACKEND` **FAIL** — automatic, never a warning |
| `cross-clock-trap/` | capture on `boottime`, publication on `uptimeMillis`, no mapping, producer age declares `clockPath: "assumed-offset"` | `FRESHNESS` **BLOCKED** — age is UNKNOWN, never a number |
| `presented-stale-results/` | 12.5 Hz session that also presents `stale` and `outputValid: false` results | `INTEGRITY` **FAIL**; the bad frames stop counting as completed work |
| `unproven-accelerator-placement/` | fast, labelled `qnn_htp`, only evidence is `QNNExecutionProvider` registered | `BACKEND` **FAIL** while `THROUGHPUT` **PASSES** — registration is not placement, and speed is not placement |

Run them all:

```
python3 scripts/evaluate-depth-session.py --self-test
python3 scripts/evaluate-depth-session.py --manifest testdata/session-fixtures/slow-session
```

Clock note: the fixtures put `boottime` exactly 3 600 000 ms ahead of `uptimeMillis`
so that the cross-clock trap produces a large *negative* age under naive subtraction.
A real device's offset differs; only the mapping's admissibility is being exercised.
