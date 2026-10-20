#!/usr/bin/env python3
"""
Acceptance checker for a recorded Depth Anything QNN harness session.

    python3 scripts/evaluate-depth-session.py --manifest testdata/session-fixtures/pass-fast-qnn
    python3 scripts/evaluate-depth-session.py --manifest session.json --json
    python3 scripts/evaluate-depth-session.py --self-test          # negative controls

Exit codes: 0 = every requirement passed, 1 = at least one failed, 2 = no verdict
possible (something blocked), 3 = the manifest could not be read.

It judges five requirements **independently**. There is deliberately no blended
score: a session that is fast but lies about its backend is not "mostly fine".

    EVIDENCE    the record is a real device session, not a synthetic one
    THROUGHPUT  >= 10 completed depth updates/s in EVERY steady-state 1-minute interval
    FRESHNESS   p95 result age <= 200 ms -- a PROPOSED HARNESS BUDGET, not a flight requirement
    INTEGRITY   nothing stale/invalid was presented, no truncated interval was silently
                accepted, the run covers its declared duration
    BACKEND     every result's recorded backend matches what was configured, and
                accelerator placement is evidenced rather than inferred

--------------------------------------------------------------------------------
MANIFEST SCHEMA (schemaVersion 1)
--------------------------------------------------------------------------------
A manifest is one JSON object. `--manifest` accepts the file or its directory
(`session.json` inside). This is a *proposed* contract: the live-capture stage
does not exist yet, so it must be reconciled against that stage's real output
before any manifest it produces is trusted.

{
  "schemaVersion": 1,                      // must be exactly 1; anything else is exit 3

  "session": {
    "id": "string",
    "synthetic": false,                    // true => EVIDENCE fails. Never set this for real data.
    "provenance": "free text: how this record was produced",
    "device": {"model": "OnePlus 9", "androidApi": 33, "soc": "Snapdragon 888"},
    "model": {"name": "string", "sha256": "hex"},
    "declaredDurationMs": 660000,          // how long the run was *declared* to be
    "warmupMs": 60000,                     // leading window excluded from every steady-state check
    "configuredBackend": "qnn_htp",        // what was actually configured, not what was hoped for
    "sessionClock": "uptimeMillis"         // the clock intervals/results are bucketed in
  },

  // OPTIONAL. Only with this may the checker convert between two clocks.
  // Without it, capture-to-publication latency is UNKNOWN -- never a number.
  "clockMapping": {
    "verified": true,                      // set only by a measurement, not by intent
    "from": "boottime", "to": "uptimeMillis",
    "offsetMs": -3600000.0,                // to_value = from_value + offsetMs (+ drift)
    "driftPpm": 4.0,
    "maxResidualMs": 1.2,
    "method": "free text: how offset and residual were obtained"
  },

  // Contiguous 1-minute buckets on sessionClock. Warmup buckets may be present;
  // they are excluded. A gap here is a FAILURE, not an average-away.
  "intervals": [
    {"index": 0, "startMs": 0, "endMs": 60000, "complete": true}
  ],

  "results": [                             // ordered by 'index'
    {
      "index": 0,
      "status": "complete",                // complete|dropped|failed|stale|invalid|truncated
      "outputValid": true,                 // false => constant/non-finite map; never presented
      "presented": true,                   // defaults to (status=="complete" && outputValid)

      "capture":    {"timeMs": 36000000.0, "clock": "boottime"},      // may be null
      "processing": {"timeMs": 36000020.0, "clock": "uptimeMillis"},  // may be null
      "published":  {"timeMs": 36000021.0, "clock": "uptimeMillis"},  // "completed" time

      // Producer-computed age. Honoured ONLY when clockPath declares HOW it was
      // obtained; "assumed-offset" and friends are rejected, because believing
      // them is exactly the cross-clock subtraction this checker refuses to do.
      "age": {"valueMs": 21.0, "clockPath": "verified-mapping"},

      "backend": {
        "label": "qnn_htp",                // must equal session.configuredBackend
        "placementProven": true,           // accelerator labels need this; registration is not proof
        "evidenceKind": "qnn-profile-node-assignment",
        "evidence": "htp_nodes=214/214 cpu_nodes=0",
        "providersRegistered": ["CPUExecutionProvider", "QNNExecutionProvider"]
      },

      "droppedSincePrevious": 0,           // telemetry, echoed
      "memory": {"javaHeapBytes": 81234567, "nativeHeapBytes": 120334455},
      "thermal": {"skinTempC": 41.5, "throttling": false}
    }
  ],

  // Free-form producer summary. ECHOED, labelled untrusted, never scored.
  "producerSummary": {"claimedUpdatesPerSecond": 12.4, "claimedP95AgeMs": 26.0}
}

Three things this checker will not do, each because getting it wrong would make
the evidence look better than it is:

  * It will not subtract a capture-clock timestamp from a processing-clock one
    without a verified clock mapping. Cross-clock ages are reported UNKNOWN.
  * It will not accept an accelerator backend because a provider registered, or
    because inference was fast. Placement must be evidenced per result.
  * It will never emit or accept a metric depth value. Output is relative,
    disparity-like, and has no metres.

--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "testdata/session-fixtures"

SCHEMA_VERSION = 1

# --- Acceptance thresholds for the v0.1 live depth gate.
TARGET_UPDATES_PER_SECOND = 10.0
FRESHNESS_BUDGET_P95_MS = 200.0
INTERVAL_MS = 60_000
# The floor is deliberately below the plan's ten-minute session: three consecutive
# steady-state minutes is the least that can show a slow minute is not averaged away.
MIN_STEADY_STATE_MS = 180_000

# --- Tolerances. Small on purpose: these compare declared numbers, not measured time.
GAP_TOLERANCE_MS = 1.0      # interval boundaries are floats; 1 ms absorbs rounding
TAIL_TOLERANCE_MS = 2_000   # the last result may lag the declared stop by one frame

# --- Clock-mapping admissibility. A mapping worse than this is not a mapping.
MAX_MAPPING_RESIDUAL_MS = 25.0
MAX_MAPPING_DRIFT_PPM = 100.0
# A p95 over a handful of samples is not a p95; refuse rather than report a
# flattering percentile of three results.
MIN_FRESHNESS_SAMPLES = 30

ACCELERATOR_BACKENDS = frozenset({"qnn", "qnn_htp", "htp", "gpu", "npu", "dsp"})
KNOWN_STATUSES = frozenset({"complete", "dropped", "failed", "stale", "invalid", "truncated"})
# The only two ways an age may have been produced. Anything else is an assumption.
ACCEPTABLE_AGE_CLOCK_PATHS = frozenset({"same-clock", "verified-mapping"})

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"


class ManifestError(Exception):
    """The manifest cannot be read, or is not the shape this checker understands."""


@dataclass
class Verdict:
    requirement: str
    status: str
    summary: str
    details: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON form. The report must serialise without a ``default=`` escape hatch,
        or a silently stringified verdict would be indistinguishable from a result."""
        return {"requirement": self.requirement, "status": self.status,
                "summary": self.summary, "details": self.details, "facts": self.facts}


# --------------------------------------------------------------------------
# manifest access
# --------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    candidate = path / "session.json" if path.is_dir() else path
    if not candidate.is_file():
        raise ManifestError(f"no session manifest at {candidate}")
    try:
        document = json.loads(candidate.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{candidate} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ManifestError(f"{candidate} must contain a JSON object")
    version = document.get("schemaVersion")
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"{candidate} declares schemaVersion={version!r}; this checker only understands {SCHEMA_VERSION}"
        )
    if not isinstance(document.get("session"), dict):
        raise ManifestError(f"{candidate} has no 'session' object")
    return document


def timestamp(node) -> tuple[float | None, str | None]:
    """Pull (value_ms, clock) out of a {"timeMs":..,"clock":..} node, or (None, None)."""
    if not isinstance(node, dict) or node.get("timeMs") is None:
        return None, None
    return float(node["timeMs"]), node.get("clock")


def completed_time(result: dict) -> tuple[float | None, str | None, str | None]:
    """The time the update was actually delivered, and which clock/node it came from.

    `published` is preferred: a depth update is completed when it is delivered,
    not when inference returned. `processing` is the fallback so a producer that
    only records inference completion still gets throughput measured rather than
    silently skipped.
    """
    for key in ("published", "processing"):
        value, clock = timestamp(result.get(key))
        if value is not None:
            return value, clock, key
    return None, None, None


def is_presented(result: dict) -> bool:
    """Was this result shown to the consumer? Defaults follow the result's own status."""
    status = result.get("status")
    valid = result.get("outputValid", True)
    default = status == "complete" and valid is not False
    return bool(result.get("presented", default))


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Deterministic and explainable in a report."""
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def steady_intervals(document: dict) -> list[tuple[dict, float, float]]:
    """Intervals clipped to the post-warmup window, as (interval, start, end)."""
    session = document["session"]
    warmup = float(session.get("warmupMs", 0.0))
    declared = float(session.get("declaredDurationMs", 0.0))
    clipped = []
    for interval in document.get("intervals", []):
        start = max(float(interval["startMs"]), warmup)
        end = min(float(interval["endMs"]), declared)
        if end > start:
            clipped.append((interval, start, end))
    return clipped


# --------------------------------------------------------------------------
# requirement checks
# --------------------------------------------------------------------------

def check_evidence(document: dict) -> Verdict:
    session = document["session"]
    if session.get("synthetic"):
        return Verdict(
            "EVIDENCE", FAIL,
            "synthetic record - not a device session, no device claim may rest on it",
            [f"session.synthetic = true; provenance: {session.get('provenance', 'unstated')}"],
            {"synthetic": True},
        )
    if not str(session.get("provenance", "")).strip():
        return Verdict(
            "EVIDENCE", BLOCKED,
            "no provenance recorded",
            ["session.provenance is empty; where this record came from is unknown"],
            {"synthetic": False, "provenance": ""},
        )
    return Verdict(
        "EVIDENCE", PASS,
        "recorded device session",
        [f"provenance: {session['provenance']}"],
        {"synthetic": False, "provenance": session["provenance"]},
    )


def check_throughput(document: dict) -> Verdict:
    session = document["session"]
    results = document.get("results", [])
    intervals = steady_intervals(document)
    warmup = float(session.get("warmupMs", 0.0))
    declared = float(session.get("declaredDurationMs", 0.0))
    detail: list[str] = []
    facts: dict = {"targetHz": TARGET_UPDATES_PER_SECOND}

    if not intervals:
        return Verdict(
            "THROUGHPUT", FAIL,
            "no steady-state interval was recorded",
            ["every interval is inside the warmup window or none exist; there is no "
             "steady-state evidence, and absent evidence is not a slow average"],
            facts,
        )

    # A gap in interval coverage is a throughput failure in its own right: the
    # unmeasured window could have been arbitrarily slow and must not be inferred.
    coverage_ms = sum(end - start for _, start, end in intervals)
    gaps = []
    for (_, _, previous_end), (_, start, _) in zip(intervals, intervals[1:]):
        if start - previous_end > GAP_TOLERANCE_MS:
            gaps.append((previous_end, start))
    coverage_ok = not gaps and coverage_ms >= MIN_STEADY_STATE_MS

    completed = []          # (time_ms, is_countable)
    bucket_source = None
    for result in results:
        value, clock, source = completed_time(result)
        if value is None:
            continue
        bucket_source = bucket_source or source
        countable = result.get("status") == "complete" and result.get("outputValid", True) is not False
        completed.append((value, countable))

    counts = []
    for interval, start, end in intervals:
        count = sum(1 for t, countable in completed if countable and start <= t < end)
        rate = count / ((end - start) / 1000.0)
        counts.append({"index": interval.get("index"), "startMs": start, "endMs": end,
                       "completed": count, "rateHz": rate})

    if len(counts) <= 12:
        for row in counts:
            detail.append(
                f"interval {row['index']} [{row['startMs']:.0f}..{row['endMs']:.0f} ms): "
                f"{row['completed']} updates = {row['rateHz']:.2f}/s"
            )
    else:
        slowest = sorted(counts, key=lambda r: r["rateHz"])[:3]
        for row in slowest:
            detail.append(
                f"slowest interval {row['index']}: {row['completed']} updates = {row['rateHz']:.2f}/s"
            )

    slowest_rate = min(r["rateHz"] for r in counts)
    slow = [r for r in counts if r["rateHz"] < TARGET_UPDATES_PER_SECOND]
    facts["intervals"] = counts
    facts["slowestHz"] = slowest_rate
    facts["steadyStateMs"] = coverage_ms
    facts["countedFrom"] = bucket_source

    if gaps:
        for gap_start, gap_end in gaps[:5]:
            detail.append(
                f"MISSING INTERVALS: no record covers {gap_start:.0f}..{gap_end:.0f} ms "
                f"({(gap_end - gap_start) / 1000.0:.1f} s unmeasured)"
            )
    if coverage_ms < MIN_STEADY_STATE_MS:
        detail.append(
            f"steady state is {coverage_ms / 1000.0:.1f} s, below the {MIN_STEADY_STATE_MS / 1000.0:.0f} s floor"
        )
    for row in slow[:5]:
        detail.append(
            f"interval {row['index']} ran {row['rateHz']:.2f}/s, below the "
            f"{TARGET_UPDATES_PER_SECOND:.1f}/s requirement"
        )
    if slow:
        detail.append(
            f"{len(slow)} of {len(counts)} intervals missed the target; intervals are "
            f"never averaged together"
        )

    if not coverage_ok or slow:
        return Verdict(
            "THROUGHPUT", FAIL,
            f"{len(slow)}/{len(counts)} intervals below {TARGET_UPDATES_PER_SECOND:.1f}/s"
            + (f", {len(gaps)} interval gap(s)" if gaps else ""),
            detail, facts,
        )
    return Verdict(
        "THROUGHPUT", PASS,
        f"{len(counts)} steady-state intervals, slowest {slowest_rate:.2f}/s "
        f"(>= {TARGET_UPDATES_PER_SECOND:.1f}/s required in EVERY interval)",
        detail, facts,
    )


def mapping_uncertainty_ms(document: dict) -> float | None:
    """Worst-case clock-mapping error over the session, or None if unverified.

    A p95 compared against a 200 ms budget while ignoring a known +/-N ms mapping
    error would overstate the margin, so the residual is added to the reported p95.
    """
    mapping = document.get("clockMapping")
    if not isinstance(mapping, dict) or mapping.get("verified") is not True:
        return None
    residual = float(mapping.get("maxResidualMs", float("inf")))
    drift_ppm = abs(float(mapping.get("driftPpm", float("inf"))))
    if residual > MAX_MAPPING_RESIDUAL_MS or drift_ppm > MAX_MAPPING_DRIFT_PPM:
        return None
    declared = float(document["session"].get("declaredDurationMs", 0.0))
    return residual + drift_ppm * 1e-6 * declared


def result_age_ms(result: dict, document: dict) -> tuple[float | None, str]:
    """Age in ms, or (None, reason). Never guesses across clocks.

    Only `published` time is used. A frame's age is how long the *consumer* has
    been looking at stale depth, which is what the 200 ms budget is about; an
    inference-completion time would flatter it by the publication and display path.
    """
    age = result.get("age")
    if isinstance(age, dict) and age.get("valueMs") is not None:
        clock_path = age.get("clockPath")
        if clock_path in ACCEPTABLE_AGE_CLOCK_PATHS:
            return float(age["valueMs"]), f"producer:{clock_path}"
        return None, f"producer age rejected: clockPath={clock_path!r} is not {sorted(ACCEPTABLE_AGE_CLOCK_PATHS)}"

    capture_ms, capture_clock = timestamp(result.get("capture"))
    published_ms, published_clock = timestamp(result.get("published"))
    if capture_ms is None or published_ms is None:
        return None, "no producer age and no capture/published timestamps"

    if capture_clock is not None and capture_clock == published_clock:
        age_ms = published_ms - capture_ms
        if age_ms < 0 or age_ms > declared_span_ms(document):
            return None, f"same-clock subtraction is implausible ({age_ms:.1f} ms)"
        return age_ms, "same-clock"

    mapping = document.get("clockMapping")
    if isinstance(mapping, dict) and mapping.get("verified") is True:
        if mapping.get("from") == capture_clock and mapping.get("to") == published_clock \
                and mapping_uncertainty_ms(document) is not None:
            age_ms = published_ms - (capture_ms + float(mapping["offsetMs"]))
            if age_ms < 0 or age_ms > declared_span_ms(document):
                return None, f"mapping-corrected age is implausible ({age_ms:.1f} ms)"
            return age_ms, "verified-mapping"

    return None, (
        f"capture clock {capture_clock!r} and publication clock {published_clock!r} "
        f"differ and no verified clock mapping is present"
    )


def declared_span_ms(document: dict) -> float:
    """Upper plausibility bound for one frame's age: the declared session length."""
    return max(float(document["session"].get("declaredDurationMs", 1000.0)), 1.0)


def check_freshness(document: dict) -> Verdict:
    results = document.get("results", [])
    ages: list[float] = []
    sources: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for result in results:
        if not is_presented(result):
            continue
        age_ms, source = result_age_ms(result, document)
        if age_ms is None:
            reasons[source] = reasons.get(source, 0) + 1
            continue
        ages.append(age_ms)
        sources[source] = sources.get(source, 0) + 1

    uncertainty = mapping_uncertainty_ms(document)
    detail: list[str] = []
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:4]:
        detail.append(f"{count} result(s) with UNKNOWN age: {reason}")
    detail.append(
        f"age computed from: {', '.join(f'{k} x{v}' for k, v in sorted(sources.items())) or 'nothing'}"
    )
    facts = {
        "budgetP95Ms": FRESHNESS_BUDGET_P95_MS,
        "budgetKind": "proposed harness freshness budget - NOT a flight requirement",
        "samples": len(ages),
        "unknownSamples": sum(reasons.values()),
        "sources": sources,
    }

    if len(ages) < MIN_FRESHNESS_SAMPLES:
        detail.append(
            f"only {len(ages)} result(s) had a derivable age, below the {MIN_FRESHNESS_SAMPLES} "
            f"needed for a p95 to mean anything"
        )
        return Verdict(
            "FRESHNESS", BLOCKED,
            f"result age UNKNOWN for {sum(reasons.values())} of {len(results)} result(s); "
            f"only {len(ages)} derivable",
            detail, facts,
        )

    p50 = percentile(ages, 0.50)
    p95 = percentile(ages, 0.95)
    upper = p95 + (uncertainty or 0.0)
    facts.update({"p50Ms": p50, "p95Ms": p95, "uncertaintyMs": uncertainty, "upperBoundP95Ms": upper})
    detail.append(f"p50 {p50:.1f} ms, p95 {p95:.1f} ms"
                  + (f" (+{uncertainty:.1f} ms clock-mapping uncertainty = {upper:.1f} ms)" if uncertainty else ""))
    detail.append(f"budget is PROPOSED FOR THE HARNESS ONLY - it is not a flight requirement "
                  f"and this checker makes no flight claim")

    if upper > FRESHNESS_BUDGET_P95_MS:
        return Verdict(
            "FRESHNESS", FAIL,
            f"p95 result age {upper:.1f} ms exceeds the {FRESHNESS_BUDGET_P95_MS:.0f} ms "
            f"proposed harness budget (not a flight requirement)",
            detail, facts,
        )
    return Verdict(
        "FRESHNESS", PASS,
        f"p95 result age {upper:.1f} ms <= {FRESHNESS_BUDGET_P95_MS:.0f} ms "
        f"(proposed harness budget, NOT a flight requirement)",
        detail, facts,
    )


def check_integrity(document: dict) -> Verdict:
    session = document["session"]
    results = document.get("results", [])
    intervals = document.get("intervals", [])
    warmup = float(session.get("warmupMs", 0.0))
    declared = float(session.get("declaredDurationMs", 0.0))
    session_clock = session.get("sessionClock")
    detail: list[str] = []
    problems = 0

    # 1. Nothing invalid may reach the consumer.
    bad_presented = []
    for result in results:
        status = result.get("status")
        valid = result.get("outputValid", True)
        if status not in KNOWN_STATUSES:
            problems += 1
            detail.append(f"result {result.get('index')} has unrecognised status {status!r}")
            continue
        if is_presented(result) and (status != "complete" or valid is False):
            bad_presented.append(result)
    for result in bad_presented[:5]:
        reason = "outputValid=false" if result.get("outputValid", True) is False else f"status={result.get('status')}"
        detail.append(f"result {result.get('index')} was PRESENTED but {reason}: a stale or invalid "
                      f"result reached the consumer")
    problems += len(bad_presented)

    # 2. Intervals must be present, contiguous, full-length and explicitly complete.
    steady = steady_intervals(document)
    gaps = []
    for (_, _, previous_end), (_, start, _) in zip(steady, steady[1:]):
        if start - previous_end > GAP_TOLERANCE_MS:
            gaps.append((previous_end, start))
    for interval in intervals:
        if interval.get("complete") is False:
            problems += 1
            detail.append(f"interval {interval.get('index')} is flagged incomplete and was recorded anyway")
        length = float(interval["endMs"]) - float(interval["startMs"])
        if length < INTERVAL_MS * 0.99:
            problems += 1
            detail.append(
                f"interval {interval.get('index')} spans {length:.0f} ms, short of a full "
                f"{INTERVAL_MS / 1000:.0f} s minute; a partial interval is not a valid interval"
            )
    for gap_start, gap_end in gaps[:5]:
        problems += 1
        detail.append(f"interval coverage gap {gap_start:.0f}..{gap_end:.0f} ms "
                      f"({(gap_end - gap_start) / 1000.0:.1f} s of the run was never bucketed)")

    # 3. The run must cover the duration it declared.
    coverage_ms = sum(end - start for _, start, end in steady)
    if declared - warmup < MIN_STEADY_STATE_MS:
        problems += 1
        detail.append(
            f"declared steady state is {(declared - warmup) / 1000.0:.1f} s, below the "
            f"{MIN_STEADY_STATE_MS / 1000.0:.0f} s minimum"
        )
    if coverage_ms < declared - warmup - GAP_TOLERANCE_MS:
        problems += 1
        detail.append(
            f"intervals cover {coverage_ms / 1000.0:.1f} s of the "
            f"{(declared - warmup) / 1000.0:.1f} s declared post-warmup window: "
            f"the run does not cover its full declared duration"
        )

    # 4. A time-ordered log that is not ordered, or repeats an index, is not evidence.
    seen_indices = set()
    duplicates = 0
    previous = None
    for result in results:
        index = result.get("index")
        if index in seen_indices:
            duplicates += 1
        seen_indices.add(index)
        value, clock, _ = completed_time(result)
        if value is None:
            continue
        if session_clock and clock and clock != session_clock:
            problems += 1
            detail.append(f"result {index} completed on clock {clock!r} but sessionClock is {session_clock!r}")
        if previous is not None and value <= previous:
            problems += 1
            detail.append(f"result {index} completes at {value:.1f} ms, not after the previous "
                          f"{previous:.1f} ms: completion timestamps are not monotonic")
        previous = value
    if duplicates:
        problems += 1
        detail.append(f"{duplicates} duplicated result index value(s)")

    facts = {
        "results": len(results),
        "presentedViolations": len(bad_presented),
        "intervalsRecorded": len(intervals),
        "steadyStateIntervals": len(steady),
        "steadyStateMs": coverage_ms,
        "declaredSteadyStateMs": declared - warmup,
        "coverageGaps": gaps,
    }
    if problems:
        return Verdict("INTEGRITY", FAIL, f"{problems} integrity problem(s)", detail, facts)
    return Verdict(
        "INTEGRITY", PASS,
        f"{len(steady)} contiguous intervals covering {coverage_ms / 1000.0:.1f} s of the declared "
        f"{(declared - warmup) / 1000.0:.1f} s steady state; no stale or invalid result presented",
        detail, facts,
    )


def check_backend(document: dict) -> Verdict:
    session = document["session"]
    results = document.get("results", [])
    detail: list[str] = []
    configured = session.get("configuredBackend")
    if not configured:
        return Verdict(
            "BACKEND", BLOCKED,
            "no configured backend was recorded",
            ["session.configuredBackend is absent; there is nothing to compare a result against, "
             "and an absent comparison is not a pass"],
            {"configured": None},
        )

    labels = [r.get("backend", {}).get("label") for r in results]
    missing = sum(1 for label in labels if label is None)
    mismatched = [r for r, label in zip(results, labels) if label is not None and label != configured]
    counts = {label: labels.count(label) for label in set(labels) if label is not None}

    # Registration is not placement. Neither is speed. An accelerator claim needs
    # per-result evidence that the work left the CPU.
    unproven = []
    if configured in ACCELERATOR_BACKENDS:
        for result in results:
            backend = result.get("backend", {})
            if backend.get("label") == configured and backend.get("placementProven") is not True:
                unproven.append(result)

    providers = sorted({p for r in results for p in r.get("backend", {}).get("providersRegistered", [])})
    detail.append(f"configured backend: {configured}; results labelled: "
                  + (", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) or "none"))
    if providers:
        detail.append(f"providers registered (NOT placement evidence): {', '.join(providers)}")
    for result in mismatched[:5]:
        detail.append(f"result {result.get('index')} is labelled "
                      f"{result['backend'].get('label')!r} but {configured!r} was configured")
    for result in unproven[:5]:
        backend = result.get("backend", {})
        detail.append(
            f"result {result.get('index')} claims {configured!r} with no placement evidence "
            f"(registered={backend.get('providersRegistered', [])}, "
            f"evidence={backend.get('evidence', 'none')!r}); a provider registering does not prove HTP ran"
        )
    facts = {
        "configured": configured,
        "labels": counts,
        "missingLabels": missing,
        "mismatched": len(mismatched),
        "unprovenPlacement": len(unproven),
        "providersRegistered": providers,
    }

    if mismatched:
        return Verdict(
            "BACKEND", FAIL,
            f"{len(mismatched)} of {len(results)} results ran on a backend other than the "
            f"configured {configured!r}",
            detail, facts,
        )
    if missing:
        return Verdict(
            "BACKEND", FAIL,
            f"{missing} result(s) carry no backend label",
            detail, facts,
        )
    if unproven:
        return Verdict(
            "BACKEND", FAIL,
            f"{configured} is claimed but placement is unproven on {len(unproven)} result(s)",
            detail, facts,
        )
    return Verdict(
        "BACKEND", PASS,
        f"all {len(results)} results labelled {configured!r} matching the configured backend"
        + ("; accelerator placement evidenced per result" if configured in ACCELERATOR_BACKENDS else ""),
        detail, facts,
    )


# --------------------------------------------------------------------------
# observation echo
# --------------------------------------------------------------------------

def observe(document: dict) -> list[str]:
    """Echo the recorded evidence. Nothing here scores anything."""
    session = document["session"]
    results = document.get("results", [])
    lines: list[str] = []

    status_counts = {s: sum(1 for r in results if r.get("status") == s) for s in sorted(KNOWN_STATUSES)}
    invalid = sum(1 for r in results if r.get("outputValid", True) is False)
    dropped = sum(r.get("droppedSincePrevious", 0) or 0 for r in results)
    status_dropped = status_counts.get("dropped", 0) + status_counts.get("failed", 0)
    lines.append(f"results: {len(results)} total; " + ", ".join(f"{k}={v}" for k, v in status_counts.items() if v)
                 + (f", outputValid=false={invalid}" if invalid else ""))
    lines.append(f"dropped frames: {status_dropped} recorded as dropped/failed, "
                 f"{dropped} reported by the producer's per-result counter")

    java = [v for r in results for v in [(r.get("memory") or {}).get("javaHeapBytes")] if v]
    native = [v for r in results for v in [(r.get("memory") or {}).get("nativeHeapBytes")] if v]
    if java or native:
        peak = []
        if java:
            peak.append(f"java heap {max(java) / 1e6:.1f} MB")
        if native:
            peak.append(f"native heap {max(native) / 1e6:.1f} MB")
        lines.append("memory (peak): " + ", ".join(peak))
    else:
        lines.append("memory: none recorded")

    skins = [v for r in results for v in [(r.get("thermal") or {}).get("skinTempC")] if v]
    throttled = sum(1 for r in results if (r.get("thermal") or {}).get("throttling"))
    lines.append(f"thermal: peak skin {max(skins):.1f} C over {len(skins)} sample(s), "
                 f"{throttled} throttling event(s)" if skins else "thermal: none recorded")

    for label, node in (("capture", "capture"), ("processing", "published")):
        values, clocks = [], set()
        for result in results:
            value, clock = timestamp(result.get(node))
            if value is not None:
                values.append(value)
                clocks.add(clock)
        if values:
            lines.append(f"{label} timestamps: {len(values)} recorded on "
                         f"{'/'.join(sorted(str(c) for c in clocks))}; "
                         f"first {min(values):.1f}, last {max(values):.1f}")
        else:
            lines.append(f"{label} timestamps: none recorded")

    ages = [result_age_ms(r, document) for r in results if is_presented(r)]
    known = sorted(a for a, _ in ages if a is not None)
    if known:
        lines.append(f"result age: p50 {percentile(known, 0.50):.1f} ms, p95 {percentile(known, 0.95):.1f} ms, "
                     f"{len(known)} known / {sum(1 for a, _ in ages if a is None)} unknown")
    else:
        lines.append(f"result age: UNKNOWN for all {len(ages)} presented result(s) - "
                     f"{ages[0][1] if ages else 'no presented results'}")

    if document.get("producerSummary"):
        claimed = ", ".join(f"{k}={v}" for k, v in sorted(document["producerSummary"].items()))
        lines.append(f"producer claims (echoed, NOT trusted or scored by this checker): {claimed}")
    return lines


def summarise(document: dict) -> tuple[list[Verdict], str]:
    verdicts = [
        check_evidence(document),
        check_throughput(document),
        check_freshness(document),
        check_integrity(document),
        check_backend(document),
    ]
    statuses = {v.status for v in verdicts}
    overall = FAIL if FAIL in statuses else (BLOCKED if BLOCKED in statuses else PASS)
    return verdicts, overall


def evaluate(path: Path) -> tuple[dict, list[Verdict], str]:
    document = load_manifest(path)
    verdicts, overall = summarise(document)
    report = {
        "schemaVersion": document["schemaVersion"],
        "sessionId": document["session"].get("id"),
        "device": document["session"].get("device", {}),
        "declaredDurationMs": document["session"].get("declaredDurationMs"),
        "warmupMs": document["session"].get("warmupMs"),
        "overall": overall,
        "requirements": {v.requirement: v.as_dict() for v in verdicts},
        "observations": observe(document),
    }
    return report, verdicts, overall


def render_text(report: dict, verdicts: list[Verdict]) -> str:
    lines = ["=" * 78, "Depth session acceptance report", "=" * 78]
    lines.append(f"session    : {report['sessionId']}")
    device = report.get("device") or {}
    if device:
        lines.append(f"device     : {device.get('model', 'unknown')} (API {device.get('androidApi', '?')}, "
                     f"{device.get('soc', '?')})")
    lines.append(f"config     : backend={report.get('configured', '?')} "
                 f"declaredDuration={report.get('declaredDurationMs')} ms "
                 f"warmup={report.get('warmupMs')} ms")
    lines.append(f"clocks     : {report.get('clockNote', 'unknown')}")
    lines.append("")
    width = max(len(v.requirement) for v in verdicts)
    for verdict in verdicts:
        lines.append(f"[{verdict.status:^7}] {verdict.requirement:<{width}}  {verdict.summary}")
        for item in verdict.details:
            lines.append(f"{'':>8}  - {item}")
    lines.append("")
    lines.append("recorded evidence:")
    for item in report["observations"]:
        lines.append(f"  {item}")
    counts = {status: sum(1 for v in verdicts if v.status == status) for status in (PASS, FAIL, BLOCKED)}
    lines.append("")
    lines.append(f"OVERALL: {report['overall']}   ({counts[PASS]} pass, {counts[FAIL]} fail, "
                 f"{counts[BLOCKED]} blocked of {len(verdicts)} requirements)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, help="recorded session manifest (file or directory)")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--self-test", action="store_true",
                        help="run the acceptance checker's own tests, including its negative controls")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.manifest:
        parser.error("--manifest is required (or use --self-test)")

    try:
        report, verdicts, overall = evaluate(args.manifest)
    except ManifestError as exc:
        print(f"CANNOT EVALUATE: {exc}", file=sys.stderr)
        return 3

    document = load_manifest(args.manifest)
    session = document["session"]
    capture_clock = next((c for r in document.get("results", [])
                          for v, c in [timestamp(r.get("capture"))] if c), "none")
    processing_clock = next((c for r in document.get("results", [])
                             for v, c in [timestamp(r.get("published"))] if c), "none")
    mapping = document.get("clockMapping") or {}
    if mapping.get("verified") is True and mapping_uncertainty_ms(document) is not None:
        clock_note = (f"capture={capture_clock} processing={processing_clock} "
                      f"mapping=verified (offsetMs={mapping.get('offsetMs')}, "
                      f"residual={mapping.get('maxResidualMs')} ms, drift={mapping.get('driftPpm')} ppm)")
    elif capture_clock != processing_clock:
        clock_note = (f"capture={capture_clock} processing={processing_clock} "
                      f"mapping=NOT VERIFIED -> cross-clock latency is UNKNOWN, not a number")
    else:
        clock_note = f"capture={capture_clock} processing={processing_clock} (same clock)"

    report["configured"] = session.get("configuredBackend")
    report["clockNote"] = clock_note

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_text(report, verdicts))
    return {PASS: 0, FAIL: 1, BLOCKED: 2}[overall]


# --------------------------------------------------------------------------
# tests -- the negative controls are the point of this file
# --------------------------------------------------------------------------

FIXTURES = {
    "pass": "pass-fast-qnn",
    "slow": "slow-session",
    "missing": "missing-intervals",
    "truncated": "truncated-session",
    "mislabelled": "mislabelled-backend",
    "crossclock": "cross-clock-trap",
    "stale": "presented-stale-results",
    "unproven": "unproven-accelerator-placement",
}


def fixture(name: str) -> Path:
    return FIXTURE_ROOT / FIXTURES[name]


class AcceptanceCheckerTests(unittest.TestCase):
    """Every fixture here is synthetic and is labelled as such inside the file.

    They prove the *checks* fire on the right evidence. None of them, and none of
    the files in testdata/session-fixtures/, can ever produce a device pass: the
    EVIDENCE requirement rejects all of them.
    """

    def run_fixture(self, name: str) -> tuple[dict, dict]:
        report, verdicts, _ = evaluate(fixture(name))
        return report, {v.requirement: v for v in verdicts}

    def require(self, name: str, requirement: str, status: str) -> Verdict:
        report, verdicts = self.run_fixture(name)
        self.assertIn(requirement, verdicts)
        self.assertEqual(verdicts[requirement].status, status,
                         f"{name}/{requirement}: expected {status}, got "
                         f"{verdicts[requirement].status} ({verdicts[requirement].summary})")
        return verdicts[requirement]

    def detail_text(self, verdict: Verdict) -> str:
        return " | ".join(verdict.details)

    def test_rejects_missing_or_slow_intervals(self):
        """A slow minute fails even when the other minutes are fast."""
        slow = self.require("slow", "THROUGHPUT", FAIL)
        self.assertIn("never averaged", self.detail_text(slow))
        rates = [row["rateHz"] for row in slow.facts["intervals"]]
        self.assertGreater(max(rates), 10.0, "fixture must contain fast intervals for the average to hide")
        self.assertLess(min(rates), 10.0)

        missing = self.require("missing", "THROUGHPUT", FAIL)
        self.assertIn("MISSING INTERVALS", self.detail_text(missing))
        self.assertTrue(missing.facts.get("coverageGaps") or any(
            "MISSING INTERVALS" in d for d in missing.details))

    def test_rejects_short_session(self):
        """A run that stops early cannot pass by having been fast while it lasted."""
        verdict = self.require("truncated", "INTEGRITY", FAIL)
        text = self.detail_text(verdict)
        self.assertIn("incomplete", text)
        self.assertIn("does not cover its full declared duration", text)
        self.require("truncated", "THROUGHPUT", FAIL)

    def test_accepts_well_formed_fast_session(self):
        """The positive control: without it, every other test would pass trivially."""
        report, verdicts = self.run_fixture("pass")
        for requirement in ("THROUGHPUT", "FRESHNESS", "INTEGRITY", "BACKEND"):
            self.assertEqual(verdicts[requirement].status, PASS,
                             f"{requirement}: {verdicts[requirement].summary} {verdicts[requirement].details}")
        self.assertGreaterEqual(verdicts["THROUGHPUT"].facts["slowestHz"], 10.0)
        self.assertEqual(verdicts["BACKEND"].facts["configured"], "qnn_htp")
        # The freshness budget must be labelled as proposed, not as a flight requirement.
        self.assertIn("NOT a flight requirement", verdicts["FRESHNESS"].summary)
        self.assertEqual(report["overall"], FAIL, "a synthetic fixture must never report an overall pass")

    def test_rejects_mislabelled_backend(self):
        verdict = self.require("mislabelled", "BACKEND", FAIL)
        self.assertIn("ran on a backend other than the configured", verdict.summary)
        self.assertIn("'cpu'", self.detail_text(verdict))
        # A labelled-vs-actual contradiction is a FAIL, never a warning.
        self.assertNotEqual(verdict.status, BLOCKED)

    def test_rejects_unproven_accelerator_placement(self):
        """A provider registering is not placement, and fast is not placement."""
        verdict = self.require("unproven", "BACKEND", FAIL)
        self.assertIn("placement is unproven", verdict.summary)
        self.assertIn("does not prove HTP ran", self.detail_text(verdict))
        # The fast one still fails on backend, so throughput cannot launder it.
        self.assertEqual(self.run_fixture("unproven")[1]["THROUGHPUT"].status, PASS)

    def test_refuses_cross_clock_latency(self):
        """No verified mapping => age is UNKNOWN, never a number derived by subtraction."""
        verdict = self.require("crossclock", "FRESHNESS", BLOCKED)
        self.assertEqual(verdict.facts["samples"], 0)
        self.assertIn("UNKNOWN age", self.detail_text(verdict))
        self.assertIn("assumed-offset", self.detail_text(verdict))
        self.assertNotIn("p95Ms", verdict.facts)
        report, _ = self.run_fixture("crossclock")
        age_line = next(line for line in report["observations"] if line.startswith("result age:"))
        self.assertIn("UNKNOWN", age_line)
        # The clock note must say why, and must not offer a latency number.
        _, freshness, overall = evaluate(fixture("crossclock"))
        by_name = {v.requirement: v for v in freshness}
        self.assertEqual(by_name["FRESHNESS"].status, BLOCKED)
        self.assertNotEqual(overall, PASS, "an unmeasurable age can never yield a pass")

    def test_verified_clock_mapping_enables_age(self):
        """Both routes to an age work, and the checker computes its own when it must."""
        verdict = self.require("pass", "FRESHNESS", PASS)
        self.assertIn("producer:verified-mapping", verdict.facts["sources"])
        self.assertGreaterEqual(verdict.facts["upperBoundP95Ms"], verdict.facts["p95Ms"])
        self.assertLessEqual(verdict.facts["upperBoundP95Ms"], 200.0)

        # Strip the producer's pre-computed ages: the checker must now derive them from
        # the capture/published timestamps through the verified mapping itself.
        document = json.loads((fixture("pass") / "session.json").read_text())
        for result in document["results"]:
            result.pop("age", None)
        derived = self.run_document(document)["requirements"]["FRESHNESS"]
        self.assertEqual(derived["status"], PASS)
        self.assertEqual(derived["facts"]["sources"], {"verified-mapping": 3000})
        self.assertAlmostEqual(derived["facts"]["p95Ms"], verdict.facts["p95Ms"], places=3)

    def test_rejects_presented_stale_and_invalid_results(self):
        verdict = self.require("stale", "INTEGRITY", FAIL)
        text = self.detail_text(verdict)
        self.assertIn("status=stale", text)
        self.assertIn("outputValid=false", text)
        # They must not be counted as completed work either: that minute's rate drops
        # below the 12.5 Hz it would otherwise have run at, while still clearing 10 Hz.
        throughput = self.run_fixture("stale")[1]["THROUGHPUT"]
        self.assertLess(throughput.facts["slowestHz"], 12.5)
        self.assertGreaterEqual(throughput.facts["slowestHz"], 10.0)

    def test_missing_required_field_blocks_rather_than_passes(self):
        """Deleting a required field must move a verdict to BLOCKED, never to PASS."""
        # run_document round-trips through JSON, so its verdicts are plain dicts.
        base = json.loads((fixture("pass") / "session.json").read_text())
        # Clear the synthetic flag so the only thing wrong with the record is the
        # defect under test, and a BLOCKED verdict is what actually drives OVERALL.
        base["session"]["synthetic"] = False
        base["session"]["provenance"] = "unit-test copy of the synthetic fixture"

        document = json.loads(json.dumps(base))
        del document["session"]["configuredBackend"]
        report = self.run_document(document)
        self.assertEqual(report["requirements"]["BACKEND"]["status"], BLOCKED)
        self.assertEqual(report["overall"], BLOCKED, "a blocked requirement must block the overall verdict")

        document = json.loads(json.dumps(base))
        del document["clockMapping"]
        for result in document["results"]:
            result["age"]["clockPath"] = "assumed-offset"
        report = self.run_document(document)
        self.assertEqual(report["requirements"]["FRESHNESS"]["status"], BLOCKED)
        self.assertEqual(report["overall"], BLOCKED)

        # A mapping that exists but is not verified is no better than no mapping.
        document = json.loads(json.dumps(base))
        document["clockMapping"]["verified"] = False
        for result in document["results"]:
            result["age"]["clockPath"] = "assumed-offset"
        self.assertEqual(
            self.run_document(document)["requirements"]["FRESHNESS"]["status"], BLOCKED)

        # Sanity: the unmodified record really does pass everything, so BLOCKED above
        # was caused by the deletion and not by the fixture failing anyway.
        self.assertEqual(self.run_document(base)["overall"], PASS)

    def run_document(self, document: dict) -> dict:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="depth-session-eval-") as tmp:
            target = Path(tmp) / "session.json"
            target.write_text(json.dumps(document))
            report, _, _ = evaluate(target)
            return json.loads(json.dumps(report, default=str))

    def test_cli_exit_codes_and_no_synthetic_pass(self):
        import subprocess
        script = str(REPO_ROOT / "scripts/evaluate-depth-session.py")
        for name in FIXTURES.values():
            completed = subprocess.run(
                [sys.executable, script, "--manifest", str(FIXTURE_ROOT / name)],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            self.assertIn(completed.returncode, (1, 2),
                          f"{name} returned {completed.returncode}: no synthetic fixture may exit 0")
            self.assertIn("OVERALL:", completed.stdout)

        missing = subprocess.run(
            [sys.executable, script, "--manifest", str(FIXTURE_ROOT / "does-not-exist")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        self.assertEqual(missing.returncode, 3)
        self.assertIn("CANNOT EVALUATE", missing.stderr)

    def test_never_emits_a_metric_depth(self):
        """Output is relative depth. No verdict, fact or observation may carry metres."""
        banned = ("metric_depth", "depthMetres", "depth_m", "meters", "metres")
        for name in FIXTURES.values():
            report, _, _ = evaluate(FIXTURE_ROOT / name)
            blob = json.dumps(report, default=str)
            for token in banned:
                self.assertNotIn(token, blob, f"{name} leaked a metric-depth token {token!r}")

    def test_echoes_the_recorded_evidence(self):
        """A verdict is worth only as much as the evidence it was measured from."""
        report, _, _ = evaluate(fixture("pass"))
        joined = "\n".join(report["observations"])
        for expected in ("dropped frames:", "memory (peak):", "thermal:", "capture timestamps:",
                         "processing timestamps:", "result age:", "NOT trusted or scored"):
            self.assertIn(expected, joined, f"missing echoed evidence: {expected}")


def run_self_test() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(AcceptanceCheckerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
