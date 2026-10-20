#!/usr/bin/env python3
"""
Acceptance checker for a captured **live depth view** session.

    python3 scripts/evaluate-live-session.py --log /tmp/live.logcat
    adb logcat -d -s DepthQNN:I | python3 scripts/evaluate-live-session.py
    python3 scripts/evaluate-live-session.py --log live.logcat --json
    python3 scripts/evaluate-live-session.py --self-test          # negative controls

Exit codes match `scripts/evaluate-depth-session.py` exactly, because these are the
same tool answering the same question about the same device:

    0 = every requirement passed
    1 = at least one requirement failed
    2 = no verdict possible (something blocked)
    3 = the capture could not be read

--------------------------------------------------------------------------------
WHAT "ACCEPTANCE" MEANS FOR A LIVE VIEW, AND WHY EACH CHECK EXISTS
--------------------------------------------------------------------------------
The still-image benchmark is gone. The screen is now a camera preview with a depth
mask composited over it, both starting automatically. The bug this mode was rebuilt
to fix was a **stale mask**: the old screen inferred one bundled JPEG and rendered
that same mask forever, so it looked plausible while never updating. A checker that
merely ran without error would not have caught that, so *completion is not a pass*
here any more than it is in the sibling script.

So each of the checks below exists because its absence is a specific way this view
can silently go back to being a still image:

    EVIDENCE           the capture contains parseable `depth rate` lines at all. A log
                       with no rate line has shown nothing about depth; it is not a slow
                       session, it is an absent one. Also rejects a synthetic fixture.
    UPDATES            enough depth updates completed for the view to have been alive.
                       Zero or near-zero inferences IS the frozen-mask bug.
    PROGRESS           the completed-inference counter actually advances. A counter that
                       repeats means the mask on screen was not being replaced.
    DROPS              `droppedBusy` is substantial relative to `offered`. The pipeline is
                       specified to DROP frames while an inference runs, never queue them.
                       Zero drops at ~0.5 Hz with ~20 offered fps means frames are piling
                       up unboundedly: the view would show depth for a scene the camera
                       left seconds ago. That is the stale mask in a new shape, and it is
                       the failure this check exists to catch.
    TOO_SMALL          must be zero. A non-zero count means frames were discarded before
                       inference because they could not yield the 518px centre crop --
                       i.e. the analysis resolution or crop maths is wrong.
    ERRORS             must be zero. A counted inference error is a frame the user saw no
                       depth for.
    SELF_CONSISTENCY   the reported `fps` equals the independently recomputed
                       `inferences / elapsedS` of the final line. A log that disagrees
                       with itself is not evidence of anything, whatever it claims.
    SPAN               the final `elapsedS` is long enough to be a measurement. A rate
                       computed over a fraction of a second is a number with no time
                       behind it; a burst of log lines can print any fps at all.
    CAMERA             the camera actually reached `Streaming`. A black preview and a dead
                       camera look identical on screen, which is exactly why the status is
                       logged and why it is judged here.

Requirements are judged **independently**. There is deliberately no blended score: a
session that streams perfectly while never completing an inference is not "mostly fine".

Three things this checker will not do, each because getting it wrong would make the
evidence look better than it is:

  * It will not trust the log's own `fps` without recomputing it from the counts.
  * It will not treat absent evidence as a pass. Missing rate lines, or a capture with
    no camera status at all, cannot clear the requirement they were supposed to clear.
  * It will never emit or accept a metric depth value. Output is relative, disparity-like,
    and has no metres.

--------------------------------------------------------------------------------
THE CAPTURE IT CONSUMES
--------------------------------------------------------------------------------
Raw logcat text — no pre-summarised JSON, because the caller has none and inventing a
format nobody produces would mean the checker never runs against reality. Lines are
matched anywhere in the text, so the logcat prefix (`MM-DD HH:MM:SS.mmm PID TID I
DepthQNN:`) may be present or absent.

The `DepthQNN` lines this checker reads, as emitted by `LiveDepthPipeline.kt` and
`MainActivity.kt`:

    depth rate inferences=10 fps=0.87 elapsedS=11.6 offered=408 droppedBusy=398 tooSmall=0 errors=0
    pipeline ready session=... backend=CPU note=CPU — QNN could not initialise, see HANDOVER 2.4
    camera status=Streaming(description=back camera bound (implementation=CAMERA2, ...))

    inferences   completed depth updates
    fps          completed / elapsedS, printed to 2 dp
    elapsedS     measured seconds since pipeline construction, including model loading
    offered      frames handed to the pipeline by the analyzer
    droppedBusy  frames dropped because an inference was already running
    tooSmall     frames too small to yield the 518px centre crop
    errors       inference errors

`pipeline ready` and `camera status` are echoed as observations. They are not scored,
except `camera status`, which is the CAMERA requirement's evidence.

A capture that contains a synthetic-fixture marker (see SYNTHETIC_MARKERS) always fails
EVIDENCE, which is what guarantees no fixture under testdata/live-session-fixtures/ can
ever exit 0.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "testdata/live-session-fixtures"

# --- Acceptance thresholds.
#
# MIN_ELAPSED_S and MIN_DROPPED_FRACTION are the two that decide whether the drop
# policy is observable at all. LiveDepthPipeline logs only every 5th inference, so
# MIN_COMPLETED_INFERENCES cannot be below 5 without being meaningless, and two rate
# lines is the least that can show a counter advancing.
MIN_RATE_LINES = 2
MIN_COMPLETED_INFERENCES = 10
MIN_ELAPSED_S = 10.0
# A camera delivering ~20 offered fps against a ~0.5 Hz inference should be dropping
# the overwhelming majority. Anything under this means frames are being queued.
MIN_DROPPED_FRACTION = 0.50

# --- Tolerances. `fps` is printed to 2 dp and `elapsedS` to 1 dp, so the recomputation
# cannot be exact: reading back the rounded elapsed can move the rate, and the error
# grows with the rate. Hence an absolute floor plus a relative allowance.
FPS_ABS_TOLERANCE = 0.05
FPS_REL_TOLERANCE = 0.05

RATE_LINE = re.compile(r"depth rate\s+(?P<rest>\S.*)$")
KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)")
CAMERA_STATUS = re.compile(r"camera status=(\w+)")
PIPELINE_READY = re.compile(r"pipeline ready\s+(?P<rest>\S.*)$")

REQUIRED_RATE_FIELDS = ("inferences", "fps", "elapsedS", "offered", "droppedBusy", "tooSmall", "errors")

# Any of these anywhere in the capture means it is a fixture. Removing one is the only
# way to make a fixture pass, which is the intent.
SYNTHETIC_MARKERS = ("synthetic fixture", "synthetic logcat", "not a recorded device session")

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"

REQUIREMENT_ORDER = (
    "EVIDENCE", "UPDATES", "PROGRESS", "DROPS",
    "TOO_SMALL", "ERRORS", "SELF_CONSISTENCY", "SPAN", "CAMERA",
)


class CaptureError(Exception):
    """The capture cannot be read at all."""


@dataclass(frozen=True)
class RateSample:
    """One `depth rate` line, as logged."""

    inferences: int
    fps: float
    elapsed_s: float
    offered: int
    dropped_busy: int
    too_small: int
    errors: int

    @property
    def recomputed_fps(self) -> float:
        """What the rate *should* read if the counts and the elapsed time agree."""
        return self.inferences / self.elapsed_s if self.elapsed_s > 0 else math.inf

    @property
    def dropped_fraction(self) -> float | None:
        return (self.dropped_busy / self.offered) if self.offered > 0 else None


@dataclass
class Capture:
    """Everything parsed out of one logcat file. Parsed facts only; no verdicts."""

    path: str
    samples: list[RateSample] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    camera_statuses: list[str] = field(default_factory=list)
    pipeline_ready: str | None = None
    synthetic_marker: str | None = None
    line_count: int = 0

    @property
    def final(self) -> RateSample | None:
        return self.samples[-1] if self.samples else None

    def has_lines(self) -> bool:
        return bool(self.samples or self.malformed)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_rate_line(rest: str) -> RateSample | None:
    """Turn the tail of a `depth rate` line into a sample, or None if it is not one.

    Every required field must be present and numeric. A line missing one is not a rate
    report, and treating a half-printed line as data is how a torn logcat buffer turns
    into a fabricated measurement.
    """
    fields = dict(KEY_VALUE.findall(rest))
    if not all(name in fields for name in REQUIRED_RATE_FIELDS):
        return None
    try:
        return RateSample(
            inferences=int(fields["inferences"]),
            fps=float(fields["fps"]),
            elapsed_s=float(fields["elapsedS"]),
            offered=int(fields["offered"]),
            dropped_busy=int(fields["droppedBusy"]),
            too_small=int(fields["tooSmall"]),
            errors=int(fields["errors"]),
        )
    except ValueError:
        return None


def parse_capture(text: str, path: str = "<stdin>") -> Capture:
    """Parse raw logcat text. Never raises on content: a bad capture yields no samples."""
    lowered = text.lower()
    capture = Capture(path=path)
    capture.synthetic_marker = next((marker for marker in SYNTHETIC_MARKERS if marker in lowered), None)

    for line in text.splitlines():
        capture.line_count += 1
        stripped = line.strip()
        if not stripped:
            continue

        match = RATE_LINE.search(stripped)
        if match:
            sample = parse_rate_line(match.group("rest"))
            if sample is None:
                capture.malformed.append(_truncate(stripped))
            else:
                capture.samples.append(sample)
            continue

        camera = CAMERA_STATUS.search(stripped)
        if camera:
            capture.camera_statuses.append(camera.group(1))

        ready = PIPELINE_READY.search(stripped)
        if ready and capture.pipeline_ready is None:
            capture.pipeline_ready = _truncate(ready.group("rest"))
    return capture


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def read_capture(path: Path | None) -> Capture:
    """Read a capture from a file, or from stdin when `path` is None or `-`."""
    if path is None or str(path) == "-":
        text = sys.stdin.read()
        return parse_capture(text, "<stdin>")
    if not path.is_file():
        raise CaptureError(f"no logcat capture at {path}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CaptureError(f"cannot read {path}: {exc}") from exc
    return parse_capture(text, str(path))


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
# requirement checks
# --------------------------------------------------------------------------

def _no_evidence(requirement: str, capture: Capture) -> Verdict:
    """The requirement's own evidence is absent. Never a pass, never invented."""
    if capture.malformed:
        return Verdict(
            requirement, FAIL,
            "every `depth rate` line in this capture is unparseable",
            [f"{len(capture.malformed)} unparseable line(s), first: {capture.malformed[0]}"],
            {"rateLines": 0, "malformedLines": len(capture.malformed)},
        )
    return Verdict(
        requirement, FAIL,
        "no `depth rate` line to judge this on",
        ["this capture contains no depth rate report at all; nothing here shows the "
         "requirement was met, and absent evidence is not a slow session"],
        {"rateLines": 0, "malformedLines": 0},
    )


def check_evidence(capture: Capture) -> Verdict:
    if capture.synthetic_marker:
        return Verdict(
            "EVIDENCE", FAIL,
            "synthetic capture - no device claim may rest on it",
            [f"capture contains the marker {capture.synthetic_marker!r}; a fixture can prove a "
             "check fires and can never stand in for a device run"],
            {"synthetic": True, "rateLines": len(capture.samples)},
        )
    if not capture.samples:
        return _no_evidence("EVIDENCE", capture)

    detail = [f"{len(capture.samples)} `depth rate` line(s) parsed from {capture.path}"]
    if capture.malformed:
        # Skipped, never counted. Reported so a torn or interleaved log is visible.
        detail.append(f"{len(capture.malformed)} unparseable rate line(s) were skipped, first: "
                      f"{capture.malformed[0]}")
    return Verdict(
        "EVIDENCE", PASS,
        f"{len(capture.samples)} depth rate report(s) parsed",
        detail,
        {"synthetic": False, "rateLines": len(capture.samples),
         "malformedLines": len(capture.malformed)},
    )


def check_updates(capture: Capture) -> Verdict:
    final = capture.final
    if final is None:
        return _no_evidence("UPDATES", capture)

    total = final.inferences
    facts = {"completedInferences": total, "floor": MIN_COMPLETED_INFERENCES}
    if total < MIN_COMPLETED_INFERENCES:
        return Verdict(
            "UPDATES", FAIL,
            f"only {total} depth update(s) completed, below the {MIN_COMPLETED_INFERENCES} "
            "needed for a live view",
            ["a depth view that never updated is the frozen-mask bug this mode was rebuilt "
             "to fix; running without error is not a pass"],
            facts,
        )
    return Verdict(
        "UPDATES", PASS,
        f"{total} depth update(s) completed (>= {MIN_COMPLETED_INFERENCES} required)",
        [f"final line: inferences={total} over {final.elapsed_s:.1f} s"],
        facts,
    )


def check_progress(capture: Capture) -> Verdict:
    samples = capture.samples
    if not samples:
        return _no_evidence("PROGRESS", capture)

    counts = [s.inferences for s in samples]
    decreases = [
        (counts[i - 1], counts[i]) for i in range(1, len(counts)) if counts[i] < counts[i - 1]
    ]
    distinct = len(set(counts))
    facts = {
        "rateLines": len(counts),
        "minRateLines": MIN_RATE_LINES,
        "firstInferences": counts[0],
        "finalInferences": counts[-1],
        "distinctCounts": distinct,
    }
    detail = [f"inferences across {len(counts)} report(s): {counts[0]} -> {counts[-1]} "
              f"({distinct} distinct value(s))"]

    if len(counts) < MIN_RATE_LINES:
        detail.append(f"only {len(counts)} rate line(s): a single report cannot show the counter "
                      "advancing, and the pipeline logs one every 5 inferences, so this capture "
                      "was too short to judge")
        return Verdict(
            "PROGRESS", FAIL,
            f"only {len(counts)} rate line(s), below the {MIN_RATE_LINES} needed to show progress",
            detail, facts,
        )
    if decreases:
        for previous, current in decreases[:3]:
            detail.append(f"inferences went {previous} -> {current}: the completed counter "
                          "decreased, which cannot happen and means the log is not one session")
        return Verdict("PROGRESS", FAIL, f"{len(decreases)} decreasing inference count(s)",
                       detail, facts)
    if distinct < 2:
        detail.append(f"every report reads inferences={counts[0]}: the mask on screen was not "
                      "being replaced, which is the stale-mask failure itself")
        return Verdict(
            "PROGRESS", FAIL,
            f"the completed-inference counter never advanced (stuck at {counts[0]})",
            detail, facts,
        )
    return Verdict(
        "PROGRESS", PASS,
        f"inference counter advanced {counts[0]} -> {counts[-1]} over {len(counts)} report(s)",
        detail, facts,
    )


def check_drops(capture: Capture) -> Verdict:
    final = capture.final
    if final is None:
        return _no_evidence("DROPS", capture)

    fraction = final.dropped_fraction
    facts = {
        "offered": final.offered,
        "droppedBusy": final.dropped_busy,
        "inferences": final.inferences,
        "droppedFraction": fraction,
        "minDroppedFraction": MIN_DROPPED_FRACTION,
    }
    detail = [f"final line: offered={final.offered} droppedBusy={final.dropped_busy} "
              f"({(fraction or 0.0) * 100:.1f}% of offered frames were dropped as busy)"]
    if fraction is None:
        detail.append("offered=0: the analyzer handed the pipeline nothing at all")
    # Accounting, reported rather than scored: a large unaccounted remainder would mean
    # some other drop path exists that this log does not describe.
    accounted = final.inferences + final.dropped_busy + final.too_small
    unaccounted = final.offered - accounted
    detail.append(f"offered {final.offered} accounted as {accounted} "
                  f"(inferences + droppedBusy + tooSmall); {unaccounted} unaccounted")
    if fraction is not None:
        detail.append(f"dropped frames are NOT queued: at most one inference runs at a time, "
                      "so the view can never show depth for a frame the camera already dropped")

    if fraction is None or fraction < MIN_DROPPED_FRACTION:
        return Verdict(
            "DROPS", FAIL,
            f"only {(fraction or 0.0) * 100:.1f}% of offered frames were dropped as busy, below "
            f"the {MIN_DROPPED_FRACTION * 100:.0f}% the drop policy requires",
            detail + ["zero or near-zero drops at a sub-1 Hz inference rate means frames are "
                      "queueing unboundedly: that is the stale mask returning in a new shape"],
            facts,
        )
    return Verdict(
        "DROPS", PASS,
        f"{(fraction or 0.0) * 100:.1f}% of offered frames dropped as busy "
        f"(>= {MIN_DROPPED_FRACTION * 100:.0f}% required)",
        detail, facts,
    )


def check_too_small(capture: Capture) -> Verdict:
    final = capture.final
    if final is None:
        return _no_evidence("TOO_SMALL", capture)
    facts = {"tooSmall": final.too_small}
    if final.too_small > 0:
        return Verdict(
            "TOO_SMALL", FAIL,
            f"{final.too_small} frame(s) were discarded as too small for the 518px centre crop",
            [f"final line: tooSmall={final.too_small}; those frames never reached inference, so "
             "the analysis resolution or the crop maths is wrong"],
            facts,
        )
    return Verdict("TOO_SMALL", PASS, "no frame was discarded for being too small",
                   [f"final line: tooSmall={final.too_small}"], facts)


def check_errors(capture: Capture) -> Verdict:
    final = capture.final
    if final is None:
        return _no_evidence("ERRORS", capture)
    facts = {"errors": final.errors}
    if final.errors > 0:
        return Verdict(
            "ERRORS", FAIL,
            f"{final.errors} inference error(s) were counted",
            [f"final line: errors={final.errors}; each one is a frame update the viewer never "
             "received, and the errors are also logged as `depth inference failed`"],
            facts,
        )
    return Verdict("ERRORS", PASS, "no inference errors", [f"final line: errors={final.errors}"],
                   facts)


def check_self_consistency(capture: Capture) -> Verdict:
    samples = capture.samples
    if not samples:
        return _no_evidence("SELF_CONSISTENCY", capture)

    mismatches = []
    worst = 0.0
    for sample in samples:
        recomputed = sample.recomputed_fps
        reported = sample.fps
        if math.isnan(reported) or math.isnan(recomputed) or math.isinf(recomputed):
            mismatches.append((sample, reported, recomputed, 0.0))
            worst = math.inf
            continue
        allowed = FPS_ABS_TOLERANCE + FPS_REL_TOLERANCE * max(abs(reported), abs(recomputed))
        delta = abs(reported - recomputed)
        worst = max(worst, delta)
        if delta > allowed:
            mismatches.append((sample, reported, recomputed, delta))

    final = samples[-1]
    facts = {
        "reportedFps": final.fps,
        "recomputedFps": final.recomputed_fps,
        "inferences": final.inferences,
        "elapsedS": final.elapsed_s,
        "absTolerance": FPS_ABS_TOLERANCE,
        "relTolerance": FPS_REL_TOLERANCE,
        "mismatchedLines": len(mismatches),
    }
    detail = [f"final line recomputes to inferences/elapsedS = {final.inferences}/{final.elapsed_s:.1f} "
              f"= {final.recomputed_fps:.4f}/s against its own reported {final.fps:.2f}/s"]

    if mismatches:
        for sample, reported, recomputed, delta in mismatches[:3]:
            detail.append(
                f"inferences={sample.inferences} elapsedS={sample.elapsed_s:.1f}: reported "
                f"{reported} but recomputes to {recomputed:.4f} (off by {delta:.4f}/s)"
                + ("" if sample is final else " (intermediate line)")
            )
        detail.append("fps is printed to 2 dp and elapsedS to 1 dp, so small differences are "
                      "rounding; a disagreement this large is the log contradicting itself")
        return Verdict(
            "SELF_CONSISTENCY", FAIL,
            f"the reported rate disagrees with the log's own counts on {len(mismatches)} line(s)",
            detail, facts,
        )
    return Verdict(
        "SELF_CONSISTENCY", PASS,
        f"reported {final.fps:.2f}/s matches the recomputed {final.recomputed_fps:.4f}/s "
        f"(worst line off by {worst:.4f}/s)",
        detail, facts,
    )


def check_span(capture: Capture) -> Verdict:
    final = capture.final
    if final is None:
        return _no_evidence("SPAN", capture)

    facts = {"elapsedS": final.elapsed_s, "minElapsedS": MIN_ELAPSED_S,
             "completedInferences": final.inferences}
    detail = [f"final line: elapsedS={final.elapsed_s:.1f} genuinely measured seconds behind "
              f"{final.inferences} completed inference(s)"]
    if final.elapsed_s < MIN_ELAPSED_S:
        detail.append(f"{final.elapsed_s:.1f} s is below the {MIN_ELAPSED_S:.0f} s floor: a rate "
                      "averaged over this little elapsed time is a number with no measurement "
                      "behind it, and a burst of log lines can print any rate at all")
        return Verdict(
            "SPAN", FAIL,
            f"the rate is backed by only {final.elapsed_s:.1f} s of elapsed time, below the "
            f"{MIN_ELAPSED_S:.0f} s floor",
            detail, facts,
        )
    return Verdict(
        "SPAN", PASS,
        f"{final.elapsed_s:.1f} s of genuinely measured elapsed time behind the reported rate",
        detail, facts,
    )


def check_camera(capture: Capture) -> Verdict:
    statuses = capture.camera_statuses
    facts = {"statuses": statuses, "streaming": "Streaming" in statuses}
    if not statuses:
        # The evidence for this requirement is absent. That is not a pass and not a
        # contradiction: the capture says nothing at all about the camera, most likely
        # because it was filtered to a tag or level that dropped the line.
        return Verdict(
            "CAMERA", BLOCKED,
            "the camera's state is UNKNOWN - no `camera status=` line is in this capture",
            ["capture the whole screen without a tag filter, or at minimum with DepthQNN:I, "
             "so the camera status line is included; a black preview and a dead camera look "
             "identical on screen"],
            facts,
        )
    if "Streaming" not in statuses:
        detail = [f"camera status reported: {', '.join(statuses)}"]
        if "Failed" in statuses:
            detail.append("the camera reported a failure; the overlay is hidden on a non-streaming "
                          "status, so this capture cannot be a live depth view")
        return Verdict(
            "CAMERA", FAIL,
            f"the camera never reached Streaming (saw: {', '.join(sorted(set(statuses)))})",
            detail, facts,
        )
    detail = [f"camera status reported: {', '.join(statuses)}"]
    if "Failed" in statuses:
        detail.append("note: a failure also appears in this capture, so the session recovered "
                      "rather than streaming throughout")
    return Verdict("CAMERA", PASS, "the camera reached Streaming", detail, facts)


# --------------------------------------------------------------------------
# observation echo
# --------------------------------------------------------------------------

def observe(capture: Capture) -> list[str]:
    """Echo what the capture recorded. Nothing here is scored."""
    lines = [f"capture: {capture.path} ({capture.line_count} line(s), "
             f"{len(capture.samples)} depth rate report(s) parsed"
             + (f", {len(capture.malformed)} unparseable" if capture.malformed else "") + ")"]

    if capture.pipeline_ready:
        lines.append(f"pipeline ready (echoed, NOT scored): {capture.pipeline_ready}")
    lines.append("backend note: the pipeline backend above is echoed, not scored. "
                 "This checker judges liveness, not placement; backend placement is out of scope "
                 "for this tool.")

    final = capture.final
    if final is None:
        lines.append("depth rate: no report parsed, so there is nothing to echo")
        return lines

    accounted = final.inferences + final.dropped_busy + final.too_small
    lines.append(
        f"final depth rate: inferences={final.inferences} fps={final.fps:.2f} "
        f"elapsedS={final.elapsed_s:.1f} offered={final.offered} "
        f"droppedBusy={final.dropped_busy} tooSmall={final.too_small} errors={final.errors}"
    )
    lines.append(
        f"frame accounting: offered {final.offered} = inferences {final.inferences} + "
        f"droppedBusy {final.dropped_busy} + tooSmall {final.too_small} + "
        f"{final.offered - accounted} unaccounted"
    )
    if final.offered > 0:
        lines.append(
            f"offer rate ~{final.offered / final.elapsed_s:.1f} frame(s)/s against a measured "
            f"{final.recomputed_fps:.2f} depth update(s)/s: a gap this wide is only safe because "
            "frames are dropped, never queued"
        )
    lines.append("depth units: relative disparity-like output; no verdict here is a metric depth "
                 "value")
    return lines


def summarise(capture: Capture) -> tuple[list[Verdict], str]:
    checks = {
        "EVIDENCE": check_evidence,
        "UPDATES": check_updates,
        "PROGRESS": check_progress,
        "DROPS": check_drops,
        "TOO_SMALL": check_too_small,
        "ERRORS": check_errors,
        "SELF_CONSISTENCY": check_self_consistency,
        "SPAN": check_span,
        "CAMERA": check_camera,
    }
    verdicts = [checks[name](capture) for name in REQUIREMENT_ORDER]
    statuses = {v.status for v in verdicts}
    overall = FAIL if FAIL in statuses else (BLOCKED if BLOCKED in statuses else PASS)
    return verdicts, overall


def evaluate(capture: Capture) -> tuple[dict, list[Verdict], str]:
    verdicts, overall = summarise(capture)
    report = {
        "capture": capture.path,
        "synthetic": capture.synthetic_marker is not None,
        "rateLines": len(capture.samples),
        "malformedLines": len(capture.malformed),
        "overall": overall,
        "requirements": {v.requirement: v.as_dict() for v in verdicts},
        "observations": observe(capture),
    }
    return report, verdicts, overall


def render_text(report: dict, verdicts: list[Verdict]) -> str:
    lines = ["=" * 78, "Live depth session acceptance report", "=" * 78]
    lines.append(f"capture    : {report['capture']}")
    lines.append(f"rate lines : {report['rateLines']} parsed, "
                 f"{report['malformedLines']} unparseable")
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
    parser.add_argument("--log", type=Path, default=None,
                        help="captured logcat file, or '-' / omitted to read stdin")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--self-test", action="store_true",
                        help="run the acceptance checker's own tests, including its negative controls")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    try:
        capture = read_capture(args.log)
    except CaptureError as exc:
        print(f"CANNOT EVALUATE: {exc}", file=sys.stderr)
        return 3

    report, verdicts, overall = evaluate(capture)
    if args.json:
        # No `default=`: a verdict must serialise as data, or a stringified one would be
        # indistinguishable from a real result.
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report, verdicts))
    return {PASS: 0, FAIL: 1, BLOCKED: 2}[overall]


# --------------------------------------------------------------------------
# tests -- the negative controls are the point of this file
# --------------------------------------------------------------------------

FIXTURES = {
    "pass": "pass-cpu-live",
    "norate": "no-rate-lines",
    "frozen": "frozen-counter",
    "nodrops": "no-drops-queueing",
    "toosmall": "too-small-frames",
    "errors": "inference-errors",
    "mismatch": "fps-mismatch",
    "burst": "rate-burst-no-elapsed",
    "camerafail": "camera-never-streamed",
    "cameramissing": "camera-status-absent",
}

# Fixtures whose whole point is one requirement, and which must leave every other
# requirement PASS. EVIDENCE is excluded: it fails on every fixture by design, because
# they are synthetic and a synthetic capture may never report an overall pass.
ISOLATED_FAILURES = {
    "frozen": "PROGRESS",
    "nodrops": "DROPS",
    "toosmall": "TOO_SMALL",
    "errors": "ERRORS",
    "mismatch": "SELF_CONSISTENCY",
    "burst": "SPAN",
    "camerafail": "CAMERA",
    "cameramissing": "CAMERA",
}


def fixture(name: str) -> Path:
    return FIXTURE_ROOT / FIXTURES[name] / "live-session.logcat"


def de_synthetic(name: str) -> Capture:
    """A fixture's log lines with its synthetic header stripped.

    Every fixture header line begins with `#`, so dropping those leaves exactly the
    logcat body. This is how a test proves a check fires on the *evidence* rather than
    on the marker, and how a BLOCKED verdict is shown to propagate to OVERALL once
    EVIDENCE is satisfied. The fixture files on disk are never modified.
    """
    path = fixture(name)
    body = "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    return parse_capture(body + "\n", str(path))


class AcceptanceCheckerTests(unittest.TestCase):
    """Every fixture here is synthetic and is labelled as such inside the file.

    They prove the *checks* fire on the right evidence. None of them, and none of the
    files in testdata/live-session-fixtures/, can ever produce a device pass: the
    EVIDENCE requirement rejects all of them.
    """

    def run_fixture(self, name: str) -> tuple[dict, dict]:
        report, verdicts, _ = evaluate(read_capture(fixture(name)))
        return report, {v.requirement: v for v in verdicts}

    def require(self, name: str, requirement: str, status: str) -> Verdict:
        _, verdicts = self.run_fixture(name)
        self.assertIn(requirement, verdicts)
        self.assertEqual(verdicts[requirement].status, status,
                         f"{name}/{requirement}: expected {status}, got "
                         f"{verdicts[requirement].status} ({verdicts[requirement].summary})")
        return verdicts[requirement]

    def detail_text(self, verdict: Verdict) -> str:
        return " | ".join(verdict.details)

    def test_parses_the_real_logcat_shape(self):
        """The positive control: the parse must survive logcat prefixes and noise."""
        report, verdicts = self.run_fixture("pass")
        statuses = verdicts["CAMERA"].facts["statuses"]
        self.assertIn("Streaming", statuses)
        self.assertGreaterEqual(report["rateLines"], 4)
        for requirement in ("UPDATES", "PROGRESS", "DROPS", "TOO_SMALL", "ERRORS",
                            "SELF_CONSISTENCY", "SPAN", "CAMERA"):
            self.assertEqual(verdicts[requirement].status, PASS,
                             f"{requirement}: {verdicts[requirement].summary} "
                             f"{verdicts[requirement].details}")
        # Without a positive control every other test here would pass trivially.
        self.assertEqual(report["overall"], FAIL,
                         "a synthetic fixture must never report an overall pass")

    def test_rejects_capture_with_no_rate_lines(self):
        """A log with no depth rate report has shown nothing about depth."""
        # On the fixture as committed the synthetic marker is the headline; the header
        # must be stripped to see the behaviour the fixture exists to prove.
        capture = de_synthetic("norate")
        self.assertEqual(capture.samples, [])
        verdicts = {v.requirement: v for v in summarise(capture)[0]}
        self.assertEqual(verdicts["EVIDENCE"].status, FAIL)
        self.assertIn("absent evidence is not a slow session", self.detail_text(verdicts["EVIDENCE"]))
        # Every requirement that depends on the rate lines must fail rather than pass.
        for requirement in ("UPDATES", "PROGRESS", "DROPS", "TOO_SMALL", "ERRORS",
                            "SELF_CONSISTENCY", "SPAN"):
            self.assertEqual(verdicts[requirement].status, FAIL,
                             f"{requirement} passed on a capture with no rate lines")

    def test_rejects_a_counter_that_never_advances(self):
        """A frozen counter is the stale-mask bug, measured directly."""
        verdict = self.require("frozen", "PROGRESS", FAIL)
        self.assertIn("stale-mask failure", self.detail_text(verdict))
        self.assertEqual(verdict.facts["distinctCounts"], 1)
        # Independence: the frozen count of 15 still clears the UPDATES floor, so only
        # PROGRESS catches this. A checker that merged the two would call it "mostly fine".
        self.assertEqual(self.run_fixture("frozen")[1]["UPDATES"].status, PASS)

    def test_rejects_queued_frames(self):
        """Zero drops at a sub-1 Hz rate is unbounded queueing, not a healthy view."""
        verdict = self.require("nodrops", "DROPS", FAIL)
        self.assertIn("queueing unboundedly", self.detail_text(verdict))
        self.assertEqual(verdict.facts["droppedBusy"], 0)
        # The view is otherwise fine, so the drop policy is the only thing that fails.
        self.assertEqual(self.run_fixture("nodrops")[1]["SELF_CONSISTENCY"].status, PASS)

    def test_rejects_discarded_frames(self):
        self.assertIn("too small for the 518px centre crop",
                      self.require("toosmall", "TOO_SMALL", FAIL).summary)

    def test_rejects_inference_errors(self):
        self.assertIn("inference error", self.require("errors", "ERRORS", FAIL).summary)

    def test_rejects_a_self_inconsistent_rate(self):
        """A log that disagrees with itself is not evidence of anything."""
        verdict = self.require("mismatch", "SELF_CONSISTENCY", FAIL)
        self.assertGreater(verdict.facts["mismatchedLines"], 0)
        self.assertIn("contradicting itself", self.detail_text(verdict))
        # The reported figure was a comfortable-looking one, which is the whole trap.
        self.assertGreater(self.run_fixture("mismatch")[1]["SELF_CONSISTENCY"].facts["reportedFps"],
                           self.run_fixture("mismatch")[1]["SELF_CONSISTENCY"].facts["recomputedFps"])

    def test_rejects_a_burst_with_no_elapsed_time(self):
        """A high rate with no time behind it is a number, not a measurement."""
        verdict = self.require("burst", "SPAN", FAIL)
        self.assertLess(verdict.facts["elapsedS"], MIN_ELAPSED_S)
        self.assertIn("burst of log lines", self.detail_text(verdict))
        # The trap only works if the other checks are fooled, so prove they were not.
        others = self.run_fixture("burst")[1]
        self.assertEqual(others["SELF_CONSISTENCY"].status, PASS)
        self.assertGreater(others["UPDATES"].facts["completedInferences"], MIN_COMPLETED_INFERENCES)
        self.assertGreaterEqual(others["DROPS"].facts["droppedFraction"], MIN_DROPPED_FRACTION)

    def test_rejects_a_camera_that_never_streamed(self):
        verdict = self.require("camerafail", "CAMERA", FAIL)
        self.assertIn("Failed", verdict.facts["statuses"])
        self.assertIn("cannot be a live depth view", self.detail_text(verdict))

    def test_absent_camera_evidence_blocks_rather_than_passes(self):
        """No camera line means the camera's state is UNKNOWN, not good."""
        self.assertEqual(self.require("cameramissing", "CAMERA", BLOCKED).facts["statuses"], [])
        # The marker is stripped so BLOCKED, not the synthetic EVIDENCE failure, is what
        # drives the overall verdict -- otherwise this test would prove nothing.
        report, _, overall = evaluate(de_synthetic("cameramissing"))
        self.assertEqual(report["requirements"]["CAMERA"]["status"], BLOCKED)
        self.assertGreater(report["rateLines"], 0, "this capture's rate lines must be healthy, "
                                                    "so only the camera evidence is missing")
        self.assertEqual(overall, BLOCKED, "a blocked requirement must block the overall verdict")

    def test_each_negative_fixture_fails_exactly_its_own_check(self):
        """One defect, one failure: otherwise a check could pass for the wrong reason."""
        for name, requirement in ISOLATED_FAILURES.items():
            expected = BLOCKED if name == "cameramissing" else FAIL
            with self.subTest(fixture=name):
                verdicts = self.run_fixture(name)[1]
                self.assertEqual(verdicts[requirement].status, expected,
                                 f"{name}/{requirement}: {verdicts[requirement].summary}")
                for other in REQUIREMENT_ORDER:
                    if other in ("EVIDENCE", requirement):
                        continue
                    self.assertEqual(verdicts[other].status, PASS,
                                     f"{name}: {other} also failed ({verdicts[other].summary}); "
                                     f"this fixture is meant to isolate {requirement}")

    def test_a_real_shaped_capture_can_pass_but_a_fixture_never_does(self):
        """The checks must be satisfiable, or a passing run means nothing."""
        text = "\n".join([
            "10-19 01:02:03.100  4242  4301 I DepthQNN: camera status=Streaming(description=back "
            "camera bound (implementation=CAMERA2, ...))",
            "10-19 01:02:04.000  4242  4301 I DepthQNN: pipeline ready session=graph backend=CPU "
            "note=CPU — QNN could not initialise, see HANDOVER 2.4",
        ] + [
            f"10-19 01:02:{10 + index:02d}.000  4242  4402 I DepthQNN: depth rate "
            f"inferences={count} fps={count / (10.0 * index):.2f} elapsedS={10.0 * index:.1f} "
            f"offered={count * 20} droppedBusy={count * 20 - count} tooSmall=0 errors=0"
            for index, count in enumerate((5, 10, 15, 20), start=1)
        ]) + "\n"
        capture = parse_capture(text, "<inline>")
        self.assertIsNone(capture.synthetic_marker)
        _, _, overall = evaluate(capture)
        self.assertEqual(overall, PASS,
                         "a genuine capture that satisfies every requirement must pass, or the "
                         "checker is unfalsifiable")

        report, verdicts, _ = evaluate(read_capture(fixture("pass")))
        self.assertEqual(report["overall"], FAIL, "a synthetic fixture must never report a pass")
        self.assertEqual({v.requirement: v for v in verdicts}["EVIDENCE"].facts["synthetic"], True)

    def test_cli_exit_codes_and_no_synthetic_pass(self):
        import subprocess
        script = str(REPO_ROOT / "scripts/evaluate-live-session.py")
        for name in FIXTURES.values():
            completed = subprocess.run(
                [sys.executable, script, "--log", str(FIXTURE_ROOT / name / "live-session.logcat")],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            self.assertIn(completed.returncode, (1, 2),
                          f"{name} returned {completed.returncode}: no synthetic fixture may exit 0")
            self.assertIn("OVERALL:", completed.stdout)

        missing = subprocess.run(
            [sys.executable, script, "--log", str(FIXTURE_ROOT / "does-not-exist")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        self.assertEqual(missing.returncode, 3)
        self.assertIn("CANNOT EVALUATE", missing.stderr)

    def test_reads_stdin(self):
        import subprocess
        script = str(REPO_ROOT / "scripts/evaluate-live-session.py")
        completed = subprocess.run(
            [sys.executable, script],
            input=fixture("pass").read_text(), capture_output=True, text=True, cwd=REPO_ROOT,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("OVERALL:", completed.stdout)
        self.assertIn("<stdin>", completed.stdout)

    def test_json_report_is_machine_readable(self):
        report, _, _ = evaluate(read_capture(fixture("pass")))
        blob = json.dumps(report, indent=2)  # no default= escape hatch
        restored = json.loads(blob)
        self.assertIn(restored["overall"], (PASS, FAIL, BLOCKED))
        self.assertEqual(list(restored["requirements"]), list(REQUIREMENT_ORDER))
        for requirement in REQUIREMENT_ORDER:
            entry = restored["requirements"][requirement]
            self.assertIn(entry["status"], (PASS, FAIL, BLOCKED), requirement)
            self.assertTrue(entry["details"], f"{requirement} reported no details to justify itself")

    def test_never_emits_a_metric_depth(self):
        """Output is relative depth. No verdict, fact or observation may carry metres."""
        banned = ("metric_depth", "depthMetres", "depth_m", "meters", "metres")
        for name in FIXTURES.values():
            report, _, _ = evaluate(read_capture(FIXTURE_ROOT / name / "live-session.logcat"))
            blob = json.dumps(report)
            for token in banned:
                self.assertNotIn(token, blob, f"{name} leaked a metric-depth token {token!r}")

    def test_echoes_the_recorded_evidence(self):
        """A verdict is worth only as much as the evidence it was measured from."""
        report, _, _ = evaluate(read_capture(fixture("pass")))
        joined = "\n".join(report["observations"])
        for expected in ("pipeline ready (echoed, NOT scored)", "backend note:", "frame accounting:",
                         "final depth rate:", "no verdict here is a metric depth value"):
            self.assertIn(expected, joined, f"missing echoed evidence: {expected}")


def run_self_test() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(AcceptanceCheckerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
