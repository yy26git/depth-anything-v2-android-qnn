#!/usr/bin/env python3
"""
Numerical regression for the Depth Anything V2 Small w8a16 conversion.

The shipped asset is a converted graph; the Qualcomm AI Hub export is the reference. A conversion
is only acceptable if the two agree on the output a caller actually observes, so that is what is
measured here, on CPU, for every fixture in the manifest:

  * the reference runs on its own uint16 boundary,
  * the candidate runs on the float boundary the conversion created,
  * both are fed the *same* input value: a uint16 code sequence pushed through the reference's
    own input scale/zero-point, which is exactly the real number the reference's boundary
    DequantizeLinear would compute, and which the candidate accepts directly,
  * the candidate's float output is requantized with the reference's output scale/zero-point and
    must land on the reference's output codes exactly.

Requantizing rather than comparing floats is the point. The reference's float arithmetic is not
bit-identical to the candidate's — the reference rounds through a QDQ pair at every activation —
so a float comparison measures the quantizer's rounding, not the conversion. The output codes
are what the caller receives, and they are comparable exactly.

Run:
    python3 scripts/verify-model-parity.py --manifest testdata/depth-parity/manifest.json
    python3 scripts/verify-model-parity.py --manifest ... --write-manifest   # re-hash and rewrite

Exit status is 0 only when every fixture matches and every file hash in the manifest is current.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper
from PIL import Image

SCHEMA = "depth-parity/1"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "testdata/depth-parity/manifest.json"

# Preprocessing is RGB/255 for BOTH exports. The graph normalizes internally: immediately after
# the input boundary it applies Sub(view_updated) then Div(view_1_updated), the ImageNet mean and
# standard deviation. A caller that pre-normalizes would have the normalization applied twice.
# Fixed here so a change to either side of the boundary shows up as a parity failure instead of
# silently producing a plausible but wrong depth map.


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Quantization:
    """The export's boundary quantization, read from the graph rather than assumed."""

    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int

    def encode_input(self, values: np.ndarray) -> np.ndarray:
        """
        Exactly-representable uint16 codes for a float image in [0,1].

        The division and rounding are done in float32, matching the precision ONNX Runtime's
        QuantizeLinear kernel computes them in. Doing this arithmetic in float64 is *more*
        accurate and is still wrong here: at a value that lands on a rounding tie the two
        precisions disagree by one code, and the reference's own output codes then appear to
        differ from an identical graph.
        """
        codes = np.rint(values.astype(np.float32) / np.float32(self.input_scale)) + self.input_zero_point
        return np.clip(codes, 0, 65535).astype(np.uint16)

    def decode_input(self, codes: np.ndarray) -> np.ndarray:
        """The float value the reference's boundary DequantizeLinear computes from `codes`."""
        return ((codes.astype(np.float32) - np.float32(self.input_zero_point)) * np.float32(self.input_scale))

    def encode_output(self, values: np.ndarray) -> np.ndarray:
        """float32 for the same reason as encode_input: this mirrors the QDQ kernel, not a math ideal."""
        codes = np.rint(values.astype(np.float32) / np.float32(self.output_scale)) + self.output_zero_point
        return np.clip(codes, 0, 65535).astype(np.uint16)


def read_quantization(model_path: Path) -> Quantization:
    graph = onnx.load(str(model_path)).graph
    arrays = {initializer.name: numpy_helper.to_array(initializer) for initializer in graph.initializer}
    return Quantization(
        input_scale=float(arrays["image_scale"]),
        input_zero_point=int(arrays["image_zero_point"]),
        output_scale=float(arrays["depth_estimates_scale"]),
        output_zero_point=int(arrays["view_zero_point"]),
    )


def session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return options


def load_fixture(path: Path, height: int, width: int) -> np.ndarray:
    """RGB/255 NCHW, batch 1, resized to the graph's fixed input with bilinear filtering."""
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((width, height), Image.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


@dataclass
class FixtureResult:
    name: str
    max_code_difference: int
    differing_codes: int
    mean_abs_float_difference: float
    quant_step: float
    exact: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "max_code_difference": self.max_code_difference,
            "differing_codes": self.differing_codes,
            "mean_abs_float_difference": self.mean_abs_float_difference,
            "quant_step": self.quant_step,
            "exact": self.exact,
        }


class ParityComparator:
    """
    Runs a reference and a candidate graph on identical inputs and compares their output codes.

    Both sessions are created once and reused across fixtures: loading a 99 MB graph costs more
    than running it, and a per-fixture session would make a sweep unusable.
    """

    def __init__(self, reference_path: Path, candidate_path: Path, threads: int = 4, reference_session=None):
        options = session_options()
        options.intra_op_num_threads = threads
        # A caller comparing several candidates against one reference pays for the reference
        # graph once; loading 99 MB costs more than running it.
        self.reference = reference_session or ort.InferenceSession(
            str(reference_path), options, providers=["CPUExecutionProvider"]
        )
        self.candidate = ort.InferenceSession(str(candidate_path), options, providers=["CPUExecutionProvider"])
        self.quantization = read_quantization(reference_path)
        self.reference_input = self.reference.get_inputs()[0].name
        self.reference_output = self.reference.get_outputs()[0].name
        self.candidate_input = self.candidate.get_inputs()[0].name
        self.candidate_output = self.candidate.get_outputs()[0].name
        self.height = int(self.reference.get_inputs()[0].shape[2])
        self.width = int(self.reference.get_inputs()[0].shape[3])

    def run(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(reference output codes, requantized candidate output codes) for one float image."""
        codes = self.quantization.encode_input(image)
        exact_input = self.quantization.decode_input(codes)
        reference_codes = self.reference.run(
            None, {self.reference_input: codes}
        )[0]
        integer_boundary = self.candidate.get_inputs()[0].type == "tensor(uint16)"
        candidate = self.candidate.run(
            None, {self.candidate_input: codes if integer_boundary else exact_input}
        )[0]
        return reference_codes, candidate if integer_boundary else self.quantization.encode_output(candidate)

    def candidate_float(self, image: np.ndarray) -> np.ndarray:
        """The candidate's raw float output, for comparing one candidate against another."""
        exact_input = self.quantization.decode_input(self.quantization.encode_input(image))
        if self.candidate.get_inputs()[0].type == "tensor(uint16)":
            output = self.candidate.run(None, {self.candidate_input: self.quantization.encode_input(image)})[0]
            return (output.astype(np.float32) - self.quantization.output_zero_point) * self.quantization.output_scale
        return self.candidate.run(None, {self.candidate_input: exact_input})[0]

    def compare(self, name: str, image: np.ndarray) -> FixtureResult:
        reference_codes, candidate_codes = self.run(image)
        if reference_codes.shape != candidate_codes.shape:
            raise SystemExit(
                f"{name}: output shape differs, reference {reference_codes.shape} "
                f"candidate {candidate_codes.shape}"
            )
        difference = np.abs(reference_codes.astype(np.int64) - candidate_codes.astype(np.int64))
        dequantized_difference = difference.astype(np.float64) * self.quantization.output_scale
        return FixtureResult(
            name=name,
            max_code_difference=int(difference.max()),
            differing_codes=int((difference != 0).sum()),
            mean_abs_float_difference=float(dequantized_difference.mean()),
            quant_step=self.quantization.output_scale,
            exact=bool(difference.max() == 0),
        )


def resolve(manifest_path: Path, manifest: dict) -> dict:
    """Absolute paths for everything the manifest names, relative to the manifest's own directory."""
    root = manifest_path.parent
    resolved = {
        "reference_onnx": (root / manifest["reference"]["onnx"]).resolve(),
        "reference_data": (root / manifest["reference"]["external_data"]).resolve(),
        "candidate_onnx": (root / manifest["candidate"]["onnx"]).resolve(),
        "fixtures": [
            {**fixture, "resolved": (root / fixture["path"]).resolve()} for fixture in manifest["fixtures"]
        ],
    }
    return resolved


def check_hashes(resolved: dict, manifest: dict) -> list[str]:
    """Every hash the manifest claims, re-verified. A stale manifest is a failure, not a warning."""
    problems = []
    pairs = [
        ("reference.onnx", resolved["reference_onnx"], manifest["reference"]["onnx_sha256"]),
        ("reference.external_data", resolved["reference_data"], manifest["reference"]["external_data_sha256"]),
        ("candidate.onnx", resolved["candidate_onnx"], manifest["candidate"]["sha256"]),
    ]
    pairs += [(f"fixture {f['name']}", f["resolved"], f["sha256"]) for f in resolved["fixtures"]]
    for label, path, expected in pairs:
        if not path.exists():
            problems.append(f"{label}: missing file {path}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{label}: sha256 {actual} != manifest {expected}")
    return problems


def verify(manifest_path: Path, fixture_filter: list[str] | None = None, threads: int = 4) -> tuple[bool, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise SystemExit(f"manifest schema {manifest.get('schema')!r} is not {SCHEMA!r}")

    resolved = resolve(manifest_path, manifest)
    problems = check_hashes(resolved, manifest)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return False, []

    fixtures = resolved["fixtures"]
    if fixture_filter:
        wanted = set(fixture_filter)
        fixtures = [fixture for fixture in fixtures if fixture["name"] in wanted]
        missing = wanted - {fixture["name"] for fixture in fixtures}
        if missing:
            raise SystemExit(f"no such fixture(s) in the manifest: {sorted(missing)}")
    if not fixtures:
        raise SystemExit("manifest lists no fixtures")

    comparator = ParityComparator(resolved["reference_onnx"], resolved["candidate_onnx"], threads=threads)
    print(
        f"reference {resolved['reference_onnx'].name} input={comparator.reference_input}"
        f"{comparator.reference.get_inputs()[0].shape} output={comparator.reference_output}"
        f"\ncandidate {resolved['candidate_onnx'].name} input={comparator.candidate_input}"
        f" output={comparator.candidate_output}"
        f"\noutput quant step {comparator.quantization.output_scale:.6e}"
        f"  input quant step {comparator.quantization.input_scale:.6e}\n"
    )

    results, ok = [], True
    for fixture in fixtures:
        result = comparator.compare(fixture["name"], load_fixture(fixture["resolved"], comparator.height, comparator.width))
        results.append(result.as_dict())
        ok = ok and result.exact
        verdict = "exact" if result.exact else "MISMATCH"
        print(
            f"  {result.name:48s} max_code_diff={result.max_code_difference:<6d} "
            f"differing={result.differing_codes:<10d} mean|Δfloat|={result.mean_abs_float_difference:.3e}  {verdict}"
        )
    print(f"\n{sum(1 for r in results if r['exact'])}/{len(results)} fixtures exact")
    return ok, results


def describe(model_path: Path) -> dict:
    model = onnx.load(str(model_path))
    graph = model.graph
    return {
        "graph_input": {
            "name": graph.input[0].name,
            "elem_type": TensorProto.DataType.Name(graph.input[0].type.tensor_type.elem_type),
            "shape": [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim],
        },
        "graph_output": {
            "name": graph.output[0].name,
            "elem_type": TensorProto.DataType.Name(graph.output[0].type.tensor_type.elem_type),
            "shape": [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim],
        },
        "node_count": len(graph.node),
        "op_histogram": {
            op: sum(1 for node in graph.node if node.op_type == op)
            for op in sorted({node.op_type for node in graph.node})
        },
    }


def write_manifest(manifest_path: Path, source_origin: dict) -> None:
    root = manifest_path.parent
    reference_onnx = root / "source/depth_anything_v2.onnx"
    reference_data = root / "source/depth_anything_v2.data"
    candidate_onnx = (REPO_ROOT / "app/src/main/assets/depth_anything_v2.onnx").resolve()
    quantization = read_quantization(reference_onnx)

    fixtures = []
    for path in sorted((root / "fixtures").iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        entry = {"name": path.name, "path": f"fixtures/{path.name}", "sha256": sha256_file(path)}
        entry.update(source_origin.get(path.name, {}))
        fixtures.append(entry)

    manifest = {
        "schema": SCHEMA,
        "note": (
            "Hashes cover the .onnx protobuf AND the external .data weights separately; hashing the "
            "protobuf alone cannot detect a change to weights stored outside it. Paths are relative "
            "to this file."
        ),
        "preprocessing": {
            "scaling": "RGB / 255",
            "reason": (
                "Both Qualcomm AI Hub exports normalize internally: immediately after the input "
                "boundary the w8a16 graph applies Sub then Div with the ImageNet mean and std "
                "(node_sub / node_div). Pre-normalizing externally would apply it twice."
            ),
            "resize": {"width": 518, "height": 518, "resample": "bilinear", "color": "RGB"},
        },
        "reference": {
            "onnx": "source/depth_anything_v2.onnx",
            "external_data": "source/depth_anything_v2.data",
            "onnx_sha256": sha256_file(reference_onnx),
            "external_data_sha256": sha256_file(reference_data),
            "provenance": source_origin.get("__source__", {}),
            "quantization": {
                "image_scale": quantization.input_scale,
                "image_zero_point": quantization.input_zero_point,
                "depth_estimates_scale": quantization.output_scale,
                "view_zero_point": quantization.output_zero_point,
            },
            **describe(reference_onnx),
        },
        "candidate": {
            "onnx": os.path.relpath(candidate_onnx, root),
            "sha256": sha256_file(candidate_onnx),
            "converter": "scripts/make-w8a16-htp-model.py",
            "converter_options": "defaults (boundary-only float I/O; no Resize or Conv rewrite)",
            **describe(candidate_onnx),
        },
        "criterion": {
            "exact_output_codes": True,
            "rationale": (
                "The candidate's float output is requantized with the reference's own output scale "
                "and zero-point, then compared to the reference's uint16 output codes. Exact codes "
                "are required because the codes are the observable output of the reference model."
            ),
            "optional_rewrite_atol": 1e-5,
            "optional_rewrite_rtol": 1e-5,
        },
        "fixtures": fixtures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path} ({len(fixtures)} fixtures)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fixture", action="append", help="limit to these fixture names")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--write-manifest", action="store_true", help="re-hash and rewrite the manifest")
    args = parser.parse_args(argv)

    if args.write_manifest:
        origin = {}
        origin_path = args.manifest.parent / "fixtures/PROVENANCE.json"
        if origin_path.exists():
            origin = json.loads(origin_path.read_text())
        write_manifest(args.manifest, origin)
        return 0

    ok, _ = verify(args.manifest, fixture_filter=args.fixture, threads=args.threads)
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
