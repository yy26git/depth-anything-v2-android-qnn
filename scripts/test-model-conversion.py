#!/usr/bin/env python3
"""
Tests for the Depth Anything V2 w8a16 conversion. The first four are negative controls: they
exist because each one of them *fails* against the converter that shipped, and they are only
worth anything as long as that stays true.

    python3 scripts/test-model-conversion.py
    python3 scripts/test-model-conversion.py --converter testdata/depth-parity/known-bad/make-w8a16-htp-model.py

The second form is the proof. `testdata/depth-parity/known-bad/` holds a verbatim copy of the
converter that produced the defective asset: it read the STRING `mode` attribute of the five
Resize nodes through `.i` (which is 0, the `nearest` enum) and replaced bilinear align-corners
upsampling with nearest-neighbour Gather, and it resolved QDQ weights by dequantizing a float
source that had never been quantized, making the head MatMul ~312x too small. Running this suite
against it must fail on `test_resize_attributes_reject_unsupported_rewrite`,
`test_qdq_weight_resolution`, `test_parity_rejects_changed_resize` and
`test_parity_rejects_scaled_weights`.

Expected constants in `test_qdq_weight_resolution` are taken from the independent audit in
outputs/conversion-audit-evidence.json, not from the implementation under test.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parent.parent
PARITY_ROOT = REPO_ROOT / "testdata/depth-parity"
SOURCE_ONNX = PARITY_ROOT / "source/depth_anything_v2.onnx"
KNOWN_BAD = PARITY_ROOT / "known-bad/make-w8a16-htp-model.py"
MANIFEST = PARITY_ROOT / "manifest.json"

# One fixture is enough for a unit test: the sweep over all of them is
# scripts/verify-model-parity.py, and this suite is about detecting a broken conversion, which
# one aerial image shows as clearly as eight.
PROBE_FIXTURE = "test_aerial_20m.jpg"

converter_path = REPO_ROOT / "scripts/make-w8a16-htp-model.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("make_w8a16_htp_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parity_module():
    spec = importlib.util.spec_from_file_location("verify_model_parity", REPO_ROOT / "scripts/verify-model-parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildCache:
    """Builds and reference sessions, shared across tests: each is expensive and immutable."""

    def __init__(self):
        self.converter = load_module(converter_path)
        self.parity = parity_module()
        self.tmp = tempfile.TemporaryDirectory(prefix="depth-conversion-tests-")
        self._builds: dict[tuple, Path] = {}
        self._reference_session = None

    def build(self, key: str, source: Path, converter=None, **flags) -> Path:
        converter = converter or self.converter
        if (key, source) not in self._builds:
            destination = Path(self.tmp.name) / f"{key}.onnx"
            converter.build(str(source), str(destination), **flags)
            self._builds[(key, source)] = destination
        return self._builds[(key, source)]

    def reference_session(self):
        if self._reference_session is None:
            options = self.parity.session_options()
            options.intra_op_num_threads = 4
            import onnxruntime as ort

            self._reference_session = ort.InferenceSession(
                str(SOURCE_ONNX), options, providers=["CPUExecutionProvider"]
            )
        return self._reference_session

    def probe_image(self):
        manifest = json.loads(MANIFEST.read_text())
        fixture = next(entry for entry in manifest["fixtures"] if entry["name"] == PROBE_FIXTURE)
        return self.parity.load_fixture((MANIFEST.parent / fixture["path"]).resolve(), 518, 518)

    def compare(self, candidate: Path):
        comparator = self.parity.ParityComparator(
            SOURCE_ONNX, candidate, threads=4, reference_session=self.reference_session()
        )
        return comparator.compare(PROBE_FIXTURE, self.probe_image())

    def mutate(self, source_model: Path, destination: Path, mutate) -> Path:
        model = onnx.load(str(source_model))
        mutate(model.graph)
        onnx.save(model, str(destination))
        return destination


CACHE: BuildCache


def setUpModule():
    global CACHE
    CACHE = BuildCache()


def weight_source_initializer(graph, tensor_name: str) -> str:
    """
    The float initializer a Convolution's weight traces back to, walking back through the QDQ
    pair. Written independently of the converter's resolver on purpose: a test that reuses the
    code under test proves nothing when that code is wrong.
    """
    inits = {initializer.name for initializer in graph.initializer}
    producers = {out: node for node in graph.node for out in node.output}
    name = tensor_name
    for _ in range(8):
        if name in inits:
            return name
        node = producers.get(name)
        if node is None:
            raise AssertionError(f"weight tensor {tensor_name} does not trace back to an initializer")
        name = node.input[0]
    raise AssertionError(f"weight tensor {tensor_name} did not terminate at an initializer")


def final_conv(graph):
    producer = {out: node for node in graph.node for out in node.output}
    name = graph.output[0].name
    for _ in range(64):
        node = producer[name]
        if node.op_type == "Conv":
            return node
        name = node.input[0]
    raise AssertionError("no Convolution on the path to the graph output")


class ResizeAttributeTests(unittest.TestCase):
    """Defect 1: STRING attributes read through `.i` look exactly like `mode=nearest`."""

    def test_resize_attributes_reject_unsupported_rewrite(self):
        source = onnx.load(str(SOURCE_ONNX))
        resizes = [node for node in source.graph.node if node.op_type == "Resize"]
        self.assertEqual(len(resizes), 5, "the w8a16 export is expected to carry five Resizes")

        # The attributes really are STRING, which is what made `.i` read as the nearest enum.
        for node in resizes:
            attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
            self.assertEqual(attrs["mode"], b"linear", f"{node.name} mode is a STRING attribute")
            self.assertEqual(
                attrs["coordinate_transformation_mode"], b"align_corners", f"{node.name} is align_corners"
            )
            self.assertNotEqual(attrs["mode"], 0, "reading mode through `.i` yields 0 = nearest")

        # Asserted against the emitted graph and the source graph, with no help from the
        # converter, so this fails on the corruption itself rather than on a missing helper. A
        # converter that replaced the Resizes with Gather emits none and has ten.
        default = CACHE.build("default", SOURCE_ONNX)
        emitted = onnx.load(str(default)).graph
        retained = [node for node in emitted.node if node.op_type == "Resize"]
        self.assertEqual(len(retained), 5, "the default conversion must retain all five Resize nodes")
        for node in retained:
            attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
            self.assertEqual(attrs["mode"], b"linear", f"{node.name} was altered")
            self.assertEqual(attrs["coordinate_transformation_mode"], b"align_corners", f"{node.name} was altered")
        gathers = [node.name for node in emitted.node if node.op_type == "Gather"]
        self.assertEqual(gathers, [], "the default conversion must not introduce Gather nodes")

        # The converter must read them the same way, not through `.i`.
        for node in resizes:
            self.assertEqual(
                CACHE.converter.resize_semantics(node),
                ("linear", "align_corners"),
                f"{node.name} was not read as linear/align_corners",
            )

        # And it must refuse the rewrite rather than substituting a nearest-neighbour gather.
        with self.assertRaises(CACHE.converter.UnsupportedRewrite):
            CACHE.converter.build(str(SOURCE_ONNX), str(Path(CACHE.tmp.name) / "refused.onnx"), rewrite_resize=True)
        self.assertFalse(Path(CACHE.tmp.name, "refused.onnx").exists(), "a refused rewrite must not write a model")


class QdqWeightTests(unittest.TestCase):
    """Defect 2: the QDQ weight resolver dequantized a float source that was never quantized."""

    def test_qdq_weight_resolution(self):
        source = onnx.load(str(SOURCE_ONNX))
        graph = source.graph
        head = next(node for node in graph.node if node.name == "node_conv2d_30")
        resolved = CACHE.converter.TensorResolver(graph).float_value(head.input[1])

        self.assertEqual(resolved.shape, (1, 32, 1, 1), "the head weight shape must survive resolution")

        # Independently produced by the audit (outputs/conversion-audit-evidence.json), not by the
        # implementation under test.
        expected_min, expected_max = -0.4106593430042267, 0.2983696758747101
        self.assertAlmostEqual(
            float(resolved.min()), expected_min, delta=abs(expected_min) * 1e-3, msg="resolved weight minimum"
        )
        self.assertAlmostEqual(
            float(resolved.max()), expected_max, delta=abs(expected_max) * 1e-3, msg="resolved weight maximum"
        )

        # The defect's signature: treating the unquantized float source as integer codes applies
        # (x - zp) * scale to it, which the audit measured at a median ratio of 0.0032 — the
        # "roughly 312x too small" figure. A correct resolver is at ratio ~1.
        float_source = numpy_helper.to_array(
            next(i for i in graph.initializer if i.name == "model.head.conv3.weight")
        )
        nonzero = float_source != 0
        median_ratio = float(np.median(resolved[nonzero] / float_source[nonzero]))
        self.assertGreater(median_ratio, 0.9, f"weights are still scaled (median ratio {median_ratio})")
        self.assertLess(median_ratio, 1.1, f"weights are not the dequantized originals (median ratio {median_ratio})")

        # Idempotence: re-quantizing an already-dequantized weight must land on the same codes,
        # which is what makes the dequantized value the one the graph itself consumes.
        scale = numpy_helper.to_array(
            next(i for i in graph.initializer if i.name == "model.head.conv3.weight_scale")
        ).reshape(-1, 1, 1, 1)
        zero_point = numpy_helper.to_array(
            next(i for i in graph.initializer if i.name == "model.head.conv3.weight_zero_point")
        ).reshape(-1, 1, 1, 1)
        recoded = np.clip(np.rint(resolved / scale) + zero_point, -128, 127).astype(np.float32)
        again = ((recoded - zero_point) * scale).astype(np.float32)
        self.assertTrue(np.array_equal(again, resolved), "dequantizing a re-quantized weight must be stable")


class ParityDetectorTests(unittest.TestCase):
    """
    The comparator must reject a graph whose arithmetic changed, and accept one whose boundary
    changed only. Both directions are asserted, because a comparator that always returned
    "exact" would satisfy every negative control above it.
    """

    def test_parity_accepts_boundary_only_adapter(self):
        result = CACHE.compare(CACHE.build("default", SOURCE_ONNX))
        self.assertEqual(result.max_code_difference, 0, f"the boundary-only conversion must be exact: {result}")
        self.assertEqual(result.differing_codes, 0, f"the boundary-only conversion must be exact: {result}")

    def test_reference_output_is_not_constant(self):
        """Guards the whole class: two constant maps compare equal no matter what."""
        image = CACHE.probe_image()
        comparator = CACHE.parity.ParityComparator(
            SOURCE_ONNX, CACHE.build("default", SOURCE_ONNX), reference_session=CACHE.reference_session()
        )
        codes, _ = comparator.run(image)
        self.assertGreater(int(codes.max()) - int(codes.min()), 1000, "reference output is nearly constant")
        self.assertGreater(float(codes.std()), 1.0, "reference output has no spread")

    def test_parity_rejects_changed_resize(self):
        self.assertTrue(CACHE.compare(CACHE.build("default", SOURCE_ONNX)).exact, "positive control")

        # The exact defect that shipped: bilinear align_corners replaced by nearest Gather.
        known_bad = CACHE.build("known_bad_resize", SOURCE_ONNX, converter=load_module(KNOWN_BAD))
        graph = onnx.load(str(known_bad))
        gathers = [node.name for node in graph.graph.node if node.op_type == "Gather"]
        self.assertTrue(gathers, "the known-bad converter is expected to emit Gather nodes")
        self.assertEqual(
            len([node for node in graph.graph.node if node.op_type == "Resize"]),
            0,
            "the known-bad converter is expected to remove every Resize",
        )

        result = CACHE.compare(known_bad)
        self.assertFalse(result.exact, "the comparator accepted a graph whose Resizes were replaced by Gather")
        self.assertGreater(result.max_code_difference, 0)

    def test_parity_rejects_scaled_weights(self):
        self.assertTrue(CACHE.compare(CACHE.build("default", SOURCE_ONNX)).exact, "positive control")

        def scale_head_weight(graph):
            conv = final_conv(graph)
            source_name = weight_source_initializer(graph, conv.input[1])
            initializer = next(i for i in graph.initializer if i.name == source_name)
            original = numpy_helper.to_array(initializer)
            # 0.0032 is the median ratio the audit measured for the shipped defect-2 weights.
            scaled = (original.astype(np.float32) * np.float32(0.0032)).astype(original.dtype)
            initializer.CopyFrom(numpy_helper.from_array(scaled, source_name))

        mutated = CACHE.mutate(
            CACHE.build("default", SOURCE_ONNX), Path(CACHE.tmp.name) / "scaled_weights.onnx", scale_head_weight
        )
        result = CACHE.compare(mutated)
        self.assertFalse(result.exact, "the comparator accepted QDQ weights scaled by 0.0032")
        self.assertGreater(result.max_code_difference, 0)


class OptionalRewriteTests(unittest.TestCase):
    """
    The opt-in rewrites are judged on float output against the boundary-only graph, which is the
    same graph they are derived from. The reference uint16 codes are the wrong yardstick here: a
    rewrite that moves a value by a thousandth of a quant step is exact as arithmetic and
    meaningless as a code comparison. Tolerances are predeclared in the manifest and are not
    widened here — a failure means the rewrite changed something.
    """

    ATOL = 1e-5
    RTOL = 1e-5

    def test_optional_conv_rewrite_preserves_float_output(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["criterion"]["optional_rewrite_atol"], self.ATOL)
        self.assertEqual(manifest["criterion"]["optional_rewrite_rtol"], self.RTOL)

        boundary = CACHE.build("default", SOURCE_ONNX)
        rewritten = CACHE.build(
            "opt_in_conv", SOURCE_ONNX, rewrite_single_channel_conv=True
        )
        image = CACHE.probe_image()

        expected = CACHE.parity.ParityComparator(
            SOURCE_ONNX, boundary, reference_session=CACHE.reference_session()
        ).candidate_float(image)
        actual = CACHE.parity.ParityComparator(
            SOURCE_ONNX, rewritten, reference_session=CACHE.reference_session()
        ).candidate_float(image)

        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        tolerance = self.ATOL + self.RTOL * np.abs(expected.astype(np.float64))
        worst = float((difference / tolerance).max())
        self.assertLessEqual(
            worst,
            1.0,
            f"the 1-channel Conv rewrite moved the output by up to {float(difference.max()):.3e}, "
            f"which exceeds atol={self.ATOL} rtol={self.RTOL} (worst tolerance ratio {worst:.3f})",
        )


class ConverterEntryPointTests(unittest.TestCase):
    """The shipped asset must be exactly what the converter emits, and the CLI must report refusal."""

    def test_cli_reports_refusal_with_non_zero_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(converter_path), str(SOURCE_ONNX), f"{tmp}/out.onnx", "--rewrite-resize"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, f"expected a refusal, got: {result.stdout}{result.stderr}")
            self.assertIn("linear", result.stdout)
            self.assertFalse((Path(tmp) / "out.onnx").exists(), "a refused rewrite must not write a model")

    def test_shipped_asset_matches_manifest(self):
        manifest = json.loads(MANIFEST.read_text())
        shipped = REPO_ROOT / "app/src/main/assets/depth_anything_v2.onnx"
        self.assertEqual(CACHE.parity.sha256_file(shipped), manifest["candidate"]["sha256"])

        graph = onnx.load(str(shipped)).graph
        self.assertEqual(graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual(graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual(len([node for node in graph.node if node.op_type == "Resize"]), 5)
        self.assertEqual(len([node for node in graph.node if node.op_type == "Conv"]), 31)
        self.assertEqual([node for node in graph.node if node.op_type == "Gather"], [])


def main() -> int:
    global converter_path
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--converter",
        type=Path,
        default=converter_path,
        help="converter under test; point this at a known-bad one to see the controls fail",
    )
    args = parser.parse_args()
    converter_path = args.converter.resolve()
    print(f"converter under test: {converter_path}")
    if converter_path == KNOWN_BAD.resolve():
        print("WARNING: this is the archived known-bad converter; the negative controls are expected to FAIL\n")
    result = unittest.main(argv=[sys.argv[0], "-v"], exit=False).result
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
