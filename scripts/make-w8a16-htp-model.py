#!/usr/bin/env python3
"""
Convert the Qualcomm AI Hub Depth Anything V2 Small w8a16 ONNX export into the asset the QNN
HTP harness runs.

The default conversion is boundary-only. The single DequantizeLinear that consumed the uint16
graph input and the single QuantizeLinear that produced the uint16 graph output are removed, and
the graph I/O is declared float32. Every arithmetic node in between — all five Resizes, all
thirty-one Convs, the whole QDQ weight graph — is retained exactly as exported.

Why boundary-only is the default: the uint16 boundary exists so Qualcomm's own runtime
reproduces their reference numbers. Nothing inside it has to change for the model to be
drivable from Kotlin, and every edit inside it is an opportunity to change the arithmetic. Two
such edits were previously applied and both corrupted the model:

  * Resize -> Gather. All five Resizes in this export carry STRING attributes `mode="linear"`
    and `coordinate_transformation_mode="align_corners"`. Reading them as `attr.i` yields 0,
    which is coincidentally the `nearest` enum, so bilinear align-corners upsampling was
    silently replaced by nearest-neighbour Gather. Bilinear align-corners is not a gather, and
    two of the five have non-integer factors (19->37 and 296->518), so no integral
    replication rewrite applies either.

  * 1-output-channel Conv -> MatMul, whose weights came out ~312x too small because the QDQ
    weight resolver dequantized a float source that had never been quantized. See
    `TensorResolver`.

Both rewrites remain available behind explicit opt-in flags. Each refuses, rather than
approximates, any semantics it cannot reproduce exactly, and `scripts/test-model-conversion.py`
holds each one to that.

Usage:
    python3 scripts/make-w8a16-htp-model.py <input.onnx> <output.onnx>
    python3 scripts/make-w8a16-htp-model.py <input.onnx> <output.onnx> --rewrite-single-channel-conv
    python3 scripts/make-w8a16-htp-model.py <input.onnx> <output.onnx> --rewrite-resize
"""

import argparse
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


class UnsupportedRewrite(Exception):
    """The requested edit cannot be shown to preserve the exported semantics exactly."""


# Integer code ranges for the tensor types a QuantizeLinear may saturate into. Used to reproduce
# the quantizer's rounding-and-clipping, never to guess it: an unknown type is refused.
CODE_RANGES = {
    TensorProto.UINT8: (0, 255),
    TensorProto.INT8: (-128, 127),
    TensorProto.UINT16: (0, 65535),
    TensorProto.INT16: (-32768, 32767),
    TensorProto.INT32: (-(2**31), 2**31 - 1),
    TensorProto.UINT32: (0, 2**32 - 1),
    TensorProto.INT64: (-(2**63), 2**63 - 1),
    TensorProto.UINT64: (0, 2**64 - 1),
}


def attributes(node) -> dict[str, Any]:
    """ONNX attributes decoded by ONNX's own reader.

    Reading `attr.i` or `attr.s` directly is what made the previous converter mis-read `mode`:
    it is a STRING attribute here, and the raw integer payload of a string attribute is 0, which
    happens to equal the `nearest` enum. `helper.get_attribute_value` returns b'linear'.
    """
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def as_text(value: Any) -> str | None:
    """Attribute payload as text, so STRING attributes read as words rather than enum numbers."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return None if value is None else str(value)


def attr_text(node, name: str, default: str) -> str:
    return as_text(attributes(node).get(name)) or default


def resize_semantics(node) -> tuple[str, str]:
    """(mode, coordinate_transformation_mode) of a Resize, read as strings."""
    attrs = attributes(node)
    return (
        as_text(attrs.get("mode")) or "nearest",
        as_text(attrs.get("coordinate_transformation_mode")) or "half_pixel",
    )


def reachable_node_indices(graph, seed_names) -> set[int]:
    """Indices of nodes transitively fed by `seed_names`. Nodes outside it are dead code."""
    producer = {out: index for index, node in enumerate(graph.node) for out in node.output}
    reachable: set[int] = set()
    pending = list(seed_names)
    while pending:
        index = producer.get(pending.pop())
        if index is None or index in reachable:
            continue
        reachable.add(index)
        pending.extend(graph.node[index].input)
    return reachable


def reachable_node_names(graph, seed_names) -> set[str]:
    return {graph.node[index].name for index in reachable_node_indices(graph, seed_names)}


def live_initializer_names(graph) -> set[str]:
    """Initializers some node consumes, or that the graph itself declares as I/O."""
    consumed = {name for node in graph.node for name in node.input if name}
    declared = {value.name for value in graph.input} | {value.name for value in graph.output}
    return {initializer.name for initializer in graph.initializer if initializer.name in consumed | declared}


class TensorResolver:
    """
    Resolves the float value of a tensor in a calibrated QDQ graph.

    A calibrated export does not store the integer codes for activations and large weights.
    Each one is a float initializer pushed through `QuantizeLinear(float, scale, zero_point)`
    and then `DequantizeLinear`, so the tensor a Convolution actually multiplies is the output
    of a DequantizeLinear whose input is the output of a QuantizeLinear.

    The QuantizeLinear step is not optional. Its output is
    `clip(round_half_to_even(x / scale) + zero_point, code_min, code_max)`, and the enclosing
    DequantizeLinear then computes `(q - zero_point) * scale`. Returning the float source from
    the QuantizeLinear instead — which is what the previous converter did, on the reasoning
    that re-running the round trip would "reintroduce the rounding we are trying to avoid" —
    hands the DequantizeLinear a tensor that was never quantized, so it applies `(x - zp) * scale`
    to real weights. Every rewritten MatMul weight came out roughly 312x too small.

    Memoised because the graph carries 861 initializers and 476 quantizers to look through.
    """

    def __init__(self, graph):
        self._inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
        self._producer = {out: node for node in graph.node for out in node.output}
        self._elem_type = {
            value.name: value.type.tensor_type.elem_type
            for value in [*graph.input, *graph.output, *graph.value_info]
            if value.type.HasField("tensor_type")
        }
        self._cache: dict[str, np.ndarray] = {}

    def float_value(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = self._compute(name)
        return self._cache[name]

    def _broadcast(self, values, ndim: int, axis: int) -> np.ndarray:
        """Per-axis quantization parameters broadcast over a tensor of rank `ndim`."""
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 0 or values.ndim == ndim:
            return values
        if values.ndim != 1:
            raise UnsupportedRewrite(f"per-axis parameter of rank {values.ndim} is not the ONNX form")
        shape = [1] * ndim
        shape[axis] = values.shape[0]
        return values.reshape(shape)

    def _code_range(self, name: str) -> tuple[int, int]:
        elem_type = self._elem_type.get(name)
        if elem_type not in CODE_RANGES:
            raise UnsupportedRewrite(
                f"quantizer output '{name}' has unknown tensor type {elem_type}; "
                "the saturating code range cannot be reproduced without guessing"
            )
        return CODE_RANGES[elem_type]

    def _compute(self, name: str) -> np.ndarray:
        if name in self._inits:
            return self._inits[name].astype(np.float32)

        node = self._producer.get(name)
        if node is None:
            raise UnsupportedRewrite(f"could not resolve a float value for tensor '{name}'")

        if node.op_type == "QuantizeLinear":
            if int(attributes(node).get("saturate", 1)) == 0:
                raise UnsupportedRewrite(f"{node.name}: non-saturating quantization is not reproduced")
            source = self.float_value(node.input[0])
            axis = int(attributes(node).get("axis", 0))
            scale = self._broadcast(self._inits[node.input[1]], source.ndim, axis)
            zero_point = self._broadcast(
                self._inits[node.input[2]] if len(node.input) > 2 and node.input[2] else 0.0,
                source.ndim,
                axis,
            )
            if np.any(scale <= 0):
                raise UnsupportedRewrite(f"{node.name}: non-positive scale")
            low, high = self._code_range(node.output[0])
            codes = np.rint(source / scale) + zero_point
            return np.clip(codes, low, high).astype(np.float32)

        if node.op_type == "DequantizeLinear":
            codes = self.float_value(node.input[0])
            axis = int(attributes(node).get("axis", 0))
            scale = self._broadcast(self._inits[node.input[1]], codes.ndim, axis)
            zero_point = self._broadcast(
                self._inits[node.input[2]] if len(node.input) > 2 and node.input[2] else 0.0,
                codes.ndim,
                axis,
            )
            return ((codes - zero_point) * scale).astype(np.float32)

        raise UnsupportedRewrite(f"tensor '{name}' is produced by unsupported op {node.op_type}")


def nearest_index_table(in_size: int, out_size: int, coordinate_mode: str, nearest_mode: str) -> np.ndarray:
    """Source index for every output position of a nearest-neighbour Resize, per the ONNX spec.

    Only reachable with `--rewrite-resize`, and only when `mode` is nearest. Every branch that
    cannot be evaluated from the attributes alone raises instead of falling back to a default.
    """
    if out_size < 1 or in_size < 1:
        raise UnsupportedRewrite(f"degenerate size {in_size}->{out_size}")

    if coordinate_mode in ("half_pixel", "pytorch_half_pixel"):
        scale = out_size / in_size
        if coordinate_mode == "pytorch_half_pixel" and out_size == 1:
            original = np.zeros(out_size, dtype=np.float64)
        else:
            original = (np.arange(out_size, dtype=np.float64) + 0.5) / scale - 0.5
    elif coordinate_mode == "asymmetric":
        original = np.arange(out_size, dtype=np.float64) * (in_size / out_size)
    elif coordinate_mode == "align_corners":
        if out_size == 1:
            original = np.zeros(out_size, dtype=np.float64)
        else:
            original = np.arange(out_size, dtype=np.float64) * ((in_size - 1) / (out_size - 1))
    else:
        raise UnsupportedRewrite(f"coordinate_transformation_mode={coordinate_mode!r} is not implemented")

    if nearest_mode == "floor":
        index = np.floor(original)
    elif nearest_mode == "ceil":
        index = np.ceil(original)
    elif nearest_mode in ("round_prefer_floor", "round_prefer_ceil"):
        index = np.round(original - 0.5) + 0.5 if nearest_mode == "round_prefer_floor" else np.round(original)
    else:
        raise UnsupportedRewrite(f"nearest_mode={nearest_mode!r} is not implemented")

    return np.clip(index, 0, in_size - 1).astype(np.int64)


def rewrite_resize_to_gather(node, shapes, inits, new_inits) -> list:
    """
    Replace one nearest-neighbour Resize with two Gathers driven by precomputed index tables.

    This is exact only for `mode=nearest`, because nearest resampling selects a single source
    element per output position. `mode=linear` averages between neighbours and cannot be
    expressed as a gather at any scale factor, so it is refused rather than approximated.
    """
    mode, coordinate_mode = resize_semantics(node)
    if mode != "nearest":
        raise UnsupportedRewrite(
            f"{node.name}: mode={mode!r} is not a gather. Bilinear and cubic resampling interpolate "
            f"between neighbours; only 'nearest' selects one source element per output position."
        )

    attrs = attributes(node)
    antialias = int(attrs.get("antialias", 0))
    if antialias != 0:
        raise UnsupportedRewrite(f"{node.name}: antialias={antialias} changes the sampling support")

    in_shape = shapes.get(node.input[0])
    if in_shape is None or len(in_shape) != 4:
        raise UnsupportedRewrite(f"{node.name}: could not infer the rank-4 input shape of {node.input[0]}")

    # Resize inputs are positional [X, roi, scales, sizes]; the unused slots are present but
    # empty, so the list must not be compacted before indexing.
    if len(node.input) > 3 and node.input[3]:
        sizes = inits[node.input[3]]
        out_h, out_w = int(sizes[2]), int(sizes[3])
    elif len(node.input) > 2 and node.input[2]:
        scales = inits[node.input[2]]
        out_h, out_w = int(round(in_shape[2] * scales[2])), int(round(in_shape[3] * scales[3]))
    else:
        raise UnsupportedRewrite(f"{node.name}: neither sizes nor scales is provided")

    in_h, in_w = in_shape[2], in_shape[3]
    nearest_mode = as_text(attrs.get("nearest_mode")) or "round_prefer_floor"
    idx_h, idx_w = f"{node.name}_idx_h", f"{node.name}_idx_w"
    mid, out = f"{node.name}_gather_h", f"{node.name}_gather_hw"
    new_inits.append(numpy_helper.from_array(nearest_index_table(in_h, out_h, coordinate_mode, nearest_mode), idx_h))
    new_inits.append(numpy_helper.from_array(nearest_index_table(in_w, out_w, coordinate_mode, nearest_mode), idx_w))
    return [
        helper.make_node("Gather", [node.input[0], idx_h], [mid], name=f"{node.name}_g_h", axis=2),
        helper.make_node("Gather", [mid, idx_w], [out], name=f"{node.name}_g_hw", axis=3),
    ]


def conv_to_matmul(node, resolver, new_inits) -> list | None:
    """
    Rewrite a 1x1 single-output-channel Convolution as Transpose -> MatMul -> Transpose.

    With one output channel and a 1x1 kernel the convolution is one dot product per pixel, so
    the MatMul form is exact. The 1-channel restriction matters: with more output channels the
    rewrite would also need the per-output-channel bias and the group/stride/pad semantics
    re-established, which is where a "harmless" rewrite starts changing arithmetic.

    Returns None when the node is not a candidate at all. Raises when it *is* a candidate whose
    semantics this rewrite cannot reproduce, so a partially-rewritten graph is never emitted.
    """
    if len(node.input) < 2:
        return None

    weight = resolver.float_value(node.input[1])
    if weight.ndim != 4 or weight.shape[0] != 1:
        return None

    attrs = attributes(node)
    # kernel_shape is optional in ONNX; when absent it is inferred from the weight tensor, so
    # reading the weight's own spatial dims is the spec-conformant default, not a fallback.
    kernel_shape = list(attrs.get("kernel_shape", weight.shape[2:4]))
    checks = {
        "kernel_shape": kernel_shape,
        "strides": list(attrs.get("strides", [1, 1])),
        "pads": list(attrs.get("pads", [0, 0, 0, 0])),
        "dilations": list(attrs.get("dilations", [1, 1])),
        "group": int(attrs.get("group", 1)),
        "auto_pad": as_text(attrs.get("auto_pad")) or "NOTSET",
    }
    if checks != {
        "kernel_shape": [1, 1],
        "strides": [1, 1],
        "pads": [0, 0, 0, 0],
        "dilations": [1, 1],
        "group": 1,
        "auto_pad": "NOTSET",
    }:
        raise UnsupportedRewrite(
            f"{node.name}: 1-channel Conv with {checks} is not reproduced exactly by a MatMul"
        )
    if weight.shape[2] != 1 or weight.shape[3] != 1:
        raise UnsupportedRewrite(f"{node.name}: weight {weight.shape} is not 1x1")

    in_channels = weight.shape[1]
    matmul_w = f"{node.name}_matmul_w"
    new_inits.append(
        numpy_helper.from_array(np.ascontiguousarray(weight.reshape(in_channels, 1), dtype=np.float32), matmul_w)
    )

    nodes = [
        helper.make_node("Transpose", [node.input[0]], [f"{node.name}_nhwc"], name=f"{node.name}_t1", perm=[0, 2, 3, 1]),
        helper.make_node("MatMul", [f"{node.name}_nhwc", matmul_w], [f"{node.name}_proj"], name=f"{node.name}_mm"),
    ]
    if len(node.input) >= 3 and node.input[2]:
        bias = resolver.float_value(node.input[2])
        if bias.size != 1:
            raise UnsupportedRewrite(f"{node.name}: bias of {bias.size} elements is not a scalar broadcast")
        # The Conv's bias is a single element, which broadcasts correctly over [1, H, W, 1] and
        # carries the dequantized QDQ bias unchanged.
        nodes.append(
            helper.make_node("Add", [f"{node.name}_proj", node.input[2]], [f"{node.name}_biased"], name=f"{node.name}_bias")
        )
        tail = f"{node.name}_biased"
    else:
        tail = f"{node.name}_proj"
    nodes.append(
        helper.make_node("Transpose", [tail], [f"{node.name}_out"], name=f"{node.name}_t2", perm=[0, 3, 1, 2])
    )
    return nodes


@dataclass
class BuildReport:
    boundary_nodes_dropped: list[str] = field(default_factory=list)
    dead_initializers_dropped: list[str] = field(default_factory=list)
    resizes_rewritten: list[tuple[str, list[int], list[int]]] = field(default_factory=list)
    convs_rewritten: list[str] = field(default_factory=list)
    resize_modes: list[tuple[str, str, str]] = field(default_factory=list)


def build(src: str, dst: str, *, rewrite_resize: bool = False, rewrite_single_channel_conv: bool = False) -> BuildReport:
    model = onnx.load(src)
    graph = model.graph
    report = BuildReport()

    graph_input = graph.input[0]
    if graph_input.type.tensor_type.elem_type != TensorProto.UINT16:
        raise UnsupportedRewrite(
            f"expected a uint16 graph input, got elem_type={graph_input.type.tensor_type.elem_type}"
        )
    graph_output = graph.output[0]
    if graph_output.type.tensor_type.elem_type != TensorProto.UINT16:
        raise UnsupportedRewrite(
            f"expected a uint16 graph output, got elem_type={graph_output.type.tensor_type.elem_type}"
        )

    # Proof, not assertion: dropping a node is only safe if nothing it produced is still read and
    # no surviving node was left stranded. Recorded before the graph is touched so the comparison
    # is against the export's own reachability, not a post-hoc impression of it.
    before = reachable_node_names(graph, [graph_input.name])

    # ---- Boundary: drop the input DequantizeLinear and the output QuantizeLinear ----------
    dropped, rewired = [], {}
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and graph_input.name in node.input:
            dropped.append(node)
            rewired[node.output[0]] = graph_input.name
        elif node.op_type == "QuantizeLinear" and graph_output.name in node.output:
            dropped.append(node)
            # Point the graph output at the float tensor the quantizer consumed.
            graph_output.name = node.input[0]
    if len(dropped) != 2:
        raise UnsupportedRewrite(f"expected exactly 2 boundary Q/DQ nodes, found {len(dropped)}")
    if any(not node.name for node in dropped):
        raise UnsupportedRewrite("a boundary node is unnamed, so it cannot be proven dead by name")

    # Identity, not protobuf value equality: two structurally equal nodes must not alias.
    removed = {id(node) for node in dropped}
    retained = [node for node in graph.node if id(node) not in removed]
    for node in retained:
        node.input[:] = [rewired.get(name, name) for name in node.input]
    for node in dropped:
        report.boundary_nodes_dropped.append(node.name)
    del graph.node[:]
    graph.node.extend(retained)

    dropped_names = set(report.boundary_nodes_dropped)
    after = reachable_node_names(graph, [graph_input.name])
    stranded = sorted(after.symmetric_difference(before - dropped_names))
    if stranded or after != before - dropped_names:
        raise UnsupportedRewrite(
            f"boundary removal changed which nodes are reachable from the graph input: {stranded}"
        )
    for node in dropped:
        for survivor in graph.node:
            if node.output and node.output[0] in survivor.input:
                raise UnsupportedRewrite(f"{survivor.name} still reads dropped tensor {node.output[0]}")

    for node in graph.node:
        if node.op_type == "Resize":
            mode, coordinate_mode = resize_semantics(node)
            report.resize_modes.append((node.name, mode, coordinate_mode))

    graph_input.type.tensor_type.elem_type = TensorProto.FLOAT
    graph_output.type.tensor_type.elem_type = TensorProto.FLOAT

    # ---- Optional rewrites, each opt-in and each refusing what it cannot reproduce ----------
    if rewrite_resize or rewrite_single_channel_conv:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        shapes = {
            value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
            for value in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]
        }
        inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
        resolver = TensorResolver(inferred.graph) if rewrite_single_channel_conv else None
        new_nodes, new_inits, aliases = [], [], {}

        for node in graph.node:
            replacement = None
            if node.op_type == "Resize" and rewrite_resize:
                replacement = rewrite_resize_to_gather(node, shapes, inits, new_inits)
            elif node.op_type == "Conv" and rewrite_single_channel_conv:
                replacement = conv_to_matmul(node, resolver, new_inits)

            if replacement is None:
                new_nodes.append(node)
                continue

            new_nodes.extend(replacement)
            aliases[node.output[0]] = replacement[-1].output[0]
            if node.op_type == "Resize":
                in_shape = shapes[node.input[0]]
                out_shape = shapes[node.output[0]]
                report.resizes_rewritten.append((node.name, in_shape, out_shape))
            else:
                report.convs_rewritten.append(node.name)

        for node in new_nodes:
            node.input[:] = [aliases.get(name, name) for name in node.input]
        del graph.node[:]
        graph.node.extend(new_nodes)
        graph.initializer.extend(new_inits)

    # ---- Prune initializers no node reads anymore -------------------------------------------
    live = live_initializer_names(graph)
    dead = [initializer.name for initializer in graph.initializer if initializer.name not in live]
    if dead:
        report.dead_initializers_dropped = dead
        keep = [initializer for initializer in graph.initializer if initializer.name in live]
        del graph.initializer[:]
        graph.initializer.extend(keep)

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--rewrite-resize",
        action="store_true",
        help="opt-in: replace nearest Resizes with Gather. Refuses any non-nearest mode.",
    )
    parser.add_argument(
        "--rewrite-single-channel-conv",
        action="store_true",
        help="opt-in: replace 1-output-channel 1x1 Convs with MatMul. Refuses everything else.",
    )
    args = parser.parse_args(argv)

    try:
        report = build(
            args.input,
            args.output,
            rewrite_resize=args.rewrite_resize,
            rewrite_single_channel_conv=args.rewrite_single_channel_conv,
        )
    except UnsupportedRewrite as error:
        print(f"refused: {error}")
        return 2

    print(f"dropped {len(report.boundary_nodes_dropped)} boundary Q/DQ node(s): {report.boundary_nodes_dropped}")
    print(f"dropped {len(report.dead_initializers_dropped)} now-unread initializer(s): {report.dead_initializers_dropped}")
    for name, mode, coordinate_mode in report.resize_modes:
        print(f"  retained Resize {name}: mode={mode} coordinate_transformation_mode={coordinate_mode}")
    for name, in_shape, out_shape in report.resizes_rewritten:
        print(f"  rewrote Resize {name}: {in_shape} -> {out_shape}")
    for name in report.convs_rewritten:
        print(f"  rewrote 1-channel Conv {name}")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
