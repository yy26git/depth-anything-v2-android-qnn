#!/usr/bin/env python3
"""
Build the Depth Anything V2 Small w8a16 ONNX into a form the QNN HTP will accept.

Two edits, both exact — no approximation and no retraining:

1. **Float I/O.** The AI Hub w8a16 export declares its graph input and output as uint16 with
   quantization carried in `image_scale` / `image_zero_point` initializers. ORT's Java API cannot
   construct a uint16 tensor at all — `ai.onnxruntime.OnnxJavaType` has no UINT16 constant — so the
   model could not be driven from Kotlin. Deleting the single boundary DequantizeLinear on the
   input and the single boundary QuantizeLinear on the output makes the graph take and return
   float32 while changing no arithmetic in between.

2. **Resize -> Gather.** The HTP execution provider will not take Resize. Depth Anything's decoder
   upsamples five times, and all five are `mode=nearest`, which is a pure gather of input rows and
   columns. Each is replaced by two Gathers (axis 2, then axis 3) driven by a precomputed index
   table, reproducing the ONNX nearest index rule exactly:

       scale      = output_size / input_size
       x_original = (i + 0.5) / scale - 0.5        # coordinate_transformation_mode = half_pixel
       src_index  = round_half_to_even(x_original), clamped to [0, input_size - 1]

   Two of the five are not integer multiples (19->37 and 296->518), so a Reshape/Expand
   upsampling would be wrong for them; the index table handles both cases uniformly.

Usage:
    python3 scripts/make-w8a16-htp-model.py <input.onnx> <output.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def nearest_index_table(in_size: int, out_size: int) -> np.ndarray:
    """ONNX Resize nearest index table, half_pixel + round_half_to_even."""
    scale = out_size / in_size
    idx = []
    for i in range(out_size):
        x_original = (i + 0.5) / scale - 0.5
        # numpy's rint is round-half-to-even, which is what the ONNX spec requires.
        j = int(np.rint(x_original))
        idx.append(min(max(j, 0), in_size - 1))
    return np.asarray(idx, dtype=np.int64)


class TensorResolver:
    """
    Resolves the float value of any tensor in a QDQ graph.

    A calibrated export does not store integer weights at all. Each one is a float initializer
    pushed through `QuantizeLinear(float, scale, zero_point)` and then `DequantizeLinear`, so the
    Convolution's weight input is the output of a DequantizeLinear whose input is itself the output
    of a QuantizeLinear. This walks that chain, with memoisation because the graph carries 861
    initializers and a few hundred Convolutions to look through.
    """

    def __init__(self, graph):
        self._inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
        self._producer = {o: n for n in graph.node for o in n.output}
        self._cache = {}

    def float_value(self, name):
        if name in self._cache:
            return self._cache[name]
        value = self._compute(name)
        self._cache[name] = value
        return value

    def _compute(self, name):
        if name in self._inits:
            return self._inits[name].astype(np.float32)

        node = self._producer.get(name)
        if node is None:
            raise SystemExit(f"could not resolve a float value for tensor '{name}'")

        if node.op_type == "QuantizeLinear":
            # The pre-quantization float source is what we want; re-running the round trip in
            # float64 and back would only reintroduce the rounding we are trying to avoid.
            return self.float_value(node.input[0])

        if node.op_type == "DequantizeLinear":
            raw = self.float_value(node.input[0])
            scale = self._inits[node.input[1]].astype(np.float32)
            zero_point = self._inits[node.input[2]] if len(node.input) > 2 else np.array(0)

            def per_channel(v):
                # Per-channel quantization is stored as a 1-D array indexed by the output-channel
                # axis, which for a [out, in, kh, kw] weight is axis 0. It must be reshaped to
                # [out, 1, 1, 1] before use: numpy would otherwise align a bare (out,) against the
                # trailing axis and fail to broadcast.
                if v.ndim > 0:
                    shape = [1] * raw.ndim
                    shape[0] = v.shape[0]
                    return np.broadcast_to(v.reshape(shape), raw.shape)
                return v

            return (raw - per_channel(np.asarray(zero_point, dtype=np.float32))) * per_channel(scale)

        raise SystemExit(f"tensor '{name}' is produced by unsupported op {node.op_type}")


def conv_to_matmul(node, resolver, new_inits):
    """
    Rewrite a single-output-channel 1x1 Convolution as Transpose -> MatMul -> Transpose.

    The HTP will not take a Conv whose output has one channel. Depth Anything's decoder ends in
    exactly one of these (`model.head.conv3`, weight [1, 32, 1, 1]), and it is the last node the
    QNN provider declines. With a 1x1 kernel and one output channel, the convolution computes one
    dot product per pixel, which is a MatMul over the channel axis — so the rewrite is exact.

    Only the plain 1x1, stride-1, one-output-channel case is rewritten. Anything else is left alone
    so the edit stays narrow and its effect stays attributable.
    """
    if len(node.input) < 2:
        return None
    strides = [1, 1]
    for attr in node.attribute:
        if attr.name == "strides":
            strides = list(attr.ints)
        elif attr.name == "kernel_shape":
            if list(attr.ints) != [1, 1]:
                return None
    if strides != [1, 1]:
        return None

    weight = resolver.float_value(node.input[1])
    if weight.ndim != 4 or weight.shape[0] != 1 or weight.shape[2] != 1 or weight.shape[3] != 1:
        return None

    in_channels = weight.shape[1]
    matmul_w = f"{node.name}_matmul_w"
    new_inits.append(numpy_helper.from_array(np.ascontiguousarray(weight.reshape(in_channels, 1)), matmul_w))

    nodes = [
        helper.make_node("Transpose", [node.input[0]], [f"{node.name}_nhwc"], name=f"{node.name}_t1", perm=[0, 2, 3, 1]),
        helper.make_node("MatMul", [f"{node.name}_nhwc", matmul_w], [f"{node.name}_proj"], name=f"{node.name}_mm"),
    ]
    # The Conv's bias is a single element, which broadcasts correctly over [1, H, W, 1].
    if len(node.input) >= 3 and node.input[2]:
        nodes.append(
            helper.make_node("Add", [f"{node.name}_proj", node.input[2]], [f"{node.name}_biased"],
                             name=f"{node.name}_bias")
        )
        tail = f"{node.name}_biased"
    else:
        tail = f"{node.name}_proj"
    nodes.append(
        helper.make_node("Transpose", [tail], [f"{node.name}_out"], name=f"{node.name}_t2", perm=[0, 3, 1, 2])
    )
    return nodes


def build(src: str, dst: str) -> None:
    model = onnx.load(src)
    graph = model.graph

    # ---- Edit 1: float graph I/O -------------------------------------------------
    graph_input = graph.input[0]
    if graph_input.type.tensor_type.elem_type != TensorProto.UINT16:
        raise SystemExit(f"expected a uint16 graph input, got elem_type={graph_input.type.tensor_type.elem_type}")
    graph_input.type.tensor_type.elem_type = TensorProto.FLOAT

    graph_output = graph.output[0]
    if graph_output.type.tensor_type.elem_type != TensorProto.UINT16:
        raise SystemExit(f"expected a uint16 graph output, got elem_type={graph_output.type.tensor_type.elem_type}")
    final_output_name = graph_output.name
    graph_output.type.tensor_type.elem_type = TensorProto.FLOAT

    boundary_to_drop = []
    rewire = {}
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and node.input and node.input[0] == graph_input.name:
            boundary_to_drop.append(node)
            rewire[node.output[0]] = graph_input.name
        elif node.op_type == "QuantizeLinear" and final_output_name in node.output:
            boundary_to_drop.append(node)
            graph_output.name = node.input[0]          # point the graph output at the float tensor
    if len(boundary_to_drop) != 2:
        raise SystemExit(f"expected exactly 2 boundary Q/DQ nodes, found {len(boundary_to_drop)}")

    # ---- Edit 2: Resize -> Gather ------------------------------------------------
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer if i.data_location != 1}
    resolver = TensorResolver(graph)

    new_nodes = []
    new_inits = []
    rewritten = []
    output_alias = {}

    for node in graph.node:
        if node in boundary_to_drop:
            continue

        if rewire:
            node.input[:] = [rewire.get(i, i) for i in node.input]

        if node.op_type != "Resize":
            if node.op_type == "Conv" and node not in boundary_to_drop:
                replacement = conv_to_matmul(node, resolver, new_inits)
                if replacement is not None:
                    new_nodes.extend(replacement)
                    output_alias[node.output[0]] = replacement[-1].output[0]
                    rewritten.append((node.name, ["1-channel Conv"], ["MatMul"]))
                    continue
            new_nodes.append(node)
            continue

        attrs = {a.name: a for a in node.attribute}
        mode = attrs["mode"].i if "mode" in attrs else 0
        if mode != 0:
            raise SystemExit(f"{node.name}: only mode=nearest is supported, got mode={mode}")

        in_shape = shapes.get(node.input[0])
        if in_shape is None:
            raise SystemExit(f"{node.name}: could not infer the input shape of {node.input[0]}")

        # Resize inputs are positional: [X, roi, scales, sizes]. The unused slot is present but
        # empty, so the list must NOT be compacted before indexing.
        raw = list(node.input)
        if len(raw) > 3 and raw[3]:
            out_h, out_w = int(inits[raw[3]][2]), int(inits[raw[3]][3])
        elif len(raw) > 2 and raw[2]:
            scales = inits[raw[2]]
            out_h, out_w = int(round(in_shape[2] * scales[2])), int(round(in_shape[3] * scales[3]))
        else:
            raise SystemExit(f"{node.name}: no sizes or scales input in {raw}")

        in_h, in_w = in_shape[2], in_shape[3]
        base = node.name
        idx_h_name, idx_w_name = f"{base}_idx_h", f"{base}_idx_w"
        mid_name, out_name = f"{base}_gather_h", f"{base}_gather_hw"
        new_inits.append(numpy_helper.from_array(nearest_index_table(in_h, out_h), idx_h_name))
        new_inits.append(numpy_helper.from_array(nearest_index_table(in_w, out_w), idx_w_name))
        new_nodes.append(helper.make_node("Gather", [node.input[0], idx_h_name], [mid_name], name=f"{base}_g_h", axis=2))
        new_nodes.append(helper.make_node("Gather", [mid_name, idx_w_name], [out_name], name=f"{base}_g_hw", axis=3))
        output_alias[node.output[0]] = out_name
        rewritten.append((base, list(in_shape), [in_shape[0], in_shape[1], out_h, out_w]))

    # Any consumer still reading a replaced Resize output now reads the Gather output instead.
    for node in new_nodes:
        node.input[:] = [output_alias.get(i, i) for i in node.input]

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)

    # The replacement nodes introduce new intermediate tensors that the export's value_info does
    # not describe. ONNX Runtime's QNN execution provider partitions with concrete shapes only, so
    # an unknown dimension reaches it as a zero and it fails late with
    # "qnn_model.cc: SetupTensors Zero tensor size!". Re-running shape inference writes those
    # dimensions into value_info before the model is ever loaded.
    del graph.value_info[:]
    model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"float I/O: dropped {len(boundary_to_drop)} boundary Q/DQ nodes")
    print(f"rewrote {len(rewritten)} Resize node(s) as Gather:")
    for name, src_shape, dst_shape in rewritten:
        print(f"  {name}: {src_shape} -> {dst_shape}")
    print(f"saved {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build(sys.argv[1], sys.argv[2])
