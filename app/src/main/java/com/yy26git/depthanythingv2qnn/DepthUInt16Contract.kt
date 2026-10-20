package com.yy26git.depthanythingv2qnn

import java.nio.ShortBuffer

/** Original Qualcomm model boundary. Short stores unsigned bits; decoding must mask the sign. */
object DepthUInt16Contract {
    const val INPUT_SCALE = 0.00002101432801282499f
    const val INPUT_ZERO_POINT = 9979
    const val OUTPUT_SCALE = 0.0002007981966016814f
    const val OUTPUT_ZERO_POINT = 0

    fun encode(value: Float): Short {
        require(value.isFinite()) { "Depth input contains a non-finite pixel" }
        // Division must round to float32 before ties-to-even rounding, matching ONNX QLinear.
        val code = Math.rint((value / INPUT_SCALE).toDouble()) + INPUT_ZERO_POINT
        return code.coerceIn(0.0, 65535.0).toInt().toShort()
    }

    fun encode(values: FloatArray, destination: ShortBuffer) {
        require(destination.capacity() == values.size) { "Depth input buffer has the wrong extent" }
        values.forEachIndexed { index, value -> destination.put(index, encode(value)) }
    }

    fun decode(code: Short): Float =
        ((code.toInt() and 0xffff) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE

    fun decode(source: ShortBuffer): FloatArray = FloatArray(source.capacity()) { decode(source.get(it)) }
}
