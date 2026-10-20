package com.yy26git.depthanythingv2qnn

import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Typed UINT16 I/O for a borrowed Java session. The factory and estimator retain all ownership.
 * The pinned ORT Java API cannot represent UINT16, so only Run/tensor creation cross JNI.
 * Caller serializes run and close; no native session, provider policy, cache or worker is created.
 */
internal class UInt16SessionRunner(
    private val session: OrtSession,
    private val inputName: String,
    private val outputName: String,
    private val inputShape: LongArray,
    private val outputShape: LongArray
) {
    private val input = bufferFor(inputShape)
    private val output = bufferFor(outputShape)

    init {
        check(nativeVersion() == BuildConfig.DEPTH_ORT_VERSION) { "UINT16 bridge requires the pinned ORT ABI" }
    }

    fun run(pixelValues: FloatArray): FloatArray {
        encodeNative(pixelValues, input, DepthUInt16Contract.INPUT_SCALE, DepthUInt16Contract.INPUT_ZERO_POINT)
        runNative(session, inputName, outputName, input, output, inputShape, outputShape)
        return decodeNative(output, DepthUInt16Contract.OUTPUT_SCALE, DepthUInt16Contract.OUTPUT_ZERO_POINT)
    }

    private fun bufferFor(shape: LongArray): ByteBuffer {
        require(shape.isNotEmpty() && shape.all { it > 0 }) { "UINT16 bridge requires concrete shapes" }
        val elements = shape.fold(1L) { product, size -> Math.multiplyExact(product, size) }
        require(elements <= Int.MAX_VALUE / 2) { "UINT16 tensor is too large" }
        return ByteBuffer.allocateDirect(elements.toInt() * 2).order(ByteOrder.nativeOrder())
    }

    companion object {
        init { System.loadLibrary("depth_uint16") }

        // Constants remain owned by DepthUInt16Contract; native code only performs bulk conversion.
        @JvmStatic external fun encodeNative(values: FloatArray, destination: ByteBuffer, scale: Float, zeroPoint: Int)
        @JvmStatic external fun decodeNative(source: ByteBuffer, scale: Float, zeroPoint: Int): FloatArray
        @JvmStatic private external fun nativeVersion(): String
        @JvmStatic private external fun runNative(
            session: OrtSession, inputName: String, outputName: String,
            input: ByteBuffer, output: ByteBuffer, inputShape: LongArray, outputShape: LongArray
        )
    }
}
