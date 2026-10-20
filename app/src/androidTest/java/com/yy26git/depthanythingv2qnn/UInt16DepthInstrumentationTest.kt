package com.yy26git.depthanythingv2qnn

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import com.yy26git.depthanythingv2qnn.settings.DepthBackend
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt

/** One foreground host, one factory fidelity session, then an isolated profile session. No camera cycling. */
@RunWith(AndroidJUnit4::class)
class UInt16DepthInstrumentationTest {
    @Test fun strictOriginalModel_matchesEightIndependentFloatReferences() {
        ActivityScenario.launch(QnnModelTestActivity::class.java).use {
            verifyBulkBoundary()
            verifyStrictModel()
        }
    }

    private fun verifyBulkBoundary() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val codes = instrumentation.context.assets.open("uint16-rgb-codes.csv").bufferedReader().use { it.readLines() }
        assertEquals(256, codes.size)
        val input = FloatArray(256) { it / 255f }
        val bytes = ByteBuffer.allocateDirect(512).order(ByteOrder.nativeOrder())
        UInt16SessionRunner.encodeNative(input, bytes, DepthUInt16Contract.INPUT_SCALE, DepthUInt16Contract.INPUT_ZERO_POINT)
        codes.forEachIndexed { index, row ->
            assertEquals(row.split(',')[1].toInt(), bytes.asShortBuffer().get(index).toInt() and 0xffff)
        }
        val edges = FloatArray(65536) { (it - DepthUInt16Contract.INPUT_ZERO_POINT + 0.5f) * DepthUInt16Contract.INPUT_SCALE }
        val buffer = ByteBuffer.allocateDirect(2 * edges.size).order(ByteOrder.nativeOrder())
        UInt16SessionRunner.encodeNative(edges, buffer, DepthUInt16Contract.INPUT_SCALE, DepthUInt16Contract.INPUT_ZERO_POINT)
        val shorts = buffer.asShortBuffer()
        edges.forEachIndexed { index, value -> assertEquals("rounding boundary $index", DepthUInt16Contract.encode(value), shorts.get(index)) }
        for (index in 0..65535) shorts.put(index, index.toShort())
        val decoded = UInt16SessionRunner.decodeNative(buffer, DepthUInt16Contract.OUTPUT_SCALE, DepthUInt16Contract.OUTPUT_ZERO_POINT)
        for (index in decoded.indices) assertEquals("unsigned code $index", DepthUInt16Contract.decode(index.toShort()), decoded[index], 0f)
        for (value in listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)) {
            assertThrows(IllegalStateException::class.java) {
                UInt16SessionRunner.encodeNative(floatArrayOf(value), ByteBuffer.allocateDirect(2), DepthUInt16Contract.INPUT_SCALE, DepthUInt16Contract.INPUT_ZERO_POINT)
            }
        }
        Log.i("DepthQNN", "u16 bulk boundary PASS independentRgb=256 roundingBoundaries=65536 unsignedDecode=65536 nonfiniteRejected=3")
    }

    private fun verifyStrictModel() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val model = context.assets.open("depth_anything_v2.onnx").use { it.readBytes() }
        val estimator = OnnxDepthEstimator.open(context, model, ExecutionMode.STRICT_QNN)
        assertEquals(DepthBackend.QNN_HTP, estimator.diagnostics.activeBackend)
        assertEquals(BuildConfig.DEPTH_ORT_VERSION, estimator.diagnostics.onnxRuntimeVersion)
        val measurements = org.json.JSONArray()
        val productionOptions = estimator.diagnostics.qnnProviderOptions
        try {
            for (index in 0 until 8) {
                fun floats(suffix: String): FloatArray {
                    val bytes = instrumentation.context.assets.open("qnn-fixtures/$index.$suffix.f32").use { it.readBytes() }
                    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                    return FloatArray(buffer.remaining()).also { buffer.get(it) }
                }
                val input = floats("input")
                val reference = floats("reference")
                val started = System.nanoTime()
                val actual = estimator.estimateDepth(input)
                val elapsedMs = (System.nanoTime() - started) / 1_000_000.0
                assertEquals(reference.size, actual.size)
                assertTrue("nonfinite output on fixture $index", actual.all(Float::isFinite))
                val xMean = reference.average()
                val yMean = actual.average()
                var covariance = 0.0
                var xVariance = 0.0
                var yVariance = 0.0
                var absoluteError = 0.0
                for (pixel in reference.indices) {
                    val x = reference[pixel] - xMean
                    val y = actual[pixel] - yMean
                    covariance += x * y
                    xVariance += x * x
                    yVariance += y * y
                    absoluteError += abs(reference[pixel].toDouble() - actual[pixel])
                }
                assertTrue("constant reference/output", xVariance > 0 && yVariance > 0)
                val correlation = covariance / sqrt(xVariance * yVariance)
                val range = reference.max() - reference.min()
                val normalizedError = absoluteError / actual.size / range
                measurements.put(org.json.JSONObject().put("fixture", index)
                    .put("pearson", correlation).put("maeOverRange", normalizedError)
                    .put("inferenceMs", elapsedMs).put("runtime", estimator.diagnostics.onnxRuntimeVersion))
                java.io.File(context.filesDir, "strict-u16-fidelity.json").writeText(measurements.toString(2))
                Log.i("DepthQNN", "u16 fidelity fixture=$index pearson=$correlation maeOverRange=$normalizedError inferenceMs=$elapsedMs runtime=${estimator.diagnostics.onnxRuntimeVersion} backend=${estimator.diagnostics.activeBackend}")
                assertTrue("fixture $index Pearson $correlation < 0.99", correlation >= 0.99)
                assertTrue("fixture $index normalized MAE $normalizedError > 0.01", normalizedError <= 0.01)
            }
        } finally { estimator.close() }
        // Java close owns the session; native code cannot keep running against a freed handle.
        assertThrows(IllegalStateException::class.java) { estimator.estimateDepth(FloatArray(3 * 518 * 518)) }
        val report = QnnPartitionProbe.probeStrict(
            context = context, modelBytes = model,
            productionProviderOptions = productionOptions,
            enableContextCache = false, runProfileInference = true
        )
        java.io.File(context.filesDir, "strict-u16-placement.txt").writeText(report.describe())
        Log.i("DepthQNN", "u16 placement session=${report.sessionCreated} strict=${report.cpuEpFallbackDisabled} " +
            "source=${report.placement.source} providers=${report.placement.perProvider}")
        assertTrue(report.initializationError ?: "profile session failed", report.sessionCreated)
        assertTrue("CPU fallback was not disabled", report.cpuEpFallbackDisabled)
        assertEquals(QnnPartitionProbe.PlacementSource.ORT_KERNEL_PROFILE, report.placement.source)
        assertTrue("No actual QNN kernel execution", (report.placement.qnnKernelEvents ?: 0) > 0)
        assertEquals("CPU kernel execution found", 0, requireNotNull(report.placement.cpuKernelEvents))
        assertTrue("Unexpected provider executed", report.placement.perProvider.keys.all { it.contains("QNN") })
    }
}
