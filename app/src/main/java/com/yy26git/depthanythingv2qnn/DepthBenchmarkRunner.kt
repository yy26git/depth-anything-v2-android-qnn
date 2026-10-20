package com.yy26git.depthanythingv2qnn

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.yy26git.depthanythingv2qnn.runtime.OnnxQnnRuntimeEnvironment
import com.yy26git.depthanythingv2qnn.settings.DepthBackend
import java.io.Closeable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt
import kotlin.system.measureNanoTime

/**
 * One execution mode to measure.
 *
 * The three modes exist because "does it run on the HTP" and "how fast is it" are different
 * questions, and on this model they have different answers. Strict mode answers the first by
 * refusing to run at all if any node falls back; the hybrid mode answers "how much of it is
 * worth having"; the CPU mode is the baseline the other two are worth nothing without.
 */
enum class ExecutionMode(
    val requestedBackend: DepthBackend,
    val allowCpuFallback: Boolean,
    val label: String,
    val intent: String
) {
    STRICT_QNN(
        requestedBackend = DepthBackend.STRICT_QNN_HTP,
        allowCpuFallback = false,
        label = "STRICT QNN HTP",
        intent = "proof of purity: no node may fall back to CPU"
    ),
    /**
     * Requests QNN with CPU fallback permitted at the *factory* level, then reports what happened.
     *
     * The name deliberately avoids "hybrid". A previous revision called this mode "QNN HTP + CPU
     * fallback" and reported it as mixed placement. That was wrong: the factory sets
     * `session.disable_cpu_ep_fallback=1` for any byte-loaded model, so QNN initialization fails
     * and the factory then retries an **entirely CPU** session. Observed on device as
     * `qnnProviderRegistered=false`, `graphCompilationStatus=CPU_GRAPH` and an empty
     * `qnnProviderOptions` — i.e. zero nodes on the HTP.
     *
     * So this mode is a *fallback* measurement, not a blend. Genuine mixed placement is measured
     * by QnnPartitionProbe, which owns a session where CPU fallback really is left on.
     */
    QNN_THEN_CPU(
        requestedBackend = DepthBackend.QNN_HTP,
        allowCpuFallback = true,
        label = "QNN attempt, then CPU retry",
        intent = "shows whether QNN could initialize at all; a CPU result here means it could not"
    ),
    CPU(
        requestedBackend = DepthBackend.CPU,
        allowCpuFallback = true,
        label = "CPU baseline",
        intent = "the comparison point; without it the QNN numbers mean nothing"
    )
}

/** Result of one [ExecutionMode] pass. Immutable. */
data class DepthPassResult(
    val mode: ExecutionMode,
    val graphDescription: String,
    val succeeded: Boolean,
    val failure: String?,
    val activeBackend: String,
    val sessionSummary: List<String>,
    val firstInferenceMs: Long,
    val warmupLatenciesMs: List<Long>,
    val measuredLatenciesMs: List<Long>,
    val preprocessMs: Long,
    val outputElementCount: Int,
    val nonFiniteCount: Int,
    val outputMin: Float,
    val outputMax: Float,
    val outputMean: Float
) {
    val p50Ms: Long get() = percentile(0.50)
    val p95Ms: Long get() = percentile(0.95)
    val meanMs: Double get() = if (measuredLatenciesMs.isEmpty()) 0.0 else measuredLatenciesMs.average()
    val effectiveFps: Float get() = if (p50Ms > 0) 1000f / p50Ms else 0f

    /** (max-min)/max over finite outputs. Near zero means a constant map, i.e. a failed run. */
    val relativeSpread: Float
        get() = if (outputMax > outputMin) (outputMax - outputMin) / outputMax else 0f

    val healthy: Boolean
        get() = succeeded && nonFiniteCount == 0 && relativeSpread >= MIN_HEALTHY_SPREAD

    private fun percentile(fraction: Double): Long {
        if (measuredLatenciesMs.isEmpty()) return 0L
        val sorted = measuredLatenciesMs.sorted()
        val index = ((sorted.size - 1) * fraction).toInt().coerceIn(0, sorted.size - 1)
        return sorted[index]
    }

    companion object {
        const val MIN_HEALTHY_SPREAD = 0.05f
    }
}

/** Everything one full benchmark cycle produced. */
data class DepthBenchmarkReport(
    val graphDescription: String,
    val modelBytes: Int,
    val sourceImageSize: String,
    val passes: List<DepthPassResult>,
    /**
     * Accelerator placement measured directly, by a probe that owns its own ORT session rather
     * than by the factory. Empty when the probe could not run at all. This is the only field in
     * this report that may speak about node placement; nothing else is evidence of it.
     */
    val partitionProbe: List<QnnPartitionProbe.QnnPartitionReport>,
    val loadedQnnLibraries: List<String>,
    val renderedDepth: Bitmap?
) {
    /** The pass whose output image is rendered: strict if it ran, else hybrid, else CPU. */
    val primaryPass: DepthPassResult?
        get() = passes.firstOrNull { it.mode == ExecutionMode.STRICT_QNN && it.healthy }
            ?: passes.firstOrNull { it.mode == ExecutionMode.QNN_THEN_CPU && it.healthy }
            ?: passes.firstOrNull { it.healthy }
}

/**
 * Owns the benchmark: a dedicated executor, a re-entrancy guard, and the measurement loop.
 *
 * Ownership: the executor is created here and closed here. MainActivity must call close() so
 * the thread cannot outlive the screen that started it.
 *
 * Why warmup runs are reported separately from measured runs: the first QNN inference finalizes
 * and caches a context graph. That cost is real but it is a one-off, so folding it into the
 * average would misrepresent steady-state latency.
 *
 * The model is read once and reused across every mode. Reading a 100 MB asset three times would
 * dominate the run and make the numbers incomparable.
 */
class DepthBenchmarkRunner(context: Context) : Closeable {

    private val appContext = context.applicationContext
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "depth-benchmark").apply { isDaemon = true }
    }
    private val runInProgress = AtomicBoolean(false)

    /** Runs at most one benchmark at a time; a second tap is rejected rather than queued. */
    fun submit(onReport: (DepthBenchmarkReport) -> Unit): Boolean {
        if (!runInProgress.compareAndSet(false, true)) return false
        executor.submit {
            val report = try {
                benchmark()
            } catch (error: Exception) {
                failureReport(error)
            }
            onReport(report)
            runInProgress.set(false)
        }
        return true
    }

    override fun close() {
        executor.shutdownNow()
    }

    private fun failureReport(error: Exception): DepthBenchmarkReport = DepthBenchmarkReport(
        graphDescription = "n/a",
        modelBytes = 0,
        sourceImageSize = "n/a",
        partitionProbe = emptyList(),
        passes = listOf(
            DepthPassResult(
                mode = ExecutionMode.STRICT_QNN,
                graphDescription = "n/a",
                succeeded = false,
                failure = "${error.javaClass.simpleName}: ${error.message}",
                activeBackend = "none",
                sessionSummary = emptyList(),
                firstInferenceMs = 0L,
                warmupLatenciesMs = emptyList(),
                measuredLatenciesMs = emptyList(),
                preprocessMs = 0L,
                outputElementCount = 0,
                nonFiniteCount = 0,
                outputMin = 0f,
                outputMax = 0f,
                outputMean = 0f
            )
        ),
        loadedQnnLibraries = OnnxQnnRuntimeEnvironment.loadedLibraries(),
        renderedDepth = null
    )

    private fun benchmark(): DepthBenchmarkReport {
        val modelBytes = appContext.assets.open(MODEL_ASSET).use { it.readBytes() }
        val sourceBitmap = appContext.assets.open(IMAGE_ASSET).use { stream ->
            BitmapFactory.decodeStream(stream)
        } ?: error("could not decode $IMAGE_ASSET")

        val passes = mutableListOf<DepthPassResult>()
        var graphDescription = "n/a"
        var renderedDepth: Bitmap? = null
        // First non-empty option map any pass observed. A CPU-only pass reports an empty map,
        // because the factory applies no QNN options at all in that mode.
        var observedProviderOptions: Map<String, String>? = null

        for (mode in ExecutionMode.entries) {
            val result = runMode(modelBytes, sourceBitmap, mode)
            if (graphDescription == "n/a") graphDescription = result.pass.graphDescription
            if (renderedDepth == null) renderedDepth = result.depth
            if (observedProviderOptions == null && result.providerOptions.isNotEmpty()) {
                observedProviderOptions = result.providerOptions
            }
            passes += result.pass
        }

        // The probe reuses production's provider options verbatim but owns its own ORT session,
        // so its placement numbers are independent of whatever the factory decided to report.
        // It runs after the passes because it needs a real session's diagnostics to source the
        // option map from; a probe fed invented options would measure a different configuration.
        val partitionProbe = runPartitionProbe(modelBytes, observedProviderOptions)

        return DepthBenchmarkReport(
            graphDescription = graphDescription,
            modelBytes = modelBytes.size,
            sourceImageSize = "${sourceBitmap.width}x${sourceBitmap.height}",
            passes = passes,
            partitionProbe = partitionProbe,
            loadedQnnLibraries = OnnxQnnRuntimeEnvironment.loadedLibraries(),
            renderedDepth = renderedDepth
        )
    }

    /**
     * Runs the partition probe, or reports nothing rather than something wrong.
     *
     * A probe that cannot run yields an empty list, which renders as "not measured" — never as
     * "no acceleration". The distinction matters: the first means the harness failed to find out,
     * the second is a claim about the hardware.
     */
    private fun runPartitionProbe(
        modelBytes: ByteArray,
        productionOptions: Map<String, String>?
    ): List<QnnPartitionProbe.QnnPartitionReport> {
        return if (productionOptions == null) {
            emptyList()
        } else {
            runCatching {
                QnnPartitionProbe.probeAll(
                    context = appContext,
                    modelBytes = modelBytes,
                    productionProviderOptions = productionOptions
                )
            }.getOrElse { emptyList() }
        }
    }

    /**
     * One measured pass plus what was observed about the backend while running it.
     *
     * [providerOptions] is the QNN option map the session factory actually applied. The probe
     * is handed this rather than a map of its own, so it measures the production configuration.
     */
    private class ModeOutcome(
        val pass: DepthPassResult,
        val depth: Bitmap?,
        val providerOptions: Map<String, String>
    )

    private fun runMode(modelBytes: ByteArray, sourceBitmap: Bitmap, mode: ExecutionMode): ModeOutcome =
        // Each mode is isolated: a strict-mode failure is a result to report, not a reason to
        // abandon the QNN-attempt and CPU passes that make it interpretable.
        runCatching { measureMode(modelBytes, sourceBitmap, mode) }
            .getOrElse { error -> ModeOutcome(failedPass(mode, error), null, emptyMap()) }

    private fun failedPass(mode: ExecutionMode, error: Throwable): DepthPassResult = DepthPassResult(
        mode = mode,
        graphDescription = "n/a",
        succeeded = false,
        failure = "${error.javaClass.simpleName}: ${error.message}",
        activeBackend = "none",
        sessionSummary = emptyList(),
        firstInferenceMs = 0L,
        warmupLatenciesMs = emptyList(),
        measuredLatenciesMs = emptyList(),
        preprocessMs = 0L,
        outputElementCount = 0,
        nonFiniteCount = 0,
        outputMin = 0f,
        outputMax = 0f,
        outputMean = 0f
    )

    private fun measureMode(modelBytes: ByteArray, sourceBitmap: Bitmap, mode: ExecutionMode): ModeOutcome {
        val estimator = OnnxDepthEstimator.open(appContext, modelBytes, mode)
        estimator.use {
            val scaled = Bitmap.createScaledBitmap(
                sourceBitmap,
                estimator.inputWidth,
                estimator.inputHeight,
                true
            )
            val preprocessStartNs = System.nanoTime()
            // RGB/255, because the graph normalizes internally: immediately after its input
            // boundary the export applies Sub then Div with the ImageNet mean and standard
            // deviation. Pre-normalizing here would apply it twice.
            val pixelValues = DepthImageCodec.toPixelValues(scaled, estimator.batchSize)
            val preprocessMs = (System.nanoTime() - preprocessStartNs) / 1_000_000L

            val warmup = ArrayList<Long>(WARMUP_RUNS)
            val measured = ArrayList<Long>(MEASURED_RUNS)
            var firstOutput: FloatArray? = null

            for (run in 0 until (WARMUP_RUNS + MEASURED_RUNS)) {
                var depth: FloatArray? = null
                val elapsedMs = measureNanoTime { depth = estimator.estimateDepth(pixelValues) } / 1_000_000L
                if (run == 0) firstOutput = depth
                if (run < WARMUP_RUNS) warmup.add(elapsedMs) else measured.add(elapsedMs)
            }

            val output = requireNotNull(firstOutput) { "benchmark produced no output" }
            val health = summarize(output)
            // The depth map's geometry is the output tensor's, not the input's. The DA-V2 decoder
            // upsamples five times before the head, so rendering at inputWidth x inputHeight is a
            // coincidence of this graph rather than a contract, and a graph that changed it would
            // produce a sheared map instead of an error. The estimator hands back a flat buffer
            // and does not publish the output shape, so the geometry is recovered from the actual
            // element count under the head's channel count and cross-checked against the input;
            // a graph that does not factor that way fails the pass rather than rendering a lie.
            val geometry = depthGeometry(
                elementCount = output.size,
                declaredWidth = estimator.outputWidth,
                declaredHeight = estimator.outputHeight,
                declaredChannels = estimator.outputChannels
            )
            val rendered = DepthImageCodec.renderDepth(output, geometry.width, geometry.height)

            return ModeOutcome(
                depth = rendered,
                providerOptions = estimator.diagnostics.qnnProviderOptions,
                pass = DepthPassResult(
                    mode = mode,
                    graphDescription = estimator.describeGraph(),
                    succeeded = true,
                    failure = null,
                    activeBackend = estimator.diagnostics.activeBackend.label,
                    sessionSummary = summarizeSession(estimator),
                    firstInferenceMs = warmup.firstOrNull() ?: 0L,
                    warmupLatenciesMs = warmup,
                    measuredLatenciesMs = measured,
                    preprocessMs = preprocessMs,
                    outputElementCount = health.count,
                    nonFiniteCount = health.nonFinite,
                    outputMin = health.min,
                    outputMax = health.max,
                    outputMean = health.mean
                )
            )
        }
    }

    private fun summarizeSession(estimator: OnnxDepthEstimator): List<String> {
        val d = estimator.diagnostics
        return listOf(
            "activeBackend=${d.activeBackend.label}",
            "qnnProviderRegistered=${d.qnnProviderRegistered}",
            "availableProviders=${d.availableProviders.joinToString()}",
            "graphCompilationStatus=${d.graphCompilationStatus}",
            "contextCacheHit=${d.contextCacheHit}",
            "sessionLoadMs=${d.modelLoadTimeMs}",
            "qnnRuntimeVersion=${d.qnnRuntimeVersion}",
            "qnnMissingLibraries=${d.qnnMissingLibraries.joinToString().ifBlank { "none" }}",
            "qnnOptions=${d.qnnProviderOptions.entries.joinToString { "${it.key}=${it.value}" }}"
        )
    }

    private data class OutputHealth(
        val count: Int,
        val nonFinite: Int,
        val min: Float,
        val max: Float,
        val mean: Float
    )

    private data class DepthGeometry(val width: Int, val height: Int)

    /**
     * Spatial extent of a depth map, from the number of elements the model actually returned.
     *
     * Depth Anything's head ends in a single-channel Convolution, so the spatial plane is the
     * element count divided by [DEPTH_HEAD_CHANNELS]. The result is required to be an exact
     * square *and* to match the input's pixel count: either check failing means the channel-count
     * assumption no longer describes this graph, which is worth an error rather than a plausible
     * looking render.
     */
    private fun depthGeometry(
        elementCount: Int,
        declaredWidth: Int,
        declaredHeight: Int,
        declaredChannels: Int
    ): DepthGeometry {
        require(elementCount > 0) { "model returned an empty depth map" }
        // The shape comes from the output tensor's declared type, and this only cross-checks that
        // the buffer actually returned matches it. Deriving the geometry from the element count
        // instead would silently mis-render any model whose head does not emit one channel at the
        // input's spatial size — and would look correct for the one model that does.
        val expected = declaredWidth.toLong() * declaredHeight.toLong() * declaredChannels.toLong()
        require(expected == elementCount.toLong()) {
            "output tensor declares ${declaredWidth}x$declaredHeight x$declaredChannels " +
                "($expected elements) but the session returned $elementCount"
        }
        return DepthGeometry(width = declaredWidth, height = declaredHeight)
    }

    private fun summarize(depthValues: FloatArray): OutputHealth {
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        var sum = 0.0
        var nonFinite = 0
        for (value in depthValues) {
            if (!value.isFinite()) {
                nonFinite++
                continue
            }
            if (value < min) min = value
            if (value > max) max = value
            sum += value
        }
        val finiteCount = depthValues.size - nonFinite
        return OutputHealth(
            count = depthValues.size,
            nonFinite = nonFinite,
            min = if (min.isFinite()) min else 0f,
            max = if (max.isFinite()) max else 0f,
            mean = if (finiteCount > 0) (sum / finiteCount).toFloat() else 0f
        )
    }

    private companion object {
        const val MODEL_ASSET = "depth_anything_v2.onnx"
        const val IMAGE_ASSET = "test_aerial_20m.jpg"
        const val WARMUP_RUNS = 3
        const val MEASURED_RUNS = 20
        const val DEPTH_HEAD_CHANNELS = 1
    }
}
