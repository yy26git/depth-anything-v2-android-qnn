package com.yy26git.depthanythingv2qnn

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import java.security.MessageDigest
import com.yy26git.depthanythingv2qnn.logging.Logger
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ThreadFactory
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Turns live camera frames into a depth mask, continuously, without ever blocking a camera thread.
 *
 * This is the component that replaced the still-image benchmark as the thing the screen actually
 * shows. The old screen inferred one bundled JPEG and rendered it forever, which is why the mask
 * looked stale: it was not a live result, it was a frozen one.
 *
 * The central constraint is that inference and frame arrival have different rates. On the verified
 * device the completed QNN profile showed one HTP partition and no CPU nodes; live depth updates
 * averaged about 3.25 per second while the camera delivered roughly 17-23 frames per second. The
 * live rate includes camera conversion, scheduling and rendering around the measured inference.
 * Three rules follow, and they are what keep this honest rather than merely working:
 *
 *  1. **Frames are dropped, never queued.** [inFlight] is the whole mechanism. If an inference is
 *     already running the arriving frame is discarded immediately and counted. A queue would grow
 *     without bound and would keep showing depth for a scene the camera left seconds ago, which is
 *     a stale mask wearing a different bug's clothes.
 *  2. **The real rate is logged, never smoothed.** No frame holding, no interpolation, no
 *     pretending. What the logcat says is what the hardware does.
 *  3. **One inference at a time, by construction.** [OnnxDepthEstimator.estimateDepth] wraps a
 *     single OrtSession and is not thread-safe. The single-thread executor plus the [inFlight]
 *     guard together guarantee at most one concurrent call, which is the only reason a plain
 *     executor would have been safe.
 *
 * The model asset is ~100 MB, so it is read and the session is opened on [worker], never on the
 * caller's thread. Callers therefore get asynchronous results through [Listener] and may be called
 * before a session exists.
 */
class LiveDepthPipeline(
    context: Context,
    private val listener: Listener,
    private val mainExecutor: java.util.concurrent.Executor
) {

    fun interface Listener {
        /** A depth mask is ready. Always called on the main thread. */
        fun onDepthMask(mask: DepthMask)
    }

    /** Presentation plus canonical depth provenance and copied camera projection metadata. */
    data class DepthMask(
        val bitmap: Bitmap,
        val frame: DepthFrame,
        val cameraGeometry: CameraFrameGeometry
    )

    private val appContext: Context = context.applicationContext
    private val logger = Logger(LOG_TAG)
    // Counter and clock share one origin, including cold model loading.
    private val pipelineStartedAtNanos = System.nanoTime()

    /**
     * True from the moment a frame is accepted until its result has been delivered.
     *
     * Checked and set as one compare-and-set so two camera threads cannot both win the race and
     * start two inferences against the same non-thread-safe session.
     */
    private val inFlight = AtomicBoolean(false)

    /** Set by [close] so a frame that arrives during teardown is refused rather than queued. */
    private val closed = AtomicBoolean(false)

    private val framesOffered = AtomicLong(0L)
    private val framesDroppedBusy = AtomicLong(0L)
    private val framesDroppedClosed = AtomicLong(0L)
    private val framesTooSmall = AtomicLong(0L)
    private val inferencesCompleted = AtomicLong(0L)
    private val inferenceErrors = AtomicLong(0L)

    /**
     * Daemon so a forgotten [close] cannot keep the process alive; the harness is evidence-gathering
     * and must never be the reason an app refuses to exit.
     */
    private val worker = Executors.newSingleThreadExecutor(
        ThreadFactory { runnable -> Thread(runnable, "depth-infer").apply { isDaemon = true } }
    )

    @Volatile
    private var estimator: OnnxDepthEstimator? = null

    @Volatile
    private var modelIdentity: ModelIdentity? = null

    init {
        // Started eagerly rather than on the first frame: the 100 MB read is slow enough that doing
        // it lazily would mean the user's first seconds of camera show no depth at all.
        submitSafely {
            estimator = openEstimator()
            logger.info(
                "pipeline ready session=${estimator?.describeGraph()} " +
                    "backend=${estimator?.diagnostics?.activeBackend} " +
                    "note=${if (estimator?.diagnostics?.providerRegistrationSucceeded == true) "HTP" else "CPU — strict QNN initialization failed"}"
            )
        }
    }

    /**
     * Opens the best session this device can actually give us.
     *
     * Strict QNN is tried first so that a working HTP would be used rather than silently skipped.
     * The original UINT16 model uses a borrowed-session bridge so its quantized boundary
     * survives Java I/O. CPU remains an explicit, logged recovery if strict initialization fails. The failure is logged, not thrown away, because "it fell back to CPU" is a finding and
     * hiding it would let a future regression read as normal.
     */
    private fun openEstimator(): OnnxDepthEstimator {
        val modelBytes = appContext.assets.open(MODEL_ASSET).use { it.readBytes() }
        modelIdentity = ModelIdentity(
            OnnxDepthEstimator.MODEL_ID,
            MessageDigest.getInstance("SHA-256").digest(modelBytes).joinToString("") { "%02x".format(it) },
            MODEL_SIDE, MODEL_SIDE
        )
        logger.info("model asset loaded bytes=${modelBytes.size}")
        return try {
            OnnxDepthEstimator.open(
                context = appContext,
                modelBytes = modelBytes,
                mode = ExecutionMode.STRICT_QNN
            )
        } catch (error: Exception) {
            logger.error("strict QNN session failed, falling back to CPU", error)
            OnnxDepthEstimator.open(
                context = appContext,
                modelBytes = modelBytes,
                mode = ExecutionMode.CPU
            )
        }
    }

    /**
     * Offers one analysed camera frame to the pipeline.
     *
     * The conversion happens on the caller's thread and the inference on [worker]. That split is
     * not an optimisation, it is forced by ownership: CameraX reuses the image buffer and the
     * caller closes the proxy the moment this returns, so the bytes must be copied out before the
     * handoff or the worker would read memory another frame has already overwritten.
     */
    fun submit(
        planes: YuvPlanes,
        geometry: CameraFrameGeometry
    ) {
        if (closed.get()) {
            framesDroppedClosed.incrementAndGet()
            return
        }
        framesOffered.incrementAndGet()

        val transform = DepthImageTransform(geometry.crop, geometry.rotation, MODEL_SIDE, MODEL_SIDE)
        if (!inFlight.compareAndSet(false, true)) {
            framesDroppedBusy.incrementAndGet()
            return
        }

        val acceptedNanos = SystemClock.elapsedRealtimeNanos()
        val pixels = try {
            DepthImageCodec.toModelPixelValues(planes, transform)
        } catch (error: Exception) {
            inFlight.set(false)
            inferenceErrors.incrementAndGet()
            logger.error("frame conversion failed for ${geometry.sourceWidth}x${geometry.sourceHeight}", error)
            return
        }

        try {
            worker.execute { runInference(pixels, transform, geometry, acceptedNanos) }
        } catch (rejected: RejectedExecutionException) {
            // close() won the race. The frame is already converted; there is simply nowhere to run it.
            inFlight.set(false)
            framesDroppedClosed.incrementAndGet()
        }
    }

    private fun runInference(
        pixels: FloatArray,
        transform: DepthImageTransform,
        geometry: CameraFrameGeometry,
        acceptedNanos: Long
    ) {
        try {
            val session = estimator
            if (session == null) {
                // Still loading the 100 MB asset. Dropping is correct: there is no session to run.
                inFlight.set(false)
                framesDroppedBusy.incrementAndGet()
                return
            }
            val startedAt = System.nanoTime()
            val depth = session.estimateDepth(pixels)
            val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
            val frame = DepthFrame(
                frameId = geometry.timestamp.nanos,
                captureTimestamp = geometry.timestamp,
                sourceWidth = geometry.sourceWidth,
                sourceHeight = geometry.sourceHeight,
                crop = geometry.crop,
                rotation = geometry.rotation,
                depthWidth = session.outputWidth,
                depthHeight = session.outputHeight,
                model = requireNotNull(modelIdentity),
                backend = if (session.diagnostics.activeBackend == com.yy26git.depthanythingv2qnn.settings.DepthBackend.CPU)
                    DepthBackend.CPU else DepthBackend.QNN_HTP,
                enqueuedNanos = acceptedNanos,
                resultNanos = SystemClock.elapsedRealtimeNanos(),
                verifiedCaptureClock = null,
                disparity = depth,
                inputTransform = transform
            )
            val mask = DepthMask(
                bitmap = DepthImageCodec.renderDepth(depth, session.outputWidth, session.outputHeight, transform),
                frame = frame,
                cameraGeometry = geometry
            )
            logger.info("depth geometry frame=${frame.frameId} source=${geometry.sourceWidth}x${geometry.sourceHeight} " +
                "crop=${geometry.crop} rotation=${geometry.rotation.degrees} validModel=" +
                "${transform.padLeft},${transform.padTop},${transform.resizedWidth},${transform.resizedHeight} " +
                "generation=${geometry.generation} inferenceMs=$elapsedMs")
            inferencesCompleted.incrementAndGet()
            reportRateIfChanged()
            mainExecutor.execute {
                if (!closed.get()) listener.onDepthMask(mask)
            }
        } catch (error: Exception) {
            inferenceErrors.incrementAndGet()
            logger.error("depth inference failed", error)
        } finally {
            // Released last, and unconditionally: the next frame must never start before this one
            // has fully finished with the session.
            inFlight.set(false)
        }
    }

    private var lastReportedCount = 0L

    /**
     * Logs the achieved rate, but only every [REPORT_EVERY] inferences so the log stays readable.
     *
     * Cumulative count and elapsed time both begin at pipeline construction. Starting the clock
     * at the fifth completion while retaining all five counts inflates the reported update rate.
     * Startup-inclusive rate is distinct from the per-inference duration logged above.
     */
    @Synchronized
    private fun reportRateIfChanged() {
        val completed = inferencesCompleted.get()
        if (completed - lastReportedCount < REPORT_EVERY) return
        lastReportedCount = completed
        val now = System.nanoTime()
        val elapsedSeconds = (now - pipelineStartedAtNanos) / 1_000_000_000.0
        val fps = if (elapsedSeconds > 0.0) completed / elapsedSeconds else 0.0
        logger.info(
            String.format(
                java.util.Locale.US,
                "depth rate inferences=%d fps=%.2f elapsedS=%.1f offered=%d droppedBusy=%d " +
                    "tooSmall=%d errors=%d",
                completed,
                fps,
                elapsedSeconds,
                framesOffered.get(),
                framesDroppedBusy.get(),
                framesTooSmall.get(),
                inferenceErrors.get()
            )
        )
    }

    /**
     * Releases the executor and the session. Safe to call more than once, because both the Activity
     * and a test teardown may reach it and a double shutdown would otherwise throw.
     */
    fun close() {
        if (!closed.compareAndSet(false, true)) return
        // Queue closure behind active inference; session.run and session.close must not race.
        worker.execute {
            estimator?.close()
            estimator = null
        }
        worker.shutdown()
    }

    private fun submitSafely(block: () -> Unit) {
        try {
            worker.execute(block)
        } catch (rejected: RejectedExecutionException) {
            logger.error("pipeline closed before startup task could run", rejected)
        }
    }

    companion object {
        private const val LOG_TAG = "DepthQNN"
        private const val MODEL_ASSET = "depth_anything_v2.onnx"

        /** Fixed tensor edge; source viewport sizes are resampled into this canvas. */
        const val MODEL_SIDE = 518

        private const val REPORT_EVERY = 5L
    }
}
