package com.yy26git.depthanythingv2qnn

import android.content.Context
import android.content.pm.PackageManager
import android.util.Size
import android.view.Surface
import androidx.camera.core.CameraInfoUnavailableException
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import com.yy26git.depthanythingv2qnn.logging.Logger
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicLong

/** Why there is no preview. Each case gets its own message; they are not interchangeable. */
enum class CameraFailure {
    /** The device reports no camera at all, so nothing can be bound. */
    NO_CAMERA_HARDWARE,

    /** The device has cameras, but not one matching the requested selector. */
    NO_BACK_CAMERA,

    /** The provider or the bind itself failed — usually the camera is held by another process. */
    BIND_FAILED
}

/** The controller's report back to whoever started it. Terminal states are terminal. */
sealed interface CameraStatus {

    /** Nothing is bound. The initial state, and the state after [CameraDepthController.unbind]. */
    object Idle : CameraStatus

    /** A bind was requested and the provider is still initializing. */
    object Starting : CameraStatus

    /** A camera is bound and the lifecycle will drive its start/stop. */
    data class Streaming(val description: String) : CameraStatus

    /** The camera will not be available. [failure] selects the message the UI shows. */
    data class Failed(val failure: CameraFailure, val detail: String) : CameraStatus
}

/**
 * Owns one CameraX binding: creating the [Preview] and [ImageAnalysis] use cases, binding them to a
 * lifecycle, and unbinding exactly what it bound.
 *
 * It still knows nothing about the model or the session — the [FrameListener] is handed raw YUV
 * planes and nothing else. That separation is what lets the preview keep working when the model
 * does not: a failure to load the 100 MB asset cannot take the camera down with it.
 *
 * What changed on 2026-10-19: this controller previously registered no frame callback at all, so
 * that nothing could later run a model on the main thread by accident. That was right when the
 * screen was a benchmark and wrong when it became a live depth view. The guarantee it used to give
 * is now structural instead — the analyzer runs on [analysisExecutor], never the main thread, and
 * the pipeline drops frames rather than queueing.
 *
 * Ownership and lifetime rules this class exists to enforce:
 *  - The [Context] is held as the application context, so the controller can never pin an Activity.
 *  - The [LifecycleOwner], the [PreviewView] and the [Listener] are parameters of [bind], never
 *    fields. Only the bound use cases and their provider are kept, and both are dropped in
 *    [unbindOwnedUseCases].
 *  - [bind] and [unbind] are each exactly one call into CameraX: every bind has its unbind.
 *  - A bind that finishes after [unbind], or after a newer bind, is refused by generation rather
 *    than applied — a late continuation must not resurrect a torn-down camera. Analysed frames are
 *    held to the same generation check.
 *  - Every [ImageProxy] is closed exactly once, on every path, because CameraX will not deliver
 *    another frame until the current one is closed.
 */
class CameraDepthController(
    context: Context,
    private val mainExecutor: Executor
) {

    fun interface Listener {
        fun onCameraStatus(status: CameraStatus)
    }

    /**
     * One analysed camera frame, delivered on the analysis thread.
     *
     * The caller must not retain [planes]: CameraX reuses the buffer and this controller closes the
     * proxy as soon as the call returns. Anything that needs those bytes later must copy them during
     * the call.
     */
    fun interface FrameListener {
        fun onFrame(
            planes: YuvPlanes,
            geometry: CameraFrameGeometry
        )
    }

    private val appContext: Context = context.applicationContext

    /**
     * Runs the YUV conversion off the camera threads.
     *
     * Separate from [mainExecutor] on purpose: the conversion touches ~800k floats per frame and must
     * never land on the main thread, while [mainExecutor] exists only for CameraX's own callbacks.
     */
    private val analysisExecutor: Executor = Executor { runnable ->
        Thread(runnable, "camera-analysis").apply { isDaemon = true }.start()
    }

    /**
     * Incremented by every bind and every unbind. A continuation captures the value it started
     * with and refuses to act once it no longer matches, so a bind whose provider was slow to
     * initialize cannot re-bind after teardown or duplicate an existing session.
     */
    private val generation = AtomicLong(0L)

    @Volatile
    private var provider: ProcessCameraProvider? = null

    @Volatile
    private var preview: Preview? = null

    @Volatile
    private var analysis: ImageAnalysis? = null

    @Volatile
    private var listener: Listener? = null

    @Volatile
    private var frameListener: FrameListener? = null

    /**
     * True between a [bind] and the moment that bind resolves either way.
     *
     * Tracked separately from [preview] because binding is asynchronous: while the provider is still
     * initializing there is no use case yet, and a caller checking only [preview] would read that as
     * "nothing is bound" and bind a second time over the top of the first.
     */
    @Volatile
    private var bindPending = false

    /** True while this controller owns a binding, including one still being established. */
    val isActive: Boolean get() = preview != null || analysis != null || bindPending

    fun isCurrentGeneration(value: Long): Boolean = value == generation.get()

    /**
     * Binds the back camera to [owner], routes its frames into [previewView], and delivers analysed
     * frames to [frameListener].
     *
     * [frameListener] may be null, in which case analysis is not bound at all and only the preview
     * runs — binding an analyzer whose output is discarded would cost power for nothing.
     *
     * Safe to call again while a binding is live: the previous one is released first, so this is a
     * replace rather than a second session on the same camera.
     */
    fun bind(
        owner: LifecycleOwner,
        previewView: PreviewView,
        listener: Listener,
        frameListener: FrameListener? = null
    ) {
        val myGeneration = generation.incrementAndGet()
        unbindOwnedUseCases()
        this.listener = listener
        this.frameListener = frameListener
        bindPending = true
        listener.onCameraStatus(CameraStatus.Starting)

        if (!appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            finish(myGeneration, CameraStatus.Failed(CameraFailure.NO_CAMERA_HARDWARE, "no camera feature"))
            return
        }

        // getInstance can throw outright if CameraX cannot initialise at all (missing native
        // dependency, no camera service). Unguarded, that would crash the screen and turn a
        // reportable camera problem into a dead app.
        val providerFuture = try {
            ProcessCameraProvider.getInstance(appContext)
        } catch (error: Exception) {
            finish(
                myGeneration,
                CameraStatus.Failed(
                    failure = CameraFailure.BIND_FAILED,
                    detail = "provider init failed: ${error.javaClass.simpleName}: ${error.message}"
                )
            )
            return
        }

        providerFuture.addListener(
            Runnable {
                // The screen may be gone, or a newer bind/unbind may have landed, while the
                // provider was initializing. Both make this continuation obsolete.
                if (myGeneration != generation.get()) return@Runnable

                val cameraProvider = try {
                    providerFuture.get()
                } catch (error: Exception) {
                    finish(
                        myGeneration,
                        CameraStatus.Failed(
                            failure = CameraFailure.BIND_FAILED,
                            detail = "provider unavailable: ${error.javaClass.simpleName}: ${error.message}"
                        )
                    )
                    return@Runnable
                }

                val hasBackCamera = try {
                    cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)
                } catch (error: CameraInfoUnavailableException) {
                    finish(
                        myGeneration,
                        CameraStatus.Failed(
                            failure = CameraFailure.NO_BACK_CAMERA,
                            detail = "camera info unavailable: ${error.message}"
                        )
                    )
                    return@Runnable
                }
                if (!hasBackCamera) {
                    finish(
                        myGeneration,
                        CameraStatus.Failed(
                            failure = CameraFailure.NO_BACK_CAMERA,
                            detail = "device exposes no back camera"
                        )
                    )
                    return@Runnable
                }

                val targetRotation = previewView.display?.rotation ?: Surface.ROTATION_0
                // Resolution belongs to the Builder; setTargetRotation and setSurfaceProvider are
                // declared on the built Preview. The two cannot share one `apply` block.
                val previewBuilder = Preview.Builder().apply {
                    // A shared viewport defines field of view; the requested sizes may fall back.
                    setResolutionSelector(sharedResolutionSelector())
                }
                val previewUseCase = previewBuilder.build().apply {
                    setTargetRotation(targetRotation)
                    setSurfaceProvider(previewView.surfaceProvider)
                }

                val analysisUseCase = frameListener?.let { buildAnalysis(myGeneration, it, targetRotation) }

                try {
                    val viewport = requireNotNull(previewView.viewPort) { "preview has not been laid out" }
                    val group = UseCaseGroup.Builder().setViewPort(viewport).addUseCase(previewUseCase)
                    analysisUseCase?.let { group.addUseCase(it) }
                    val camera = cameraProvider.bindToLifecycle(
                        owner, CameraSelector.DEFAULT_BACK_CAMERA, group.build()
                    )
                    provider = cameraProvider
                    preview = previewUseCase
                    analysis = analysisUseCase
                    finish(
                        myGeneration,
                        CameraStatus.Streaming(
                            description = "back camera bound " +
                                "(implementation=${camera.cameraInfo.implementationType}, " +
                                "sensorRotation=${camera.cameraInfo.sensorRotationDegrees}, " +
                                "analysis=${analysisUseCase != null})"
                        )
                    )
                } catch (error: IllegalArgumentException) {
                    // CameraX raises this when the selector matches no camera at bind time.
                    finish(
                        myGeneration,
                        CameraStatus.Failed(
                            failure = CameraFailure.NO_BACK_CAMERA,
                            detail = "${error.javaClass.simpleName}: ${error.message}"
                        )
                    )
                } catch (error: Exception) {
                    // The common case is CameraUnavailableException.CAMERA_IN_USE: another process
                    // holds the camera. That is reported, not worked around.
                    finish(
                        myGeneration,
                        CameraStatus.Failed(
                            failure = CameraFailure.BIND_FAILED,
                            detail = "${error.javaClass.simpleName}: ${error.message}"
                        )
                    )
                }
            },
            mainExecutor
        )
    }

    /**
     * Releases the camera. The matching half of [bind], and safe to call when nothing is bound.
     *
     * After this returns, every in-flight continuation from an earlier bind has been refused and
     * the controller holds no reference to the Activity, its view, or its listener.
     */
    fun unbind() {
        generation.incrementAndGet()
        bindPending = false
        unbindOwnedUseCases()
        listener = null
        frameListener = null
    }

    private fun finish(myGeneration: Long, status: CameraStatus) {
        if (myGeneration != generation.get()) return
        val target = listener ?: return
        bindPending = false
        if (status is CameraStatus.Failed) {
            // In-flight depth belongs to the failed binding and must not re-show its mask.
            generation.incrementAndGet()
            unbindOwnedUseCases()
        }
        target.onCameraStatus(status)
    }

    /** Resolution is a preference; ViewPort supplies the authoritative visible crop. */
    private fun sharedResolutionSelector(): ResolutionSelector = ResolutionSelector.Builder()
        .setResolutionStrategy(
            ResolutionStrategy(
                Size(ANALYSIS_SIDE, ANALYSIS_SIDE),
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
            )
        )
        .build()

    private fun buildAnalysis(
        myGeneration: Long,
        consumer: FrameListener,
        targetRotation: Int
    ): ImageAnalysis? {
        if (myGeneration != generation.get()) return null
        return ImageAnalysis.Builder()
            .setResolutionSelector(sharedResolutionSelector())
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            // Mandatory while inference runs at roughly 0.5 Hz against a ~20 fps camera. Without it
            // CameraX hands us every frame and they queue without bound, so the mask would keep
            // showing depth for a scene the camera left long ago — the stale mask all over again.
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(targetRotation)
            .build()
            .apply {
                setAnalyzer(analysisExecutor) { image -> deliverFrame(image, myGeneration, consumer) }
            }
    }

    /**
     * Hands one frame to [consumer] and closes it exactly once.
     *
     * The `use` block is the whole ownership story: CameraX will not deliver another frame until this
     * one is closed, so a missed close on any path stalls analysis permanently rather than merely
     * leaking. Every early return below therefore sits inside the `use`.
     */
    private fun deliverFrame(image: ImageProxy, myGeneration: Long, consumer: FrameListener) {
        image.use { proxy ->
            // A frame that arrived after unbind, or after a newer bind, belongs to a torn-down
            // session. Dropping it is the only safe answer.
            if (myGeneration != generation.get()) return

            val width = proxy.width
            val height = proxy.height
            val planeList = proxy.planes
            if (planeList.size < 3) return
            val planes = try {
                YuvPlanes(
                    width = width,
                    height = height,
                    y = YuvPlane(planeList[0].buffer, planeList[0].rowStride, planeList[0].pixelStride),
                    u = YuvPlane(planeList[1].buffer, planeList[1].rowStride, planeList[1].pixelStride),
                    v = YuvPlane(planeList[2].buffer, planeList[2].rowStride, planeList[2].pixelStride)
                )
            } catch (error: Exception) {
                Logger(LOG_TAG).error("could not address the YUV planes", error)
                return
            }

            val rotation = try {
                Rotation.ofDegrees(proxy.imageInfo.rotationDegrees)
            } catch (error: IllegalArgumentException) {
                // ofDegrees rejects anything that is not a multiple of 90. Dropping beats guessing a
                // rotation nobody asked for, which would silently spin the whole mask.
                Logger(LOG_TAG).error("unusable rotation ${proxy.imageInfo.rotationDegrees}", error)
                return
            }

            val rect = proxy.cropRect
            if (rect.width() <= 0 || rect.height() <= 0) return
            val matrix = FloatArray(9)
            proxy.imageInfo.sensorToBufferTransformMatrix.getValues(matrix)
            val geometry = CameraFrameGeometry(
                sourceWidth = width,
                sourceHeight = height,
                crop = SourceCrop(rect.left, rect.top, rect.width(), rect.height()),
                rotation = rotation,
                timestamp = CaptureTimestamp(proxy.imageInfo.timestamp, ClockDomain.CAMERA_SENSOR_NANOS),
                generation = myGeneration,
                sensorToBuffer = matrix.toList()
            )
            consumer.onFrame(planes = planes, geometry = geometry)
        }
    }

    private fun unbindOwnedUseCases() {
        val boundPreview = preview
        val boundAnalysis = analysis
        val boundProvider = provider
        preview = null
        analysis = null
        provider = null
        if (boundProvider != null) {
            // unbind(this use case), never unbindAll(). The provider is a process-wide singleton, so
            // tearing down every session would take any other CameraX consumer in this process with
            // it — that is not this controller's session to end.
            listOfNotNull(boundPreview, boundAnalysis).forEach { useCase ->
                runCatching { boundProvider.unbind(useCase) }
            }
        }
    }

    private companion object {
        /** Preferred stream size, not a guarantee of square output. */
        const val ANALYSIS_SIDE = 640

        /** How often a too-small frame reaches the log, so one misconfiguration is one line. */
        const val SHORT_FRAME_REPORT_EVERY = 100L

        const val LOG_TAG = "DepthQNN"
    }
}
