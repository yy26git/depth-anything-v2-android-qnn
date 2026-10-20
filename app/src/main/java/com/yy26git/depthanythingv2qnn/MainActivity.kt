package com.yy26git.depthanythingv2qnn

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.yy26git.depthanythingv2qnn.logging.Logger
import kotlin.math.roundToInt

/**
 * The screen: a live camera preview with the depth mask registered on top of it.
 *
 * That is the whole job. Everything this class used to do — three benchmark modes, a long
 * diagnostics report, buttons to start and stop the camera and the inference — was removed at the
 * product owner's direction on 2026-10-19. The measurement classes (`DepthBenchmarkRunner`,
 * `QnnPartitionProbe`) are still in the repo and still produce their evidence; they are just no
 * longer what this window shows. Their output goes to logcat, which is where evidence belongs when
 * the thing on screen is a picture.
 *
 * The revision this replaces rendered depth from one bundled JPEG, so the mask could never change.
 * That was the stale-mask bug: not a caching problem, a missing live path.
 *
 * Both the camera and the inference start automatically in `onCreate`. There is no start/stop
 * control, because there is no state worth choosing between — the screen always wants to be live.
 *
 * The one thing that is *not* automatic is permission, which the user has to grant from the system.
 * A denial therefore still gets a visible recovery action; that is required UX, not diagnostics.
 *
 * ComponentActivity rather than android.app.Activity: CameraX binds to a LifecycleOwner, which the
 * platform Activity does not implement.
 */
class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: ImageView
    private lateinit var statusView: TextView
    private lateinit var recoveryButton: TextView
    private lateinit var cameraController: CameraDepthController
    private lateinit var pipeline: LiveDepthPipeline

    private val logger = Logger(LOG_TAG)

    /** Latest result owns its capture geometry; never reconstruct it from current buffer sizes. */
    private var currentMask: LiveDepthPipeline.DepthMask? = null

    /** Set in [onDestroy]. Every callback checks it before touching a view. */
    private var destroyed = false

    private var permissionRequested = false

    /**
     * Registered as a field, before onCreate: the launcher must exist before the activity reaches
     * STARTED, which a lazily created property would not guarantee.
     */
    private val cameraPermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            hidePermissionRecovery()
            startCamera()
        } else {
            showPermissionRecovery()
        }
    }

    private val cameraListener = CameraDepthController.Listener { status -> onCameraStatus(status) }

    private val pipelineListener = LiveDepthPipeline.Listener { mask -> onDepthMask(mask) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Keep this live camera viewer visible while foreground; background windows do not hold it.
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val mainExecutor = ContextCompat.getMainExecutor(applicationContext)
        cameraController = CameraDepthController(
            // Built from the application context, not `this`. The controller keeps its executors for
            // its whole life, and executors derived from the Activity would pin the screen long
            // after onDestroy.
            context = applicationContext,
            mainExecutor = mainExecutor
        )
        pipeline = LiveDepthPipeline(
            context = applicationContext,
            listener = pipelineListener,
            mainExecutor = mainExecutor
        )

        setContentView(buildLayout())

        // The model load is already running inside the pipeline; the camera starts regardless of
        // whether it will succeed, so a model failure can never present as a camera failure.
        startCamera()
    }

    private fun buildLayout(): View {
        val padding = (16 * resources.displayMetrics.density).roundToInt()

        previewView = PreviewView(this).apply {
            tag = TAG_PREVIEW
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
            setBackgroundColor(Color.BLACK)
        }
        overlayView = ImageView(this).apply {
            tag = TAG_OVERLAY
            // The mask must never distort: a stretch would register to nothing.
            scaleType = ImageView.ScaleType.MATRIX
            // renderDepth emits a fully opaque ARGB bitmap, which is right for a report that has to
            // be legible on its own. Blending it into a live view is a different job: at full
            // opacity it hides the camera completely, and the whole point is to read depth *against*
            // the scene. The blend therefore belongs to the view layer, leaving the codec's output
            // faithful for the tooling that renders the same buffer standalone.
            alpha = OVERLAY_ALPHA
            visibility = View.INVISIBLE
        }
        statusView = TextView(this).apply {
            tag = TAG_CAMERA_STATUS
            textSize = 13f
            setTextColor(Color.WHITE)
            setBackgroundColor(SEMITRANSPARENT_BACKGROUND)
            setPadding(padding, padding / 2, padding, padding / 2)
        }
        recoveryButton = TextView(this).apply {
            tag = TAG_PERMISSION_RECOVERY
            textSize = 14f
            setTextColor(Color.WHITE)
            setBackgroundColor(SEMITRANSPARENT_BACKGROUND)
            setPadding(padding, padding, padding, padding)
            visibility = View.GONE
        }

        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        root.addView(
            previewView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )
        // Added after the preview so it draws on top.
        root.addView(
            overlayView,
            FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        )
        root.addView(
            statusView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP
            )
        )
        root.addView(
            recoveryButton,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM
            )
        )
        previewView.addOnLayoutChangeListener { _, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
            if (right - left != oldRight - oldLeft || bottom - top != oldBottom - oldTop) {
                currentMask = null
                overlayView.visibility = View.INVISIBLE
                cameraController.unbind()
                previewView.post { if (!destroyed) startCamera() }
            }
        }
        return root
    }

    override fun onResume() {
        super.onResume()
        // CameraX pauses and resumes with this lifecycle, so an ordinary background/foreground
        // cycle needs no work. The case that does is CAMERA being granted from system settings while
        // we were away, which is why permission is re-checked and only a genuinely idle controller
        // is re-bound. isActive, not "is a use case bound", because a bind issued from onCreate may
        // still be initializing when this runs.
        if (hasCameraPermission() && !cameraController.isActive) {
            startCamera()
        }
    }

    override fun onDestroy() {
        destroyed = true
        // The pipeline owns the session and the inference thread; the controller owns the camera.
        // Neither may outlive the screen that started them.
        cameraController.unbind()
        pipeline.close()
        super.onDestroy()
    }

    // ---------------------------------------------------------------- camera

    private fun startCamera() {
        if (destroyed || cameraController.isActive) return
        if (previewView.width == 0 || previewView.height == 0) {
            previewView.post { if (!destroyed) startCamera() }
            return
        }
        if (!hasCameraPermission()) {
            requestCameraPermission()
            return
        }
        hidePermissionRecovery()
        cameraController.bind(
            owner = this,
            previewView = previewView,
            listener = cameraListener,
            frameListener = CameraDepthController.FrameListener { planes, geometry ->
                pipeline.submit(planes = planes, geometry = geometry)
            }
        )
    }

    private fun onCameraStatus(status: CameraStatus) {
        // A status can still arrive from a binding that was already torn down; touching the views
        // then would mean writing to a destroyed window.
        if (destroyed) return
        // Logged because a black preview is otherwise indistinguishable from a dead camera: the
        // screen looks the same whether the sensor is delivering or nothing is bound at all.
        logger.info("camera status=$status")
        statusView.text = when (status) {
            CameraStatus.Idle -> CAMERA_STOPPED
            CameraStatus.Starting -> CAMERA_STARTING
            is CameraStatus.Streaming -> CAMERA_RUNNING
            // Each failure gets its own sentence. "Camera failed" is not an actionable message, and
            // a device with no back camera needs a different action from one another app is holding.
            is CameraStatus.Failed -> when (status.failure) {
                CameraFailure.NO_CAMERA_HARDWARE ->
                    "This device has no camera, so there is nothing to show."
                CameraFailure.NO_BACK_CAMERA ->
                    "This device has no back camera, so there is nothing to show."
                CameraFailure.BIND_FAILED ->
                    "The camera could not start: ${status.detail}. " +
                        "Another app holding the camera is the usual cause."
            }
        }
        if (status !is CameraStatus.Streaming) {
            // A dead camera must not leave a frozen mask sitting on screen pretending to be live.
            overlayView.visibility = View.INVISIBLE
        }
    }

    // -------------------------------------------------------------- overlay

    /** Projects the exact accepted viewport, including input rotation and letterboxing. */
    private fun onDepthMask(mask: LiveDepthPipeline.DepthMask) {
        if (destroyed || !cameraController.isCurrentGeneration(mask.cameraGeometry.generation)) {
            mask.bitmap.recycle()
            return
        }
        currentMask = mask
        overlayView.setImageBitmap(mask.bitmap)
        applyOverlayGeometry()
    }

    private fun applyOverlayGeometry() {
        if (destroyed) return
        val mask = currentMask ?: return
        if (!cameraController.isCurrentGeneration(mask.cameraGeometry.generation)) return
        val transform = mask.frame.inputTransform ?: return
        val sensorToView = previewView.sensorToViewTransform ?: run {
            overlayView.visibility = View.INVISIBLE
            return
        }
        val sensorToBuffer = Matrix().apply { setValues(mask.cameraGeometry.sensorToBuffer.toFloatArray()) }
        val bufferToSensor = Matrix()
        if (!sensorToBuffer.invert(bufferToSensor)) {
            overlayView.visibility = View.INVISIBLE
            return
        }
        val bufferToView = Matrix().apply { setConcat(sensorToView, bufferToSensor) }
        val width = mask.bitmap.width.toFloat()
        val height = mask.bitmap.height.toFloat()
        val bitmapCorners = floatArrayOf(0f, 0f, width, 0f, width, height, 0f, height)
        val sourceCorners = FloatArray(8)
        for (index in 0..3) {
            val (x, y) = transform.modelToSource(
                bitmapCorners[2 * index] * transform.modelWidth.toDouble() / width,
                bitmapCorners[2 * index + 1] * transform.modelHeight.toDouble() / height
            )
            sourceCorners[2 * index] = x.toFloat()
            sourceCorners[2 * index + 1] = y.toFloat()
        }
        bufferToView.mapPoints(sourceCorners)
        val bitmapToView = Matrix()
        check(bitmapToView.setPolyToPoly(bitmapCorners, 0, sourceCorners, 0, 4))
        overlayView.imageMatrix = bitmapToView
        overlayView.visibility = View.VISIBLE
        logger.info("overlay projected frame=${mask.frame.frameId} view=${previewView.width}x${previewView.height} " +
            "generation=${mask.cameraGeometry.generation} corners=${sourceCorners.joinToString()}")
    }

    override fun onStop() {
        currentMask = null
        overlayView.visibility = View.INVISIBLE
        cameraController.unbind()
        super.onStop()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) applyOverlayGeometry()
    }

    // ------------------------------------------------------------ permission

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        statusView.text = CAMERA_PERMISSION_NEEDED
        // Ask once per launch. After that the recovery action owns the interaction, so a user who
        // said no is not immediately prompted again by an unrelated resume.
        if (permissionRequested) {
            showPermissionRecovery()
            return
        }
        permissionRequested = true
        cameraPermissionRequest.launch(Manifest.permission.CAMERA)
    }

    private fun showPermissionRecovery() {
        if (destroyed) return
        // shouldShowRequestPermissionRationale goes false once the system stops showing the dialog,
        // so re-firing the request would return "denied" instantly and look broken. Settings is the
        // only remaining route in that state.
        val canAskAgain = shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)
        statusView.text = CAMERA_PERMISSION_DENIED
        recoveryButton.visibility = View.VISIBLE
        recoveryButton.text =
            if (canAskAgain) "Tap to grant camera access" else "Tap to open app settings"
        recoveryButton.setOnClickListener {
            if (canAskAgain) {
                cameraPermissionRequest.launch(Manifest.permission.CAMERA)
            } else {
                openAppSettings()
            }
        }
    }

    private fun hidePermissionRecovery() {
        recoveryButton.visibility = View.GONE
    }

    private fun openAppSettings() {
        // Wrapped because the settings activity is not guaranteed on every image; a crash here would
        // be a worse answer than leaving the recovery row up.
        try {
            startActivity(
                Intent(
                    Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    Uri.fromParts("package", packageName, null)
                )
            )
        } catch (error: Exception) {
            logger.error("could not open app settings: ${error.message}")
        }
    }

    companion object {
        /**
         * Stable view handles for the instrumentation tests. The tests walk the real view tree
         * looking for these instead of the activity exposing test-only accessors, so the production
         * surface stays exactly what the screen needs.
         */
        const val TAG_PREVIEW = "depthqnn/preview"
        const val TAG_OVERLAY = "depthqnn/overlay"
        const val TAG_CAMERA_STATUS = "depthqnn/camera-status"
        const val TAG_PERMISSION_RECOVERY = "depthqnn/permission-recovery"

        private const val LOG_TAG = "DepthQNN"
        private const val SEMITRANSPARENT_BACKGROUND = 0x99000000.toInt()

        /**
         * Blend factor for the depth mask over the live preview. Low enough that scene detail stays
         * readable through it — a mask you cannot see the world behind defeats its own purpose.
         */
        private const val OVERLAY_ALPHA = 0.55f

        private const val CAMERA_STOPPED = "Camera stopped."
        private const val CAMERA_STARTING = "Starting camera…"
        private const val CAMERA_RUNNING = "Live depth"
        private const val CAMERA_PERMISSION_NEEDED = "Waiting for camera permission…"
        private const val CAMERA_PERMISSION_DENIED =
            "Camera permission is required to show live depth."
    }
}
