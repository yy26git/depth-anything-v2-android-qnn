package com.yy26git.depthanythingv2qnn

import android.Manifest
import android.content.Context
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.util.concurrent.atomic.AtomicLong
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device tests for the screen as it is now: a live camera preview with the depth mask registered
 * on top of it.
 *
 * Every test here is written against what the rewrite changed, and every one of them would have
 * failed against the dashboard that came before it. That is the bar. The previous screen had a
 * preview, a "run benchmark" button, a camera start/stop toggle and a long text report; the mask
 * it rendered came from one bundled JPEG, so it could never change. A test that passes on both
 * screens is decoration, so none of these check anything that survived the rewrite unchanged.
 *
 * The tags on `MainActivity` are the only handle these tests use. The view tree is walked exactly
 * as a user's screen lays out, rather than reaching through test-only accessors on the Activity,
 * so the assertions run against the same objects that are actually on the glass.
 *
 * None of this has been executed. No Gradle build and no adb run was performed while writing it, so
 * nothing here is a claim that these pass — the parent builds and runs them.
 */
@RunWith(AndroidJUnit4::class)
class DepthOverlayInstrumentationTest {

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    private val packageName: String
        get() = context.packageName

    @After
    fun clearPermissionState() {
        // The device is shared with the rest of the suite and with manual runs. Leaving CAMERA
        // granted would silently disarm a later denial test, and leaving it user-fixed would put
        // the next manual launch straight into the recovery screen with no way back.
        revokeCameraPermission()
        clearPermissionUserFixed()
    }

    /**
     * The overlay has to be *on* the preview, not near it and not after it.
     *
     * Against the dashboard this fails at `assertSame`: the mask was rendered into a second
     * surface of its own, below the preview rather than across it, so the user saw a picture and a
     * preview in sequence instead of depth read against the scene.
     */
    @Test
    fun overlay_sharesOneContainerWithThePreview_andDrawsAboveIt() {
        grantCameraPermission()

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val preview = requireView<PreviewView>(activity, MainActivity.TAG_PREVIEW)
                val overlay = requireView<ImageView>(activity, MainActivity.TAG_OVERLAY)
                val container = screenContainer(activity)

                assertSame(
                    "the overlay and the preview must share one FrameLayout. A mask on a separate " +
                        "surface below the preview is the layout the live view replaced.",
                    container,
                    preview.parent
                )

                // In a ViewGroup the later child is drawn last, so child order is the draw order.
                // This is the assertion that says 'overlay' rather than 'a second thing'.
                val overlayIndex = childIndex(container, overlay)
                val previewIndex = childIndex(container, preview)
                assertTrue(
                    "the overlay must be added after the preview so it draws on top of it " +
                        "(preview at index $previewIndex, overlay at index $overlayIndex)",
                    overlayIndex > previewIndex
                )
            }
        }
    }

    /**
     * The mask is blended into the preview, never laid over it opaquely.
     *
     * This regression really happened and was really fixed: `renderDepth` emits a fully opaque
     * ARGB bitmap, which is correct for a report that has to be legible on its own and completely
     * wrong for an overlay. At full opacity the mask hides the camera entirely, so the screen still
     * "worked" — it just showed a picture with no scene behind it, which is the opposite of
     * reading depth against the world.
     */
    @Test
    fun overlay_isBlended_ratherThanOpaque() {
        grantCameraPermission()

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val overlay = requireView<ImageView>(activity, MainActivity.TAG_OVERLAY)
                val alpha = overlay.alpha

                assertTrue(
                    "the depth mask must be blended into the live preview, but the overlay alpha " +
                        "was $alpha. A value of 1f hides the camera completely; a value of 0f " +
                        "shows no depth at all.",
                    alpha > 0f && alpha < 1f
                )
            }
        }
    }

    /**
     * No debug controls survive anywhere in the tree.
     *
     * Against the dashboard this fails twice over: there was a Button for the benchmark and a Button
     * for the camera start/stop toggle. The second assertion is what stops either from coming back
     * as some other widget type — the screen is pinned to exactly the four views it owns, so a new
     * control of any kind has to be a deliberate edit here rather than a quiet regression.
     */
    @Test
    fun noDebugControlsRemain_anywhereInTheViewTree() {
        grantCameraPermission()

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val buttons = collectViews(activity.window.decorView).filterIsInstance<Button>()
                assertTrue(
                    "the live depth screen has no buttons; found ${buttons.size} tagged " +
                        "${buttons.map { it.tag }}",
                    buttons.isEmpty()
                )

                val container = screenContainer(activity)
                val childTags = (0 until container.childCount)
                    .map { container.getChildAt(it).tag?.toString() }
                    .toSet()
                assertEquals(
                    "the screen must hold only the preview, the overlay, the status line and the " +
                        "permission recovery row, but held $childTags",
                    setOf(
                        MainActivity.TAG_PREVIEW,
                        MainActivity.TAG_OVERLAY,
                        MainActivity.TAG_CAMERA_STATUS,
                        MainActivity.TAG_PERMISSION_RECOVERY
                    ),
                    childTags
                )
                assertEquals(
                    "the screen must not have gained or lost a child, which would mean a control " +
                        "or a panel came back",
                    4,
                    container.childCount
                )
            }
        }
    }

    /**
     * Launching is the whole interaction: no button is pressed because there is none.
     *
     * A visible `PreviewView` proves nothing here — one with a dead binding looks identical to a
     * live one. So the evidence is frames actually arriving, which the dashboard could not produce
     * without pressing its start toggle first.
     */
    @Test
    fun launch_startsTheCameraOnItsOwn_withNoInteraction() {
        grantCameraPermission()

        val frames = AtomicLong(0)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val preview = requireView<PreviewView>(activity, MainActivity.TAG_PREVIEW)
                preview.setFrameUpdateListener(ContextCompat.getMainExecutor(activity)) { _ ->
                    frames.incrementAndGet()
                }
            }
            awaitFrames(frames, "preview frames to prove the camera is streaming unprompted")

            // The permission path has to have completed on its own too: granted, so the recovery
            // row is dismissed without the user ever having been asked.
            awaitFromTestThread("the permission recovery row to be dismissed") {
                viewVisibility(scenario, MainActivity.TAG_PERMISSION_RECOVERY) == View.GONE
            }

            scenario.onActivity { activity ->
                val status = requireView<TextView>(activity, MainActivity.TAG_CAMERA_STATUS)
                assertTrue(
                    "the status line must say something once the camera is live, was " +
                        "'${status.text}'",
                    status.text.isNotBlank()
                )
            }
        }
    }

    /**
     * The stale-mask bug, caught at the only place it can be: the overlay must receive depth from
     * the live camera, and must receive it with no button pressed.
     *
     * The dashboard had a mask too — it just never appeared until the benchmark button was tapped,
     * and then it rendered one bundled JPEG forever. Pressing nothing here and still getting a mask
     * is the difference between a live view and a frozen picture.
     *
     * The wait is long because the work is: a ~100 MB model asset is read and the session is opened
     * on a worker thread before the first inference can run, and that first inference is CPU bound
     * at roughly 1400-2000 ms because the QNN EP cannot partition this graph. It is a bounded wait
     * for a real state, never a sleep — on success this returns as soon as the mask lands.
     */
    @Test
    fun liveFrames_reachTheOverlay_asAMask_withNoUserAction() {
        grantCameraPermission()

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            awaitFromTestThread(
                what = "the first live depth mask to be drawn over the preview",
                timeoutMs = MASK_TIMEOUT_MS
            ) {
                viewVisibility(scenario, MainActivity.TAG_OVERLAY) == View.VISIBLE
            }

            scenario.onActivity { activity ->
                val overlay = requireView<ImageView>(activity, MainActivity.TAG_OVERLAY)
                assertNotNull(
                    "a visible overlay must actually carry a mask bitmap",
                    overlay.drawable
                )
            }
        }
    }

    /** Exercise the real result bitmap and the Activity's projection, not just view visibility. */
    @Test
    fun liveDepth_validPixelsCoverEveryPreviewEdge() {
        grantCameraPermission()
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            awaitFromTestThread("depth projection", timeoutMs = MASK_TIMEOUT_MS) {
                viewVisibility(scenario, MainActivity.TAG_OVERLAY) == View.VISIBLE
            }
            scenario.onActivity { activity ->
                val preview = requireView<PreviewView>(activity, MainActivity.TAG_PREVIEW)
                val overlay = requireView<ImageView>(activity, MainActivity.TAG_OVERLAY)
                val bitmap = (overlay.drawable as android.graphics.drawable.BitmapDrawable).bitmap
                var left = bitmap.width
                var top = bitmap.height
                var right = -1
                var bottom = -1
                for (y in 0 until bitmap.height) for (x in 0 until bitmap.width) {
                    if (android.graphics.Color.alpha(bitmap.getPixel(x, y)) != 0) {
                        left = minOf(left, x)
                        top = minOf(top, y)
                        right = maxOf(right, x)
                        bottom = maxOf(bottom, y)
                    }
                }
                assertTrue("no valid depth pixels", right >= left && bottom >= top)
                val bounds = android.graphics.RectF(left.toFloat(), top.toFloat(), (right + 1).toFloat(), (bottom + 1).toFloat())
                overlay.imageMatrix.mapRect(bounds)
                assertTrue("left uncovered: $bounds", bounds.left <= 1f)
                assertTrue("top uncovered: $bounds", bounds.top <= 1f)
                assertTrue("right uncovered: $bounds vs ${preview.width}", bounds.right >= preview.width - 1f)
                assertTrue("bottom uncovered: $bounds vs ${preview.height}", bounds.bottom >= preview.height - 1f)
            }
        }
    }

    /**
     * A denial stays a visible, actionable state — required UX, not diagnostics, and it has to
     * survive the rewrite rather than being collateral damage from it.
     *
     * This partly overlaps the sibling preview suite on purpose: that test was written against the
     * dashboard and had to give up its "the benchmark stays reachable" clause when the button
     * went away. The requirement underneath it — a stated problem and a visible recovery action —
     * did not go away with it.
     */
    @Test
    fun deniedPermission_showsTheRecoveryAction_andLeavesNoMaskOnScreen() {
        revokeCameraPermission()
        markPermissionUserFixed()

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            // The permission result lands after onCreate, so the recovery row is not there the
            // instant the scenario reports RESUMED. Waiting here, from the test thread which is
            // allowed to block, is the difference between a real assertion and a race.
            awaitFromTestThread("the permission recovery row to appear") {
                viewVisibility(scenario, MainActivity.TAG_PERMISSION_RECOVERY) == View.VISIBLE
            }

            scenario.onActivity { activity ->
                val status = requireView<TextView>(activity, MainActivity.TAG_CAMERA_STATUS)
                assertTrue(
                    "a denial must be stated in the status line, was '${status.text}'",
                    status.text.isNotBlank()
                )

                val recovery = requireView<TextView>(activity, MainActivity.TAG_PERMISSION_RECOVERY)
                assertEquals(
                    "the recovery action must be visible after a denial",
                    View.VISIBLE,
                    recovery.visibility
                )
                assertTrue(
                    "the recovery action must say what it does, was '${recovery.text}'",
                    recovery.text.isNotBlank()
                )

                // It has to be in the same container as the preview, or it is not reachable by the
                // user who needs it.
                assertSame(
                    "the recovery action must live in the same container as the preview",
                    screenContainer(activity),
                    recovery.parent
                )

                // No camera means no frames means no mask. Anything visible here would be the
                // frozen still-image mask the live path was built to replace.
                val overlay = requireView<ImageView>(activity, MainActivity.TAG_OVERLAY)
                assertEquals(
                    "a denied camera must not leave a depth mask sitting on screen",
                    View.INVISIBLE,
                    overlay.visibility
                )
            }
        }
    }

    /**
     * Removing the debug report has to be enforced, not merely intended.
     *
     * A status line that grows a stack trace or a file path has started reporting again, and
     * nobody notices on a phone screen — it is three lines of small grey text that a user reads
     * past. Against the dashboard this fails immediately: the report it rendered was hundreds of
     * characters of class names and paths.
     */
    @Test
    fun statusText_whileStreaming_isShortAndCarriesNoDiagnostics() {
        grantCameraPermission()

        val frames = AtomicLong(0)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val preview = requireView<PreviewView>(activity, MainActivity.TAG_PREVIEW)
                preview.setFrameUpdateListener(ContextCompat.getMainExecutor(activity)) { _ ->
                    frames.incrementAndGet()
                }
            }
            // Scoped to the streaming state on purpose: the status text is only required to be
            // plain while the screen is doing its job, and frames arriving is the only honest
            // proof that it is.
            awaitFrames(frames, "preview frames to confirm the camera is streaming")

            scenario.onActivity { activity ->
                val status = requireView<TextView>(activity, MainActivity.TAG_CAMERA_STATUS)
                val text = status.text.toString()

                assertTrue("the status line must not be blank while streaming", text.isNotBlank())
                assertTrue(
                    "the status line must be a short sentence now the report is gone, but it was " +
                        "${text.length} characters: '$text'",
                    text.length <= MAX_STATUS_CHARS
                )
                assertFalse(
                    "the status line must be one line, not a stack trace or a dump: '$text'",
                    text.contains('\n')
                )
                assertFalse(
                    "the status line must not contain a stack frame: '$text'",
                    STACK_FRAME_PATTERN.containsMatchIn(text)
                )
                assertFalse(
                    "the status line must not name a class from this app, which is a diagnostic " +
                        "leftover: '$text'",
                    OWN_PACKAGE_PATTERN.containsMatchIn(text)
                )
                assertFalse(
                    "the status line must not contain a file path: '$text'",
                    text.contains('/')
                )
            }
        }
    }

    // ------------------------------------------------------------- helpers

    /**
     * The FrameLayout the screen builds its content into, reached through the tagged overlay.
     *
     * `require` rather than `!!`: reaching the cast is exactly what proves the layout is the one
     * the screen promises, and a container of the wrong type should say so rather than throw a bare
     * ClassCastException from somewhere further down.
     */
    private fun screenContainer(activity: MainActivity): FrameLayout {
        val overlay = requireView<ImageView>(activity, MainActivity.TAG_OVERLAY)
        val parent = requireNotNull(overlay.parent) { "the overlay is attached to nothing" }
        require(parent is FrameLayout) {
            "the overlay's container is a ${parent.javaClass.simpleName}, not a FrameLayout"
        }
        return parent
    }

    private fun childIndex(parent: ViewGroup, target: View): Int {
        for (index in 0 until parent.childCount) {
            if (parent.getChildAt(index) === target) return index
        }
        throw AssertionError("the view is not a child of ${parent.javaClass.simpleName}")
    }

    /** Every view in the tree, so a control cannot hide behind a container. */
    private fun collectViews(root: View): List<View> {
        val found = mutableListOf(root)
        if (root is ViewGroup) {
            for (index in 0 until root.childCount) {
                found.addAll(collectViews(root.getChildAt(index)))
            }
        }
        return found
    }

    private inline fun <reified T : View> requireView(activity: MainActivity, tag: String): T {
        val found = findByTagNode(activity.window.decorView, tag)
        assertNotNull("no view tagged '$tag' — the screen layout changed", found)
        // Checked here rather than left to a bare cast, so a tag that survives onto a view of the
        // wrong type fails with a sentence naming the tag instead of a ClassCastException.
        assertTrue(
            "view tagged '$tag' is a ${found?.javaClass?.simpleName}, not a ${T::class.java.simpleName}",
            found is T
        )
        @Suppress("UNCHECKED_UNCHECKED_CAST")
        return found as T
    }

    /**
     * Walks the real view tree instead of using a test-only accessor on the Activity, so the
     * assertions above run against the same objects a user touches.
     *
     * Deliberately neither `inline` nor `reified`: the traversal calls itself, and Kotlin forbids a
     * recursive reified inline function. It returns the untyped match and lets [requireView] do the
     * cast, which is what keeps the type check in one place.
     *
     * This is the same traversal the sibling preview suite carries. It is repeated rather than
     * shared because both helpers are `private` to their own test class, and widening them is a
     * change to that file, which this stage does not own.
     */
    private fun findByTagNode(root: View, tag: String): View? {
        if (root.tag == tag) return root
        if (root is ViewGroup) {
            for (index in 0 until root.childCount) {
                findByTagNode(root.getChildAt(index), tag)?.let { return it }
            }
        }
        return null
    }

    /**
     * Polls [condition] from the test thread.
     *
     * `onActivity` hands its block to the main thread, so blocking inside that block would wait on
     * the very thread the wait depends on. Anything asynchronous has to be waited for out here.
     */
    private fun awaitFromTestThread(
        what: String,
        timeoutMs: Long = UI_TIMEOUT_MS,
        condition: () -> Boolean
    ) {
        val deadline = SystemClock.uptimeMillis() + timeoutMs
        while (SystemClock.uptimeMillis() <= deadline) {
            if (condition()) return
            SystemClock.sleep(POLL_MS)
        }
        fail("Timed out after ${timeoutMs}ms waiting for $what")
    }

    private fun viewVisibility(scenario: ActivityScenario<MainActivity>, tag: String): Int {
        var visibility = View.GONE
        scenario.onActivity { activity ->
            visibility = findByTagNode(activity.window.decorView, tag)?.visibility ?: View.GONE
        }
        return visibility
    }

    /**
     * Waits for the first preview frame. A live PreviewView and a dead one look the same to the
     * user, so arrival is the only signal that says anything.
     */
    private fun awaitFrames(counter: AtomicLong, what: String, timeoutMs: Long = FRAME_TIMEOUT_MS) {
        val deadline = SystemClock.uptimeMillis() + timeoutMs
        while (counter.get() <= 0L) {
            if (SystemClock.uptimeMillis() > deadline) {
                fail(
                    "No preview frames arrived within ${timeoutMs}ms waiting for $what. A visible " +
                        "PreviewView with no frames is the failure this test exists to catch."
                )
            }
            SystemClock.sleep(POLL_MS)
        }
    }

    private fun grantCameraPermission() {
        shell("pm grant $packageName ${Manifest.permission.CAMERA}")
    }

    private fun revokeCameraPermission() {
        // runCatching: revoke fails on an app that never held the permission, which is not a test
        // failure — it is the state the test wanted anyway.
        runCatching { shell("pm revoke $packageName ${Manifest.permission.CAMERA}") }
    }

    /**
     * Marks the permission as permanently denied.
     *
     * Without this the app's own request raises the system dialog on top of the test, and the test
     * would be asserting against a modal it cannot dismiss. `user-fixed` makes the next request
     * return denied immediately, which is the real-world state this test is about.
     */
    private fun markPermissionUserFixed() {
        runCatching {
            shell(
                "pm set-permission-flags $packageName ${Manifest.permission.CAMERA} user-fixed"
            )
        }
    }

    private fun clearPermissionUserFixed() {
        runCatching {
            shell(
                "pm clear-permission-flags $packageName ${Manifest.permission.CAMERA} user-fixed"
            )
        }
    }

    private fun shell(command: String) {
        val descriptor = InstrumentationRegistry.getInstrumentation().uiAutomation
            .executeShellCommand(command)
        // executeShellCommand hands back the READ end of the pipe, so this must be an input stream.
        // Draining matters: a shell command whose output is never read blocks once the pipe fills.
        ParcelFileDescriptor.AutoCloseInputStream(descriptor).use { it.readBytes() }
    }

    private companion object {
        const val FRAME_TIMEOUT_MS = 20_000L
        const val UI_TIMEOUT_MS = 10_000L

        /**
         * Model load plus first inference, which is the honest cost of the live path on this
         * device. Generous on purpose: it is a ceiling for a failure, not a wait that is paid on a
         * passing run.
         */
        const val MASK_TIMEOUT_MS = 180_000L

        const val POLL_MS = 100L

        /** Comfortably above any sentence the screen needs to say and far below a debug report. */
        const val MAX_STATUS_CHARS = 120

        /** `at com.yy26git.depthanythingv2qnn.MainActivity.onCreate(MainActivity.kt:91)` — a stack frame. */
        val STACK_FRAME_PATTERN = Regex("""\bat\s+[\w.]+(?:\.[\w]+)+\(.*:\d+\)""")

        /** Any name rooted in this app's own package — a class name leaking into user-facing text. */
        val OWN_PACKAGE_PATTERN = Regex("""\bcom\.yy26git\.depthanythingv2qnn\.[\w.]+""")
    }
}
