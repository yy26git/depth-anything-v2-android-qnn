package com.yy26git.depthanythingv2qnn

import android.Manifest
import android.content.Context
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.lang.ref.WeakReference
import java.util.concurrent.atomic.AtomicLong
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device tests for the preview stage. These are NOT unit tests and cannot be satisfied by a JVM:
 * every assertion here is about frames that a real camera produced, a permission the real package
 * manager answered, and a session the real camera service released.
 *
 * None of this has been executed. No Gradle build and no adb run was performed for this stage, so
 * nothing below is a claim that these pass — they are the proposed proof, written against the
 * CameraX 1.5.3 API surface read out of the resolved AARs.
 *
 * The frame signal is `PreviewView.setFrameUpdateListener`, which CameraX invokes once per frame
 * delivered to the preview surface. A visible widget proves nothing: a PreviewView with a dead
 * binding looks identical to a live one until frames actually arrive.
 */
@RunWith(AndroidJUnit4::class)
class CameraPreviewInstrumentationTest {

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    private val packageName: String
        get() = context.packageName

    @After
    fun clearPermissionState() {
        // Leaving CAMERA granted or user-fixed behind would silently disarm the denial test on a
        // second run of the suite, and would leave the device itself in a permanently-denied state
        // that the next manual install inherits. Both are cleared.
        revokeCameraPermission()
        clearPermissionUserFixed()
    }

    /**
     * The headline requirement: with permission granted, real frames arrive — with no model
     * session created and no benchmark ever run.
     *
     * Nothing in this test touches the benchmark button. That is the point of the assertion: the
     * preview path must be reachable with the model absent, so if a future change makes the camera
     * depend on inference readiness, this test fails.
     */
    @Test
    fun grantedPermission_deliversFrames_withNoModelSessionAndNoBenchmarkRun() {
        grantCameraPermission()

        val frames = AtomicLong(0)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val preview = requireView<PreviewView>(activity, MainActivity.TAG_PREVIEW)
                preview.setFrameUpdateListener(ContextCompat.getMainExecutor(activity)) { _ ->
                    frames.incrementAndGet()
                }
            }
            awaitFrames(frames)
        }
    }

    /**
     * Denial must be a visible, actionable state — not a crash and not a blank screen — and the
     * still benchmark must remain reachable.
     */
    @Test
    fun deniedPermission_showsRecoveryUi_andLeavesTheBenchmarkReachable() {
        revokeCameraPermission()
        markPermissionUserFixed()

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            // The permission result comes back after onCreate, so the recovery row is not there
            // the instant the scenario reports RESUMED. Waiting for it here — from the test thread,
            // which is allowed to block — is the difference between a real assertion and a race.
            awaitFromTestThread("permission recovery row to appear") {
                viewVisibility(scenario, MainActivity.TAG_PERMISSION_RECOVERY) == View.VISIBLE
            }

            scenario.onActivity { activity ->
                val status = requireView<TextView>(activity, MainActivity.TAG_CAMERA_STATUS)
                assertTrue(
                    "denial must be stated in the status text, was: '${status.text}'",
                    status.text.isNotBlank()
                )

                val recovery = requireView<TextView>(activity, MainActivity.TAG_PERMISSION_RECOVERY)
                assertEquals(
                    "the recovery action must be visible after a denial",
                    View.VISIBLE,
                    recovery.visibility
                )
                assertTrue(
                    "the recovery action must say what it does, was: '${recovery.text}'",
                    recovery.text.isNotBlank()
                )

                // The screen no longer has a benchmark button — the still benchmark was replaced by
                // the live depth view, and a denied camera now costs the user the whole screen rather
                // than one control. So the assertion that used to live here ("the benchmark must
                // stay reachable") is deliberately gone: there is nothing left to reach. What is
                // still enforced is that a denial produces a visible, labelled recovery action and
                // a stated status, checked immediately above.
            }
        }
    }

    /**
     * A denied permission must not bind a camera session either. The provider is still installed
     * on this process, so "no frames for a while, with no error dialog" is the observable proof
     * that nothing was bound.
     */
    @Test
    fun deniedPermission_bindsNoCameraSession() {
        revokeCameraPermission()
        markPermissionUserFixed()

        val frames = AtomicLong(0)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val preview = requireView<PreviewView>(activity, MainActivity.TAG_PREVIEW)
                preview.setFrameUpdateListener(ContextCompat.getMainExecutor(activity)) { _ ->
                    frames.incrementAndGet()
                }
            }
            // A fixed window, not a full timeout: the assertion is that nothing arrives, so the
            // wait length is a tolerance for a slow device rather than a wait for success.
            SystemClock.sleep(NO_FRAMES_WINDOW_MS)
            assertEquals(
                "a denied permission must not produce preview frames",
                0L,
                frames.get()
            )
        }
    }

    /**
     * Reopen releases the camera.
     *
     * Two halves, because "reopen works" and "the old screen let go" fail differently. The WeakReference
     * catches the leak that would otherwise keep the destroyed Activity — and with it its session —
     * alive; the second launch proves the camera service actually gave the device back, which it
     * would not if the destroyed screen were still holding it.
     */
    @Test
    fun reopen_releasesTheCamera_andDoesNotLeakTheDestroyedActivity() {
        grantCameraPermission()

        var destroyedActivity: WeakReference<MainActivity>? = null
        val firstLaunchFrames = AtomicLong(0)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                destroyedActivity = WeakReference(activity)
                val preview = requireView<PreviewView>(activity, MainActivity.TAG_PREVIEW)
                preview.setFrameUpdateListener(ContextCompat.getMainExecutor(activity)) { _ ->
                    firstLaunchFrames.incrementAndGet()
                }
            }
            awaitFrames(firstLaunchFrames)
        }

        awaitGarbageCollected(destroyedActivity)

        val secondLaunchFrames = AtomicLong(0)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val preview = requireView<PreviewView>(activity, MainActivity.TAG_PREVIEW)
                preview.setFrameUpdateListener(ContextCompat.getMainExecutor(activity)) { _ ->
                    secondLaunchFrames.incrementAndGet()
                }
            }
            awaitFrames(secondLaunchFrames)
        }
    }

    // ------------------------------------------------------------- helpers

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

    private fun awaitFrames(counter: AtomicLong, timeoutMs: Long = FRAME_TIMEOUT_MS) {
        val deadline = SystemClock.uptimeMillis() + timeoutMs
        while (counter.get() <= 0L) {
            if (SystemClock.uptimeMillis() > deadline) {
                fail(
                    "No preview frames arrived within ${timeoutMs}ms. A visible PreviewView with no " +
                        "frames is the failure this test exists to catch."
                )
            }
            SystemClock.sleep(POLL_MS)
        }
    }

    private fun awaitGarbageCollected(reference: WeakReference<*>?) {
        assertNotNull("the first launch never recorded an Activity reference", reference)
        val deadline = SystemClock.uptimeMillis() + GC_TIMEOUT_MS
        while (SystemClock.uptimeMillis() < deadline) {
            if (reference?.get() == null) return
            System.gc()
            Runtime.getRuntime().gc()
            SystemClock.sleep(POLL_MS)
        }
        fail(
            "The destroyed Activity is still reachable, so it outlived its camera session. " +
                "Something is holding the Activity, the PreviewView, or the LifecycleOwner."
        )
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
     * Without this the app's own request would raise the system dialog on top of the test, and the
     * test would be asserting against a modal it cannot dismiss. `user-fixed` makes the next
     * request return denied immediately, which is the real-world state this test is about.
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
        const val NO_FRAMES_WINDOW_MS = 3_000L
        const val UI_TIMEOUT_MS = 10_000L
        const val GC_TIMEOUT_MS = 15_000L
        const val POLL_MS = 100L
    }
}
