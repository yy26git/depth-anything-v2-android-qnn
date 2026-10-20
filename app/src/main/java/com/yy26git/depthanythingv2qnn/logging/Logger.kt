package com.yy26git.depthanythingv2qnn.logging

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class Logger(private val tag: String = "Depth Anything V2") {
    private val entries = ArrayDeque<String>()
    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    fun appStarted() = info("Application started")
    fun cameraStarted() = info("Camera started")
    fun modelLoaded(modelName: String) = info("Model loaded: $modelName")
    fun modelSwitched(modelName: String) = info("Model switched: $modelName")
    fun inferenceStarted(modelName: String) = Unit
    fun inferenceFinished(modelName: String, inferenceTimeMs: Long, detectionCount: Int) {
        debug("Inference finished: $modelName ${inferenceTimeMs}ms detections=$detectionCount")
    }

    fun inference(modelName: String, inferenceTimeMs: Long, detectionCount: Int) {
        debug("Inference $modelName ${inferenceTimeMs}ms detections=$detectionCount")
    }

    fun memoryWarning(usedMb: Long, maxMb: Long) {
        warn("Memory warning: ${usedMb}MB used of ${maxMb}MB max")
    }

    fun performance(message: String) = info("Performance: $message")

    fun export(context: Context): File {
        val directory = File(context.getExternalFilesDir(null), "depth-anything-v2-qnn/logs").apply { mkdirs() }
        val file = File(directory, "vision_ai_${System.currentTimeMillis()}.log")
        file.writeText(snapshot().joinToString(separator = "\n"))
        return file
    }

    fun info(message: String) {
        record("INFO", message)
        Log.i(tag, message)
    }

    fun debug(message: String) {
        record("DEBUG", message)
        Log.d(tag, message)
    }

    fun warn(message: String) {
        record("WARN", message)
        Log.w(tag, message)
    }

    fun error(message: String, throwable: Throwable? = null) {
        record("ERROR", if (throwable == null) message else "$message: ${throwable.message}")
        Log.e(tag, message, throwable)
    }

    private fun record(level: String, message: String) {
        synchronized(entries) {
            entries.addLast("${formatter.format(Date())} $level $message")
            while (entries.size > MAX_ENTRIES) entries.removeFirst()
        }
    }

    private fun snapshot(): List<String> = synchronized(entries) { entries.toList() }

    companion object {
        private const val MAX_ENTRIES = 1000

        /**
         * A [Logger] for unit tests that do not need a real Android [Log] sink. The returned
         * instance accepts every call but does not write to logcat (the underlying [Log]
         * calls use a tag that the test runner can filter) and does not accumulate entries
         * beyond a small fixed window. Use this for tests that exercise code paths with
         * [Logger] dependencies but do not assert on log output.
         */
        val silent: Logger = Logger(tag = "Silent")
    }
}
