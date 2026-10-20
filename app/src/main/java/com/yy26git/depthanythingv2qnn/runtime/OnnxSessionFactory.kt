package com.yy26git.depthanythingv2qnn.runtime

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import android.system.Os
import android.util.Log
import com.yy26git.depthanythingv2qnn.logging.Logger
import com.yy26git.depthanythingv2qnn.settings.DepthSettings
import com.yy26git.depthanythingv2qnn.settings.DepthBackend
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

data class OnnxSessionHandle(
    val environment: OrtEnvironment,
    val session: OrtSession,
    val diagnostics: DepthBackendDiagnostics
) : AutoCloseable {
    @Volatile
    private var closed: Boolean = false

    val isClosed: Boolean
        get() = closed

    override fun close() {
        if (closed) return
        closed = true
        runCatching { session.close() }
    }
}

data class DepthBackendDiagnostics(
    val modelId: String,
    val purpose: String,
    val requestedBackend: DepthBackend,
    val activeBackend: DepthBackend,
    val providerRegistrationAttempted: Boolean,
    val providerRegistrationSucceeded: Boolean,
    val sessionCreated: Boolean,
    val fallbackBackend: DepthBackend?,
    val fallbackReason: String?,
    val qnnRuntimeLibrariesPresent: Boolean,
    val qnnBackendLoaded: Boolean,
    val modelCompatibilityStatus: String,
    val lastInitializationError: String?,
    val availableProviders: List<String>,
    val qnnProviderCompiled: Boolean,
    val qnnRuntimeVersion: String,
    val qnnRequiredLibraries: List<String>,
    val qnnMissingLibraries: List<String>,
    val qnnBackendPath: String,
    val qnnProviderOptions: Map<String, String>,
    val loadedQnnLibraries: List<String>,
    val graphCompilationStatus: String,
    val contextCacheHit: Boolean,
    val contextCachePath: String,
    val profileOutputPath: String,
    val modelLoadTimeMs: Long,
    val firstInferenceTimeMs: Long = 0L,
    val averageInferenceTimeMs: Float = 0f,
    val inferenceCount: Long = 0L,
    val htpInferenceConfirmed: Boolean = false,
    val lastInferenceError: String? = null,
    val onnxRuntimeVersion: String = ORT_VERSION
) {
    val qnnProviderRegistered: Boolean
        get() = providerRegistrationSucceeded

    val htpInitialized: Boolean
        get() = activeBackend == DepthBackend.QNN_HTP &&
            providerRegistrationSucceeded &&
            sessionCreated &&
            graphCompilationStatus == "FULL_QNN_GRAPH_COMPILED"

    val cpuFallbackReason: String?
        get() = if (activeBackend == DepthBackend.CPU && requestedBackend != DepthBackend.CPU) fallbackReason else null

    companion object {
        const val ORT_VERSION = "1.30.0"
    }
}

data class OnnxQnnRuntimeStatus(
    val available: Boolean,
    val nativeLibraryDir: String,
    val backendPath: String,
    val requiredLibraries: List<String>,
    val missingLibraries: List<String>,
    val reason: String?,
    val qnnRuntimeVersion: String,
    val detectedSoc: String,
    val htpArchitecture: String,
    val environmentConfigured: Boolean
)

object OnnxQnnRuntimeInspector {
    const val ORT_VERSION = DepthBackendDiagnostics.ORT_VERSION
    const val QNN_SDK_VERSION = "Qualcomm AI Runtime / AI Engine Direct SDK 2.50.0.260828"
    const val QNN_HTP_LIBRARY = "libQnnHtp.so"
    val REQUIRED_QNN_LIBRARIES = listOf(
        "libQnnHtp.so",
        "libQnnHtpPrepare.so",
        "libQnnHtpV68Stub.so",
        "libQnnHtpV68.so",
        "libQnnHtpV68Skel.so",
        "libQnnSystem.so",
        "libQnnIr.so",
        "libQnnSaver.so"
    )

    fun inspect(context: Context): OnnxQnnRuntimeStatus {
        val nativeDir = File(context.applicationInfo.nativeLibraryDir.orEmpty())
        val backend = File(nativeDir, QNN_HTP_LIBRARY)
        val missing = REQUIRED_QNN_LIBRARIES.filterNot { File(nativeDir, it).exists() }
        val soc = detectedSoc()
        val isV68 = soc.contains("SM8350", ignoreCase = true) ||
            Build.HARDWARE.contains("lahaina", ignoreCase = true)
        val environmentConfigured = OnnxQnnRuntimeEnvironment.isConfigured()
        val reason = when {
            missing.isNotEmpty() ->
                "Missing Qualcomm QNN HTP runtime libraries: ${missing.joinToString()} in ${nativeDir.absolutePath}"
            !isV68 ->
                "Packaged QNN runtime targets HTP V68/SM8350, but this device reports SoC=$soc hardware=${Build.HARDWARE}."
            !environmentConfigured ->
                "QNN native environment failed to initialize. Check the startup library-load log for the exact dlopen error."
            else -> null
        }
        return OnnxQnnRuntimeStatus(
            available = missing.isEmpty() && isV68 && environmentConfigured,
            nativeLibraryDir = nativeDir.absolutePath,
            backendPath = backend.absolutePath,
            requiredLibraries = REQUIRED_QNN_LIBRARIES,
            missingLibraries = missing,
            reason = reason,
            qnnRuntimeVersion = QNN_SDK_VERSION,
            detectedSoc = soc,
            htpArchitecture = if (isV68) "V68" else "unsupported",
            environmentConfigured = environmentConfigured
        )
    }

    fun startupDiagnostic(context: Context): String {
        val status = inspect(context)
        return buildString {
            appendLine("=== QNN STARTUP DIAGNOSTIC ===")
            appendLine("onnxRuntimeVersion=$ORT_VERSION")
            appendLine("qnnRuntimeVersion=${status.qnnRuntimeVersion}")
            appendLine("device=${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("soc=${status.detectedSoc}")
            appendLine("htpArchitecture=${status.htpArchitecture}")
            appendLine("nativeLibraryDir=${status.nativeLibraryDir}")
            appendLine("environmentConfigured=${status.environmentConfigured}")
            appendLine("runtimeLibrariesPresent=${status.missingLibraries.isEmpty()}")
            appendLine("missingLibraries=${status.missingLibraries.joinToString().ifBlank { "none" }}")
            appendLine("qnnAvailable=${status.available}")
            append("reason=${status.reason ?: "none"}")
        }
    }

    private fun detectedSoc(): String =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else Build.HARDWARE
}

object OnnxQnnRuntimeEnvironment {
    private val configured = AtomicBoolean(false)

    fun configure(context: Context, logger: Logger): Boolean {
        if (configured.get()) return true
        val nativeDir = context.applicationInfo.nativeLibraryDir.orEmpty()
        val adspPath = listOf(
            nativeDir,
            "/odm/lib/rfsa/adsp",
            "/odm/lib/rfsa/adsp/aiboost",
            "/vendor/dsp/cdsp",
            "/vendor/lib/rfsa/adsp",
            "/system/lib/rfsa/adsp",
            "/dsp"
        ).joinToString(";")
        val ldPath = listOf(
            nativeDir,
            "/odm/lib64",
            "/vendor/lib64",
            System.getenv("LD_LIBRARY_PATH").orEmpty()
        ).filter { it.isNotBlank() }.joinToString(":")

        return runCatching {
            Os.setenv("ADSP_LIBRARY_PATH", adspPath, true)
            Os.setenv("LD_LIBRARY_PATH", ldPath, true)
            System.setProperty("ADSP_LIBRARY_PATH", adspPath)
            System.setProperty("LD_LIBRARY_PATH", ldPath)
            val loadOrder = listOf(
                "libQnnSystem.so",
                "libQnnHtpPrepare.so",
                "libQnnHtpV68Stub.so",
                "libQnnHtp.so"
            )
            loadOrder.forEach { library ->
                val file = File(nativeDir, library)
                require(file.isFile) { "Required QNN library is missing: ${file.absolutePath}" }
                System.load(file.absolutePath)
                logger.info("QNN native library loaded: ${file.absolutePath}")
            }
            configured.set(true)
            logger.info("QNN environment configured ADSP_LIBRARY_PATH=$adspPath LD_LIBRARY_PATH=$ldPath")
            true
        }.getOrElse { error ->
            configured.set(false)
            logger.error("QNN environment initialization failed", error)
            false
        }
    }

    fun isConfigured(): Boolean = configured.get()

    fun loadedLibraries(): List<String> = runCatching {
        File("/proc/self/maps").useLines { lines ->
            lines.mapNotNull { line ->
                line.substringAfterLast(' ', missingDelimiterValue = "")
                    .takeIf { it.contains("Qnn", ignoreCase = true) }
            }.distinct().sorted().toList()
        }
    }.getOrDefault(emptyList())
}

object OnnxRuntimeDiagnosticsStore {
    private val states = ConcurrentHashMap<String, DepthBackendDiagnostics>()
    private val timing = ConcurrentHashMap<String, TimingState>()

    fun recordLoad(diagnostics: DepthBackendDiagnostics) {
        states[diagnostics.modelId] = diagnostics
        timing.remove(diagnostics.modelId)
    }

    fun recordInference(modelId: String, inferenceTimeMs: Long) {
        val current = states[modelId] ?: return
        val updatedTiming = timing.compute(modelId) { _, previous ->
            (previous ?: TimingState()).record(inferenceTimeMs)
        } ?: TimingState().record(inferenceTimeMs)
        val firstQnnInference = current.activeBackend == DepthBackend.QNN_HTP && current.inferenceCount == 0L
        val loadedLibraries = if (firstQnnInference) {
            OnnxQnnRuntimeEnvironment.loadedLibraries()
        } else {
            current.loadedQnnLibraries
        }
        states[modelId] = current.copy(
            firstInferenceTimeMs = updatedTiming.firstInferenceTimeMs,
            averageInferenceTimeMs = updatedTiming.averageInferenceTimeMs,
            inferenceCount = updatedTiming.count,
            htpInferenceConfirmed = current.activeBackend == DepthBackend.QNN_HTP,
            loadedQnnLibraries = loadedLibraries,
            lastInferenceError = null
        )
        if (firstQnnInference) {
            Log.i(
                QNN_LOG_TAG,
                "HTP inference confirmed model=$modelId firstInferenceMs=$inferenceTimeMs " +
                    "loadedLibraries=${loadedLibraries.joinToString()}"
            )
        }
    }

    fun recordInferenceFailure(modelId: String, error: Throwable) {
        val current = states[modelId] ?: return
        states[modelId] = current.copy(lastInferenceError = error.message ?: error.javaClass.simpleName)
        Log.e(QNN_LOG_TAG, "ONNX inference failed model=$modelId activeBackend=${current.activeBackend.label}", error)
    }

    fun latest(modelId: String): DepthBackendDiagnostics? = states[modelId]

    fun snapshot(): List<DepthBackendDiagnostics> =
        states.values.sortedWith(compareBy<DepthBackendDiagnostics> { it.purpose }.thenBy { it.modelId })

    private data class TimingState(
        val count: Long = 0L,
        val totalMs: Long = 0L,
        val firstInferenceTimeMs: Long = 0L
    ) {
        val averageInferenceTimeMs: Float
            get() = if (count > 0L) totalMs.toFloat() / count else 0f

        fun record(timeMs: Long): TimingState =
            copy(
                count = count + 1L,
                totalMs = totalMs + timeMs,
                firstInferenceTimeMs = if (count == 0L) timeMs else firstInferenceTimeMs
            )
    }

    private const val QNN_LOG_TAG = "DepthQNN"
}

class OnnxSessionFactory(
    context: Context,
    private val logger: Logger
) {
    private val appContext = context.applicationContext

    fun createSession(
        modelId: String,
        purpose: String,
        modelBytes: ByteArray?,
        modelPath: String? = null,
        isEpContext: Boolean = false,
        settings: DepthSettings? = null,
        requestedBackend: DepthBackend? = null,
        allowCpuFallback: Boolean = true,
        profilingOutputPath: String? = null,
    ): OnnxSessionHandle {
        require(modelBytes != null || modelPath != null) { "Either modelBytes or modelPath must be provided." }
        val selectedBackend = requestedBackend ?: settings?.onnxBackend ?: readPersistedBackend()
        if (selectedBackend != DepthBackend.CPU) {
            OnnxQnnRuntimeEnvironment.configure(appContext, logger)
        }
        val strictQnn = selectedBackend == DepthBackend.STRICT_QNN_HTP
        val effectiveAllowCpuFallback = allowCpuFallback && !strictQnn
        val attempts = when (selectedBackend) {
            DepthBackend.AUTO -> if (effectiveAllowCpuFallback) listOf(DepthBackend.QNN_HTP, DepthBackend.CPU) else listOf(DepthBackend.QNN_HTP)
            DepthBackend.QNN_HTP -> if (effectiveAllowCpuFallback) listOf(DepthBackend.QNN_HTP, DepthBackend.CPU) else listOf(DepthBackend.QNN_HTP)
            DepthBackend.STRICT_QNN_HTP -> listOf(DepthBackend.QNN_HTP)
            DepthBackend.CPU -> listOf(DepthBackend.CPU)
        }
        var fallbackReason: String? = null
        attempts.forEach { backend ->
            if (backend == DepthBackend.QNN_HTP) {
                val qnnRuntime = OnnxQnnRuntimeInspector.inspect(appContext)
                if (!qnnRuntime.available) {
                    fallbackReason = qnnRuntime.reason
                    val message = "ONNX QNN unavailable for $modelId: $fallbackReason"
                    logger.warn(message)
                    if (!effectiveAllowCpuFallback) {
                        throw IllegalStateException("Strict QNN requested for $modelId. $message")
                    }
                    return@forEach
                }
            }
            val result = runCatching {
                createSessionForBackend(
                    modelId = modelId,
                    purpose = purpose,
                    modelBytes = modelBytes,
                    modelPath = modelPath,
                    isEpContext = isEpContext,
                    requestedBackend = selectedBackend,
                    activeBackend = backend,
                    fallbackReason = fallbackReason,
                    settings = settings,
                    profilingOutputPath = profilingOutputPath,
                )
            }
            if (result.isSuccess) {
                return result.getOrThrow()
            }
            val error = result.exceptionOrNull()
            fallbackReason = error?.message ?: "ONNX ${backend.label} session creation failed."
            logger.warn("ONNX ${backend.label} failed for $modelId: $fallbackReason")
            if (!effectiveAllowCpuFallback && backend == DepthBackend.QNN_HTP) {
                throw IllegalStateException("Strict QNN requested for $modelId. $fallbackReason", error)
            }
        }
        throw IllegalStateException("ONNX session creation failed for $modelId: ${fallbackReason ?: "unknown error"}")
    }

    private fun createSessionForBackend(
        modelId: String,
        purpose: String,
        modelBytes: ByteArray?,
        modelPath: String? = null,
        isEpContext: Boolean = false,
        requestedBackend: DepthBackend,
        activeBackend: DepthBackend,
        fallbackReason: String?,
        settings: DepthSettings?,
        profilingOutputPath: String? = null,
    ): OnnxSessionHandle {
        val environment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        // EPContext models carry a pre-compiled QNN binary — skip ORT's own context cache (the binary
        // is already embedded). A float model.data graph (modelPath, not EPContext) DOES use the
        // cache: ORT generates a QNN context on first load and reloads it on subsequent loads.
        val cache = when {
            isEpContext -> QnnContextCache(hit = false, file = java.io.File(""))
            modelBytes != null -> qnnContextCache(modelId, modelBytes)
            modelPath != null -> qnnContextCacheForPath(modelId, modelPath)
            else -> QnnContextCache(hit = false, file = java.io.File(""))
        }
        val useCachedContext = !isEpContext && activeBackend == DepthBackend.QNN_HTP && cache.hit
        val qnnRuntime = OnnxQnnRuntimeInspector.inspect(appContext)
        val qnnProviderOptions = if (activeBackend == DepthBackend.QNN_HTP) {
            qnnProviderOptions(qnnRuntime.backendPath, isEpContext)
        } else {
            emptyMap()
        }
        val availableProviders = availableProviderNames()
        val loadStarted = System.nanoTime()
        try {
            profilingOutputPath?.let { options.enableProfiling(it) }
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            options.setIntraOpNumThreads(settings?.inferenceThreads?.coerceIn(1, 8) ?: DEFAULT_CPU_THREADS)
            options.setInterOpNumThreads(1)
            options.setLoggerId("DepthAnythingV2-$modelId-${activeBackend.id}")
            var qnnRegistered = false
            if (activeBackend == DepthBackend.QNN_HTP) {
                // VERBOSE logs ~8 lines per inference from the QNN EP, on the inference thread.
                // At 30 FPS that is 240 logd writes/sec of pure overhead in the hot path, so it
                // stays off unless QNN_VERBOSE_LOGGING is flipped for a debugging session.
                if (QNN_VERBOSE_LOGGING) {
                    options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
                    options.setSessionLogVerbosityLevel(1)
                } else {
                    options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
                }
                // A pre-compiled EPContext model already embeds its QNN context binary — register the
                // QNN EP to CONSUME it, and do NOT enable context generation. (A model.data float
                // graph is NOT EPContext: it still needs ORT to generate + cache a context.)
                configureQnn(options, cache, qnnRuntime, qnnProviderOptions, useCachedContext, isEpContext = isEpContext)
                // A float external-weights graph (model.data) offloads most ops to the HTP but keeps
                // a few on CPU (per-node partitioning) — do NOT disable CPU EP fallback for it, or the
                // whole QNN session fails. EPContext + fully-quantized models stay pure-QNN.
                val isExternalWeightsFloat = modelPath != null && !isEpContext
                if (!isExternalWeightsFloat) {
                    options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                }
                qnnRegistered = true
            } else {
                options.addCPU(true)
            }
            val session = when {
                useCachedContext -> environment.createSession(cache.file.readBytes(), options)
                modelPath != null -> environment.createSession(modelPath, options)
                else -> environment.createSession(modelBytes!!, options)
            }
            val loadedQnnLibraries = OnnxQnnRuntimeEnvironment.loadedLibraries()
            val diagnostics = DepthBackendDiagnostics(
                modelId = modelId,
                purpose = purpose,
                requestedBackend = requestedBackend,
                activeBackend = activeBackend,
                providerRegistrationAttempted = activeBackend == DepthBackend.QNN_HTP,
                providerRegistrationSucceeded = qnnRegistered,
                sessionCreated = true,
                fallbackBackend = if (activeBackend == DepthBackend.CPU && requestedBackend != DepthBackend.CPU) DepthBackend.CPU else null,
                fallbackReason = if (activeBackend == DepthBackend.CPU && requestedBackend != DepthBackend.CPU) fallbackReason else null,
                qnnRuntimeLibrariesPresent = qnnRuntime.available,
                qnnBackendLoaded = activeBackend == DepthBackend.QNN_HTP && qnnRegistered,
                modelCompatibilityStatus = modelCompatibilityStatus(activeBackend, qnnRuntime),
                lastInitializationError = null,
                availableProviders = availableProviders,
                qnnProviderCompiled = availableProviders.any { it.equals("QNN", ignoreCase = true) },
                qnnRuntimeVersion = qnnRuntime.qnnRuntimeVersion,
                qnnRequiredLibraries = qnnRuntime.requiredLibraries,
                qnnMissingLibraries = qnnRuntime.missingLibraries,
                qnnBackendPath = qnnRuntime.backendPath,
                qnnProviderOptions = qnnProviderOptions,
                loadedQnnLibraries = loadedQnnLibraries,
                graphCompilationStatus = if (activeBackend == DepthBackend.QNN_HTP) {
                    "FULL_QNN_GRAPH_COMPILED"
                } else {
                    "CPU_GRAPH"
                },
                contextCacheHit = useCachedContext,
                contextCachePath = cache.file.absolutePath,
                profileOutputPath = qnnProviderOptions["profiling_file_path"].orEmpty(),
                modelLoadTimeMs = elapsedMs(loadStarted)
            )
            OnnxRuntimeDiagnosticsStore.recordLoad(diagnostics)
            logger.info(diagnostics.toDiagnosticBlock())
            logger.info(
                "ONNX session loaded model=$modelId purpose=$purpose requested=${requestedBackend.label} " +
                    "active=${activeBackend.label} loadMs=${diagnostics.modelLoadTimeMs} " +
                    "cache=${if (useCachedContext) "hit" else "miss"} " +
                    "graph=${diagnostics.graphCompilationStatus} fallback=${diagnostics.cpuFallbackReason ?: "none"}"
            )
            return OnnxSessionHandle(environment, session, diagnostics)
        } catch (error: Throwable) {
            if (activeBackend == DepthBackend.QNN_HTP) {
                logger.error(
                    "QNN HTP graph compilation failed model=$modelId " +
                        "providerOptions=${qnnProviderOptions.entries.joinToString()} " +
                        "loadedLibraries=${OnnxQnnRuntimeEnvironment.loadedLibraries().joinToString()} " +
                        "classification=${classifyQnnFailure(error)}",
                    error
                )
            }
            throw IllegalStateException("ONNX ${activeBackend.label} initialization failed for $modelId: ${error.message}", error)
        } finally {
            options.close()
        }
    }

    private fun configureQnn(
        options: OrtSession.SessionOptions,
        cache: QnnContextCache,
        qnnRuntime: OnnxQnnRuntimeStatus,
        qnnOptions: Map<String, String>,
        useCachedContext: Boolean,
        isEpContext: Boolean = false
    ) {
        if (!qnnRuntime.available) {
            throw OrtException(qnnRuntime.reason)
        }
        configureQnnLibrarySearchPath(qnnRuntime.nativeLibraryDir)
        when {
            // The model already IS a compiled EPContext binary — just consume it. Turning on
            // context generation here would try to recompile it and dump to a bogus path.
            isEpContext -> options.addConfigEntry("ep.context_enable", "0")
            useCachedContext -> options.addConfigEntry("ep.context_enable", "0")
            else -> {
                options.addConfigEntry("ep.context_enable", "1")
                options.addConfigEntry("ep.context_embed_mode", "1")
                options.addConfigEntry("ep.context_file_path", cache.file.absolutePath)
            }
        }
        options.addQnn(qnnOptions)
    }

    private fun configureQnnLibrarySearchPath(nativeLibraryDir: String) {
        if (!OnnxQnnRuntimeEnvironment.configure(appContext, logger)) {
            throw IllegalStateException("QNN library search path or native library initialization failed for $nativeLibraryDir")
        }
    }

    private fun qnnProviderOptions(backendPath: String, isEpContext: Boolean = false): Map<String, String> {
        return linkedMapOf(
            "backend_path" to backendPath,
            "htp_performance_mode" to "burst",
            "soc_model" to SM8350_SOC_MODEL,
            "htp_arch" to HTP_V68_ARCH,
            "device_id" to "0",
            "htp_graph_finalization_optimization_mode" to "3",
            "offload_graph_io_quantization" to if (isEpContext) "1" else "0"
        )
    }

    private fun qnnContextCache(modelId: String, modelBytes: ByteArray): QnnContextCache =
        qnnContextCacheForIdentity(modelId, sha256(modelBytes))

    /** Cache for a filesystem-resident model (external-weights float ONNX): key off path + size. */
    private fun qnnContextCacheForPath(modelId: String, modelPath: String): QnnContextCache {
        val f = File(modelPath)
        return qnnContextCacheForIdentity(modelId, "path:${f.absolutePath}:size:${f.length()}")
    }

    private fun qnnContextCacheForIdentity(modelId: String, contentIdentity: String): QnnContextCache {
        val optionsIdentity =
            "backend=QNN_HTP;performance=burst;soc=$SM8350_SOC_MODEL;arch=$HTP_V68_ARCH;" +
                "finalization=3;graphIoQuantization=0;ort=${OnnxQnnRuntimeInspector.ORT_VERSION};qnn=${qnnRuntimeIdentity()}"
        val identity = listOf(
            modelId,
            contentIdentity,
            OnnxQnnRuntimeInspector.ORT_VERSION,
            qnnRuntimeIdentity(),
            Build.MANUFACTURER,
            Build.MODEL,
            Build.DEVICE,
            Build.VERSION.SDK_INT.toString(),
            optionsIdentity
        ).joinToString("|")
        val cacheDir = File(appContext.filesDir, "onnx/qnn_context_cache").apply { mkdirs() }
        val file = File(cacheDir, "${modelId.sanitizeFileName()}-${sha256(identity.toByteArray()).take(16)}-ctx.onnx")
        return QnnContextCache(file = file, hit = file.exists() && file.length() > 0L)
    }

    private fun classifyQnnFailure(error: Throwable): String {
        val message = generateSequence(error) { it.cause }
            .mapNotNull { it.message }
            .joinToString(" | ")
        return when {
            message.contains("assigned to the default CPU EP", ignoreCase = true) ||
                message.contains("Unable to compile any nodes", ignoreCase = true) ->
                "UNSUPPORTED_GRAPH_OR_ZERO_QNN_PARTITIONS"
            message.contains("QNN_DEVICE_ERROR_INVALID_CONFIG", ignoreCase = true) ->
                "HTP_DEVICE_CONFIGURATION_REJECTED"
            message.contains("dlopen", ignoreCase = true) ||
                message.contains("cannot locate", ignoreCase = true) ->
                "QNN_NATIVE_LIBRARY_LOAD_FAILURE"
            message.contains("context", ignoreCase = true) ->
                "QNN_CONTEXT_CACHE_OR_COMPILATION_FAILURE"
            else -> "QNN_INITIALIZATION_FAILURE"
        }
    }

    private fun qnnRuntimeIdentity(): String {
        val nativeDir = File(appContext.applicationInfo.nativeLibraryDir.orEmpty())
        val files = OnnxQnnRuntimeInspector.REQUIRED_QNN_LIBRARIES.map { File(nativeDir, it) }
        if (files.none { it.exists() }) return "qnn-sdk-unavailable"
        return files.filter { it.exists() }
            .joinToString(separator = ";") { "${it.name}:${it.length()}:${it.lastModified()}" }
    }

    private fun readPersistedBackend(): DepthBackend {
        val prefs = appContext.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        return DepthBackend.fromId(prefs.getString(KEY_ONNX_BACKEND, DepthBackend.AUTO.id).orEmpty())
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private fun String.sanitizeFileName(): String =
        lowercase(Locale.US).replace(Regex("[^a-z0-9_.-]"), "_")

    private fun elapsedMs(startNs: Long): Long = (System.nanoTime() - startNs) / 1_000_000L

    private fun availableProviderNames(): List<String> =
        runCatching { OrtEnvironment.getAvailableProviders().map(OrtProvider::name).sorted() }
            .getOrElse { listOf("unavailable: ${it.message ?: it.javaClass.simpleName}") }

    private fun modelCompatibilityStatus(activeBackend: DepthBackend, qnnRuntime: OnnxQnnRuntimeStatus): String =
        when {
            activeBackend != DepthBackend.QNN_HTP && !qnnRuntime.available ->
                "QNN unavailable: ${qnnRuntime.reason}"
            activeBackend == DepthBackend.QNN_HTP ->
                "QNN session created with CPU EP fallback disabled. Runtime logs must show HTP device setup and graph assignment."
            else ->
                "CPU session. QNN model compatibility not evaluated."
        }

    private fun DepthBackendDiagnostics.toDiagnosticBlock(): String = buildString {
        appendLine("=== DETECTOR BACKEND DIAGNOSTIC ===")
        appendLine("model=$modelId")
        appendLine("purpose=$purpose")
        appendLine("modelFormat=ONNX")
        appendLine("modelQuantization=${if (modelId.contains("qdq", ignoreCase = true) || modelId.contains("int8", ignoreCase = true)) "QDQ INT8" else "metadata/report required"}")
        appendLine("onnxRuntimeVersion=$onnxRuntimeVersion")
        appendLine("requestedBackend=${requestedBackend.label}")
        appendLine("availableProviders=${availableProviders.joinToString()}")
        appendLine("qnnProviderCompiled=$qnnProviderCompiled")
        appendLine("qnnProviderRegistrationAttempted=$providerRegistrationAttempted")
        appendLine("qnnProviderRegistered=$providerRegistrationSucceeded")
        appendLine("qnnRuntimeVersion=$qnnRuntimeVersion")
        appendLine("qnnRequiredLibraries=${qnnRequiredLibraries.joinToString()}")
        appendLine("qnnMissingLibraries=${qnnMissingLibraries.joinToString().ifBlank { "none" }}")
        appendLine("qnnBackendPath=$qnnBackendPath")
        appendLine("qnnProviderOptions=${qnnProviderOptions.entries.joinToString { "${it.key}=${it.value}" }.ifBlank { "none" }}")
        appendLine("profileOutputPath=${profileOutputPath.ifBlank { "none" }}")
        appendLine("loadedQnnLibraries=${loadedQnnLibraries.joinToString().ifBlank { "none" }}")
        appendLine("sessionCreated=$sessionCreated")
        appendLine("graphCompilationStatus=$graphCompilationStatus")
        appendLine("htpInferenceConfirmed=$htpInferenceConfirmed")
        appendLine("activeBackend=${activeBackend.label}")
        appendLine("fallbackBackend=${fallbackBackend?.label ?: "none"}")
        appendLine("fallbackReason=${fallbackReason ?: "none"}")
        appendLine("modelCompatibilityStatus=$modelCompatibilityStatus")
        appendLine("lastInitializationError=${lastInitializationError ?: "none"}")
        appendLine("initializationDurationMs=$modelLoadTimeMs")
        append("===================================")
    }

    private data class QnnContextCache(
        val file: File,
        val hit: Boolean
    )

    companion object {
        private const val PREF_NAME = "depth_qnn_settings"
        private const val KEY_ONNX_BACKEND = "onnx_backend"
        private const val DEFAULT_CPU_THREADS = 4

        /** Flip to true only to debug QNN graph partitioning; costs ~8 log lines per inference. */
        private const val QNN_VERBOSE_LOGGING = false
        private const val SM8350_SOC_MODEL = "30"
        private const val HTP_V68_ARCH = "68"
    }
}
