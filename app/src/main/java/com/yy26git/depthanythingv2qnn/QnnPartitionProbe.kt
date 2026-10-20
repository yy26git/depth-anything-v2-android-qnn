package com.yy26git.depthanythingv2qnn

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.TensorInfo
import android.content.Context
import com.yy26git.depthanythingv2qnn.runtime.OnnxQnnRuntimeEnvironment
import com.yy26git.depthanythingv2qnn.runtime.OnnxQnnRuntimeInspector
import com.yy26git.depthanythingv2qnn.logging.Logger
import org.json.JSONObject
import org.json.JSONArray
import java.io.File
import java.nio.FloatBuffer

/**
 * Harness-only diagnostic that answers two questions the benchmark cannot:
 *
 *  1. How much of this graph does the QNN EP actually take?
 *  2. What exactly fails when QNN initialization fails?
 *
 * ## Why this exists at all
 *
 * The benchmark's non-strict "QNN HTP + CPU fallback" mode does not measure a mixed session.
 * OnnxSessionFactory sets `session.disable_cpu_ep_fallback=1` for any byte-loaded model, so the
 * non-strict QNN attempt still refuses CPU fallback; it fails QNN initialization outright and the
 * factory then retries a wholly CPU session. The mode label says "hybrid" and the session that
 * results is pure CPU. That number is not a hybrid number and must not be read as one.
 *
 * This probe therefore builds its own session with the one line that makes a mixed session real —
 * CPU EP fallback left enabled — and reports node placement as its own, separate measurement.
 * It does not relabel the benchmark's modes; DepthBenchmarkRunner owns those.
 *
 * ## Division of ownership with the session factory
 *
 * The probe is measurement scaffolding, not a second QNN configuration:
 *
 *  - It NEVER loads QNN libraries. `System.load` ordering, ADSP_LIBRARY_PATH and the isV68 gate
 *    belong to `OnnxQnnRuntimeEnvironment`/`OnnxQnnRuntimeInspector`. The probe only *reads* their
 *    state and refuses to run if the factory has not already configured the runtime. Forking the
 *    loader would make the probe a shadow implementation that could pass while the product's
 *    loader drifted.
 *  - It NEVER builds provider options itself. The HTP option map is taken verbatim from the
 *    factory's own diagnostics (`DepthBackendDiagnostics.qnnProviderOptions`), so the probe
 *    partitions under exactly the options production uses. If that map is empty the probe stops
 *    with a named precondition failure rather than inventing defaults.
 *  - It uses its own `SessionOptions` and its own `OrtSession`, on its own thread, and never
 *    touches the factory's context cache.
 *
 * ## Why placement is measured from an ORT kernel profile
 *
 * ORT's Java API exposes no node-placement accessor. The kernel profile does: each executed node
 * emits an event named `<op>_kernel_time` carrying `args.provider`, which is the literal provider
 * string ("QNNExecutionProvider" / "CPUExecutionProvider"). Counting those over a single inference
 * gives per-provider node executions for that inference.
 *
 * That is the only placement evidence used, and its absence is reported as UNKNOWN rather than
 * inferred. It is deliberately NOT inferred from provider registration, from
 * `graphCompilationStatus`, or from latency:
 *  - registration says the EP was added to the session, not that it was assigned any node;
 *  - `graphCompilationStatus` is a hardcoded literal in the factory ("FULL_QNN_GRAPH_COMPILED"
 *    whenever QNN is selected) and carries no information at all;
 *  - latency cannot distinguish a mixed session from a CPU one.
 */
object QnnPartitionProbe {

    private const val LOG_TAG = "QNN-Probe"

    /**
     * Session log level. VERBOSE is correct *here* even though the factory keeps it off: the probe
     * creates a session and runs at most one inference, and the QNN EP's load-time messages —
     * notably "Number of partitions supported by QNN EP: N" — are emitted at that level. The
     * factory suppresses them because it runs inference in a hot loop; the probe does not.
     */
    private val SESSION_LOG_LEVEL = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE

    /**
     * Refuses to synthesize a probe input larger than this. The depth graph is ~800k floats, so
     * the guard never fires in practice; it exists so a malformed dynamic shape cannot turn a
     * diagnostic into an OutOfMemoryError.
     */
    private const val MAX_SYNTHETIC_INPUT_ELEMENTS = 16_000_000L

    /**
     * How the probe's session options differ from the factory's.
     *
     * Kept as data so the report can print exactly what was applied — a diagnostic whose own
     * configuration is not reported is not a diagnostic.
     */
    private data class ProbeConfiguration(
        val mode: ProbeMode,
        val disableCpuEpFallback: Boolean,
        val contextCacheEnabled: Boolean,
        val profilingRequested: Boolean
    )

    /**
     * Which partition question a probe run is asking.
     *
     * [disableCpuEpFallback] is the whole difference between the two, and it is the same single
     * config entry the factory toggles.
     */
    enum class ProbeMode(
        val id: String,
        val label: String,
        val disableCpuEpFallback: Boolean,
        val intent: String
    ) {
        /**
         * CPU EP fallback left ON. A partial partition is allowed to succeed, so a session that
         * opens here is a genuinely mixed QNN/CPU session and its profile gives real node
         * placement. This is the run that measures "how much of the graph does QNN take".
         */
        MIXED(
            id = "mixed",
            label = "MIXED QNN+CPU (CPU fallback ON)",
            disableCpuEpFallback = false,
            intent = "allow a partial partition so node placement becomes measurable"
        ),

        /**
         * CPU EP fallback OFF, mirroring the factory's strict path. Session creation fails if
         * partitioning leaves ANY node on the CPU, so a session that opens here proves full
         * acceptance. Failure here is the expected outcome for a graph with residual CPU nodes.
         */
        STRICT(
            id = "strict",
            label = "STRICT QNN (CPU fallback OFF)",
            disableCpuEpFallback = true,
            intent = "proof of purity: no node may fall back to the CPU EP"
        )
    }

    /** Where a placement count came from. UNAVAILABLE means unknown, not zero. */
    enum class PlacementSource {
        ORT_KERNEL_PROFILE,
        UNAVAILABLE
    }

    /**
     * Per-provider node executions observed in a single probe inference.
     *
     * The counts are nullable so "unknown" cannot be confused with "zero". A count of zero is a
     * measurement; a null is the absence of one.
     */
    data class QnnNodePlacement(
        val source: PlacementSource,
        val qnnKernelEvents: Int?,
        val cpuKernelEvents: Int?,
        val perProvider: Map<String, Int>
    ) {
        val known: Boolean
            get() = source != PlacementSource.UNAVAILABLE &&
                qnnKernelEvents != null &&
                cpuKernelEvents != null

        companion object {
            val UNKNOWN = QnnNodePlacement(
                source = PlacementSource.UNAVAILABLE,
                qnnKernelEvents = null,
                cpuKernelEvents = null,
                perProvider = emptyMap()
            )
        }
    }

    /**
     * Everything one probe run produced. Every field is independent evidence: no field is derived
     * from another, and none is derived from `graphCompilationStatus`.
     */
    data class QnnPartitionReport(
        val mode: ProbeMode,
        val probeModelId: String,
        /** True only when the ORT build exposes a QNN provider AND addQnn() was accepted. */
        val qnnProviderRegistered: Boolean,
        val availableProviders: List<String>,
        val providerOptionsApplied: Map<String, String>,
        val cpuEpFallbackDisabled: Boolean,
        val contextCacheEnabled: Boolean,
        val sessionCreated: Boolean,
        val sessionLoadMs: Long,
        /** Full chained error text, or null on success. Never truncated or replaced. */
        val initializationError: String?,
        val errorClassification: String?,
        val placement: QnnNodePlacement,
        /**
         * True only when a profile was read AND it showed zero QNN kernel events. Null whenever
         * the session never opened or no profile was produced.
         */
        val fellBackToCpuEntirely: Boolean?,
        val profilePath: String?,
        val notes: List<String>
    ) {

        /**
         * Human-readable block for logcat and the report screen. Fields are printed separately and
         * labelled, so a reader cannot mistake registration for placement.
         */
        fun describe(): String = buildString {
            appendLine("=== QNN PARTITION PROBE ===")
            appendLine("mode=${mode.label} [${mode.intent}]")
            appendLine("probeModelId=$probeModelId")
            appendLine("qnnProviderRegistered=$qnnProviderRegistered")
            appendLine("availableProviders=${availableProviders.joinToString().ifBlank { "none" }}")
            appendLine("providerOptions=${providerOptionsApplied.entries.joinToString { "${it.key}=${it.value}" }.ifBlank { "none" }}")
            appendLine("cpuEpFallbackDisabled=$cpuEpFallbackDisabled")
            appendLine("contextCacheEnabled=$contextCacheEnabled")
            appendLine("sessionCreated=$sessionCreated")
            appendLine("sessionLoadMs=$sessionLoadMs")
            appendLine("qnnKernelEvents=${placement.qnnKernelEvents ?: "UNKNOWN"}")
            appendLine("cpuKernelEvents=${placement.cpuKernelEvents ?: "UNKNOWN"}")
            appendLine("placementSource=${placement.source}")
            appendLine("perProviderKernelEvents=${placement.perProvider.entries.joinToString { "${it.key}=${it.value}" }.ifBlank { "none" }}")
            appendLine("fellBackToCpuEntirely=${fellBackToCpuEntirely ?: "UNKNOWN"}")
            appendLine("profilePath=${profilePath ?: "none"}")
            appendLine("errorClassification=${errorClassification ?: "none"}")
            appendLine("initializationError=${initializationError ?: "none"}")
            notes.forEach { appendLine("note=$it") }
            append("===========================")
        }
    }

    /**
     * Runs both partition paths, mixed first.
     *
     * Order is fixed so a truncated log still shows the run that matters most: [ProbeMode.MIXED] is
     * the one that can produce placement numbers.
     */
    fun probeAll(
        context: Context,
        modelBytes: ByteArray,
        productionProviderOptions: Map<String, String>,
        enableContextCache: Boolean = false,
        runProfileInference: Boolean = true,
        logger: Logger = Logger(LOG_TAG)
    ): List<QnnPartitionReport> = ProbeMode.entries.map { mode ->
        probe(
            context = context,
            modelBytes = modelBytes,
            mode = mode,
            productionProviderOptions = productionProviderOptions,
            enableContextCache = enableContextCache,
            runProfileInference = runProfileInference,
            logger = logger
        )
    }

    /** Convenience wrapper for the mixed path. */
    fun probeMixed(
        context: Context,
        modelBytes: ByteArray,
        productionProviderOptions: Map<String, String>,
        enableContextCache: Boolean = false,
        runProfileInference: Boolean = true,
        logger: Logger = Logger(LOG_TAG)
    ): QnnPartitionReport = probe(
        context = context,
        modelBytes = modelBytes,
        mode = ProbeMode.MIXED,
        productionProviderOptions = productionProviderOptions,
        enableContextCache = enableContextCache,
        runProfileInference = runProfileInference,
        logger = logger
    )

    /** Convenience wrapper for the strict path. */
    fun probeStrict(
        context: Context,
        modelBytes: ByteArray,
        productionProviderOptions: Map<String, String>,
        enableContextCache: Boolean = false,
        runProfileInference: Boolean = true,
        logger: Logger = Logger(LOG_TAG)
    ): QnnPartitionReport = probe(
        context = context,
        modelBytes = modelBytes,
        mode = ProbeMode.STRICT,
        productionProviderOptions = productionProviderOptions,
        enableContextCache = enableContextCache,
        runProfileInference = runProfileInference,
        logger = logger
    )

    /**
     * Creates one probe session and reports what the QNN EP did with it.
     *
     * Must be called on a worker thread: session creation compiles on the HTP and blocks. The
     * caller owns the serial ordering (the benchmark's own worker).
     *
     * @param productionProviderOptions the `qnnProviderOptions` map from the factory's
     *   diagnostics. Passed in rather than rebuilt so the probe partitions under the same HTP
     *   options production uses; an empty map is refused, not defaulted.
     * @param enableContextCache defaults to false so the probe never reads or writes the factory's
     *   context cache and always measures fresh load-time partitioning rather than a cached
     *   context load. When enabled the probe still writes to its own directory under its own
     *   filename.
     * @param runProfileInference runs one zero-filled inference and reads ORT's kernel profile.
     *   This is the only source of placement counts.
     */
    fun probe(
        context: Context,
        modelBytes: ByteArray,
        mode: ProbeMode,
        productionProviderOptions: Map<String, String>,
        enableContextCache: Boolean = false,
        runProfileInference: Boolean = true,
        logger: Logger = Logger(LOG_TAG)
    ): QnnPartitionReport {
        // A probe identifier distinct from any production modelId. The factory keys its context
        // cache on modelId, so a distinct id also guarantees no production cache file is ever
        // read or overwritten by this probe.
        val probeModelId = "$PROBE_MODEL_ID_PREFIX${mode.id}"
        val environment = OrtEnvironment.getEnvironment()
        val availableProviders = runCatching {
            OrtEnvironment.getAvailableProviders().map { it.name }.sorted()
        }.getOrElse { error -> listOf("unavailable: ${error.message ?: error.javaClass.simpleName}") }
        val qnnInBuild = availableProviders.any { it.contains("QNN", ignoreCase = true) }
        val configuration = ProbeConfiguration(
            mode = mode,
            disableCpuEpFallback = mode.disableCpuEpFallback,
            contextCacheEnabled = enableContextCache,
            profilingRequested = runProfileInference
        )

        val precondition = preconditionFailure(
            context = context,
            qnnInBuild = qnnInBuild,
            productionProviderOptions = productionProviderOptions
        )
        if (precondition != null) {
            logger.warn("QNN partition probe precondition failed mode=${mode.label}: $precondition")
            return QnnPartitionReport(
                mode = mode,
                probeModelId = probeModelId,
                qnnProviderRegistered = false,
                availableProviders = availableProviders,
                providerOptionsApplied = emptyMap(),
                cpuEpFallbackDisabled = configuration.disableCpuEpFallback,
                contextCacheEnabled = configuration.contextCacheEnabled,
                sessionCreated = false,
                sessionLoadMs = 0L,
                initializationError = precondition.toString(),
                errorClassification = precondition.classification,
                placement = QnnNodePlacement.UNKNOWN,
                fellBackToCpuEntirely = null,
                profilePath = null,
                notes = listOf(
                    "No session was attempted, so no placement count exists. Placement is UNKNOWN," +
                        " not zero.",
                    "Provider registration, graphCompilationStatus and latency are not substitutes."
                )
            )
        }

        val notes = mutableListOf<String>()
        notes += "Probe does not load QNN libraries; it reuses the runtime the factory configured."
        notes += "Provider options are the factory's own qnnProviderOptions, passed through verbatim."
        notes += "CPU EP is NOT explicitly added: ORT already registers it as the lowest-priority" +
            " fallback. Calling addCPU(true) would give the CPU priority over QNN and could starve" +
            " the QNN partition, which would invert the thing being measured."

        var options: SessionOptions? = null
        var session: OrtSession? = null
        var providerRegistered = false
        var sessionCreated = false
        var sessionLoadMs = 0L
        var profilePath: String? = null
        var profileInferenceCompleted = false
        var failure: Throwable? = null
        var loadStartedNs = System.nanoTime()

        try {
            // Local non-null handles are used for the calls that need them; the nullable vars above
            // exist only so `finally` can close whatever was reached. Smart-casting a captured var
            // is not something this file should depend on.
            val sessionOptions = SessionOptions()
            options = sessionOptions
            configure(
                options = sessionOptions,
                context = context,
                probeModelId = probeModelId,
                configuration = configuration,
                productionProviderOptions = productionProviderOptions
            )
            // addQnn returning without throwing is the only claim "provider registered" can make.
            // It is recorded as such and never promoted into a placement statement.
            providerRegistered = true

            if (configuration.profilingRequested) {
                // Profiling is requested as an option BEFORE the session exists, because that is
                // the only point at which it can be enabled. endProfiling() is reached only if the
                // session opens; a failed init produces no profile and none is claimed.
                sessionOptions.enableProfiling(File(profileDirectory(context), mode.id).absolutePath)
            }

            loadStartedNs = System.nanoTime()
            val openedSession = environment.createSession(modelBytes, sessionOptions)
            session = openedSession
            sessionCreated = true
            sessionLoadMs = (System.nanoTime() - loadStartedNs) / 1_000_000L

            if (configuration.profilingRequested) {
                val inferenceProblem = runOneProbeInference(openedSession)
                profileInferenceCompleted = inferenceProblem == null
                if (inferenceProblem != null) {
                    notes += "Placement inference skipped: $inferenceProblem. The profile below is" +
                        " incomplete and node counts derived from it are UNKNOWN."
                }
                // endProfiling must run before the session is closed, and its return value is the
                // authoritative path — never the prefix passed to enableProfiling.
                profilePath = runCatching { openedSession.endProfiling() }.getOrElse { error ->
                    notes += "endProfiling failed: ${error.message ?: error.javaClass.simpleName}"
                    null
                }
            }
        } catch (error: Throwable) {
            failure = error
            if (!sessionCreated) sessionLoadMs = (System.nanoTime() - loadStartedNs) / 1_000_000L
            logger.error(
                "QNN partition probe failed mode=${mode.label} fallbackDisabled=${configuration.disableCpuEpFallback}" +
                    " classification=${classify(error)}",
                error
            )
            notes += "Session creation failed, so no profile exists and placement is UNKNOWN."
        } finally {
            // Ownership: options and session are created here and released here, in both cases and
            // on every path including the failure path above.
            runCatching { session?.close() }
                .onFailure { logger.error("QNN partition probe session close failed", it) }
            runCatching { options?.close() }
                .onFailure { logger.error("QNN partition probe options close failed", it) }
        }

        // Resolved through a local val rather than a null-check on profilePath directly, so no
        // smart cast on the captured var is involved.
        val profileFile = profilePath?.let { path -> File(path) }
        val placement = if (profileFile != null && profileInferenceCompleted) {
            readPlacement(profileFile)
        } else {
            QnnNodePlacement.UNKNOWN
        }

        val fellBackEntirelyToCpu: Boolean? = when {
            !sessionCreated || !placement.known -> null
            else -> placement.qnnKernelEvents == 0
        }

        if (sessionCreated && !placement.known) {
            notes += "The session opened but produced no readable kernel profile. Node placement is" +
                " UNKNOWN — it is not inferred from successful initialization."
        }

        val report = QnnPartitionReport(
            mode = mode,
            probeModelId = probeModelId,
            qnnProviderRegistered = providerRegistered && qnnInBuild,
            availableProviders = availableProviders,
            providerOptionsApplied = productionProviderOptions,
            cpuEpFallbackDisabled = configuration.disableCpuEpFallback,
            contextCacheEnabled = configuration.contextCacheEnabled,
            sessionCreated = sessionCreated,
            sessionLoadMs = sessionLoadMs,
            initializationError = failure?.let { errorChain(it) },
            errorClassification = failure?.let { classify(it) },
            placement = placement,
            fellBackToCpuEntirely = fellBackEntirelyToCpu,
            profilePath = profilePath,
            notes = notes
        )
        logger.info(report.describe())
        return report
    }

    private class PreconditionFailure(val classification: String, val message: String) {
        override fun toString(): String = "$classification: $message"
    }

    /**
     * Refuses to run a probe that could not possibly answer the question, instead of producing a
     * confusing downstream error.
     */
    private fun preconditionFailure(
        context: Context,
        qnnInBuild: Boolean,
        productionProviderOptions: Map<String, String>
    ): PreconditionFailure? = when {
        !OnnxQnnRuntimeEnvironment.isConfigured() -> PreconditionFailure(
            classification = "QNN_RUNTIME_NOT_CONFIGURED",
            message = "The QNN native environment has not been configured. The probe deliberately" +
                " does not load the libraries itself — OnnxQnnRuntimeEnvironment owns that — so run" +
                " a factory session first, on the same worker."
        )
        !OnnxQnnRuntimeInspector.inspect(context).available -> PreconditionFailure(
            classification = "QNN_RUNTIME_UNAVAILABLE",
            message = "OnnxQnnRuntimeInspector reports the QNN runtime as unavailable on this device."
        )
        !qnnInBuild -> PreconditionFailure(
            classification = "QNN_PROVIDER_NOT_IN_BUILD",
            message = "This ONNX Runtime build exposes no QNN execution provider."
        )
        productionProviderOptions.isEmpty() -> PreconditionFailure(
            classification = "PRODUCTION_PROVIDER_OPTIONS_EMPTY",
            message = "qnnProviderOptions from the factory diagnostics is empty, which means the" +
                " factory did not build a QNN session for the diagnostics being passed in. The probe" +
                " does not synthesize replacement options, because guessing them would measure a" +
                " configuration the product never runs."
        )
        else -> null
    }

    /**
     * Applies the probe's session options.
     *
     * The option set is deliberately a subset of the factory's:
     *  - `addQnn` with production's own map;
     *  - `session.disable_cpu_ep_fallback` set explicitly to this mode's value, so the difference
     *    between the two probe modes is a visible single line rather than an implicit default;
     *  - `ep.context_enable` off by default, so partitioning is measured fresh at load time and no
     *    context file is read or written.
     *
     * VERBOSE logging is on for the reasons documented on [SESSION_LOG_LEVEL].
     */
    private fun configure(
        options: SessionOptions,
        context: Context,
        probeModelId: String,
        configuration: ProbeConfiguration,
        productionProviderOptions: Map<String, String>
    ) {
        options.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
        options.setInterOpNumThreads(1)
        options.setIntraOpNumThreads(INTRA_OP_THREADS)
        // Distinct logger id so probe log lines are separable from production ones in logcat.
        options.setLoggerId("DepthQNN-$probeModelId")
        options.setSessionLogLevel(SESSION_LOG_LEVEL)
        options.setSessionLogVerbosityLevel(1)

        options.addQnn(productionProviderOptions)
        options.addConfigEntry("session.disable_cpu_ep_fallback", if (configuration.disableCpuEpFallback) "1" else "0")

        if (configuration.contextCacheEnabled) {
            // Probe-owned path under a probe-owned directory with a probe-owned filename, so the
            // factory's cache directory is never touched. embed_mode 0 writes a standalone context
            // binary to that file; embed_mode 1 would try to write into the model itself, which
            // this probe does not own.
            val contextFile = File(profileDirectory(context), "${configuration.mode.id}-probe-context.onnx")
            options.addConfigEntry("ep.context_enable", "1")
            options.addConfigEntry("ep.context_embed_mode", "0")
            options.addConfigEntry("ep.context_file_path", contextFile.absolutePath)
        } else {
            options.addConfigEntry("ep.context_enable", "0")
        }
    }

    /**
     * One zero-filled inference, run only to give the profiler node events to count.
     *
     * The values are zeros on purpose: the probe measures placement, not output quality. A correct
     * result is not required and is not checked.
     *
     * @return null on success, or a reason the inference could not be run — in which case the
     *   caller must not derive node counts from whatever profile exists.
     */
    private fun runOneProbeInference(session: OrtSession): String? {
        val entry = session.inputInfo.entries.firstOrNull()
            ?: return "the session exposes no inputs"
        val info = entry.value.info as? TensorInfo
            ?: return "input '${entry.key}' is not a tensor (${entry.value.info})"
        val shape = info.shape
        if (shape.any { it <= 0 } || shape.fold(1L) { a, b -> a * b } > MAX_SYNTHETIC_INPUT_ELEMENTS) {
            return "probe input has an invalid or excessive extent"
        }
        val output = session.outputInfo.entries.singleOrNull() ?: return "probe requires one output"
        val outputInfo = output.value.info as? TensorInfo ?: return "probe output is not a tensor"
        if (info.onnxType != TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
            outputInfo.onnxType != TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
            return "probe requires the original UINT16 graph"
        }
        return try {
            UInt16SessionRunner(session, entry.key, output.key, shape, outputInfo.shape)
                .run(FloatArray(shape.fold(1L) { a, b -> a * b }.toInt()))
            null
        } catch (error: Exception) {
            "probe inference failed: ${error.message ?: error.javaClass.simpleName}"
        }
    }

    /**
     * Counts per-provider node executions from an ORT kernel profile.
     *
     * ORT names each node event `<op>_kernel_time` and records the assigned provider in
     * `args.provider`. Any event shape this does not recognise is skipped rather than guessed at,
     * and a profile with no recognisable node events yields [QnnNodePlacement.UNKNOWN].
     */
    private fun readPlacement(profileFile: File): QnnNodePlacement {
        if (!profileFile.isFile || profileFile.length() == 0L) return QnnNodePlacement.UNKNOWN
        val text = runCatching { profileFile.readText() }.getOrNull() ?: return QnnNodePlacement.UNKNOWN
        val events = runCatching {
            // ORT emits a top-level array; also accept a Chrome trace wrapper.
            if (text.trimStart().startsWith("[")) JSONArray(text)
            else JSONObject(text).optJSONArray("traceEvents")
        }.getOrNull() ?: return QnnNodePlacement.UNKNOWN

        val perProvider = mutableMapOf<String, Int>()
        for (index in 0 until events.length()) {
            val event = events.optJSONObject(index) ?: continue
            val name = event.optString("name")
            if (!name.endsWith(KERNEL_EVENT_SUFFIX)) continue
            val arguments = event.optJSONObject("args") ?: continue
            val provider = arguments.optString("provider")
            if (provider.isBlank()) continue
            perProvider[provider] = (perProvider[provider] ?: 0) + 1
        }
        if (perProvider.isEmpty()) return QnnNodePlacement.UNKNOWN

        val qnn = perProvider.filterKeys { it.contains("QNN", ignoreCase = true) }.values.sum()
        val cpu = perProvider.filterKeys { it.contains("CPU", ignoreCase = true) }.values.sum()
        return QnnNodePlacement(
            source = PlacementSource.ORT_KERNEL_PROFILE,
            qnnKernelEvents = qnn,
            cpuKernelEvents = cpu,
            perProvider = perProvider.toSortedMap()
        )
    }

    /**
     * Full chained error text. The QNN failure this probe exists to diagnose is buried several
     * levels down (OrtException -> IllegalStateException -> native status), so the outermost
     * message alone routinely reports nothing useful.
     */
    private fun errorChain(error: Throwable): String = generateSequence(error) { it.cause }
        .map { "${it.javaClass.simpleName}: ${it.message ?: "no message"}" }
        .joinToString(separator = " <- ")

    /**
     * Maps a failure onto a named class.
     *
     * `QNN_PARTITION_NO_GRAPH_INPUT_TENSORS` is the live hypothesis for this graph: ORT 1.27's
     * qnn_model.cc SetupTensors returns on an empty tensor-wrapper vector, so a partition whose
     * node inputs are all initializers produces no graph-input wrappers at all. That is a
     * hypothesis the probe is built to test, not an established cause — which is exactly why the
     * raw text is reported alongside it.
     */
    private fun classify(error: Throwable): String {
        val message = generateSequence(error) { it.cause }
            .mapNotNull { it.message }
            .joinToString(separator = " | ")
        return when {
            message.contains("Zero tensor size", ignoreCase = true) ||
                message.contains("Failed to setup QNN input tensors", ignoreCase = true) ->
                "QNN_PARTITION_NO_GRAPH_INPUT_TENSORS"
            message.contains("Fallback to CPU EP is disabled", ignoreCase = true) ->
                "CPU_FALLBACK_DISABLED_WITH_RESIDUAL_NODES"
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

    /** Probe-owned output directory. Never the factory's `onnx/qnn_context_cache`. */
    private fun profileDirectory(context: Context): File =
        File(context.filesDir, "onnx/qnn_partition_probe").apply { mkdirs() }

    private const val PROBE_MODEL_ID_PREFIX = "depth_partition_probe_"
    private const val KERNEL_EVENT_SUFFIX = "_kernel_time"

    /** Matches the factory's DEFAULT_CPU_THREADS so the probe is not the odd one out. */
    private const val INTRA_OP_THREADS = 4
}
