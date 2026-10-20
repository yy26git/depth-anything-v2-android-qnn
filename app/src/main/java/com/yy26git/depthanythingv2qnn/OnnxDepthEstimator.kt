package com.yy26git.depthanythingv2qnn

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.yy26git.depthanythingv2qnn.runtime.DepthBackendDiagnostics
import com.yy26git.depthanythingv2qnn.runtime.OnnxRuntimeDiagnosticsStore
import com.yy26git.depthanythingv2qnn.runtime.OnnxSessionFactory
import com.yy26git.depthanythingv2qnn.logging.Logger
import java.io.Closeable

/** Owns one factory-created session; native code borrows it solely for UINT16 tensor I/O. */
class OnnxDepthEstimator private constructor(
    private val session: OrtSession,
    val diagnostics: DepthBackendDiagnostics,
    val inputName: String,
    val outputName: String,
    val inputHeight: Int,
    val inputWidth: Int,
    val batchSize: Int,
    val outputHeight: Int,
    val outputWidth: Int,
    val outputChannels: Int,
    private val runner: UInt16SessionRunner
) : Closeable {
    private var closed = false

    @Synchronized
    fun estimateDepth(pixelValues: FloatArray): FloatArray {
        check(!closed) { "Depth session is closed" }
        return runner.run(pixelValues)
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runCatching { session.close() }
            .onFailure { Logger("DepthQNN").error("session close failed", it) }
    }

    fun describeGraph(): String =
        "input='$inputName' [$batchSize,3,$inputHeight,$inputWidth] UINT16 " +
            "output='$outputName' [$batchSize,$outputChannels,$outputHeight,$outputWidth] UINT16"

    companion object {
        const val MODEL_ID = "depth_anything_v2"
        const val MODEL_PURPOSE = "depth_estimation"

        fun open(context: android.content.Context, modelBytes: ByteArray, mode: ExecutionMode): OnnxDepthEstimator {
            // Byte loading is intentional: the session factory enforces
            // session.disable_cpu_ep_fallback=1 here, unlike external-weights path loading.
            val handle = OnnxSessionFactory(context, Logger("DepthQNN")).createSession(
                // Namespace the factory-owned cache because its product version labels are fixed.
                modelId = "${MODEL_ID}_u16_ort${BuildConfig.DEPTH_ORT_VERSION}_qairt${BuildConfig.DEPTH_QAIRT_VERSION}_${BuildConfig.DEPTH_RUNTIME_ID}",
                purpose = MODEL_PURPOSE, modelBytes = modelBytes,
                isEpContext = false, settings = null, requestedBackend = mode.requestedBackend,
                allowCpuFallback = mode.allowCpuFallback
            )
            val session = handle.session
            try {
                require(session.inputInfo.size == 1 && session.outputInfo.size == 1) { "Depth model must have one input/output" }
                val input = session.inputInfo.entries.single()
                val output = session.outputInfo.entries.single()
                val inputInfo = input.value.info as? TensorInfo ?: error("Depth input is not a tensor")
                val outputInfo = output.value.info as? TensorInfo ?: error("Depth output is not a tensor")
                val uint16 = TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
                require(inputInfo.onnxType == uint16 && outputInfo.onnxType == uint16) { "Original UINT16 model boundary required" }
                val inputShape = inputInfo.shape
                val outputShape = outputInfo.shape
                require(inputShape.contentEquals(longArrayOf(1, 3, 518, 518))) { "Unexpected depth input shape" }
                require(outputShape.contentEquals(longArrayOf(1, 1, 518, 518))) { "Unexpected depth output shape" }
                val observed = handle.diagnostics.copy(
                    onnxRuntimeVersion = OrtEnvironment.getEnvironment().version,
                    qnnRuntimeVersion = "Qualcomm QAIRT ${BuildConfig.DEPTH_QAIRT_VERSION}"
                )
                OnnxRuntimeDiagnosticsStore.recordLoad(observed)
                Logger("DepthQNN").info("typed session runtime=${observed.onnxRuntimeVersion} qairt=${BuildConfig.DEPTH_QAIRT_VERSION} runtimeId=${BuildConfig.DEPTH_RUNTIME_ID} io=UINT16 strict=${!mode.allowCpuFallback} backend=${observed.activeBackend}")
                return OnnxDepthEstimator(
                    session = session, diagnostics = observed,
                    inputName = input.key, outputName = output.key,
                    inputHeight = inputShape[2].toInt(), inputWidth = inputShape[3].toInt(), batchSize = 1,
                    outputHeight = outputShape[2].toInt(), outputWidth = outputShape[3].toInt(), outputChannels = 1,
                    runner = UInt16SessionRunner(session, input.key, output.key, inputShape, outputShape)
                )
            } catch (error: Throwable) {
                handle.close()
                throw error
            }
        }
    }
}
