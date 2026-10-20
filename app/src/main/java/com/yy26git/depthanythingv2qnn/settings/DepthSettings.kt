package com.yy26git.depthanythingv2qnn.settings

/** Backend selection and inference options used by the session factory. */
enum class DepthBackend(val id: String, val label: String) {
    AUTO("auto", "Automatic"),
    QNN_HTP("qnn_htp", "Qualcomm QNN"),
    STRICT_QNN_HTP("strict_qnn_htp", "Strict Qualcomm QNN"),
    CPU("cpu", "CPU");

    companion object {
        fun fromId(id: String): DepthBackend = entries.firstOrNull { it.id == id } ?: AUTO
    }
}

data class DepthSettings(
    val onnxBackend: DepthBackend = DepthBackend.AUTO,
    val inferenceThreads: Int = 4
)
