package com.yy26git.depthanythingv2qnn

import kotlin.math.min
import kotlin.math.roundToInt

/** One crop/rotation/letterbox contract, shared by sampling and result projection.
 * Coordinates are continuous pixel edges; samplers pass x+0.5, y+0.5.
 */
data class DepthImageTransform(
    val crop: SourceCrop,
    val rotation: Rotation,
    val modelWidth: Int = 518,
    val modelHeight: Int = 518
) {
    val uprightWidth = if (rotation.swapsAxes) crop.height else crop.width
    val uprightHeight = if (rotation.swapsAxes) crop.width else crop.height
    private val scale = min(modelWidth.toDouble() / uprightWidth, modelHeight.toDouble() / uprightHeight)
    val resizedWidth = (uprightWidth * scale).roundToInt().coerceIn(1, modelWidth)
    val resizedHeight = (uprightHeight * scale).roundToInt().coerceIn(1, modelHeight)
    val padLeft = (modelWidth - resizedWidth) / 2
    val padTop = (modelHeight - resizedHeight) / 2

    init {
        require(modelWidth > 0 && modelHeight > 0)
    }

    fun containsModelPoint(x: Double, y: Double): Boolean =
        x >= padLeft && x < padLeft + resizedWidth && y >= padTop && y < padTop + resizedHeight

    /** Maps even boundary points; callers must use containsModelPoint to reject padding. */
    fun modelToSource(x: Double, y: Double): Pair<Double, Double> {
        val ux = (x - padLeft) * uprightWidth / resizedWidth
        val uy = (y - padTop) * uprightHeight / resizedHeight
        val local = when (rotation) {
            Rotation.NONE -> ux to uy
            Rotation.CLOCKWISE_90 -> uy to crop.height - ux
            Rotation.CLOCKWISE_180 -> crop.width - ux to crop.height - uy
            Rotation.CLOCKWISE_270 -> crop.width - uy to ux
        }
        return local.first + crop.left to local.second + crop.top
    }

    fun sourceToModel(x: Double, y: Double): Pair<Double, Double> {
        val cx = x - crop.left
        val cy = y - crop.top
        val upright = when (rotation) {
            Rotation.NONE -> cx to cy
            Rotation.CLOCKWISE_90 -> crop.height - cy to cx
            Rotation.CLOCKWISE_180 -> crop.width - cx to crop.height - cy
            Rotation.CLOCKWISE_270 -> cy to crop.width - cx
        }
        return padLeft + upright.first * resizedWidth / uprightWidth to
            padTop + upright.second * resizedHeight / uprightHeight
    }
}

/** Snapshot taken before ImageProxy closes. Matrices are copied to immutable lists. */
data class CameraFrameGeometry(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val crop: SourceCrop,
    val rotation: Rotation,
    val timestamp: CaptureTimestamp,
    val generation: Long,
    val sensorToBuffer: List<Float>
) {
    init {
        require(sensorToBuffer.size == 9)
    }
}
