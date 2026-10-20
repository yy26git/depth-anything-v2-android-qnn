package com.yy26git.depthanythingv2qnn

import android.graphics.Bitmap
import android.graphics.Color
import java.util.Locale

/**
 * Conversions between a Bitmap and the tensors Depth Anything consumes and produces.
 *
 * Both directions live together because they are the two halves of one boundary contract —
 * what goes in must be shaped and scaled exactly the way the model was exported, and what
 * comes out has to be made legible. Splitting them would invite the two halves to drift.
 *
 * Preprocessing is RGB/255, and that is a property of the exported graphs rather than a free
 * choice. Immediately after its input boundary each Qualcomm AI Hub export normalizes internally:
 * a `Sub` by `view_updated` followed by a `Div` by `view_1_updated`, carrying the ImageNet mean
 * and standard deviation. An earlier version of this file claimed the float export wanted
 * externally ImageNet-normalised input and offered a normalization mode to do it; that was
 * wrong, it would have applied the mean/std twice, and the mode has been removed rather than
 * left available. Both exports want the same raw image in [0,1].
 *
 * Scope limit: these conversions are an execution-fidelity approximation, not a claim about
 * depth accuracy. The real DA-V2 processor resizes preserving aspect ratio to a multiple of 14;
 * this one resizes to the model's declared static input.
 */
object DepthImageCodec {

    /** Bitmap -> NCHW float in [0,1], replicated across the batch. */
    fun toPixelValues(bitmap: Bitmap, batchSize: Int): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val planeStride = width * height
        val values = FloatArray(batchSize * CHANNELS * planeStride)
        for (batch in 0 until batchSize) {
            val planeBase = batch * CHANNELS * planeStride
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val pixel = pixels[y * width + x]
                    values.putNchwPixel(
                        planeBase = planeBase,
                        planeStride = planeStride,
                        width = width,
                        row = y,
                        column = x,
                        red = (pixel shr 16) and 0xFF,
                        green = (pixel shr 8) and 0xFF,
                        blue = pixel and 0xFF
                    )
                }
            }
        }
        return values
    }

    /**
     * `YUV_420_888` frame -> NCHW float in [0,1], cropped and rotated to upright.
     *
     * Exists so live camera frames and still bitmaps feed the model through the *same* layout and
     * the same /255 scaling. The claim is only true because both entry points land in
     * [putNchwPixel]; two loops that merely agree today are exactly the thing that drifts.
     *
     * **Output geometry.** The result is the [crop] rectangle after [rotation], so a 90 or 270
     * degree rotation swaps the axes. It is upright — the orientation a Bitmap decoded from the
     * same capture would already be in — which is what makes this comparable to the path above
     * rather than merely similar to it.
     *
     * **Layout assumptions**, stated once and enforced in [YuvPlanes]: plane roles come from the
     * `YUV_420_888` contract (Y, U, V), and no memory layout is assumed at all. Row and pixel
     * strides address every byte, so NV12, NV21 and planar inputs all read correctly. Chroma is
     * cosited on the even grid, as camera hardware produces it, so chroma for source pixel (x, y)
     * is the sample at (x/2, y/2) of the *full frame* — computed from the absolute coordinate
     * rather than from the crop-local one, which is what keeps an odd crop origin from shifting
     * chroma by half a sample.
     *
     * **Colour.** BT.601 limited range (studio swing) in fixed point — the range Camera2
     * `YUV_420_888` streams carry. This is deliberately not bit-identical to
     * `YuvImage.compressToJpeg`, which is a JPEG encoder path with its own rounding; it is a
     * documented, exactly reproducible transform, which matters more here than matching a decoder
     * nobody is using. It remains an execution-fidelity approximation, not an accuracy claim.
     */
    fun toPixelValues(
        planes: YuvPlanes,
        crop: SourceCrop,
        rotation: Rotation,
        batchSize: Int = 1
    ): FloatArray {
        require(batchSize > 0) { "batchSize $batchSize cannot form a tensor batch" }
        require(crop.right <= planes.width && crop.bottom <= planes.height) {
            "crop ${crop.left},${crop.top} ${crop.width}x${crop.height} falls outside the " +
                "${planes.width}x${planes.height} frame"
        }
        // Checked once up front rather than per pixel: a stride misread that throws halfway through
        // would leave a half-filled tensor looking like a dark image, which is indistinguishable
        // from real data once it reaches the screen.
        require(
            planes.y.canAddress(planes.height, planes.width)
        ) {
            "luma plane addresses ${planes.y.requiredCapacity(planes.height, planes.width)} bytes " +
                "for ${planes.width}x${planes.height} but holds ${planes.y.capacity()}"
        }
        require(
            planes.u.canAddress(planes.chromaHeight, planes.chromaWidth)
        ) { "U plane is too small for the ${planes.chromaWidth}x${planes.chromaHeight} chroma grid" }
        require(
            planes.v.canAddress(planes.chromaHeight, planes.chromaWidth)
        ) { "V plane is too small for the ${planes.chromaWidth}x${planes.chromaHeight} chroma grid" }

        val outWidth = if (rotation.swapsAxes) crop.height else crop.width
        val outHeight = if (rotation.swapsAxes) crop.width else crop.height
        val planeStride = outWidth * outHeight
        val values = FloatArray(batchSize * CHANNELS * planeStride)

        for (batch in 0 until batchSize) {
            val planeBase = batch * CHANNELS * planeStride
            for (y in 0 until outHeight) {
                for (x in 0 until outWidth) {
                    // Output pixel -> crop pixel, by the inverse of the clockwise rotation. cw/ch
                    // are the crop's own dimensions, which are what the crop-local coordinates run in.
                    val cropX: Int
                    val cropY: Int
                    when (rotation) {
                        Rotation.NONE -> {
                            cropX = x
                            cropY = y
                        }
                        Rotation.CLOCKWISE_90 -> {
                            cropX = y
                            cropY = crop.height - 1 - x
                        }
                        Rotation.CLOCKWISE_180 -> {
                            cropX = crop.width - 1 - x
                            cropY = crop.height - 1 - y
                        }
                        Rotation.CLOCKWISE_270 -> {
                            cropX = crop.width - 1 - y
                            cropY = x
                        }
                    }
                    val sourceX = crop.left + cropX
                    val sourceY = crop.top + cropY
                    val rgb = readRgb(planes, sourceX, sourceY)
                    values.putNchwPixel(
                        planeBase = planeBase,
                        planeStride = planeStride,
                        width = outWidth,
                        row = y,
                        column = x,
                        red = (rgb shr 16) and 255,
                        green = (rgb shr 8) and 255,
                        blue = rgb and 255
                    )
                }
            }
        }
        return values
    }

    /** Samples the entire visible crop, using the same transform later inverted by rendering. */
    fun toModelPixelValues(planes: YuvPlanes, transform: DepthImageTransform): FloatArray {
        require(transform.crop.right <= planes.width && transform.crop.bottom <= planes.height)
        require(planes.y.canAddress(planes.height, planes.width))
        require(planes.u.canAddress(planes.chromaHeight, planes.chromaWidth))
        require(planes.v.canAddress(planes.chromaHeight, planes.chromaWidth))
        val stride = transform.modelWidth * transform.modelHeight
        // Padding normalizes to zero inside the unchanged model.
        val values = FloatArray(3 * stride)
        values.fill(0.485f, 0, stride)
        values.fill(0.456f, stride, 2 * stride)
        values.fill(0.406f, 2 * stride, 3 * stride)
        for (y in transform.padTop until transform.padTop + transform.resizedHeight) {
            for (x in transform.padLeft until transform.padLeft + transform.resizedWidth) {
                val (sx, sy) = transform.modelToSource(x + 0.5, y + 0.5)
                val px = (sx - 0.5).coerceIn(transform.crop.left.toDouble(), (transform.crop.right - 1).toDouble())
                val py = (sy - 0.5).coerceIn(transform.crop.top.toDouble(), (transform.crop.bottom - 1).toDouble())
                val x0 = kotlin.math.floor(px).toInt()
                val y0 = kotlin.math.floor(py).toInt()
                val x1 = (x0 + 1).coerceAtMost(transform.crop.right - 1)
                val y1 = (y0 + 1).coerceAtMost(transform.crop.bottom - 1)
                val rgb00 = readRgb(planes, x0, y0)
                val rgb10 = readRgb(planes, x1, y0)
                val rgb01 = readRgb(planes, x0, y1)
                val rgb11 = readRgb(planes, x1, y1)
                val fx = px - x0
                val fy = py - y0
                for (channel in 0..2) {
                    val shift = 16 - channel * 8
                    val a = (rgb00 shr shift) and 255
                    val b = (rgb10 shr shift) and 255
                    val c = (rgb01 shr shift) and 255
                    val d = (rgb11 shr shift) and 255
                    values[channel * stride + y * transform.modelWidth + x] =
                        (((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy) / 255).toFloat()
                }
            }
        }
        return values
    }

    private fun readRgb(planes: YuvPlanes, x: Int, y: Int): Int {
        val luma = planes.y.byteAt(y, x) - Y_BIAS
        val cb = planes.u.byteAt(y / 2, x / 2) - CHROMA_BIAS
        val cr = planes.v.byteAt(y / 2, x / 2) - CHROMA_BIAS
        val red = ((Y_RANGE * luma + CR_R * cr + ROUNDING) shr 8).coerceIn(0, 255)
        val green = ((Y_RANGE * luma - CB_G * cb - CR_G * cr + ROUNDING) shr 8).coerceIn(0, 255)
        val blue = ((Y_RANGE * luma + CB_B * cb + ROUNDING) shr 8).coerceIn(0, 255)
        return (red shl 16) or (green shl 8) or blue
    }

    /**
     * Depth values -> Bitmap, scaled to its own observed range so any affine output is visible.
     *
     * [width] and [height] must be the geometry of the depth buffer itself. The buffer is
     * checked against them rather than padded or cropped: a short buffer zero-filled into the
     * bottom of the image renders as a partly-black map, which is indistinguishable from a
     * legitimate depth reading once it reaches the screen.
     */
    fun renderDepth(
        depthValues: FloatArray, width: Int, height: Int, transform: DepthImageTransform? = null
    ): Bitmap {
        require(width > 0 && height > 0) { "depth geometry ${width}x$height is not renderable" }
        require(depthValues.size == width * height) {
            "depth buffer holds ${depthValues.size} values but ${width}x$height needs ${width * height}"
        }
        val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        output.density = Bitmap.DENSITY_NONE
        fun isValid(x: Int, y: Int): Boolean = transform == null || transform.containsModelPoint(
            (x + 0.5) * transform.modelWidth / width,
            (y + 0.5) * transform.modelHeight / height
        )
        var observedMin = Float.POSITIVE_INFINITY
        var observedMax = Float.NEGATIVE_INFINITY
        for (y in 0 until height) for (x in 0 until width) {
            val value = depthValues[y * width + x]
            if (isValid(x, y) && value.isFinite()) {
                observedMin = minOf(observedMin, value)
                observedMax = maxOf(observedMax, value)
            }
        }
        val range = if (observedMax > observedMin) observedMax - observedMin else 1f
        val pixels = IntArray(width * height)
        for (y in 0 until height) for (x in 0 until width) {
            val value = depthValues[y * width + x]
            pixels[y * width + x] = if (isValid(x, y) && value.isFinite()) {
                colorFor(((value - observedMin) / range).coerceIn(0f, 1f))
            } else Color.TRANSPARENT
        }
        output.setPixels(pixels, 0, width, 0, 0, width, height)
        return output
    }

    private fun colorFor(t: Float): Int {
        val red = (255 * t * t).toInt().coerceIn(0, 255)
        val green = (255 * 2f * t * (1f - t) * 1.6f).toInt().coerceIn(0, 255)
        val blue = (255 * (0.35f + 0.65f * t)).toInt().coerceIn(0, 255)
        return Color.rgb(red, green, blue)
    }

    fun format(value: Float): String = String.format(Locale.US, "%.4f", value)

    /**
     * Writes one RGB pixel into the NCHW tensor.
     *
     * This is the "one preprocessing implementation" the two entry points share: same plane order,
     * same /255 scaling, same indexing. Anything the model sees goes through here.
     */
    private fun FloatArray.putNchwPixel(
        planeBase: Int,
        planeStride: Int,
        width: Int,
        row: Int,
        column: Int,
        red: Int,
        green: Int,
        blue: Int
    ) {
        val withinPlane = row * width + column
        this[planeBase + withinPlane] = red / 255f
        this[planeBase + planeStride + withinPlane] = green / 255f
        this[planeBase + 2 * planeStride + withinPlane] = blue / 255f
    }

    // BT.601 limited range in 8.8 fixed point. Coefficients are the standard studio-swing set;
    // the +128 on every channel is the rounding term for the >>8, which is why it is named.
    private const val CHANNELS = 3
    private const val Y_RANGE = 298
    private const val CR_R = 409
    private const val CB_G = 100
    private const val CR_G = 208
    private const val CB_B = 516
    private const val Y_BIAS = 16
    private const val CHROMA_BIAS = 128
    private const val ROUNDING = 128
}
