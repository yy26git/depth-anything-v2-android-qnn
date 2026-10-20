package com.yy26git.depthanythingv2qnn

import java.nio.ByteBuffer

/**
 * What one disparity map is a measurement *of*.
 *
 * A depth map on its own is unattributable: at ~1.4-2.0 s per frame on this device, a map that
 * arrives with no record of which exposure produced it, how old it is, which model made it and on
 * which backend is indistinguishable from a map of something else. This type exists so that claim
 * is carried by the data rather than reconstructed afterwards from logs.
 *
 * Three decisions in here are deliberate and should not be "simplified" later:
 *
 *  1. **Capture time and processing time are different clock domains.** `ImageProxy.timestamp` is
 *     nanoseconds on the camera HAL's clock; everything this harness stamps work with comes from
 *     `SystemClock.elapsedRealtimeNanos()`. Subtracting one from the other yields a number that
 *     looks like a latency and is not. So the subtraction happens only through
 *     [captureToResult], and only when a [CaptureClockMapping] says the two domains were related on
 *     a real device. Otherwise it is [CrossDomainDuration.Unknown] — a first-class answer, not a
 *     zero and not a default offset. See [CrossDomainDuration] for why that matters here.
 *
 *  2. **The output is disparity, not distance.** Depth Anything V2 Small emits *relative inverse
 *     depth*. There is deliberately no scale, no focal length, no baseline and no unit field on
 *     this type, because each one invites a reader to divide by something and call the result
 *     metres. Nothing in this harness can support that, so nothing here suggests it.
 *
 *  3. **The disparity buffer is copied in and never handed out.** See [disparityAt] and
 *     [copyDisparity].
 *
 * Not responsible for: decoding pixels (DepthImageCodec) or running the model
 * (OnnxDepthEstimator). This type only describes what came out.
 */
class DepthFrame(
    /** Unique within one camera session. Conventionally the capture timestamp, or a frame counter. */
    val frameId: Long,
    /** When the exposure was taken, in whatever clock domain the producer reported. */
    val captureTimestamp: CaptureTimestamp,
    /** Full frame dimensions as delivered by the camera, before any crop or rotation. */
    val sourceWidth: Int,
    val sourceHeight: Int,
    /** The sub-rectangle actually fed to the model, in [sourceWidth]x[sourceHeight] coordinates. */
    val crop: SourceCrop,
    /** Clockwise rotation that brings the frame upright. Mirrors `ImageProxy.getRotationDegrees()`. */
    val rotation: Rotation,
    /** Disparity grid size, in the UPRIGHT orientation — the same orientation the tensor input was in. */
    val depthWidth: Int,
    val depthHeight: Int,
    /** Which model, which asset bytes. */
    val model: ModelIdentity,
    /** Which backend actually executed, as evidence rather than as intent. */
    val backend: DepthBackend,
    /** When the frame entered the pipeline, `SystemClock.elapsedRealtimeNanos()`. */
    val enqueuedNanos: Long,
    /** When disparity came back, `SystemClock.elapsedRealtimeNanos()`. */
    val resultNanos: Long,
    /**
     * A *measured* offset from [CaptureTimestamp.domain] into the monotonic domain, or null when
     * nobody has measured one on this device. Null is the expected value today; passing a guess
     * would produce a plausible frame age that no evidence supports.
     */
    verifiedCaptureClock: CaptureClockMapping?,
    /** Disparity values, row-major, `depthWidth * depthHeight` of them. Copied; see [disparityAt]. */
    disparity: FloatArray,
    val inputTransform: DepthImageTransform? = null
) {
    private val values: FloatArray

    /**
     * The capture timestamp translated into the monotonic domain, or null when it cannot be.
     *
     * The domain check is not belt-and-braces: a mapping captured for one domain and applied to a
     * frame stamped in another would produce a confident, wrong age. Matching on the domain is
     * what makes a mapping safe to pass around.
     */
    private val captureMonotonicNanos: Long?

    /** How long this frame's disparity was in flight. Same domain on both ends, so always a number. */
    val processingLatencyNanos: Long
        get() = resultNanos - enqueuedNanos

    /**
     * Capture to disparity-available, or [CrossDomainDuration.Unknown].
     *
     * This is also the only frame *age* this harness can state honestly. It is the age at the
     * moment depth became available, not the age now — and "age now" would need a monotonic
     * reading paired with a capture stamp in the monotonic domain, which is precisely the pairing
     * that does not exist without a verified mapping.
     */
    val captureToResult: CrossDomainDuration

    init {
        require(sourceWidth > 0 && sourceHeight > 0) {
            "frame ${frameId} has no dimensions: ${sourceWidth}x$sourceHeight"
        }
        require(crop.right <= sourceWidth && crop.bottom <= sourceHeight) {
            "frame ${frameId} crop ${crop.left},${crop.top} ${crop.width}x${crop.height} " +
                "falls outside its own ${sourceWidth}x$sourceHeight source"
        }
        require(depthWidth > 0 && depthHeight > 0) {
            "frame ${frameId} declares an empty disparity grid: ${depthWidth}x$depthHeight"
        }
        require(resultNanos >= enqueuedNanos) {
            "frame ${frameId} completed at $resultNanos before it was enqueued at $enqueuedNanos"
        }
        // A multi-channel output is refused rather than reduced. Depth Anything emits one channel;
        // silently taking channel 0 of a three-channel output would render a plausible map of the
        // wrong thing, which is the failure mode this harness exists to avoid.
        require(disparity.size == depthWidth * depthHeight) {
            "frame ${frameId} declares a ${depthWidth}x$depthHeight disparity grid but carries " +
                "${disparity.size} values"
        }

        // Copied so the producer may recycle its inference buffer the instant this call returns.
        // At 518x518 that is ~1 MB against a 1.4-2.0 s inference, so the copy is free; aliasing
        // it would not be, because the producer's buffer is the one ORT reuses next time.
        this.values = disparity.copyOf()

        this.captureMonotonicNanos = verifiedCaptureClock
            ?.takeIf { it.captureDomain == captureTimestamp.domain }
            ?.let { captureTimestamp.nanos + it.offsetNanos }

        this.captureToResult = captureMonotonicNanos
            ?.let { CrossDomainDuration.Mapped(resultNanos - it) }
            ?: CrossDomainDuration.Unknown
    }

    /**
     * One disparity value, bounds-checked.
     *
     * Read this rather than reaching for the array: the array is private precisely so that the
     * "everyone shares one buffer" mistake cannot be made by accident.
     */
    fun disparityAt(x: Int, y: Int): Float {
        require(x in 0 until depthWidth && y in 0 until depthHeight) {
            "disparity pixel ($x,$y) is outside the ${depthWidth}x$depthHeight grid"
        }
        return values[y * depthWidth + x]
    }

    /**
     * A private copy of the whole disparity grid, for a consumer that needs an array.
     *
     * The copy is the ownership transfer: the caller may keep and mutate it freely. It stays heap
     * memory and is reclaimed by the GC when the last reference dies — there is no pool and no
     * `release()` to call, because nothing here is native memory.
     */
    fun copyDisparity(): FloatArray = values.copyOf()

    /**
     * Maps a disparity pixel back to the pixel of the CAPTURED frame it describes.
     *
     * The chain, stated once so it does not have to be re-derived at each call site:
     *
     *   disparity(dx, dy)
     *     -> upright crop pixel, by proportional scaling across [depthWidth]x[depthHeight] onto the
     *        upright crop, which is [crop.width]x[crop.height] unless the rotation swaps the axes
     *     -> crop pixel, by the inverse of [rotation] (the crop is stored pre-rotation)
     *     -> source pixel, by adding [SourceCrop.left] and [SourceCrop.top]
     *
     * Scaling is floor division, so the mapping is exact when the disparity grid matches the
     * upright crop and approximate otherwise — which is the honest description of a model that
     * predicts at 518x518 for a 4032x3024 crop.
     *
     * Returns null for a disparity pixel outside the grid rather than clamping, because a clamped
     * answer is a real pixel that the depth map does not actually describe.
     */
    fun mapDisparityToSource(dx: Int, dy: Int): Pixel? {
        if (dx !in 0 until depthWidth || dy !in 0 until depthHeight) return null
        inputTransform?.let { transform ->
            val mx = (dx + 0.5) * transform.modelWidth / depthWidth
            val my = (dy + 0.5) * transform.modelHeight / depthHeight
            if (!transform.containsModelPoint(mx, my)) return null
            val (sx, sy) = transform.modelToSource(mx, my)
            return Pixel(kotlin.math.floor(sx).toInt(), kotlin.math.floor(sy).toInt())
        }
        val uprightWidth = if (rotation.swapsAxes) crop.height else crop.width
        val uprightHeight = if (rotation.swapsAxes) crop.width else crop.height
        val uprightX = dx * uprightWidth / depthWidth
        val uprightY = dy * uprightHeight / depthHeight
        val cropX: Int
        val cropY: Int
        when (rotation) {
            // Inverse of the forward rotation applied by DepthImageCodec. cw/ch are crop dimensions.
            Rotation.NONE -> {
                cropX = uprightX
                cropY = uprightY
            }
            Rotation.CLOCKWISE_90 -> {
                cropX = uprightY
                cropY = crop.height - 1 - uprightX
            }
            Rotation.CLOCKWISE_180 -> {
                cropX = crop.width - 1 - uprightX
                cropY = crop.height - 1 - uprightY
            }
            Rotation.CLOCKWISE_270 -> {
                cropX = crop.width - 1 - uprightY
                cropY = uprightX
            }
        }
        return Pixel(x = crop.left + cropX, y = crop.top + cropY)
    }

    override fun toString(): String =
        "DepthFrame(id=$frameId, capture=${captureTimestamp.domain}@${captureTimestamp.nanos}ns, " +
            "source=${sourceWidth}x$sourceHeight, crop=${crop.left},${crop.top} ${crop.width}x${crop.height}, " +
            "rotation=${rotation.degrees}, depth=${depthWidth}x$depthHeight, " +
            "model=${model.modelId}@${model.assetSha256.take(SHA_PREFIX)}/${model.inputWidth}x${model.inputHeight}, " +
            "backend=$backend, latency=${processingLatencyNanos}ns, captureToResult=$captureToResult)"

    private companion object {
        const val SHA_PREFIX = 12
    }
}

/** A point in the captured frame's own pixel grid. */
data class Pixel(val x: Int, val y: Int)

/**
 * A sub-rectangle of the captured frame, in that frame's pre-rotation coordinates.
 *
 * Pre-rotation on purpose: this is the space the camera delivers, and it is the space the YUV
 * planes are indexed by. Converting to upright is [DepthFrame.mapDisparityToSource]'s job.
 */
data class SourceCrop(val left: Int, val top: Int, val width: Int, val height: Int) {
    init {
        require(left >= 0 && top >= 0) { "crop origin ${left},$top is negative" }
        require(width > 0 && height > 0) { "crop ${width}x$height is empty" }
    }

    val right: Int get() = left + width
    val bottom: Int get() = top + height

    companion object {
        /** The whole frame, used when a producer applies no crop of its own. */
        fun full(width: Int, height: Int): SourceCrop =
            SourceCrop(left = 0, top = 0, width = width, height = height)
    }
}

/**
 * Clockwise rotation that brings a captured frame upright.
 *
 * Mirrors `ImageProxy.getRotationDegrees()`, which reports degrees clockwise rather than the
 * EXIF counter-clockwise convention, so the two agree and no sign flip happens at the boundary.
 */
enum class Rotation(val degrees: Int) {
    NONE(0),
    CLOCKWISE_90(90),
    CLOCKWISE_180(180),
    CLOCKWISE_270(270);

    /** True when rotating swaps width and height. */
    val swapsAxes: Boolean
        get() = degrees == 90 || degrees == 270

    companion object {
        /**
         * Rejects anything else rather than rounding. Camera rotation is always a multiple of 90,
         * so a value that is not is a bug in the caller, and quietly rounding it would rotate a
         * frame by an amount nobody asked for.
         */
        fun ofDegrees(degrees: Int): Rotation = requireNotNull(entries.firstOrNull { it.degrees == degrees }) {
            "rotation must be 0, 90, 180 or 270 degrees, was $degrees"
        }
    }
}

/**
 * A duration across two clock domains — which is a number only if the domains were related.
 *
 * The alternative designs were tried and are worse. A nullable `Long?` invites `?: 0L` at the call
 * site, which turns "we did not measure this" into "this frame was instantaneous". A zero default
 * is worse still, because a run of frames all reading zero latency is superficially the best
 * possible result. Making the unknown case its own value means a caller has to decide what to do
 * about it, and no expression can quietly turn it into a number.
 */
sealed interface CrossDomainDuration {
    /** Subtraction performed across a [CaptureClockMapping] that was verified on a real device. */
    data class Mapped(val nanos: Long) : CrossDomainDuration

    /** No mapping exists, so no duration exists. Not zero — absent. */
    data object Unknown : CrossDomainDuration
}

/**
 * A camera capture timestamp and the clock it is counted on.
 *
 * The domain travels with the value on purpose. `ImageProxy.timestamp` is nanoseconds on the
 * camera HAL's clock, which is *not* `SystemClock.elapsedRealtimeNanos()` even though both are
 * nominally monotonic, and the offset between them is a property of the device and driver rather
 * than of the app. Carrying the domain means a timestamp can never be silently compared against
 * the wrong clock.
 */
data class CaptureTimestamp(val nanos: Long, val domain: ClockDomain)

enum class ClockDomain {
    /**
     * What `ImageProxy.timestamp` reports: nanoseconds on the camera HAL's own clock. Not
     * comparable to any application clock without a measured offset.
     */
    CAMERA_SENSOR_NANOS,

    /** `SystemClock.elapsedRealtimeNanos()` — the domain every processing stamp in this harness uses. */
    HARNESS_ELAPSED_REALTIME_NANOS,

    /**
     * A domain nobody has identified. A duration across it is always UNKNOWN: guessing is how a
     * fabricated latency gets reported as a measurement.
     */
    UNVERIFIED
}

/**
 * A measured translation from a capture clock domain into the harness's monotonic domain:
 * `monotonicNanos = captureNanos + offsetNanos`.
 *
 * "Measured" is the whole requirement. A mapping is only valid for the [captureDomain] it was
 * measured against, which is why it is carried rather than being a bare offset — see
 * [DepthFrame.captureMonotonicNanos].
 */
data class CaptureClockMapping(val captureDomain: ClockDomain, val offsetNanos: Long)

/**
 * Which model produced a disparity map, pinned to the exact bytes.
 *
 * The hash is not ceremony. The harness loads a 100.2 MB inlined asset, and "the same model" across
 * two runs is otherwise an assumption: a re-export with identical dimensions produces identically
 * shaped output and a different map.
 */
data class ModelIdentity(
    val modelId: String,
    val assetSha256: String,
    val inputWidth: Int,
    val inputHeight: Int
)

/**
 * Which backend actually executed.
 *
 * [UNKNOWN] is the correct value whenever only the *requested* backend is known. The graph reports
 * `graphCompilationStatus` as a hardcoded `FULL_QNN_GRAPH_COMPILED` string rather than a
 * measurement, so honouring that field here would record intent as fact — the same reason the
 * harness treats a successful run under CPU fallback as unproven HTP execution.
 */
enum class DepthBackend {
    CPU,
    QNN_HTP,
    UNKNOWN
}

/**
 * The three planes of a `YUV_420_888` frame, addressed the way `Image.Plane` addresses them.
 *
 * This mirrors the platform type field for field (`buffer`, `rowStride`, `pixelStride`) so that the
 * adapter from `ImageProxy` is mechanical and cannot quietly change the layout assumption. It
 * exists as a plain JVM type rather than an `Image.Plane` so the conversion is testable without a
 * device — which is the only way the stride arithmetic below can be proven rather than reviewed.
 *
 * **Layout assumption.** The *roles* of the three planes come from the `YUV_420_888` contract:
 * index 0 is Y, index 1 is U (Cb), index 2 is V (Cr). Their *memory layout* is not fixed by that
 * contract and is not assumed here either — semi-planar NV12, semi-planar NV21 and fully planar
 * frames are all read correctly, because every byte is addressed through its own plane's
 * `rowStride` and `pixelStride` rather than by advancing a packed pointer. NV12 and NV21 differ
 * only in which of the two chroma planes starts at the lower address, which is carried by *which
 * plane object* was passed, not by an ordering guess inside this codec.
 *
 * Byte offsets are absolute from index 0 of [YuvPlane.buffer], matching `Image.Plane.getBuffer()`
 * with position 0. Ownership of the underlying memory stays with the caller — CameraX reuses its
 * image buffers — so the planes are only valid for the duration of the conversion call.
 */
class YuvPlanes(
    val width: Int,
    val height: Int,
    /** Luma. */
    val y: YuvPlane,
    /** Chroma blue difference, U / Cb. */
    val u: YuvPlane,
    /** Chroma red difference, V / Cr. */
    val v: YuvPlane
) {
    init {
        require(width > 0 && height > 0) { "YUV frame is ${width}x$height" }
    }

    /** Chroma grid extent. 4:2:0 halves each axis, rounding up so odd sizes stay addressable. */
    val chromaWidth: Int get() = (width + 1) / 2
    val chromaHeight: Int get() = (height + 1) / 2
}

/**
 * One plane's bytes plus how to walk it.
 *
 * Duplicated rather than retained so a caller repositioning the buffer mid-conversion cannot move
 * the ground under the reader. This is *not* a defensive copy of the data — duplicating a
 * `ByteBuffer` shares its storage — so the caller must keep the memory alive and unmodified for
 * as long as the conversion runs, and must not reuse it afterwards.
 */
class YuvPlane(
    buffer: ByteBuffer,
    val rowStride: Int,
    val pixelStride: Int
) {
    private val buffer: ByteBuffer = buffer.duplicate()

    init {
        require(rowStride >= 0) { "rowStride $rowStride is negative" }
        require(pixelStride >= 1) { "pixelStride $pixelStride cannot address a pixel" }
    }

    /** Absolute offset of one byte, from index 0. */
    fun offsetOf(row: Int, column: Int): Int = row * rowStride + column * pixelStride

    /** Bytes available to address. */
    fun capacity(): Int = buffer.capacity()

    /** One unsigned byte value. Bytes are unsigned in every YUV plane by definition. */
    fun byteAt(row: Int, column: Int): Int = buffer.get(offsetOf(row, column)).toInt() and 0xFF

    /** Bytes this plane must be able to address to cover a [rows]x[columns] grid. */
    fun requiredCapacity(rows: Int, columns: Int): Int =
        if (rows <= 0 || columns <= 0) 0 else (rows - 1) * rowStride + (columns - 1) * pixelStride

    /** Whether this plane can address a [rows]x[columns] grid at all. */
    fun canAddress(rows: Int, columns: Int): Boolean = requiredCapacity(rows, columns) < buffer.capacity()
}
