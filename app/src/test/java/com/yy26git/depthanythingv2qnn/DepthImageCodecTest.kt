package com.yy26git.depthanythingv2qnn

import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

/**
 * JVM unit tests for the pixel boundary: YUV_420_888 and Bitmap must both reach the model through
 * one layout, and a depth map must be traceable back to the frame that produced it.
 *
 * These run on a plain JVM on purpose. Every assertion is about arithmetic a device would let you
 * skip past — a stride honoured or ignored, a plane order guessed or read, a rotation applied or not
 * — and none of it is observable by looking at a rendered image. A wrong stride produces a
 * plausible picture.
 *
 * The fixtures are hand-computed, not captured from a camera, so every expected value here can be
 * checked against the BT.601 limited-range coefficients in `DepthImageCodec` by hand. The fixture is
 * built to punish the two mistakes this code is most likely to make:
 *
 *   - **Strides are real.** The luma plane has rowStride 6 for a 4-wide frame, so every row after
 *     the first is preceded by two `0xFF` padding bytes. Ignoring rowStride does not produce an
 *     error, it produces luma 255. [yuvRowStrideIsHonoured] pins the exact wrong answer so the test
 *     fails loudly rather than looking like a lighting difference.
 *   - **Plane order is not memory layout.** Chroma is ONE NV12 buffer with U and V interleaved and
 *     the V plane starting one byte later, which is how a real semi-planar frame arrives.
 *     [yuvChromaOrderComesFromThePlaneNotTheLayout] swaps the offsets to NV21 and shows the result
 *     changes, proving the codec reads *which plane it was given* rather than a hardcoded order.
 *
 * EXECUTED, but not by Gradle. These were compiled and run with the standalone embeddable Kotlin
 * compiler (kotlin-compiler-embeddable 2.2.10) against `android.jar` plus junit 4.13.2, because the
 * build is serialized and this stage must not start one. All 12 pass there. That run is evidence
 * about the arithmetic; it is NOT a substitute for the module's own `testDebugUnitTest`, which
 * still has to confirm the wiring and the classpath.
 */
class DepthImageCodecTest {

    @Test
    fun fullViewportSamplingMatchesExistingCodecWhenNoResizeIsNeeded() {
        val crop = SourceCrop(0, 0, 4, 4)
        for (rotation in Rotation.entries) {
            val expected = DepthImageCodec.toPixelValues(yuvFrame(), crop, rotation)
            val actual = DepthImageCodec.toModelPixelValues(yuvFrame(), DepthImageTransform(crop, rotation, 4, 4))
            assertEquals(expected.toList(), actual.toList())
        }
    }

    @Test
    fun rectangularViewportHasMeanPaddingAndRetainsBothEnds() {
        val transform = DepthImageTransform(SourceCrop(0, 0, 4, 2), Rotation.NONE, 4, 4)
        val actual = DepthImageCodec.toModelPixelValues(yuvFrame(), transform)
        assertEquals(1, transform.padTop)
        assertEquals(0.485f, actual[0], 0f)
        assertEquals(0.456f, actual[16], 0f)
        assertEquals(0.406f, actual[32], 0f)
        val expected = DepthImageCodec.toPixelValues(yuvFrame(), transform.crop, Rotation.NONE)
        for (channel in 0..2) for (index in 0 until 8) {
            assertEquals(expected[channel * 8 + index], actual[channel * 16 + 4 + index], 0f)
        }
    }

    @Test
    fun letterboxedDepthFrameRejectsPaddingAndMapsToSource() {
        val transform = DepthImageTransform(SourceCrop(2, 1, 3, 2), Rotation.NONE, 6, 6)
        val frame = DepthFrame(
            frameId = 1, captureTimestamp = CaptureTimestamp(1, ClockDomain.CAMERA_SENSOR_NANOS),
            sourceWidth = 8, sourceHeight = 6, crop = transform.crop, rotation = Rotation.NONE,
            depthWidth = 6, depthHeight = 6,
            model = ModelIdentity("test", "0".repeat(64), 6, 6), backend = DepthBackend.CPU,
            enqueuedNanos = 1, resultNanos = 2, verifiedCaptureClock = null,
            disparity = FloatArray(36), inputTransform = transform
        )
        assertNull(frame.mapDisparityToSource(0, 0))
        assertEquals(Pixel(2, 1), frame.mapDisparityToSource(0, 1))
        assertEquals(Pixel(4, 2), frame.mapDisparityToSource(5, 4))
    }

    // ------------------------------------------------------------------ yuv strides and crop

    @Test
    fun yuvStridesAndCrop() {
        val values = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 1, top = 0, width = 2, height = 2),
            rotation = Rotation.NONE,
            batchSize = 1
        )

        // A 2x2 crop of a 3-channel tensor is 12 floats: R plane, then G, then B, each row-major.
        assertEquals(12L, values.size.toLong())

        // Crop origin (1,0) means output pixel (0,0) is source pixel (1,0): Y=70, U=100, V=200.
        // Every channel below is the fixed-point result of those three numbers.
        assertRgb(values, planeStride = 4, index = 0, expected = listOf(178, 15, 6))
        assertRgb(values, planeStride = 4, index = 1, expected = listOf(126, 44, 99))
        assertRgb(values, planeStride = 4, index = 2, expected = listOf(224, 62, 53))
        assertRgb(values, planeStride = 4, index = 3, expected = listOf(172, 90, 145))

        // Scaling is /255 and nothing else — no ImageNet normalization, matching the Bitmap path.
        assertEquals(178 / 255f, values[0], 0f)
        assertEquals(6 / 255f, values[8], 0f)
    }

    /**
     * The fixture only means something if a packed reader would get a different answer. The luma
     * bytes are laid out identically in both frames; only the *declared* rowStride changes, so the
     * second one reads the `0xFF` padding as luma for rows 1+. The conversion then clamps rather
     * than overflowing. Row 0 is identical either way, so the top row alone would hide the bug.
     */
    @Test
    fun yuvRowStrideIsHonoured() {
        val honest = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 1, top = 0, width = 2, height = 2),
            rotation = Rotation.NONE,
            batchSize = 1
        )
        val packed = DepthImageCodec.toPixelValues(
            planes = yuvFrame(declaredLumaRowStride = 4),
            crop = SourceCrop(left = 1, top = 0, width = 2, height = 2),
            rotation = Rotation.NONE,
            batchSize = 1
        )

        assertRgb(packed, planeStride = 4, index = 0, expected = listOf(178, 15, 6))
        assertRgb(packed, planeStride = 4, index = 1, expected = listOf(126, 44, 99))
        // Padding read as luma: 255, 231, 222 is luma 255 saturating through the BT.601 matrix.
        assertRgb(packed, planeStride = 4, index = 2, expected = listOf(255, 231, 222))
        assertRgb(packed, planeStride = 4, index = 3, expected = listOf(149, 67, 122))

        assertFalse(
            "a codec ignoring rowStride produced an identical tensor, so this fixture proves nothing",
            packed.contentEquals(honest)
        )
    }

    /**
     * NV12 (U at the lower address) and NV21 (V at the lower address) are the same bytes in a
     * different order. The codec must follow the plane it was handed, so the two decode differently
     * — swapped chroma is a plausible-looking wrong colour, not an exception.
     */
    @Test
    fun yuvChromaOrderComesFromThePlaneNotTheLayout() {
        val nv12 = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 1, top = 0, width = 2, height = 2),
            rotation = Rotation.NONE,
            batchSize = 1
        )
        val nv21 = DepthImageCodec.toPixelValues(
            planes = yuvFrame(blueDiffOffset = 1, redDiffOffset = 0),
            crop = SourceCrop(left = 1, top = 0, width = 2, height = 2),
            rotation = Rotation.NONE,
            batchSize = 1
        )

        assertRgb(nv12, planeStride = 4, index = 0, expected = listOf(178, 15, 6))
        assertRgb(nv21, planeStride = 4, index = 0, expected = listOf(18, 57, 208))
        assertFalse("swapping chroma order changed nothing", nv12.contentEquals(nv21))
    }

    /**
     * Chroma is cosited on the even grid, so it must be indexed from the absolute source coordinate,
     * not the crop-local one. A crop at x=1 reads chroma column 0; a crop at x=2 reads column 1.
     */
    @Test
    fun yuvChromaIsSitedOnTheFullFrameNotTheCrop() {
        val atOne = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 1, top = 0, width = 1, height = 1),
            rotation = Rotation.NONE,
            batchSize = 1
        )
        val atTwo = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 2, top = 0, width = 1, height = 1),
            rotation = Rotation.NONE,
            batchSize = 1
        )
        assertRgb(atOne, planeStride = 1, index = 0, expected = listOf(178, 15, 6))
        assertRgb(atTwo, planeStride = 1, index = 0, expected = listOf(126, 44, 99))
    }

    /** A crop outside the frame is refused, not clipped. A clipped crop is a different image. */
    @Test
    fun yuvCropOutsideTheFrameIsRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            DepthImageCodec.toPixelValues(
                planes = yuvFrame(),
                crop = SourceCrop(left = 3, top = 0, width = 2, height = 2),
                rotation = Rotation.NONE,
                batchSize = 1
            )
        }
    }

    // ------------------------------------------------------------------ rotation

    /**
     * 180 degrees on a square crop is exactly a pixel reversal *within each channel plane*. It is
     * not the reversal of the whole tensor: that would also reorder R, G and B, and a tensor with
     * its channels swapped still passes any check that only looks at sizes.
     */
    @Test
    fun yuvRotation180ReversesPixels() {
        val upright = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 1, top = 0, width = 2, height = 2),
            rotation = Rotation.NONE,
            batchSize = 1
        )
        val turned = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 1, top = 0, width = 2, height = 2),
            rotation = Rotation.CLOCKWISE_180,
            batchSize = 1
        )
        assertEquals(rotatedHalfTurn(upright, planeStride = 4), turned.toList())
    }

    /**
     * A 4x2 crop rotated a quarter turn comes out 2 wide and 4 tall. The axis swap is the whole
     * point: feeding a model the right pixels in the wrong orientation is invisible to any check
     * that only compares tensor sizes.
     */
    @Test
    fun yuvRotationQuarterTurnSwapsAxes() {
        val clockwise = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 0, top = 0, width = 4, height = 2),
            rotation = Rotation.CLOCKWISE_90,
            batchSize = 1
        )
        val counterClockwise = DepthImageCodec.toPixelValues(
            planes = yuvFrame(),
            crop = SourceCrop(left = 0, top = 0, width = 4, height = 2),
            rotation = Rotation.CLOCKWISE_270,
            batchSize = 1
        )

        assertEquals(24L, clockwise.size.toLong())
        assertEquals(24L, counterClockwise.size.toLong())

        // Output (x, y) reads source crop pixel (y, height-1-x). The four corners of the 2x4 output
        // are therefore the four corners of the 4x2 crop, walked in reading order.
        assertRgb(clockwise, planeStride = 8, index = 0, expected = listOf(213, 50, 41))
        assertRgb(clockwise, planeStride = 8, index = 1, expected = listOf(166, 4, 0))
        assertRgb(clockwise, planeStride = 8, index = 6, expected = listOf(184, 102, 157))
        assertRgb(clockwise, planeStride = 8, index = 7, expected = listOf(137, 55, 110))

        // 270 is 90 about the same axis, so the two differ by a per-plane reversal.
        assertEquals(rotatedHalfTurn(clockwise, planeStride = 8), counterClockwise.toList())
    }

    // ------------------------------------------------------------------ frame metadata

    /**
     * The headline mapping claim: a disparity pixel names a pixel of the captured frame, and across
     * all four rotations that naming is a bijection onto the crop — no pixel claimed twice, none
     * missed. A rotation bug shows up here as a pixel landing outside the crop.
     */
    @Test
    fun rotationMapsDepthToCapturedFrame() {
        // Metadata round-trips: every enum value survives a trip through its own degree count, and
        // an angle the camera cannot produce is refused rather than rounded.
        for (rotation in Rotation.values()) {
            assertEquals(rotation, Rotation.ofDegrees(rotation.degrees))
        }
        assertThrows(IllegalArgumentException::class.java) { Rotation.ofDegrees(45) }
        assertThrows(IllegalArgumentException::class.java) { Rotation.ofDegrees(-90) }

        // Crop at (2,1) size 3x2 inside an 8x6 source. The disparity grid matches the UPRIGHT crop,
        // so it is 3x2 unrotated and 2x3 for a quarter turn.
        val crop = SourceCrop(left = 2, top = 1, width = 3, height = 2)
        val expectedByRotation = mapOf(
            Rotation.NONE to listOf(
                Pixel(2, 1), Pixel(3, 1), Pixel(4, 1),
                Pixel(2, 2), Pixel(3, 2), Pixel(4, 2)
            ),
            Rotation.CLOCKWISE_90 to listOf(
                Pixel(2, 2), Pixel(2, 1), Pixel(3, 2),
                Pixel(3, 1), Pixel(4, 2), Pixel(4, 1)
            ),
            Rotation.CLOCKWISE_180 to listOf(
                Pixel(4, 2), Pixel(3, 2), Pixel(2, 2),
                Pixel(4, 1), Pixel(3, 1), Pixel(2, 1)
            ),
            Rotation.CLOCKWISE_270 to listOf(
                Pixel(4, 1), Pixel(4, 2), Pixel(3, 1),
                Pixel(3, 2), Pixel(2, 1), Pixel(2, 2)
            )
        )

        for ((rotation, expected) in expectedByRotation) {
            val (depthWidth, depthHeight) = if (rotation.swapsAxes) 2 to 3 else 3 to 2
            val frame = frame(
                rotation = rotation,
                crop = crop,
                depthWidth = depthWidth,
                depthHeight = depthHeight
            )

            val actual = buildList {
                for (y in 0 until depthHeight) {
                    for (x in 0 until depthWidth) {
                        add(requireNotNull(frame.mapDisparityToSource(x, y)))
                    }
                }
            }
            assertEquals("mapping for $rotation", expected, actual)
            assertEquals("mapping for $rotation is not onto the crop", cropPixelSet(crop), actual.toSet())
            assertEquals(
                "mapping for $rotation is not injective",
                crop.width.toLong() * crop.height,
                actual.distinct().size.toLong()
            )
        }
    }

    /** A disparity pixel outside the grid is refused rather than clamped to an edge pixel. */
    @Test
    fun disparityOutsideTheGridMapsToNothing() {
        val frame = frame(rotation = Rotation.NONE, depthWidth = 3, depthHeight = 2)
        assertNull(frame.mapDisparityToSource(-1, 0))
        assertNull(frame.mapDisparityToSource(0, -1))
        assertNull(frame.mapDisparityToSource(3, 0))
        assertNull(frame.mapDisparityToSource(0, 2))
    }

    /**
     * Frame age across clock domains is UNKNOWN until somebody measures a mapping on the device.
     * The point of the sealed type is that no arithmetic at the call site can turn it into a 0.
     */
    @Test
    fun captureAgeIsUnknownWithoutAVerifiedClockMapping() {
        val unverified = frame(rotation = Rotation.NONE, depthWidth = 3, depthHeight = 2)
        assertEquals(
            "a capture stamp from the camera clock must not be differenced against a monotonic stamp",
            CrossDomainDuration.Unknown,
            unverified.captureToResult
        )
        // Same-domain work is still a real number; only the cross-domain step is unknown.
        assertEquals(2_000_000L, unverified.processingLatencyNanos)

        // The offset below places the 5e9 sensor stamp at the 4e6 enqueue instant, so capture-to-
        // result is the same 2e6 the pipeline measured for its own work.
        val verified = frame(
            rotation = Rotation.NONE,
            depthWidth = 3,
            depthHeight = 2,
            verifiedCaptureClock = CaptureClockMapping(
                captureDomain = ClockDomain.CAMERA_SENSOR_NANOS,
                offsetNanos = -4_996_000_000L
            )
        )
        assertEquals(CrossDomainDuration.Mapped(2_000_000L), verified.captureToResult)

        // A mapping for the wrong domain is ignored rather than applied, so a stale offset cannot
        // produce a confident wrong age.
        val mismatched = frame(
            rotation = Rotation.NONE,
            depthWidth = 3,
            depthHeight = 2,
            verifiedCaptureClock = CaptureClockMapping(
                captureDomain = ClockDomain.UNVERIFIED,
                offsetNanos = -4_996_000_000L
            )
        )
        assertEquals(CrossDomainDuration.Unknown, mismatched.captureToResult)
    }

    /** The disparity buffer is copied on the way in, so the producer may recycle its own. */
    @Test
    fun disparityIsCopiedOnConstructionAndHandedOutByCopy() {
        val disparity = FloatArray(6) { it / 6f }
        val frame = frame(
            rotation = Rotation.NONE,
            depthWidth = 3,
            depthHeight = 2,
            disparity = disparity
        )

        disparity[0] = 99f
        assertEquals(0f, frame.disparityAt(0, 0), 0f)

        val handedOut = frame.copyDisparity()
        handedOut[0] = -1f
        assertEquals(0f, frame.disparityAt(0, 0), 0f)
        assertEquals(6L, handedOut.size.toLong())
    }

    /** A grid that disagrees with its buffer is refused rather than padded or cropped. */
    @Test
    fun disparityGridMustMatchItsBuffer() {
        assertThrows(IllegalArgumentException::class.java) {
            frame(rotation = Rotation.NONE, depthWidth = 4, depthHeight = 2, disparity = FloatArray(6))
        }
    }

    // ------------------------------------------------------------------ fixtures

    /**
     * The 4x4 synthetic frame.
     *
     * Luma: rowStride 6 for a 4-wide frame, so two `0xFF` padding bytes precede every row after the
     * first. Chroma: one NV12 buffer, rowStride 4, pixelStride 2, U and V interleaved as
     * `U00,V00,U10,V10` then `U01,V01,U11,V11` — with the V plane handed to the codec as the same
     * storage sliced one byte later.
     *
     * [declaredLumaRowStride] is a parameter so the stride trap can reuse these exact bytes while
     * telling the codec a different stride.
     */
    private fun yuvFrame(
        declaredLumaRowStride: Int = LUMA_ROW_STRIDE,
        blueDiffOffset: Int = 0,
        redDiffOffset: Int = 1
    ): YuvPlanes {
        val lumaRows = listOf(
            intArrayOf(60, 70, 80, 90),
            intArrayOf(100, 110, 120, 130),
            intArrayOf(140, 150, 160, 170),
            intArrayOf(180, 190, 200, 210)
        )
        // Always laid out at LUMA_ROW_STRIDE, padding included, so the padding bytes exist in every
        // fixture. Only the declared stride varies — that is what a codec ignoring it would read.
        val luma = ByteArray(lumaRows.size * LUMA_ROW_STRIDE) { 0xFF.toByte() }
        lumaRows.forEachIndexed { row, pixels ->
            pixels.forEachIndexed { column, value ->
                luma[row * LUMA_ROW_STRIDE + column] = value.toByte()
            }
        }

        // Interleaved NV12: U00,V00,U10,V10 then U01,V01,U11,V11.
        val chroma = intArrayOf(
            100, 200, 140, 160,
            160, 120, 180, 100
        ).map { it.toByte() }.toByteArray()
        val chromaBuffer = ByteBuffer.wrap(chroma)

        // The V plane is the same storage as the U plane, starting one byte later — which is what a
        // semi-planar NV12 frame actually is. `slice()` is what makes index 0 of that plane mean
        // V00 rather than U00. The cast is required because the android.jar this module compiles
        // against declares ByteBuffer.slice() as returning Buffer.
        fun chromaPlaneAt(offset: Int, pixelStride: Int): YuvPlane = YuvPlane(
            buffer = chromaBuffer.duplicate().apply { position(offset) }.slice() as ByteBuffer,
            rowStride = 4,
            pixelStride = pixelStride
        )

        return YuvPlanes(
            width = 4,
            height = 4,
            y = YuvPlane(ByteBuffer.wrap(luma), rowStride = declaredLumaRowStride, pixelStride = 1),
            u = chromaPlaneAt(offset = blueDiffOffset, pixelStride = 2),
            v = chromaPlaneAt(offset = redDiffOffset, pixelStride = 2)
        )
    }

    private fun frame(
        rotation: Rotation,
        crop: SourceCrop = SourceCrop(left = 2, top = 1, width = 3, height = 2),
        depthWidth: Int,
        depthHeight: Int,
        disparity: FloatArray = FloatArray(depthWidth * depthHeight) { it / 12f },
        verifiedCaptureClock: CaptureClockMapping? = null
    ): DepthFrame = DepthFrame(
        frameId = 7L,
        captureTimestamp = CaptureTimestamp(
            nanos = 5_000_000_000L,
            domain = ClockDomain.CAMERA_SENSOR_NANOS
        ),
        sourceWidth = 8,
        sourceHeight = 6,
        crop = crop,
        rotation = rotation,
        depthWidth = depthWidth,
        depthHeight = depthHeight,
        model = ModelIdentity(
            modelId = "depth_anything_v2",
            assetSha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            inputWidth = 518,
            inputHeight = 518
        ),
        backend = DepthBackend.UNKNOWN,
        enqueuedNanos = 4_000_000L,
        resultNanos = 6_000_000L,
        verifiedCaptureClock = verifiedCaptureClock,
        disparity = disparity
    )

    private fun cropPixelSet(crop: SourceCrop): Set<Pixel> = buildSet {
        for (y in crop.top until crop.bottom) {
            for (x in crop.left until crop.right) {
                add(Pixel(x, y))
            }
        }
    }

    /** One pixel of an NCHW tensor, read the way the model reads it: plane-major, then row, then column. */
    private fun assertRgb(
        values: FloatArray,
        planeStride: Int,
        index: Int,
        expected: List<Int>
    ) {
        expected.forEachIndexed { channel, channelValue ->
            assertEquals(
                "pixel $index channel $channel",
                channelValue / 255f,
                values[channel * planeStride + index],
                0f
            )
        }
    }

    /**
     * What a half turn does to an NCHW tensor: pixel order reverses inside each channel plane, and
     * the planes themselves stay in R, G, B order.
     */
    private fun rotatedHalfTurn(source: FloatArray, planeStride: Int): List<Float> =
        (0 until CHANNELS).flatMap { channel ->
            val base = channel * planeStride
            source.sliceArray(base until base + planeStride).reversedArray().toList()
        }

    private companion object {
        /** The real luma row stride of the fixture: 4-wide rows carrying 2 padding bytes. */
        const val LUMA_ROW_STRIDE = 6
        const val CHANNELS = 3
    }
}
