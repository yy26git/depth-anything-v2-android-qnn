package com.yy26git.depthanythingv2qnn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DepthImageTransformTest {
    @Test
    fun raw1024x768At90MapsToUpright768x1024() {
        val t = DepthImageTransform(SourceCrop(0, 0, 1024, 768), Rotation.CLOCKWISE_90)
        assertEquals(768, t.uprightWidth)
        assertEquals(1024, t.uprightHeight)
        assertEquals(389, t.resizedWidth)
        assertEquals(518, t.resizedHeight)
        assertPoint(0.0 to 768.0, t.modelToSource(t.padLeft.toDouble(), 0.0))
        assertPoint(1024.0 to 0.0, t.modelToSource((t.padLeft + t.resizedWidth).toDouble(), 518.0))
    }

    @Test
    fun wholeVisibleRegionAndOffCenterCropRoundTripAcrossAllRotations() {
        val crop = SourceCrop(123, 55, 700, 389)
        for (rotation in Rotation.entries) {
            val t = DepthImageTransform(crop, rotation)
            for ((x, y) in listOf(123.0 to 55.0, 823.0 to 444.0, 200.25 to 180.75)) {
                val p = t.sourceToModel(x, y)
                assertPoint(x to y, t.modelToSource(p.first, p.second))
            }
        }
    }

    @Test
    fun paddingMapsToNoSourcePixel() {
        val t = DepthImageTransform(SourceCrop(0, 0, 320, 640), Rotation.NONE)
        assertFalse(t.containsModelPoint(0.5, 259.5))
        assertTrue(t.containsModelPoint(t.padLeft + 0.5, 0.5))
        assertFalse(t.containsModelPoint((t.padLeft + t.resizedWidth).toDouble(), 100.0))
    }

    @Test
    fun smallSourceIsResizedInsteadOfRejected() {
        val t = DepthImageTransform(SourceCrop(0, 0, 240, 320), Rotation.NONE)
        assertEquals(518, t.resizedHeight)
        assertEquals(389, t.resizedWidth)
    }

    @Test
    fun oldCenterCropCannotCoverWholeVisibleRegion() {
        val old = DepthImageTransform(SourceCrop(253, 125, 518, 518), Rotation.CLOCKWISE_90)
        val top = old.sourceToModel(512.0, 0.0)
        assertFalse(old.containsModelPoint(top.first, top.second))
        val full = DepthImageTransform(SourceCrop(0, 0, 1024, 768), Rotation.CLOCKWISE_90)
        val corrected = full.sourceToModel(512.0, 0.5)
        assertTrue(full.containsModelPoint(corrected.first, corrected.second))
    }

    private fun assertPoint(expected: Pair<Double, Double>, actual: Pair<Double, Double>) {
        assertEquals(expected.first, actual.first, 1e-4)
        assertEquals(expected.second, actual.second, 1e-4)
    }
}
