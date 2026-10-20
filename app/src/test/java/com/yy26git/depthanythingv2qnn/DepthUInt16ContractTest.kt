package com.yy26git.depthanythingv2qnn

import org.junit.Assert.*
import org.junit.Test
import java.nio.ShortBuffer

class DepthUInt16ContractTest {
    @Test fun everyCameraByteMatchesIndependentOnnxEncoding() {
        val resource = checkNotNull(javaClass.getResourceAsStream("/uint16-rgb-codes.csv"))
        val rows = resource.bufferedReader().use { it.readLines() }
        assertEquals(256, rows.size)
        rows.forEach { row ->
            val (pixel, expected) = row.split(',').map(String::toInt)
            assertEquals("RGB byte $pixel", expected, DepthUInt16Contract.encode(pixel / 255f).toInt() and 0xffff)
        }
    }
    @Test fun saturatesInsteadOfWrappingAndRejectsNonfinite() {
        assertEquals(0, DepthUInt16Contract.encode(-100f).toInt() and 0xffff)
        assertEquals(65535, DepthUInt16Contract.encode(100f).toInt() and 0xffff)
        for (value in listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)) {
            assertThrows(IllegalArgumentException::class.java) { DepthUInt16Contract.encode(value) }
        }
    }
    @Test fun decodesUnsignedHighBitAndAllCodesMonotonically() {
        val decoded = DepthUInt16Contract.decode(ShortBuffer.wrap(ShortArray(65536) { it.toShort() }))
        assertEquals(0f, decoded.first(), 0f)
        assertEquals(13.15931f, decoded.last(), 0.00001f)
        for (i in 1 until decoded.size) assertTrue("code $i", decoded[i] > decoded[i - 1])
    }
    @Test fun rejectsWrongBufferExtentBeforeNativeRun() {
        assertThrows(IllegalArgumentException::class.java) {
            DepthUInt16Contract.encode(floatArrayOf(0f), ShortBuffer.allocate(2))
        }
    }
}
