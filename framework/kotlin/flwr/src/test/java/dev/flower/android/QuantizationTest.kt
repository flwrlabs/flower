package dev.flower.android

import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class QuantizationTest {

    @Test
    fun testEmptyArrayRoundtrip() {
        val empty = FloatArray(0)
        val quantized = Quantization.quantizeToInt4(empty)
        assertEquals(0, quantized.originalLength)
        assertEquals(0, quantized.buffer.remaining())
        assertEquals(0, quantized.scales.size)

        val restored = Quantization.dequantizeFromInt4(quantized)
        assertEquals(0, restored.size)
    }

    @Test
    fun testZeroArrayRoundtrip() {
        val zeros = FloatArray(128) { 0.0f }
        val quantized = zeros.quantizeInt4(blockSize = 64)
        val restored = quantized.dequantize()

        assertEquals(zeros.size, restored.size)
        for (i in zeros.indices) {
            assertEquals(0.0f, restored[i], 1e-6f)
        }
    }

    @Test
    fun testLowMagnitudeNonzeroBlocksPreserved() {
        // Values below previous 1e-8f threshold must not round to zero
        val lowValues = FloatArray(64) { 1e-9f }
        val quantized = lowValues.quantizeInt4(blockSize = 64)
        val restored = quantized.dequantize()

        assertEquals(lowValues.size, restored.size)
        for (i in lowValues.indices) {
            assertTrue("Expected nonzero value, got ${restored[i]}", restored[i] > 0.0f)
            assertEquals(1e-9f, restored[i], 1e-10f)
        }
    }

    @Test
    fun testLargeBlockSizeNoIntegerOverflow() {
        // Test ceiling calculation does not overflow when blockSize is near Int.MAX_VALUE
        val data = FloatArray(3) { 1.0f }
        val largeEvenBlockSize = (Int.MAX_VALUE - 1) // 2147483646, positive even integer
        val quantized = data.quantizeInt4(blockSize = largeEvenBlockSize)
        assertEquals(1, quantized.scales.size)
        assertEquals(3, quantized.originalLength)

        val restored = quantized.dequantize()
        assertEquals(3, restored.size)
        for (i in data.indices) {
            assertEquals(1.0f, restored[i], 0.2f)
        }
    }

    @Test
    fun testTruncatedBufferRejected() {
        val original = FloatArray(64) { 1.0f }
        val quantized = original.quantizeInt4(blockSize = 64)

        // Provide a truncated buffer with fewer bytes than required (32 required, give 10)
        val truncatedBuffer = ByteBuffer.allocate(10)
        val invalidQuantized = QuantizedBuffer(
            buffer = truncatedBuffer,
            scales = quantized.scales,
            originalLength = 64,
            blockSize = 64
        )

        assertThrows(IllegalArgumentException::class.java) {
            Quantization.dequantizeFromInt4(invalidQuantized)
        }
    }

    @Test
    fun testInvalidBlockSize() {
        val data = FloatArray(16) { 1.0f }
        assertThrows(IllegalArgumentException::class.java) {
            Quantization.quantizeToInt4(data, blockSize = 0)
        }
        assertThrows(IllegalArgumentException::class.java) {
            Quantization.quantizeToInt4(data, blockSize = 63)
        }
        assertThrows(IllegalArgumentException::class.java) {
            Quantization.quantizeToInt4(data, blockSize = -2)
        }
    }

    @Test
    fun testNonFiniteValuesRejected() {
        val nanData = floatArrayOf(Float.NaN, 0.0f)
        assertThrows(IllegalArgumentException::class.java) {
            nanData.quantizeInt4()
        }

        val infData = floatArrayOf(Float.POSITIVE_INFINITY, 1.0f)
        assertThrows(IllegalArgumentException::class.java) {
            infData.quantizeInt4()
        }

        val negInfData = floatArrayOf(0.5f, Float.NEGATIVE_INFINITY)
        assertThrows(IllegalArgumentException::class.java) {
            negInfData.quantizeInt4()
        }
    }

    @Test
    fun testOverflowingScaleRejectedInDequantize() {
        // A scale near Float.MAX_VALUE would overflow Float when multiplied by non-zero nibbles
        val buffer = ByteBuffer.wrap(byteArrayOf(0xFF.toByte()))
        val invalidQuantized = QuantizedBuffer(
            buffer = buffer,
            scales = floatArrayOf(Float.MAX_VALUE),
            originalLength = 2,
            blockSize = 64
        )
        assertThrows(IllegalArgumentException::class.java) {
            Quantization.dequantizeFromInt4(invalidQuantized)
        }
    }

    @Test
    fun testOddAndEvenLengthsFidelity() {
        val rng = Random(42)
        val testSizes = intArrayOf(1, 2, 63, 64, 65, 128, 257)

        for (size in testSizes) {
            val original = FloatArray(size) { (rng.nextFloat() * 2.0f) - 1.0f }
            val quantized = original.quantizeInt4(blockSize = 64)
            val restored = quantized.dequantize()

            assertEquals(size, restored.size)

            var sumSquaredError = 0.0
            for (i in original.indices) {
                val diff = original[i] - restored[i]
                sumSquaredError += (diff * diff)
            }
            val mse = sumSquaredError / size
            assertTrue("MSE too high for size $size: $mse", mse < 0.05)
        }
    }

    @Test
    fun testSmallMagnitudeWithZeroValues() {
        // Values in range [Float.MIN_NORMAL, 7f / Float.MAX_VALUE] alongside zeros
        // must not produce Infinity or NaN in scaling
        val data = floatArrayOf(1.5e-38f, 0.0f, -1.5e-38f, 0.0f)
        val quantized = data.quantizeInt4(blockSize = 64)
        val restored = quantized.dequantize()
        assertEquals(4, restored.size)
        assertEquals(1.5e-38f, restored[0], 1e-39f)
        assertEquals(0.0f, restored[1], 1e-45f)
        assertEquals(-1.5e-38f, restored[2], 1e-39f)
        assertEquals(0.0f, restored[3], 1e-45f)
    }

    @Test
    fun testCompressionPayloadRatio() {
        val count = 1024
        val rawBytes = count * 4 // float32 = 4 bytes per float -> 4096 bytes
        val data = FloatArray(count) { it.toFloat() * 0.01f }

        val quantized = data.quantizeInt4(blockSize = 64)
        val packedBytes = quantized.buffer.remaining() // 512 bytes

        assertEquals(rawBytes / 8, packedBytes)
    }
}
