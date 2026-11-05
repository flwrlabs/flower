package dev.flower.android

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
    fun testCompressionPayloadRatio() {
        val count = 1024
        val rawBytes = count * 4 // float32 = 4 bytes per float -> 4096 bytes
        val data = FloatArray(count) { it.toFloat() * 0.01f }

        val quantized = data.quantizeInt4(blockSize = 64)
        val packedBytes = quantized.buffer.remaining() // 512 bytes

        assertEquals(rawBytes / 8, packedBytes)
    }
}
