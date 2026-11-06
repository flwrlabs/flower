package dev.flower.android

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Encapsulates packed INT4 quantized tensor data with per-block scale factors.
 *
 * @property buffer Packed byte buffer holding two 4-bit signed values per byte.
 * @property scales Per-block float32 scale factors.
 * @property originalLength Number of scalar float elements before quantization.
 * @property blockSize Number of elements per quantization block.
 */
data class QuantizedBuffer(
    val buffer: ByteBuffer,
    val scales: FloatArray,
    val originalLength: Int,
    val blockSize: Int = 64
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as QuantizedBuffer

        if (originalLength != other.originalLength) return false
        if (blockSize != other.blockSize) return false
        if (!scales.contentEquals(other.scales)) return false
        if (buffer != other.buffer) return false

        return true
    }

    override fun hashCode(): Int {
        var result = buffer.hashCode()
        result = 31 * result + scales.contentHashCode()
        result = 31 * result + originalLength
        result = 31 * result + blockSize
        return result
    }
}

/**
 * Utility functions for block-wise INT4 tensor quantization and dequantization
 * to reduce edge-to-server communication payload size in mobile federated learning.
 */
object Quantization {

    private const val MIN_NORMAL_POSITIVE_FLOAT = 1.17549435e-38f

    /**
     * Quantize a FloatArray into packed INT4 nibbles with per-block scaling.
     *
     * @param floats Input array of floating-point values.
     * @param blockSize Number of elements per block. Must be positive and even.
     * @return [QuantizedBuffer] containing packed bytes and per-block scales.
     */
    fun quantizeToInt4(floats: FloatArray, blockSize: Int = 64): QuantizedBuffer {
        require(blockSize > 0 && blockSize % 2 == 0) {
            "blockSize must be a positive even integer, got $blockSize"
        }

        val totalElements = floats.size
        if (totalElements == 0) {
            return QuantizedBuffer(
                buffer = ByteBuffer.allocate(0),
                scales = FloatArray(0),
                originalLength = 0,
                blockSize = blockSize
            )
        }

        // Avoid integer overflow in ceiling division
        val numBlocks = totalElements / blockSize + if (totalElements % blockSize != 0) 1 else 0
        val scales = FloatArray(numBlocks)
        val packedBytesCount = totalElements / 2 + (totalElements % 2)
        val packedBytes = ByteArray(packedBytesCount)

        var byteIdx = 0
        var blockIdx = 0

        while (blockIdx < numBlocks) {
            val blockStart = blockIdx * blockSize
            val blockEnd = minOf(blockStart + blockSize, totalElements)

            var maxAbs = 0.0f
            for (i in blockStart until blockEnd) {
                val a = abs(floats[i])
                if (a > maxAbs) maxAbs = a
            }

            // Zero-safe scale preserving low-magnitude blocks down to numerical limits
            val scale: Float
            val invScale: Float
            if (maxAbs == 0.0f || maxAbs < MIN_NORMAL_POSITIVE_FLOAT) {
                scale = 0.0f
                invScale = 0.0f
            } else {
                scale = maxAbs / 7.0f
                invScale = 7.0f / maxAbs
            }
            scales[blockIdx] = scale

            var i = blockStart
            while (i < blockEnd) {
                val q0 = ((floats[i] * invScale).roundToInt().coerceIn(-8, 7) + 8) and 0x0F
                val q1 = if (i + 1 < blockEnd) {
                    ((floats[i + 1] * invScale).roundToInt().coerceIn(-8, 7) + 8) and 0x0F
                } else {
                    8
                }

                packedBytes[byteIdx++] = ((q0 shl 4) or q1).toByte()
                i += 2
            }
            blockIdx++
        }

        val byteBuffer = ByteBuffer.wrap(packedBytes).order(ByteOrder.LITTLE_ENDIAN)
        return QuantizedBuffer(
            buffer = byteBuffer,
            scales = scales,
            originalLength = totalElements,
            blockSize = blockSize
        )
    }

    /**
     * Dequantize packed INT4 nibbles back into a full FloatArray.
     *
     * @param quantized [QuantizedBuffer] containing packed bytes and block scales.
     * @return Reconstructed FloatArray.
     * @throws IllegalArgumentException If the buffer is truncated or metadata is inconsistent.
     */
    fun dequantizeFromInt4(quantized: QuantizedBuffer): FloatArray {
        val totalElements = quantized.originalLength
        if (totalElements == 0) return FloatArray(0)

        require(totalElements > 0) {
            "originalLength must be positive, got $totalElements"
        }
        val blockSize = quantized.blockSize
        require(blockSize > 0 && blockSize % 2 == 0) {
            "blockSize must be a positive even integer, got $blockSize"
        }

        val expectedBytes = totalElements / 2 + (totalElements % 2)
        val remainingBytes = quantized.buffer.remaining()
        require(remainingBytes >= expectedBytes) {
            "Truncated packed buffer: expected at least $expectedBytes bytes for $totalElements elements, got $remainingBytes"
        }

        val expectedBlocks = totalElements / blockSize + if (totalElements % blockSize != 0) 1 else 0
        require(quantized.scales.size >= expectedBlocks) {
            "Insufficient scales: expected at least $expectedBlocks block scales, got ${quantized.scales.size}"
        }

        val restored = FloatArray(totalElements)
        val buffer = quantized.buffer.duplicate()

        var elemIdx = 0
        var blockIdx = 0

        while (elemIdx < totalElements) {
            val scale = quantized.scales[blockIdx]
            val blockEnd = minOf((blockIdx + 1) * blockSize, totalElements)

            while (elemIdx < blockEnd) {
                val b = buffer.get().toInt()
                val q0 = ((b shr 4) and 0x0F) - 8
                restored[elemIdx++] = q0 * scale

                if (elemIdx < blockEnd) {
                    val q1 = (b and 0x0F) - 8
                    restored[elemIdx++] = q1 * scale
                }
            }
            blockIdx++
        }

        return restored
    }
}

fun FloatArray.quantizeInt4(blockSize: Int = 64): QuantizedBuffer =
    Quantization.quantizeToInt4(this, blockSize)

fun QuantizedBuffer.dequantize(): FloatArray =
    Quantization.dequantizeFromInt4(this)
