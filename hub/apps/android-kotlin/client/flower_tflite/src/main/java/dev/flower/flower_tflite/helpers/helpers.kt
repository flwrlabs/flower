package dev.flower.flower_tflite.helpers

import android.content.Context
import android.util.Log
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

@Throws(IOException::class)
fun loadMappedFile(file: File): MappedByteBuffer {
    Log.i("Loading mapped file", "$file")
    val accessFile = RandomAccessFile(file, "r")
    val channel = accessFile.channel
    return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
}

@Throws(IOException::class)
fun loadMappedAssetFile(context: Context, filePath: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(filePath)
    val fileChannel = fileDescriptor.createInputStream().channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

infix fun <T, R> Iterable<T>.lazyZip(other: Array<out R>): Sequence<Pair<T, R>> {
    val ours = iterator()
    val theirs = other.iterator()

    return sequence {
        while (ours.hasNext() && theirs.hasNext()) {
            yield(ours.next() to theirs.next())
        }
    }
}

fun FloatArray.argmax(): Int = indices.maxBy { this[it] }

@Throws(AssertionError::class)
fun assertIntsEqual(expected: Int, actual: Int) {
    if (expected != actual) {
        throw AssertionError("Test failed: expected `$expected`, got `$actual` instead.")
    }
}
