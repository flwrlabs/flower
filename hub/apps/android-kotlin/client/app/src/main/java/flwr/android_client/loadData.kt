package flwr.android_client

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import dev.flower.flower_tflite.FlowerClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.ExecutionException

suspend fun readAssetLines(
    context: Context,
    fileName: String,
    call: suspend (Int, String) -> Unit
) {
    withContext(Dispatchers.IO) {
        BufferedReader(InputStreamReader(context.assets.open(fileName))).useLines {
            it.forEachIndexed { i, l -> launch { call(i, l) } }
        }
    }
}

/**
 * Load training data from disk.
 */
@Throws
suspend fun loadData(
    context: Context,
    flowerClient: FlowerClient<Float3DArray, FloatArray>,
    device_id: Int
) {
    readAssetLines(context, "data/partition_${device_id - 1}_train.txt") { index, line ->
        if (index % 500 == 499) {
            Log.i(TAG, index.toString() + "th training image loaded")
        }
        addSample(context, flowerClient, "data/$line", true)
    }
    readAssetLines(context, "data/partition_${device_id - 1}_test.txt") { index, line ->
        if (index % 500 == 499) {
            Log.i(TAG, index.toString() + "th test image loaded")
        }
        addSample(context, flowerClient, "data/$line", false)
    }
}

@Throws
private fun addSample(
    context: Context,
    flowerClient: FlowerClient<Float3DArray, FloatArray>,
    photoPath: String,
    isTraining: Boolean
) {
    val options = BitmapFactory.Options()
    options.inPreferredConfig = Bitmap.Config.ARGB_8888
    val bitmap = BitmapFactory.decodeStream(context.assets.open(photoPath), null, options)!!
    val sampleClass = getClass(photoPath)

    // get rgb equivalent and class
    val rgbImage = prepareImage(bitmap)

    // add to the list.
    try {
        flowerClient.addSample(rgbImage, classToLabel(sampleClass), isTraining)
    } catch (e: ExecutionException) {
        throw RuntimeException("Failed to add sample to model", e.cause)
    } catch (e: InterruptedException) {
        // no-op
    }
}

fun getClass(path: String): String {
    return path.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()[2]
}

/**
 * Normalizes a camera image to [0; 1], cropping it
 * to size expected by the model and adjusting for camera rotation.
 */
private fun prepareImage(bitmap: Bitmap): Float3DArray {
    val normalizedRgb = Array(IMAGE_SIZE) { Array(IMAGE_SIZE) { FloatArray(3) } }
    for (y in 0 until IMAGE_SIZE) {
        for (x in 0 until IMAGE_SIZE) {
            val rgb = bitmap.getPixel(x, y)
            val r = (rgb shr 16 and LOWER_BYTE_MASK) * (1 / 255.0f)
            val g = (rgb shr 8 and LOWER_BYTE_MASK) * (1 / 255.0f)
            val b = (rgb and LOWER_BYTE_MASK) * (1 / 255.0f)
            normalizedRgb[y][x][0] = r
            normalizedRgb[y][x][1] = g
            normalizedRgb[y][x][2] = b
        }
    }
    return normalizedRgb
}

private const val TAG = "Load Data"
const val LOWER_BYTE_MASK = 0xFF

/**
 * CIFAR10 image size. This cannot be changed as the TFLite model's input layer expects
 * a 32x32x3 input.
 */
const val IMAGE_SIZE = 32

val CLASSES = listOf(
    "cat",
    "dog",
    "truck",
    "bird",
    "airplane",
    "ship",
    "frog",
    "horse",
    "deer",
    "automobile"
)

fun classToLabel(className: String): FloatArray {
    return CLASSES.map {
        if (className == it) 1f else 0f
    }.toFloatArray()
}
