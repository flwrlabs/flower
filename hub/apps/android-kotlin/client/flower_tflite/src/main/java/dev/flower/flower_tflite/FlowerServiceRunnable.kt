package dev.flower.flower_tflite

import android.util.Log
import com.google.protobuf.ByteString
import dev.flower.flower_tflite.helpers.assertIntsEqual
import flwr.android_client.ClientMessage
import flwr.android_client.FlowerServiceGrpc
import flwr.android_client.Parameters
import flwr.android_client.Scalar
import flwr.android_client.ServerMessage
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import io.grpc.stub.StreamObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch

/**
 * Start communication with Flower server and training in the background.
 * Note: constructing an instance of this class **immediately** starts training.
 *
 * Use [createFlowerService] to create a [FlowerServiceRunnable] instance using Flower server address.
 * @param flowerServerChannel Channel already connected to Flower server.
 * @param callback Called with information on training events.
 */
class FlowerServiceRunnable<X : Any, Y : Any>
@Throws constructor(
    flowerServerChannel: ManagedChannel,
    val flowerClient: FlowerClient<X, Y>,
    val callback: (String) -> Unit
) {
    private val sampleSize: Int
        get() = flowerClient.trainingSamples.size
    val finishLatch = CountDownLatch(1)

    val asyncStub = FlowerServiceGrpc.newStub(flowerServerChannel)!!
    val requestObserver = asyncStub.join(object : StreamObserver<ServerMessage> {
        override fun onNext(msg: ServerMessage) {
            try {
                handleMessage(msg)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        override fun onError(t: Throwable) {
            t.printStackTrace()
            finishLatch.countDown()
            Log.e(TAG, t.message!!)
        }

        override fun onCompleted() {
            finishLatch.countDown()
            Log.d(TAG, "Done")
        }
    })!!

    @Throws
    fun handleMessage(message: ServerMessage) {
        val clientMessage = if (message.hasGetParametersIns()) {
            handleGetParamsIns()
        } else if (message.hasFitIns()) {
            handleFitIns(message)
        } else if (message.hasEvaluateIns()) {
            handleEvaluateIns(message)
        } else if (message.hasReconnectIns()) {
            return requestObserver.onCompleted()
        } else {
            throw Error("Unreachable! Unknown client message")
        }
        requestObserver.onNext(clientMessage)
        callback("Response sent to the server")
    }

    @Throws
    fun handleGetParamsIns(): ClientMessage {
        Log.d(TAG, "Handling GetParameters")
        callback("Handling GetParameters message from the server.")
        return weightsAsProto(weightsByteBuffers())
    }

    @Throws
    fun handleFitIns(message: ServerMessage): ClientMessage {
        Log.d(TAG, "Handling FitIns")
        callback("Handling Fit request from the server.")
        val layers = message.fitIns.parameters.tensorsList
        assertIntsEqual(layers.size, flowerClient.layersSizes.size)
        val epochConfig = message.fitIns.configMap.getOrDefault(
            "local_epochs", Scalar.newBuilder().setSint64(1).build()
        )!!
        val epochs = epochConfig.sint64.toInt()
        val newWeights = weightsFromLayers(layers)
        flowerClient.updateParameters(newWeights.toTypedArray())
        flowerClient.fit(
            epochs,
            lossCallback = { callback("Average loss: ${it.average()}.") })
        return fitResAsProto(weightsByteBuffers(), sampleSize)
    }

    @Throws
    fun handleEvaluateIns(message: ServerMessage): ClientMessage {
        Log.d(TAG, "Handling EvaluateIns")
        callback("Handling Evaluate request from the server")
        val layers = message.evaluateIns.parameters.tensorsList
        assertIntsEqual(layers.size, flowerClient.layersSizes.size)
        val newWeights = weightsFromLayers(layers)
        flowerClient.updateParameters(newWeights.toTypedArray())
        val (loss, accuracy) = flowerClient.evaluate()
        callback("Test Accuracy after this round = $accuracy")
        return evaluateResAsProto(loss, sampleSize)
    }

    private fun weightsByteBuffers() = flowerClient.getParameters()

    private fun weightsFromLayers(layers: List<ByteString>) =
        layers.map { ByteBuffer.wrap(it.toByteArray()) }

    companion object {
        private const val TAG = "Flower Service Runnable"
    }
}

fun weightsAsProto(weights: Array<ByteBuffer>): ClientMessage {
    val layers = weights.map { ByteString.copyFrom(it) }
    val p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build()
    val res = ClientMessage.GetParametersRes.newBuilder().setParameters(p).build()
    return ClientMessage.newBuilder().setGetParametersRes(res).build()
}

fun fitResAsProto(weights: Array<ByteBuffer>, training_size: Int): ClientMessage {
    val layers: MutableList<ByteString> = ArrayList()
    for (weight in weights) {
        layers.add(ByteString.copyFrom(weight))
    }
    val p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build()
    val res =
        ClientMessage.FitRes.newBuilder().setParameters(p).setNumExamples(training_size.toLong())
            .build()
    return ClientMessage.newBuilder().setFitRes(res).build()
}

fun evaluateResAsProto(accuracy: Float, testing_size: Int): ClientMessage {
    val res = ClientMessage.EvaluateRes.newBuilder().setLoss(accuracy)
        .setNumExamples(testing_size.toLong()).build()
    return ClientMessage.newBuilder().setEvaluateRes(res).build()
}

/**
 * Create a [FlowerServiceRunnable] with address to the Flower server.
 * @param flowerServerAddress Like "dns:///$host:$port".
 */
suspend fun <X : Any, Y : Any> createFlowerService(
    flowerServerAddress: String,
    useTLS: Boolean,
    flowerClient: FlowerClient<X, Y>,
    callback: (String) -> Unit
): FlowerServiceRunnable<X, Y> {
    val channel = createChannel(flowerServerAddress, useTLS)
    return FlowerServiceRunnable(channel, flowerClient, callback)
}

/**
 * @param address Address of the gRPC server, like "dns:///$host:$port".
 */
suspend fun createChannel(address: String, useTLS: Boolean = false): ManagedChannel {
    val channelBuilder =
        ManagedChannelBuilder.forTarget(address).maxInboundMessageSize(HUNDRED_MEBIBYTE)
    if (!useTLS) {
        channelBuilder.usePlaintext()
    }
    return withContext(Dispatchers.IO) {
        channelBuilder.build()
    }
}

const val HUNDRED_MEBIBYTE = 100 * 1024 * 1024
