package flwr.android_client

import android.icu.text.SimpleDateFormat
import android.os.Bundle
import android.text.TextUtils
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.util.Patterns
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.room.Room
import dev.flower.flower_tflite.FlowerClient
import dev.flower.flower_tflite.FlowerServiceRunnable
import dev.flower.flower_tflite.SampleSpec
import dev.flower.flower_tflite.createFlowerService
import dev.flower.flower_tflite.helpers.classifierAccuracy
import dev.flower.flower_tflite.helpers.loadMappedAssetFile
import dev.flower.flower_tflite.helpers.negativeLogLikelihoodLoss
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.util.*

@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity() {
    private val scope = MainScope()
    lateinit var flowerClient: FlowerClient<Float3DArray, FloatArray>
    lateinit var flowerServiceRunnable: FlowerServiceRunnable<Float3DArray, FloatArray>
    private lateinit var ip: EditText
    private lateinit var port: EditText
    private lateinit var loadDataButton: Button
    private lateinit var trainButton: Button
    private lateinit var resultText: TextView
    private lateinit var deviceId: EditText
    lateinit var db: Db

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        db = Room.databaseBuilder(this, Db::class.java, "model-db").build()
        setContentView(R.layout.activity_main)
        resultText = findViewById(R.id.grpc_response_text)
        resultText.movementMethod = ScrollingMovementMethod()
        deviceId = findViewById(R.id.device_id_edit_text)
        ip = findViewById(R.id.serverIP)
        port = findViewById(R.id.serverPort)
        loadDataButton = findViewById(R.id.load_data)
        trainButton = findViewById(R.id.trainFederated)
        createFlowerClient()
        scope.launch { restoreInput() }
    }

    private fun createFlowerClient() {
        val buffer = loadMappedAssetFile(this, "model/cifar10.tflite")
        val layersSizes = intArrayOf(1800, 24, 9600, 64, 768000, 480, 40320, 336, 3360, 40)
        val sampleSpec = SampleSpec<Float3DArray, FloatArray>(
            { it.toTypedArray() },
            { it.toTypedArray() },
            { Array(it) { FloatArray(CLASSES.size) } },
            ::negativeLogLikelihoodLoss,
            ::classifierAccuracy,
        )
        flowerClient = FlowerClient(buffer, layersSizes, sampleSpec)
    }

    suspend fun restoreInput() {
        val input = db.inputDao().get() ?: return
        runOnUiThread {
            deviceId.text.append(input.device_id)
            ip.text.append(input.ip)
            port.text.append(input.port)
        }
    }

    fun setResultText(text: String) {
        val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.GERMANY)
        val time = dateFormat.format(Date())
        resultText.append("\n$time   $text")
    }

    suspend fun runWithStacktrace(call: suspend () -> Unit) {
        try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
        }
    }

    suspend fun <T> runWithStacktraceOr(or: T, call: suspend () -> T): T {
        return try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
            or
        }
    }

    fun loadData(@Suppress("UNUSED_PARAMETER") view: View) {
        if (deviceId.text.isEmpty() || !(1..10).contains(deviceId.text.toString().toInt())) {
            Toast.makeText(
                this,
                "Please enter a client partition ID between 1 and 10 (inclusive)",
                Toast.LENGTH_LONG
            ).show()
        } else {
            hideKeyboard()
            setResultText("Loading the local training dataset in memory. It will take several seconds.")
            deviceId.isEnabled = false
            loadDataButton.isEnabled = false
            scope.launch {
                loadDataInBackground()
            }
            scope.launch {
                db.inputDao().upsertAll(
                    Input(
                        1,
                        deviceId.text.toString(),
                        ip.text.toString(),
                        port.text.toString()
                    )
                )
            }
        }
    }

    suspend fun loadDataInBackground() {
        val result = runWithStacktraceOr("Failed to load training dataset.") {
            loadData(this, flowerClient, deviceId.text.toString().toInt())
            "Training dataset is loaded in memory. Ready to train!"
        }
        runOnUiThread {
            setResultText(result)
            trainButton.isEnabled = true
        }
    }

    fun runGrpc(@Suppress("UNUSED_PARAMETER") view: View) {
        val host = ip.text.toString()
        val portStr = port.text.toString()
        if (TextUtils.isEmpty(host) || TextUtils.isEmpty(portStr) || !Patterns.IP_ADDRESS.matcher(
                host
            ).matches()
        ) {
            Toast.makeText(
                this,
                "Please enter the correct IP and port of the FL server",
                Toast.LENGTH_LONG
            ).show()
        } else {
            val port = if (TextUtils.isEmpty(portStr)) 0 else portStr.toInt()
            scope.launch {
                runWithStacktrace {
                    runGrpcInBackground(host, port)
                }
            }
            hideKeyboard()
            ip.isEnabled = false
            this.port.isEnabled = false
            trainButton.isEnabled = false
            setResultText("Started training.")
        }
    }

    suspend fun runGrpcInBackground(host: String, port: Int) {
        val address = "dns:///$host:$port"
        val result = runWithStacktraceOr("Failed to connect to the FL server \n") {
            flowerServiceRunnable = createFlowerService(address, false, flowerClient) {
                runOnUiThread {
                    setResultText(it)
                }
            }
            "Connection to the FL server successful \n"
        }
        runOnUiThread {
            setResultText(result)
            trainButton.isEnabled = false
        }
    }

    fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        var view = currentFocus
        if (view == null) {
            view = View(this)
        }
        imm.hideSoftInputFromWindow(view.windowToken, 0)
    }
}

private const val TAG = "MainActivity"

typealias Float3DArray = Array<Array<FloatArray>>
