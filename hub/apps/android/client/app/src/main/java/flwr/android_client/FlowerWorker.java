package flwr.android_client;



import static android.content.Context.NOTIFICATION_SERVICE;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.icu.text.SimpleDateFormat;
import android.os.Build;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import android.util.Log;
import android.util.Pair;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import  flwr.android_client.FlowerServiceGrpc.FlowerServiceStub;
import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.HashMap;
import java.util.Map;
import androidx.core.app.NotificationCompat;
import androidx.work.Data;
import androidx.work.ForegroundInfo;
import androidx.work.WorkManager;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

public class FlowerWorker extends Worker {

    private ManagedChannel channel;
    public FlowerClient fc;
    private StreamObserver<ClientMessage> UniversalRequestObserver;
    private static final String TAG = "Flower";
    String serverIp = "00:00:00";
    String serverPort = "0000";
    String dataslice = "1";
    public static String start_time;
    public static String end_time;
    // following variables are just to send the worker routine to the
    public static String workerStartTime = "";

    public static String workerEndTime = "";

    public static String workerEndReason = "worker ended";

    public String getTime() {
        // Extract hours, minutes, and seconds
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
            String formattedDateTime = sdf.format(new Date());
            return formattedDateTime;
        }
        return "";
    }
    private NotificationManager notificationManager;

    private static String PROGRESS = "PROGRESS";

    public FlowerWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
        FlowerWorker worker = this;
        notificationManager = (NotificationManager)
                context.getSystemService(NOTIFICATION_SERVICE);
        fc = new FlowerClient(context.getApplicationContext());
    }

    @NonNull
    @Override
    public Result doWork() {

        Data checkData = getInputData();
        serverIp = checkData.getString("server");
        serverPort = checkData.getString("port");
        dataslice = checkData.getString("dataslice");

        // Creating Foreground Notification Service about the Background Worker FL tasks
        setForegroundAsync(createForegroundInfo("Progress"));
        try {
            workerStartTime = getTime();
            // Ensuring whether the connection is establish or not with the given gRPC IP & port
            boolean resultConnect = connect();
            if(resultConnect)
            {
                loadData();
                CompletableFuture<Void> grpcFuture = runGrpc();
                grpcFuture.get();
                return Result.success();
            }
            else
            {
                workerEndReason = "GRPC Connection failed";
                return Result.failure();
            }

        } catch (Exception e) {
            // To handle any exceptions and return a failure result
            // Failure if there is any OOM or midway connection error
            workerEndReason = "Unknown Error occured in main try catch";
            Log.e(TAG, "Error executing flower code: " + e.getMessage(), e);
            return Result.failure();
        }
    }

    @Override
    public void onStopped() {
        super.onStopped();
        // Worker is canceled, stopping the global requestObserver if it's not null
        Throwable cancellationCause = new Throwable("Worker canceled");
        if (UniversalRequestObserver != null) {
            UniversalRequestObserver.onError(cancellationCause); // Signal to the server that communication is done
        }
    }

    public boolean connect() {
        int port = Integer.parseInt(serverPort);
        try {
            channel = ManagedChannelBuilder.forAddress(serverIp, port)
                    .maxInboundMessageSize(10 * 1024 * 1024)
                    .usePlaintext()
                    .build();
            fc.writeStringToFile(getApplicationContext(), "FlowerResults.txt" , "Connection : Successful with " + serverIp + " : " + serverPort + " : " + dataslice);
            return true; // connection is successful
        } catch (Exception e) {
            Log.e(TAG, "Failed to connect to the server: " + e.getMessage(), e);
            fc.writeStringToFile(getApplicationContext(), "FlowerResults.txt" , "Connection : Failed with " + serverIp + " : " + serverPort + " : " + dataslice);
            return false; // connection failed
        }
    }

    public void loadData() {
        try {
            fc.loadData(Integer.parseInt(dataslice));
            Log.d("LOAD", "Loading is complete");
            fc.writeStringToFile(getApplicationContext(), "FlowerResults.txt", "Loading Bit Images : Success" );
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            Log.d("LOAD_ERROR", "Error occured in Loading");
            fc.writeStringToFile(getApplicationContext(), "FlowerResults.txt", "Loading Bit Images : Failed" );
        }
    }

    public CompletableFuture<Void> runGrpc() {

        CompletableFuture<Void> future = new CompletableFuture<>();
        FlowerWorker worker = this;
        ExecutorService executor = Executors.newSingleThreadExecutor();

        ProgressUpdater progressUpdater = new ProgressUpdater();

        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    CountDownLatch latch = new CountDownLatch(1);

                    (new FlowerServiceRunnable()).run(FlowerServiceGrpc.newStub(channel), worker, latch , progressUpdater , getApplicationContext());

                    latch.await(); // Wait for the latch to count down
                    future.complete(null); // Complete the future when the latch is counted down

                    Log.d("GRPC", "inside GRPC");
                } catch (Exception e) {
                    StringWriter sw = new StringWriter();
                    PrintWriter pw = new PrintWriter(sw);
                    e.printStackTrace(pw);
                    pw.flush();
                    Log.e("GRPC", "Failed to connect to the FL server \n" + sw);
                    future.completeExceptionally(e); // Complete the future with an exception
                }
            }
        });

        return future;
    }


    @NonNull
    private ForegroundInfo createForegroundInfo(@NonNull String progress) {
        // Building a notification using bytesRead and contentLength
        Context context = getApplicationContext();
        String id = context.getString(R.string.notification_channel_id);
        String title = context.getString(R.string.notification_title);
        String cancel =context.getString(R.string.cancel_download);
        // Creating a PendingIntent that can be used to cancel the worker
        PendingIntent intent = WorkManager.getInstance(context)
                .createCancelPendingIntent(getId());
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            createChannel();
        }
        Notification notification = new NotificationCompat.Builder(context, id)
                .setContentTitle(title)
                .setTicker(title)
                .setSmallIcon(R.drawable.ic_logo)
                .setOngoing(true)
                // Add the cancel action to the notification which can
                // be used to cancel the worker
                .addAction(android.R.drawable.ic_delete, cancel, intent)
                .build();
        int notificationId = 1002;
        return new ForegroundInfo(notificationId, notification);
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private void createChannel() {
        Context context = getApplicationContext();
        String channelId = context.getString(R.string.notification_channel_id);
        String channelName = context.getString(R.string.notification_title);
        int importance = NotificationManager.IMPORTANCE_DEFAULT;

        NotificationChannel channel = new NotificationChannel(channelId, channelName, importance);
        // Configure the channel
        channel.setDescription("Channel description");
        // Set other properties of the channel as needed if needed ...
        NotificationManager notificationManager = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        notificationManager.createNotificationChannel(channel);
    }


    public class ProgressUpdater {
        public void setProgress() {
            // Aim of this class is to allow static FlowerServiceRunnable Object to notifiy Main Activity about the changes in real time to be displayed to User
            Log.d("DATA-BACKGROUND","Sending it to the main activity");
            setProgressAsync(new Data.Builder().putInt("progress", 0).build());

        }
    }

    private static class FlowerServiceRunnable{
        protected Throwable failed;
        public void run(FlowerServiceStub asyncStub, FlowerWorker worker ,  CountDownLatch latch , ProgressUpdater progressUpdater , Context context) {
            join(asyncStub, worker , latch , progressUpdater , context);
        }

        public void writeStringToFile( Context context , String fileName, String content) {
            try {
                // Getting the app-specific external storage directory
                File directory = context.getExternalFilesDir(null);

                if (directory != null) {
                    File file = new File(directory, fileName);

                    // Checking if the file exists
                    boolean fileExists = file.exists();

                    // Open a FileWriter in append mode
                    FileWriter writer = new FileWriter(file, true);

                    // If the file exists and is not empty, add a new line
                    if (fileExists && file.length() > 0) {
                        writer.append("\n");
                    }

                    // Write the string to the file
                    writer.append(content);

                    // Close the FileWriter
                    writer.close();
                }
            } catch (IOException e) {
                e.printStackTrace(); // Handle the exception as needed
            }
        }

        private void join(FlowerServiceStub asyncStub, FlowerWorker worker, CountDownLatch latch , ProgressUpdater progressUpdater , Context context)
                throws RuntimeException {
            final CountDownLatch finishLatch = new CountDownLatch(1);

            worker.UniversalRequestObserver = asyncStub.join(new StreamObserver<ServerMessage>() {
                @Override
                public void onNext(ServerMessage msg) {
                    handleMessage(msg, worker , progressUpdater , context);
                }

                @Override
                public void onError(Throwable t) {
                    t.printStackTrace();
                    failed = t;
                    finishLatch.countDown();
                    latch.countDown();
                    // Error handling for timeout & other GRPC communication related Errors
                    workerEndReason = t.getMessage();
                    writeStringToFile( context ,"FlowerResults.txt", workerEndReason);
                    Log.e(TAG, t.getMessage());
                }

                @Override
                public void onCompleted() {
                    finishLatch.countDown();
                    latch.countDown();
                    Log.e(TAG, "Done");
                }
            });


            try {
                finishLatch.await();
            } catch (InterruptedException e) {
                Log.e(TAG, "Interrupted while waiting for gRPC communication to finish: " + e.getMessage(), e);
                Thread.currentThread().interrupt();
            }
        }

        private void handleMessage(ServerMessage message, FlowerWorker worker , ProgressUpdater progressUpdater , Context context) {

            try {
                ByteBuffer[] weights;
                ClientMessage c = null;

                if (message.hasGetParametersIns()) {
                    Log.e(TAG, "Handling GetParameters");

                    weights = worker.fc.getWeights();
                    c = weightsAsProto(weights);
                } else if (message.hasFitIns()) {

                    SimpleDateFormat sdf = null;
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
                    }

                    // Get the current date and time
                    Date currentDate = new Date();

                    // Format the date and time using the SimpleDateFormat object
                    // String formattedDate = null;
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        start_time = sdf.format(currentDate);
                    }
                    Log.e(TAG, "Handling FitIns");

                    List<ByteString> layers = message.getFitIns().getParameters().getTensorsList();

                    Scalar epoch_config = null;
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        epoch_config = message.getFitIns().getConfigMap().getOrDefault("local_epochs", Scalar.newBuilder().setSint64(1).build());
                    }

                    assert epoch_config != null;
                    int local_epochs = (int) epoch_config.getSint64();

                    // Our model has 10 layers
                    ByteBuffer[] newWeights = new ByteBuffer[10] ;
                    for (int i = 0; i < 10; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }

                    Pair<ByteBuffer[], Integer> outputs = worker.fc.fit(newWeights, local_epochs);
                    currentDate = new Date();
                    // Format the date and time using the SimpleDateFormat object
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        end_time = sdf.format(currentDate);
                    }
                    Log.d("FIT-RESPONSE", "ABOUT TO SEND FIT RESPONSE");
                    c = fitResAsProto(outputs.first, outputs.second);
                } else if (message.hasEvaluateIns()) {
                    Log.e(TAG, "Handling EvaluateIns");

                    SimpleDateFormat sdf = null;
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
                    }
                    Date currentDate = new Date();
                    // Format the date and time using the SimpleDateFormat object
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        start_time = sdf.format(currentDate);
                    }
                    List<ByteString> layers = message.getEvaluateIns().getParameters().getTensorsList();
                    // Our model has 10 layers
                    ByteBuffer[] newWeights = new ByteBuffer[10] ;
                    for (int i = 0; i < 10; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }
                    Pair<Pair<Float, Float>, Integer> inference = worker.fc.evaluate(newWeights);
                    float loss = inference.first.first;
                    float accuracy = inference.first.second;
                    int test_size = inference.second;
                    currentDate = new Date();
                    // Format the date and time using the SimpleDateFormat object
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        end_time = sdf.format(currentDate);
                    }
                    Log.d("EVALUATE-RESPONSE", "ABOUT TO SEND EVALUATE RESPONSE");
                    String newMessage = "Time : " + end_time + " , " + " Round Accuracy : " + String.valueOf(accuracy);
                    writeStringToFile( context ,"FlowerResults.txt", newMessage);
                    progressUpdater.setProgress();
                    c = evaluateResAsProto(loss , accuracy , test_size);
                }
                worker.UniversalRequestObserver.onNext(c);
            }
            catch (Exception e){
                Log.e("Exception","Exception occured in GRPC Connection");
                Log.e(TAG, e.getMessage());
            }
        }
    }

    private static ClientMessage weightsAsProto(ByteBuffer[] weights){
        List<ByteString> layers = new ArrayList<>();
        for (ByteBuffer weight : weights) {
            layers.add(ByteString.copyFrom(weight));
        }
        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.GetParametersRes res = ClientMessage.GetParametersRes.newBuilder().setParameters(p).build();
        return ClientMessage.newBuilder().setGetParametersRes(res).build();
    }

    private static ClientMessage fitResAsProto(ByteBuffer[] weights, int training_size){
        List<ByteString> layers = new ArrayList<>();
        for (ByteBuffer weight : weights) {
            layers.add(ByteString.copyFrom(weight));
        }

        Log.d("ENDTIME", end_time);
        Log.d("STARTTIME", start_time);

        // An example portraying how to upload data to the server via FLower Server side GRPC
        Map<String, Scalar> metrics = new HashMap<>();

        metrics.put("start_time", Scalar.newBuilder().setString(start_time).build());
        metrics.put("end_time", Scalar.newBuilder().setString(end_time).build());
        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.FitRes res = ClientMessage.FitRes.newBuilder().setParameters(p).setNumExamples(training_size).putAllMetrics(metrics).build();
        return ClientMessage.newBuilder().setFitRes(res).build();
    }



    private static ClientMessage evaluateResAsProto(float loss, float accuracy ,int testing_size){

        // attempting to send accuracy to the server :
        Map<String, Scalar> metrics = new HashMap<>();


        Log.d("ENDTIME", end_time);
        Log.d("STARTTIME", start_time);

        Log.d("Accuracy", String.valueOf(accuracy));
        Log.d("Loss", String.valueOf(loss));


        // An example portraying how to upload data to the server via FLower Server side GRPC
        metrics.put("Accuracy", Scalar.newBuilder().setString(String.valueOf(accuracy)).build());
        metrics.put("Loss" , Scalar.newBuilder().setString(String.valueOf(loss)).build());
        metrics.put("start_time", Scalar.newBuilder().setString(start_time).build());
        metrics.put("end_time", Scalar.newBuilder().setString(end_time).build());


        ClientMessage.EvaluateRes res = ClientMessage.EvaluateRes.newBuilder().setLoss(loss).setNumExamples(testing_size).putAllMetrics(metrics).build();
        return ClientMessage.newBuilder().setEvaluateRes(res).build();
    }


}







