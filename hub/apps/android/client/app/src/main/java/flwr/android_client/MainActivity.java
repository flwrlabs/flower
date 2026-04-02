package flwr.android_client;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.LifecycleOwner;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.work.Constraints;
import androidx.work.Data;
import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkInfo;
import androidx.work.WorkManager;
import android.os.PowerManager;
import android.provider.Settings;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.lifecycle.Observer;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;



public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Flower";
    private static final int REQUEST_WRITE_PERMISSION = 786;
    private Button batteryOptimisationButton;
    MessageAdapter messageAdapter;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        RecyclerView recyclerView = findViewById(R.id.recyclerView);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        messageAdapter = new MessageAdapter(readStringFromFile( getApplicationContext() , "FlowerResults.txt")); // Create your custom adapter
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(messageAdapter);

        requestPermission();

        LifecycleOwner lifecycleOwner = this ;
        WorkManager.getInstance(getApplicationContext()).getWorkInfosForUniqueWorkLiveData("my_unique_periodic_work").observe(lifecycleOwner, new Observer<List<WorkInfo>>() {
            @Override
            public void onChanged(List<WorkInfo> workInfos) {
                if (workInfos.size() > 0) {
                    WorkInfo info = workInfos.get(0);
                    int progress = info.getProgress().getInt("progress", -1);
                    // You can receive any message from the Worker Thread
                    refreshRecyclerView();
                }
            }
        });
        // code for functionality of permission buttons :
        batteryOptimisationButton = findViewById(R.id.battery_optimisation);
        batteryOptimisationButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toggleBatteryOptimization();
            }
        });
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_PERMISSION);
            createEmptyFile("FlowerResults.txt");
        }
        else
        {
            createEmptyFile("FlowerResults.txt");
        }
    }
    private List<String> readStringFromFile(Context context, String fileName) {
        List<String> lines = new ArrayList<>();

        try {
            File directory = context.getExternalFilesDir(null);

            if (directory != null) {
                File file = new File(directory, fileName);

                // Checking if the file exists
                if (!file.exists()) {
                    return lines; // File doesn't exist then return an empty list
                }
                // Opening a FileReader to read the file
                FileReader reader = new FileReader(file);
                BufferedReader bufferedReader = new BufferedReader(reader);
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    lines.add(line);
                }
                // Closing the readers
                bufferedReader.close();
                reader.close();
            }
        } catch (IOException e) {
            e.printStackTrace(); // Handle the exception as needed
        }

        return lines;
    }


    private void clearFileContents(Context context, String fileName) {
        try {
            File directory = context.getExternalFilesDir(null);

            if (directory != null) {
                File file = new File(directory, fileName);

                // Checking if the file exists
                if (!file.exists()) {
                    // File doesn't exist, so there's nothing to clear
                    return;
                }

                // Opens a FileWriter with append mode set to false (this will clear the file)
                FileWriter writer = new FileWriter(file, false);
                writer.write(""); // Write an empty string to clear the file
                writer.close();

                refreshRecyclerView();
            }
        } catch (IOException e) {
            e.printStackTrace(); // Handle the exception as needed
        }
    }



    public void startWorker(View view) {

        // ensuring all inputs are entered :

        EditText deviceIdEditText = findViewById(R.id.device_id_edit_text);
        EditText serverIPEditText = findViewById(R.id.serverIP);
        EditText serverPortEditText = findViewById(R.id.serverPort);

        // Get the text from the EditText widgets
        String dataSlice = deviceIdEditText.getText().toString();
        String serverIP = serverIPEditText.getText().toString();
        String serverPort = serverPortEditText.getText().toString();

        if (TextUtils.isEmpty(dataSlice) || TextUtils.isEmpty(serverIP) || TextUtils.isEmpty(serverPort)) {
            // Display a toast message indicating that fields are omitted
            Toast.makeText(this, "Please fill in all fields", Toast.LENGTH_SHORT).show();
        } else {

            // Launching the Worker :
            Constraints constraints = new Constraints.Builder()
                    // Add constraints if needed (e.g., network connectivity)
                    .build();

            PeriodicWorkRequest workRequest = new PeriodicWorkRequest.Builder(
                    FlowerWorker.class, 15, TimeUnit.MINUTES)
                    .setInitialDelay(0, TimeUnit.MILLISECONDS)
                    .setInputData(new Data.Builder()
                            .putString( "dataslice", deviceIdEditText.getText().toString() )
                            .putString( "server", serverIPEditText.getText().toString())
                            .putString( "port" , serverPortEditText.getText().toString())
                            .build())
                    .setConstraints(constraints)
                    .build();

            String uniqueWorkName = "my_unique_periodic_work";

            WorkManager.getInstance(getApplicationContext())
                    .enqueueUniquePeriodicWork(uniqueWorkName, ExistingPeriodicWorkPolicy.KEEP, workRequest);

            // Providing user feedback, e.g., a toast message
            Toast.makeText(this, "Worker started!", Toast.LENGTH_SHORT).show();
        }
    }

    // Listener function for the "Stop" button
    public void stopWorker(View view) {
        // Cancel the worker
        WorkManager.getInstance(getApplicationContext()).cancelAllWork();
        // Providing user feedback again, e.g., a toast message
        Toast.makeText(this, "Worker stopped!", Toast.LENGTH_SHORT).show();
    }


    // Another Listener function to refresh the updates :

    public void refresh(View view)
    {
        refreshRecyclerView();
    }

    // Another Listener to clear the contents of the File :

    public void clear(View view)
    {
        clearFileContents(getApplicationContext() , "FlowerResults.txt");
    }


    private void refreshRecyclerView() {
        // Get messages from MessageRepository using the getMessagesArray method
        List<String> messages = readStringFromFile( getApplicationContext() ,"FlowerResults.txt");

        // Update the data source of your adapter with the new messages
        messageAdapter.setData(messages);

        // Notify the adapter that the data has changed
        messageAdapter.notifyDataSetChanged();

    }



    // following code is for just the permissions :
    private void toggleBatteryOptimization() {
        if (isBatteryOptimizationEnabled()) {
            disableBatteryOptimization();
        } else {
            requestBatteryOptimization();
        }
    }

    private boolean isBatteryOptimizationEnabled() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            String packageName = getPackageName();
            PowerManager powerManager = (PowerManager) getSystemService(Context.POWER_SERVICE);
            return powerManager.isIgnoringBatteryOptimizations(packageName);
        }
        // Battery optimization is not available on versions prior to M, so return false.
        return false;
    }

    private void disableBatteryOptimization() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Intent intent = new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS);
            startActivity(intent);
        }
    }

    private void requestBatteryOptimization() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
            intent.setData(Uri.parse("package:" + getPackageName()));
            startActivity(intent);
//            startActivityForResult(intent, BATTERY_OPTIMIZATION_REQUEST_CODE);
        }
    }

    public void createEmptyFile(String fileName) {
        try {
            File file = new File(fileName);

            // Create the file if it doesn't exist
            if (!file.exists()) {
                file.createNewFile();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


