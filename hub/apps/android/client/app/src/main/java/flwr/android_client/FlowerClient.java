package flwr.android_client;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.os.ConditionVariable;
import android.util.Log;
import android.util.Pair;

import androidx.lifecycle.MutableLiveData;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;

public class FlowerClient {

    private TransferLearningModelWrapper tlModel;
    private static final int LOWER_BYTE_MASK = 0xFF;
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private Context context;
    private final ConditionVariable isTraining = new ConditionVariable();
    private static String TAG = "Flower";
    private int local_epochs = 1;

    public FlowerClient(Context context) {
        this.tlModel = new TransferLearningModelWrapper(context);
        this.context = context;
    }

    public ByteBuffer[] getWeights() {
        return tlModel.getParameters();
    }

    public Pair<ByteBuffer[], Integer> fit(ByteBuffer[] weights, int epochs) {

        this.local_epochs = epochs;
        tlModel.updateParameters(weights);
        isTraining.close();
        tlModel.train(this.local_epochs);
        tlModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
        isTraining.block();
        return Pair.create(getWeights(), tlModel.getSize_Training());
    }

    public Pair<Pair<Float, Float>, Integer> evaluate(ByteBuffer[] weights) {
        tlModel.updateParameters(weights);
        tlModel.disableTraining();
        return Pair.create(tlModel.calculateTestStatistics(), tlModel.getSize_Testing());
    }

    public void setLastLoss(int epoch, float newLoss) {
        if (epoch == this.local_epochs - 1) {
            Log.e(TAG, "Training finished after epoch = " + epoch);
            lastLoss.postValue(newLoss);
            tlModel.disableTraining();
            isTraining.open();
        }
    }

    public void loadData(int device_id) {
        try {
            Log.d("FLOWERCLIENT_LOAD", "loadData: ");
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/partition_" + (device_id - 1) + "_train.txt")));
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i + "th training image loaded");
                addSample("data/" + line, true);
            }
            reader.close();

            i = 0;
            reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/partition_" +  (device_id - 1)  + "_test.txt")));
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i + "th test image loaded");
                addSample("data/" + line, false);
            }
            reader.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void addSample(String photoPath, Boolean isTraining) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap =  BitmapFactory.decodeStream(this.context.getAssets().open(photoPath), null, options);
        String sampleClass = get_class(photoPath);

        // get rgb equivalent and class
        float[] rgbImage = prepareImage(bitmap);

        // add to the list.
        try {
            this.tlModel.addSample(rgbImage, sampleClass, isTraining).get();
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
        } catch (InterruptedException e) {
            // no-op
        }
    }

    public String get_class(String path) {
        String label = path.split("/")[2];
        return label;
    }

    /**
     * Normalizes a camera image to [0; 1], cropping it
     * to size expected by the model and adjusting for camera rotation.
     */
    private static float[] prepareImage(Bitmap bitmap)  {
        int modelImageSize = TransferLearningModelWrapper.IMAGE_SIZE;

        float[] normalizedRgb = new float[modelImageSize * modelImageSize * 3];
        int nextIdx = 0;
        for (int y = 0; y < modelImageSize; y++) {
            for (int x = 0; x < modelImageSize; x++) {
                int rgb = bitmap.getPixel(x, y);

                float r = ((rgb >> 16) & LOWER_BYTE_MASK) * (1 / 255.0f);
                float g = ((rgb >> 8) & LOWER_BYTE_MASK) * (1 / 255.0f);
                float b = (rgb & LOWER_BYTE_MASK) * (1 / 255.0f);

                normalizedRgb[nextIdx++] = r;
                normalizedRgb[nextIdx++] = g;
                normalizedRgb[nextIdx++] = b;
            }
        }

        return normalizedRgb;
    }

    // function to write to a file :

    public void writeStringToFile( Context context , String fileName, String content) {
        try {
            // Get the app-specific external storage directory
            File directory = context.getExternalFilesDir(null);

            if (directory != null) {
                File file = new File(directory, fileName);

                // Check if the file exists
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


}
