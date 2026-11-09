---
Title: "FedFitTech: A Baseline in Federated Learning for Fitness Tracking"
url: "http://arxiv.org/abs/2506.16840"
Labels:
  - TinyHAR
  - "https://publikationen.bibliothek.kit.edu/1000150216"
Dataset:
  - WEAR
  - "https://mariusbock.github.io/wear/"
---

# FedFitTech: A Baseline in Federated Learning for Fitness Tracking

<p align="justify">  This repository introduces FedFitTech, which is a Federated Learning baseline specifically designed for Fitness Technology (FitTech) using the Flower framework, enabling reproducible experiments and benchmarking. As an example usage of FedFitTech, we also included a case study that incorporates a client-side early stopping strategy.  </p>

**Paper:** [http://arxiv.org/abs/2506.16840](http://arxiv.org/abs/2506.16840)

**Authors** [Zeyneddin Oz](https://orcid.org/0000-0002-4216-9854), [Shreyas Korde](https://orcid.org/0009-0000-3771-3096), [Marius Bock](https://orcid.org/0000-0001-7401-928X), [Kristof Van Laerhoven](https://orcid.org/0000-0001-5296-5347)

**Affiliation:** [**University of Siegen**](https://www.uni-siegen.de/start/), [Ubiquitous Computing](https://ubi29.informatik.uni-siegen.de/usi/)

  
**Abstract:** <p align="justify">  Rapid evolution of sensors and resource-efficient machine learning models have spurred the widespread adoption of wearable fitness devices. Equipped with inertial sensors, such devices continuously capture physical movements for fitness technology (FitTech), enabling applications from sports optimization to preventive healthcare. Traditional centralized learning approaches to detect fitness activities struggle with data privacy concerns, regulatory constraints, and communication inefficiencies. In contrast, Federated Learning enables a decentralized model training by communicating model updates (see Fig. 1) rather than potentially private wearable sensor data. Applying Federated Learning to FitTech presents unique challenges, such as data imbalance, heterogeneous user activity patterns, and trade-offs between personalization and generalization. To research this, we present FedFitTech, a Flower framework that can be used as a baseline for Federated Learning in FitTech. As a case study that illustrates its usage, we designed a system with FedFitTech that incorporates a client-side early stopping strategy. This system empowers wearable devices to optimize the trade-off between capturing common fitness activity patterns and preserving individuals' nuances, thereby enhancing both the scalability and efficiency of fitness tracking applications. Empirical results demonstrate that this early stopping strategy has shown to reduce overall redundant communications by 13%, while maintaining the overall recognition performance at a negligible recognition cost by 1%.   </p>

<p align="center">
  <img src="/baselines/fedfittech/Results_to_compare/FedFitTech_png.png" width="70%" /> 
</p>
<p align="center">
  <em>Figure 1: In FedFitTech, wearable fitness devices locally train a global classification model for fitness activities (green model) shared by a server. Tuned local models (blue models) are shared with the server for aggregation to update a global model across all participants. This global model is then redistributed for further training rounds, leading to a private system that benefits from external patterns.</em>
</p>



## 1. About this baseline

**1.1 What's implemented:** <p align="justify">  The code in this directory provides the FedFitTech baseline, enables reproducible experiments and benchmarking of Fitness Tracking in the Federated Learning setting, as well as a case study of FedFitTech. </p>


**1.2 Dataset:** <p align="justify"> The [WEAR](https://mariusbock.github.io/wear/) (Wearable and Egocentric Activity Recognition) dataset includes 22 participants, 24 subjects, 18 activities + NULL class. The activities are grouped into three categories: 

* Jogging (5 activities: normal, rotating arms, skipping, sidesteps, butt-kicks).

* Stretching (5 activities: triceps, lunging, shoulders, hamstrings, lumbar rotation). 

* Strength (8 activities: push-ups, push-ups (complex), sit-ups, sit-ups (complex), burpees, lunges, lunges (complex), bench dips). </p>



**1.3 Hardware Setup:** <p align="justify">  The experiments were conducted on a machine equipped with 32 GB RAM, 14 CPU cores, and an Nvidia RTX 3060 GPU with 6 GB of VRAM.
</p>

> [!NOTE]
> <p align="justify"> By default, the code will run on a GPU if available; otherwise, it will run on the CPU. This example runs much faster when the `ClientApp` have access to a GPU. If your system has one, try running the example with GPU support.
  
> Although it is not compulsory to use a GPU for training the TinyHAR model, as it can be efficiently trained on a CPU with minimal computation time.</p>

**Contributors**: Shreyas Korde, Zeyneddin Oz, Marius Bock

## 2. Experimental Setup

**2.1 Task:** Human Activity Recognition


**2.2 Model:** <p align="justify"> TinyHAR, which is a lightweight deep learning model designed for Human Activity Recognition (HAR) tasks. This model architecture achieves competitive performance with fewer parameters, making it suitable for resource-constrained devices.</p>


**2.3 Data splitting:** <p align="justify">In the WEAR dataset, participants perform the same activities multiple times across different time periods to simulate realistic scenarios. This variability means that activity patterns can change over time due to factors like fatigue, weather conditions, and ground changes. To account for this, the dataset is split such that the first 20% of each labeled activity in the time series is used for testing, and the remaining 80% is used for training. This approach ensures that the testing data includes varied patterns, such as a person jogging both at the beginning and later in the exercise session.</p>
If the `server.py` script fails to download the inertial data automatically due to a server timeout or other issues, you can download it manually. The data used for this baseline is available [here](https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/). Click on the link to manually download the WEAR dataset. Ensure you create an `inertial_data` folder in the `/baselines/fedfittech/` directory. The updated repository structure should look like this: `/baselines/fedfittech/inertial_data/*.csv`.

**2.4 Client setting:** In the dataset, 2 participants were re-recorded in a different season and environment, resulting in 24 subjects. These 2 additional subjects are treated as separate devices for the same individuals to reflect real-world FitTech scenarios, thus setting the number of clients to 24.

| Dataset  | classes (including NULL) | clients | 
| :------- | :------: | :------: | 
| WEAR |    19    |  24  |  

**2.5 FedFitTech Baseline:** <p align="justify"> FedFitTech Baseline is your foundational setup for fitness technology applications using Federated Learning (FL). To get started, use the default hyperparameters provided, which ensure a stable starting point for your experiments. Feel free to explore different configurations by adjusting these hyperparameters to optimize performance and achieve better results tailored to your specific needs.  </p>
    
**2.6 FL Training Hyperparameters:** <p align="justify">By default, the main hyperparameters used are shown in the table below. For a complete list of hyperparameters, please refer to the configuration file in `/baselines/fedfittech/config/base.yaml`.</p>


| Description | Value |
| --------------- | --------- |
|  Min available clients |  24     |
|Learning rate| 0.001|
|Local Epoch| 1 |
| Batch size | 32 |
| Optimizer | Adam |
| Window size | 100 |
| Global round | 100 |
| Number of Features | 12 |
| Number of Filters | 24 |
| Preference for NULL | True |
| Fraction fit| 1.0 |
| Fraction Evaluate | 1.0 |
| Early Stopping | False |


## 3. Envirnomental Setup

**3.1 Clone Repository**
Change the working directory to FedFitTech where project.toml is located.
```
cd baselines/fedfittech
```

> \[!TIP\]
> The structure of the project has explained below.



**3.2 Create and Activate the environment**

```bash
# Create the virtual environment
pyenv virtualenv 3.10.14 FedFitTech

# Activate it
pyenv activate FedFitTech

# Install the project dependencies
pip install -e .

```


> [!NOTE]
> `fedfittech` refers to the name of the environment.

## 4. Project Structure



```shell
FedFitTech/ 
├── FedFitTech/
│   ├── flwr_utils/
│   │   ├── TinyHAR.py                      # Defines TinyHAR Model
│   │   ├── client_utils.py                 # Utility function for client 
│   │   ├── my_strategy_plotting.py         # Defines plotting functions
│   │   ├── server_plotting_function.py     # Defines server side plotting methods
│   │   └── utils_for_tinyhar.py            # Defines functions for data cleaning and processing
│   ├── client_app.py                       # Defines ClientApp
│   ├── my_strategy.py                      # Defines custom FedAvg strategy for ServerApp
│   ├── server_app.py                       # Defines ServerApp
│   └── task.py
├── config/                                 # Configuration for entire project
│   └── base.yaml
├── Results_to_compare/                     # Results for comparison with FedFitTech Baseline
│   ├── fedfittech_original_results/        # original baseline results
│   │   └── *.csv
│   ├── plots_with_name/                    # Labeled results plots
│   └── *fedfittech_original plots       
├── pyproject.toml                          # Project metadata like dependencies and configs
├── requirements.txt                        # Other Requirements for the project
├── LICENSE
└── README.md
```

## 5. Running Experiment


> \[!TIP\]
> To run your `ClientApps` on GPU or to adjust the degree or parallelism of your simulation, edit the `[tool.flwr.federations.local-simulation]` section in the `pyproject.toml`. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```




## 6. Evaluation Results

**6.1 FedFitTech Result**

<p align="justify">
This repository contains the necessary code and resources to reproduce the baseline results presented in the experimental section of the original paper: FedFitTech. Note that values such as F1 scores and server rounds may vary slightly due to differences in hardware, software environments, and random initialization. For a comprehensive overview, detailed illustrations of the graphs and results are provided below.
</p>

> \[!Note\]
> Upon running the experiment, a `/baselines/fedfittech/Flower_log` folder will be generated, containing all the plots and CSV files.

Figure 2 illustrates the mean F1 score for all clients using FedFitTech over 100 global rounds.
<p align="center">
  <img src="/baselines/fedfittech/Results_to_compare/F1_scores_baseline.svg" width="85%" />
</p>
<p align="center">
  <em>Figure 2: F1 score vs Clients for FedFitTech baseline</em>
</p>

Figure 3 illustrates the Convergence of the F1 score over 100 global rounds for the FedFitTech baseline having **mean F1 score of 0.68**.
<p align="center">
  <img src="/baselines/fedfittech/Results_to_compare/F1_scores_convergence_linegraph_baseline.svg" width="85%" />
</p>
<p align="center">
  <em>Figure 3: Convergence of the F1 score over 100 global rounds for the FedFitTech baseline.</em>
</p>

Figures 4 depict the label-based F1-scores for the FedFitTech baseline. The actual label names in these figures are as follows:</p>
<p align="center">
  <img src="/baselines/fedfittech/Results_to_compare/clients_vs_label_F1_scores_heatmaps_Normal.svg" width="85%" />
</p>
<p align="center">
  <em>Figure 4 (Client-Label Based F1-Scores of the FedFitTech)</em>
</p>
<p align="justify">
A: NULL
B: Jogging
C: Jogging (rotating arms)
D: Jogging (skipping)
E: Jogging (sidesteps)
F: Jogging (butt-kicks)
G: Stretching (triceps)
H: Stretching (lunging)
I: Stretching (shoulders)
J: Stretching (hamstrings)
K: Stretching (lumbar rotation)
L: Push-ups
M: Push-ups (complex)
N: Sit-ups
O: Sit-ups (complex)
P: Burpees
Q: Lunges
R: Lunges (complex)
S: Bench-dips
 </p>

---

## 7. FedFitTech with Case Study: Client-side Early Stopping

**7.1 Early stopping Strategy:** <p align="justify"> Our case study focuses on the FitTech domain and employs early stopping based on the validation F1-score over a sliding window. Although simple to implement, this method ensures that the global model retains valuable patterns from all clients, including those with limited generalization capability.</p>


**7.2 Implementation of FedFitTech case study** 

<p align="justify"> The FedFitTech case study presents an improved version of FedFitTech, where clients stop learning once they have converged, enhancing efficiency.
</p>

> \[!Tip\]
> To perform an experiment with the case study, you just need to change the `Early Stopping` flag in the configuration file located at `/baselines/fedfittech/config/base.yaml`. The updated hyperparameters will be as shown below. 


| Description | Value |
| --------------- | --------- |
|  Min available clients |  24     |
|Learning rate| 0.001|
|Local Epoch| 1 |
| Batch size | 32 |
| Optimizer | Adam |
| Window size | 100 |
| Global round | 100 |
| Number of Features | 12 |
| Number of Filters | 24 |
| Preference for NULL | True |
| Fraction fit| 1.0 |
| Fraction Evaluate | 1.0 |
| Early Stopping | **True** |

> \[!Note\]
> For logs related to early stopping, an `/baselines/fedfittech/Early_stoppigit ng_logs` folder will be created. This folder will store client-specific early stopping metadata

**7.3 Early stopping metadata**

<p align="justify">
  
Initially, the server round value will be set to `np.nan`. Once early stopping is triggered for a specific client, the corresponding `server round` value will be recorded in the early stopping log. </p>



```
                        Early Stopping in Fit Log.                        

+++++++++ Early stopping triggered for Client 5 at Fit. No further training required. +++++++++
Config records for Client Id 5: Best Validation F1 score 0.30341428672751103, Counter value 0, for server round 25 Has_converged = True
+++++++++ Early stopping triggered for Client 15 at Fit. No further training required. +++++++++
Config records for Client Id 15: Best Validation F1 score 0.5544575872359317, Counter value 0, for server round 27 Has_converged = True
+++++++++ Early stopping triggered for Client 10 at Fit. No further training required. +++++++++
Config records for Client Id 10: Best Validation F1 score 0.4166345252593648, Counter value 0, for server round 28 Has_converged = True
+++++++++ Early stopping triggered for Client 7 at Fit. No further training required. +++++++++
Config records for Client Id 7: Best Validation F1 score 0.6434438207487768, Counter value 0, for server round 34 Has_converged = True
+++++++++ Early stopping triggered for Client 14 at Fit. No further training required. +++++++++
Config records for Client Id 14: Best Validation F1 score 0.5862694737076439, Counter value 0, for server round 68 Has_converged = True
+++++++++ Early stopping triggered for Client 17 at Fit. No further training required. +++++++++
Config records for Client Id 17: Best Validation F1 score 0.7534962813629125, Counter value 0, for server round 74 Has_converged = True
+++++++++ Early stopping triggered for Client 4 at Fit. No further training required. +++++++++
Config records for Client Id 4: Best Validation F1 score 0.48267210946367445, Counter value 0, for server round 80 Has_converged = True
+++++++++ Early stopping triggered for Client 18 at Fit. No further training required. +++++++++
Config records for Client Id 18: Best Validation F1 score 0.750440226645707, Counter value 0, for server round 93 Has_converged = True

```

**7.4 Evaluation Results for FedFitTech Case study**

<p align="justify"> The evaluation of this work includes a compelling comparison between FedFitTech and its client-side early stopping case study, focusing on both communication loads and the global model's performance on local data. </p>

  **7.4.1 Communication Cost**

<p align="justify"> Figure 5 illustrates that Notably, 9 out of 24 clients stopped training the global model early, with the earliest stop at round 40 and some clients stopping after more than 80 rounds. This highlights that prolonged participation does not guarantee continued benefit. </p>



<p align="center">
  <img src="/baselines/fedfittech/Results_to_compare/Comp_saved_Global_rounds_vs_clients_single_bar_plot.svg" width="85%" />
</p>
<p align="center">
  <em>Figure 5: The figure shows the number of training rounds each client attended in the case study. The Y-axis represents the F1-score, and the X-axis represents client IDs. Red indicates communication cost, green shows saved communication rounds, and the dashed black line is the mean. </em>
</p>


<p align="justify">
Figure 6 illustrates that some clients did not improve their local F1-score even after 30 rounds of training. Despite some clients being dropped, the overall global model F1-score continued to increase, as shown by the black dashed line.
</p>

<p align="center">
  <img src="/baselines/fedfittech/Results_to_compare/F1_scores_convergence_with_early_stopping_linegraph.svg" alt="Figure 2" width="85%" />
</p>
<p align="center">
  <em> Figure 6: Communication rounds (X-axis) vs. F1-scores (Y-axis). The dashed black line shows the mean values of all local performances, and triangles depict early-stopped clients' rounds. </em>
</p>


  **7.4.2 Model Performance**

<p align="justify"> Figure 7 shows the difference of F1-score (y-axis) over clients(x-axis) </p>

<p align="center">
  <img src="/baselines/fedfittech/Results_to_compare/F1_scores_comparison_double_bar_plot.svg" width="85%" />
</p>
<p align="center">
  <em>Figure 7: F1-Score Comparison: FedFitTech vs. Case Study </em>
</p>

<p align="justify"> Figure 7 depicts the client-based F1-score changes. FedFitTech achieved a mean F1-score of 68% overall for clients, while the early stopping case study's mean F1-score is 67%. Notably, some clients even have better F1-scores in the case study, specifically clients with IDs: 2, 7, 9, 10, 11, 12, 14, 17, 19, 21, and 22. As a result, 11 out of 24 clients obtained better performance compared to FedFitTech. </p>

<p align="justify">
Figures 4 and 8 depict the label-based F1-scores for the FedFitTech baseline and the case study, respectively. The actual label names in these figures are as follows:</p>

<p align="center">
  <img src="/baselines/fedfittech/Results_to_compare/clients_vs_label_F1_scores_heatmaps_Normal.svg" width="49%" />
  <img src="/baselines/fedfittech/Results_to_compare/clients_vs_label_F1_scores_heatmaps_Early_Stopping.svg" width="49%" />
</p>
<p align="center">
  <em>Figure 4 (Client-Label Based F1-Scores of the FedFitTech), Figure 8 (Client-Label Based F1-Scores of the Case Study)</em>
</p>

<p align="justify">
A: NULL
B: Jogging
C: Jogging (rotating arms)
D: Jogging (skipping)
E: Jogging (sidesteps)
F: Jogging (butt-kicks)
G: Stretching (triceps)
H: Stretching (lunging)
I: Stretching (shoulders)
J: Stretching (hamstrings)
K: Stretching (lumbar rotation)
L: Push-ups
M: Push-ups (complex)
N: Sit-ups
O: Sit-ups (complex)
P: Burpees
Q: Lunges
R: Lunges (complex)
S: Bench-dips
 </p>



> [!NOTE]
> If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.




### For any questions, feel free to contact us via email.

**Contact Information**

Shreyas Korde: shreyas.korde@student.uni-siegen.de

Zeyneddin Oz: zeyneddin.oez@uni-siegen.de
