"""FedFitTech: A Flower / Client PyTorch app."""

import json
import os
from datetime import datetime

import numpy as np
import torch

from fedfittech.flwr_utils.client_utils import get_net_and_config, load_data_for_client
from fedfittech.flwr_utils.utils_for_tinyhar import (
    evaluation_functions,
    training_functions,
)
from fedfittech.task import get_weights, set_weights
from flwr.client import ClientApp, NumPyClient
from flwr.common import ConfigRecord, Context


# Defin Flower Client
class FlowerClient(NumPyClient):
    """Standard Flower client."""

    # pylint: disable=too-many-arguments
    def __init__(self, context: Context, net, trainloader, valloader, config):
        self.cfg = config
        self.net = net
        self.trainloader = trainloader[0]
        self.valloader = valloader[0]
        self.patience = 5
        self.threshold = 0.01
        self.f1_target = 0.69
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cuda is available for client = {torch.cuda.is_available()}")

        self.client_state = context.state

        if "early_stop_metrics" not in self.client_state:
            self.client_state["early_stop_metrics"] = ConfigRecord()

            record = self.client_state.get("early_stop_metrics")
            if not isinstance(record, ConfigRecord):
                record = ConfigRecord()
                self.client_state["early_stop_metrics"] = record

            context_early_stop: ConfigRecord = record

            context_early_stop["context_best_val_f1_score"] = 0.0
            context_early_stop["counter"] = 0
            context_early_stop["has_converged"] = False
            context_early_stop["print_status"] = False
            file_date1 = datetime.now().strftime("%d-%m-%Y_%H-%M")
            context_early_stop["log_file_name"] = f"Early_stopping_{file_date1}.txt"
            context_early_stop["f1_scores_list"] = []
            context_early_stop["Training_stop_round"] = np.nan
        # print(f"Configs records{self.client_state['early_stop_metrics']}")

    def fit(self, parameters, config):
        """Implement fit function for a given client."""
        server_round = config.get("server_round", 0)
        set_weights(self.net, parameters)
        # Dictconfig to store early stopping metrics
        context_early_stop = self.client_state["early_stop_metrics"]

        config_reco_msg = (
            f"Config records for Client Id {self.cfg.sub_id}: "
            f"Best Val F1 score {context_early_stop['context_best_val_f1_score']}, "
            f"Counter value {context_early_stop['counter']}, "
            f"for server round {context_early_stop['Training_stop_round']} "
            f"Has_converged = {context_early_stop['has_converged']}"
        )
        print(config_reco_msg)

        if context_early_stop["has_converged"]:
            fit_log_msg = (
                "++++++++ Early stopping triggered for Client "
                f"{self.cfg.sub_id} at Fit. No further training required. ++++++++\n"
            )
            print(fit_log_msg)

            if not context_early_stop["print_status"]:
                context_early_stop["Training_stop_round"] = server_round
                msg_wth_stp_rnd = (
                    f"Config records for Client Id {self.cfg.sub_id}: "
                    f"Best Val F1 {context_early_stop['context_best_val_f1_score']}, "
                    f"Counter value {context_early_stop['counter']}, "
                    f"for server round {context_early_stop['Training_stop_round']} "
                    f"Has_converged = {context_early_stop['has_converged']}"
                )

                # Define log file path inside the Flower logs directory
                log_dir = "./Early_stopping_logs/"
                os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
                file_name = context_early_stop["log_file_name"]
                fit_log_file = os.path.join(log_dir, file_name)

                # Ensure file exists before writing
                if not os.path.exists(fit_log_file):
                    with open(
                        fit_log_file, "w"
                    ) as f:  # "w" mode creates the file if missing
                        f.write("             Early Stopping in Fit Log.         \n\n")
                # Append the new log entry
                with open(fit_log_file, "a") as f:
                    f.write(fit_log_msg)
                    f.write(msg_wth_stp_rnd + "\n")
                context_early_stop["print_status"] = True

            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {},
            )

        elif not context_early_stop["has_converged"]:
            loss, accuracy = training_functions.train_model_federated(
                self.net,
                self.trainloader,
                num_epochs=self.cfg.LOCAL_EPOCH,
                opt=self.cfg.OPTIMIZER,
                learning_rate=self.cfg.LEARNING_RATE,
                device=self.DEVICE,
            )

            # Prepare outputs
            outputs = {
                "Local_train_loss": "{:.4f}".format(loss),
                "Local Accuracy": "{:.4f}".format(accuracy),
                "client_id": self.cfg.sub_id,
            }
            self.train_examples = len(self.trainloader.dataset)

            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                outputs,
            )

    def evaluate(self, parameters, config):
        """Implement evaluate function for a given client."""
        server_round = config.get("server_round", 0)
        set_weights(self.net, parameters)
        # Dictconfig to store early stopping metrics
        context_early_stop = self.client_state["early_stop_metrics"]

        loss, accuracy, precision, recall, fscore = evaluation_functions.evaluate_model(
            model=self.net, testloader=self.valloader, device=self.DEVICE, cfg=self.cfg
        )
        metrics = {
            "Validation loss": "{:.4f}".format(loss),
            "Validation Accuracy": "{:.2f}".format(accuracy),
            "Validation Precision": "{:.4f}".format(precision),
            "Validation Recall": "{:.4f}".format(recall),
            "Validation F1 score": "{:.2f}".format(fscore),
            "Client_id": self.cfg.sub_id,
            "Number of Training Examples": self.cfg.num_train_examples,
            "Training_stop_round": context_early_stop["Training_stop_round"],
        }

        # Colllect F1 scores for each label after final round
        if server_round == self.cfg.GLOBAL_ROUND:
            real_labels_all = []
            pred_labels_all = []
            all_labels, all_predictions = (
                evaluation_functions.get_ground_truth_and_predictions(
                    model=self.net, testloader=self.valloader, DEVICE=self.DEVICE
                )
            )
            real_labels_all.append(all_labels)
            pred_labels_all.append(all_predictions)
            labelwise_result_dict = evaluation_functions.get_label_based_results(
                all_labels, all_predictions, self.cfg.reversed_labels_set
            )
            complex_labelwise_result_dict = json.dumps(
                labelwise_result_dict
            )  # Converts dict to str
            metrics = {
                "Validation loss": "{:.4f}".format(loss),
                "Validation Accuracy": "{:.2f}".format(accuracy),
                "Validation Precision": "{:.4f}".format(precision),
                "Validation Recall": "{:.4f}".format(recall),
                "Validation F1 score": "{:.2f}".format(fscore),
                "Client_id": self.cfg.sub_id,
                "Number of Training Examples": self.cfg.num_train_examples,
                "Training_stop_round": context_early_stop["Training_stop_round"],
                "F1_labels_result": complex_labelwise_result_dict,
            }

        # Early stopping logic if Early stopping flag is = 'true' , '1', 'yes'
        f1_scores = context_early_stop["f1_scores_list"]

        if self.cfg.Early_stopping in ["true", "yes", "1"]:
            f1_scores.append(fscore)
            if len(f1_scores) > self.patience:
                f1_scores_window = f1_scores[-self.patience :]
                context_early_stop["context_best_val_f1_score"] = fscore
                if (
                    round(max(f1_scores_window) - min(f1_scores_window), 2)
                    < self.threshold
                ):
                    context_early_stop["has_converged"] = True
                    print(
                        f"+++++++++++++++++++++++++++++ Early stopping triggerd for "
                        f"client {self.cfg.sub_id} +++++++++++++++++++++++++++++++"
                    )

        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    """Flower Client Function represetinng a single organization."""
    partition_id = context.node_config["partition-id"]
    net, added_cfg = get_net_and_config()
    trainloader, valloader, cfg = load_data_for_client(added_cfg, user_num=partition_id)
    config = cfg

    # Return Client instance
    return FlowerClient(context, net, trainloader, valloader, config).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
