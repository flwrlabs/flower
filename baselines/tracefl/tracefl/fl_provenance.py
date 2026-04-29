"""Flower provenance analysis for TraceFL."""

import logging
import time
import traceback

import torch

from .model import test_neural_network
from .neuron_provenance import NeuronProvenance
from .utils import get_prov_eval_metrics, normalize_contributions


# pylint: disable=too-many-instance-attributes
class FlowerProvenance:
    """Flower provenance analysis for TraceFL.

    This class handles provenance analysis for federated learning rounds, including
    faulty client detection and contribution analysis.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        cfg,
        round_id,
        server_test_data,
        global_model,
        client2model,
        client2num_examples,
        client2class,
    ):
        """Initialize a FlowerProvenance instance.

        Parameters
        ----------
        cfg : object
            Configuration object.
        round_id : int
            Current training round ID.
        server_test_data : object
            Test data from the central server.
        global_model : torch.nn.Module
            Global model.
        client2model : dict
            Dictionary mapping client IDs to their models.
        client2num_examples : dict
            Dictionary mapping client IDs to number of examples.
        """
        self.cfg = cfg
        self.round_id = round_id
        self.server_test_data = server_test_data
        self.global_model = global_model
        self.client2model = client2model
        self.client2num_examples = client2num_examples
        self.client2class = self._normalize_client2class(client2class)
        logging.debug("Original client2class: %s", client2class)
        logging.debug("Normalized client2class: %s", self.client2class)
        self.device = cfg.device

        self.faulty_clients_ids = [
            int(cid)
            for cid in getattr(cfg, "faulty_clients_ids", [])
            if isinstance(cid, int | float | str) and str(cid) != ""
        ]
        self._faulty_client_strings = {str(cid) for cid in self.faulty_clients_ids}
        self.label2flip = {
            int(k): int(v)
            for k, v in getattr(cfg, "label2flip", {}).items()
            if str(k) != "" and str(v) != ""
        }
        self._faulty_mode = bool(self.faulty_clients_ids and self.label2flip)

        # Initialize all_rounds_clients2class from provided client-to-class mappings
        self.all_rounds_clients2class = {
            client_id: self.client2class.get(client_id, {})
            for client_id in self.client2model.keys()
        }
        self.participating_clients_labels = (
            []
        )  # Will be set in _setParticipatingClientsLabels
        self._set_participating_clients_labels()

        # Initialize attributes that will be set later
        self.subset_test_data = None
        self.loss = None
        self.acc = None

        # Select provenance data
        self._select_provenance_data()

        logging.info(
            "\n\n             ----------Round key  %s -------------- \n",
            round_id,
        )

    def _normalize_client2class(self, client2class):
        """Ensure client-to-class mappings use integer client IDs and string labels."""
        normalized = {}
        for cid, label_counts in (client2class or {}).items():
            try:
                client_id = int(cid)
            except (TypeError, ValueError):
                client_id = cid

            normalized_counts = {}
            for label, count in (label_counts or {}).items():
                if isinstance(count, torch.Tensor):
                    count = count.item()
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                label_int = int(label.item() if hasattr(label, "item") else label)
                normalized_counts[label_int] = count_int

            normalized[client_id] = normalized_counts

        return normalized

    def _set_participating_clients_labels(self):
        """Set participating client labels for provenance analysis.

        In baseline without tracking, use all possible labels.
        """
        labels = set()
        for class_counts in self.all_rounds_clients2class.values():
            for label in class_counts.keys():
                try:
                    labels.add(int(label))
                except (TypeError, ValueError):
                    continue

        if not labels:
            labels = set(range(self.cfg.data_dist.num_classes))
        self.participating_clients_labels = list(labels)
        logging.debug(
            "participating_clients_labels: %s",
            self.participating_clients_labels,
        )

    def _select_faulty_provenance_data(self, min_per_label: int = 10) -> None:
        """Select misclassified samples matching the configured label flips."""
        wrong_indices, actual_labels, predicted_labels = (
            self._eval_and_extract_wrong_preds(self.server_test_data)
        )

        logging.info("Total wrong predictions: %s", len(wrong_indices))
        logging.info("label2flip mapping: %s", self.label2flip)

        if hasattr(actual_labels, "tolist"):
            actual_labels = actual_labels.tolist()
        if hasattr(predicted_labels, "tolist"):
            predicted_labels = predicted_labels.tolist()
        if hasattr(wrong_indices, "tolist"):
            wrong_indices = wrong_indices.tolist()

        allowed_true = {int(k) for k in self.label2flip.keys()}
        allowed_pred = {int(v) for v in self.label2flip.values()}

        selected_wrong_indices = []
        for index_i in wrong_indices:
            if index_i >= len(actual_labels) or index_i >= len(predicted_labels):
                continue
            true_label = int(actual_labels[index_i])
            predicted_label = int(predicted_labels[index_i])

            if predicted_label in allowed_pred and true_label in allowed_true:
                selected_wrong_indices.append(int(index_i))
            if len(selected_wrong_indices) >= min_per_label:
                break

        logging.info("Selected wrong indices: %s", selected_wrong_indices)

        if hasattr(self.server_test_data, "select"):
            self.subset_test_data = self.server_test_data.select(selected_wrong_indices)
        else:
            self.subset_test_data = torch.utils.data.Subset(
                self.server_test_data, selected_wrong_indices
            )

        if selected_wrong_indices:
            selected_actual = [actual_labels[i] for i in selected_wrong_indices]
            selected_pred = [predicted_labels[i] for i in selected_wrong_indices]
            logging.info("Selected actual labels: %s", selected_actual)
            logging.info("Selected predicted labels: %s", selected_pred)
        else:
            logging.info("No incorrect predictions available for provenance analysis")

    def _select_provenance_data(self, min_per_label: int = 2):
        """Select a subset of test data for provenance analysis based on balanced,
        correctly predicted samples. This matches the FederatedProvTrue logic from
        original TraceFL.

        Parameters
        ----------
        min_per_label : int, optional
            Minimum number of samples per label to select (default is 2).

        Returns
        -------
        None
        """
        if self._faulty_mode:
            self._select_faulty_provenance_data(max(min_per_label, 10))
            return

        # Evaluate global model on test data and get correct predictions
        correct_indices, actual_labels = self._eval_and_extract_correct_preds(
            self.server_test_data
        )

        logging.info("Total correct predictions: %s", len(correct_indices))

        # Balance dataset by label (same as original TraceFL)
        balanced_indices = self._balance_dataset_by_label(
            correct_indices, actual_labels, min_per_label
        )

        if len(balanced_indices) > 0:
            # Create subset dataset
            if hasattr(self.server_test_data, "select"):
                # Hugging Face dataset
                self.subset_test_data = self.server_test_data.select(balanced_indices)
            else:
                # PyTorch dataset - create subset
                self.subset_test_data = torch.utils.data.Subset(
                    self.server_test_data, balanced_indices
                )

            logging.info(
                "Selected %s balanced samples for provenance analysis",
                len(self.subset_test_data),
            )
        else:
            logging.info("No correct predictions found")

    def _eval_and_extract_wrong_preds(self, test_data):
        """Evaluate the model on test data and extract indices and labels for incorrect
        predictions.

        Parameters
        ----------
        test_data : object
            Test dataset to be evaluated.

        Returns
        -------
        tuple
            A tuple containing:
                - torch.Tensor: Indices of incorrectly predicted samples.
                - torch.Tensor: Actual labels of the samples.
                - torch.Tensor: Predicted labels of the samples.
        """
        # Determine architecture from model type
        arch = "cnn"  # Default
        if hasattr(self.global_model, "config"):
            arch = "transformer"
        elif any(
            name.startswith("resnet") for name in [type(self.global_model).__name__]
        ):
            arch = "resnet"
        elif any(
            name.startswith("densenet") for name in [type(self.global_model).__name__]
        ):
            arch = "densenet"

        # Use test_neural_network function (same as original TraceFL)
        d = test_neural_network(arch, {"model": self.global_model}, test_data)
        self.loss = d["loss"]
        self.acc = d["accuracy"]
        logging.debug("Accuracy on test data: %s", self.acc)

        return (
            d["eval_incorrect_indices"],
            d["eval_actual_labels"],
            d["eval_predicted_labels"],
        )

    def _eval_and_extract_correct_preds(self, test_data):
        """Evaluate the model on test data and extract indices and labels for correct
        predictions. This matches the FederatedProvTrue logic from original TraceFL.

        Parameters
        ----------
        test_data : object
            Test dataset to be evaluated.

        Returns
        -------
        tuple
            A tuple containing:
                - torch.Tensor: Indices of correctly predicted samples.
                - torch.Tensor: Actual labels of the samples.
        """
        # Determine architecture from model type
        arch = "cnn"  # Default
        if hasattr(self.global_model, "config"):
            arch = "transformer"
        elif any(
            name.startswith("resnet") for name in [type(self.global_model).__name__]
        ):
            arch = "resnet"
        elif any(
            name.startswith("densenet") for name in [type(self.global_model).__name__]
        ):
            arch = "densenet"

        # Use test_neural_network function (same as original TraceFL)
        d = test_neural_network(arch, {"model": self.global_model}, test_data)
        self.loss = d["loss"]
        self.acc = d["accuracy"]
        logging.debug("Accuracy on test data: %s", self.acc)

        return d["eval_correct_indices"], d["eval_actual_labels"]

    def _balance_dataset_by_label(self, correct_indices, dataset_labels, min_per_label):
        """Balance the dataset by selecting a minimum number of samples per label from
        correctly predicted indices. This matches the _balanceDatasetByLabel logic from
        original TraceFL.

        Parameters
        ----------
        correct_indices : torch.Tensor
            Indices corresponding to correctly predicted samples.
        dataset_labels : torch.Tensor
            Labels for the dataset corresponding to the indices.
        min_per_label : int
            Minimum number of samples required for each label.

        Returns
        -------
        torch.Tensor
            A tensor of indices representing a balanced subset of the dataset.
        """
        balanced_indices = []
        logging.debug(
            "participating_clients_labels %s",
            self.participating_clients_labels,
        )

        for label in self.participating_clients_labels:
            selected_labels = dataset_labels[correct_indices]
            temp_bools = selected_labels == label
            temp_correct_indices = correct_indices[temp_bools]
            if len(temp_correct_indices) >= min_per_label:
                balanced_indices.append(temp_correct_indices[:min_per_label])

        if len(balanced_indices) > 0:
            balanced_indices = torch.cat(balanced_indices)

        return balanced_indices

    def calculate_client_contributions(self):
        """Calculate client contributions using neuron provenance analysis.

        Returns
        -------
        dict
            Dictionary containing provenance results and client contributions.
        """
        if len(self.subset_test_data) == 0:
            logging.warning("No test data available for provenance analysis")
            return {"error": "No test data available"}

        try:
            # Add time tracking (matching original TraceFL)
            start_time = time.time()

            # Initialize NeuronProvenance
            neuron_prov = NeuronProvenance(
                cfg=self.cfg,
                arch=self.cfg.data_dist.model_architecture,
                test_data=self.subset_test_data,
                gmodel=self.global_model,
                c2model=self.client2model,
                num_classes=self.cfg.data_dist.num_classes,
                c2nk=self.client2num_examples,
            )

            # Compute input provenance
            input2prov = neuron_prov.compute_input_provenance()

            # Compute evaluation metrics (matching original TraceFL)
            eval_metrics = self._compute_eval_metrics(input2prov)
            end_time = time.time()

            # Aggregate client contributions across all inputs
            client_contributions = self._aggregate_client_contributions(input2prov)

            # Normalize contributions
            normalized_contributions = normalize_contributions(client_contributions)

            # Find top contributor
            top_contributor = max(
                normalized_contributions, key=normalized_contributions.get
            )

            results = {
                "round_id": self.round_id,
                "samples_analyzed": len(self.subset_test_data),
                "client_contributions": normalized_contributions,
                "top_contributor": top_contributor,
                "input_provenance": input2prov,
                "eval_metrics": eval_metrics,
                "test_data_acc": self.acc,
                "test_data_loss": self.loss,
                "avg_prov_time_per_input": (end_time - start_time)
                / len(self.subset_test_data),
            }

            # Log results (matching original TraceFL format)
            logging.info(
                "[Round %s] TraceFL Localization Accuracy = %s || "
                "Total Inputs Used In Prov: %s || GM_(loss, acc) (%s,%s)",
                self.round_id,
                eval_metrics["Accuracy"] * 100,
                len(self.subset_test_data),
                self.loss,
                self.acc,
            )

            return results

        except (ValueError, RuntimeError, AttributeError) as e:
            logging.error("Error in provenance calculation: %s", e)
            logging.error("Traceback:\n%s", traceback.format_exc())
            return {"error": str(e)}

    def _aggregate_client_contributions(self, input2prov):
        """Aggregate client contributions across all inputs.

        Parameters
        ----------
        input2prov : list
            List of provenance data for each input.

        Returns
        -------
        dict
            Dictionary mapping client IDs to aggregated contributions.
        """
        client_contributions = {
            client_id: 0.0 for client_id in self.client2model.keys()
        }

        for input_prov in input2prov:
            client2prov = input_prov.get("client2prov", {})
            for client_id, contrib in client2prov.items():
                if client_id in client_contributions:
                    client_contributions[client_id] += contrib

        return client_contributions

    def _compute_eval_metrics(self, input2prov):
        """Compute evaluation metrics based on provenance information. This matches the
        _computeEvalMetrics method from original TraceFL.

        Parameters
        ----------
        input2prov : list of dict
            List of provenance data dictionaries.

        Returns
        -------
        dict
            Dictionary of evaluation metrics.
        """
        data_loader = torch.utils.data.DataLoader(self.subset_test_data, batch_size=1)

        target_labels = []
        for data in data_loader:
            if isinstance(data, dict):
                label = data.get("labels") or data.get("label")
                if label is not None:
                    target_labels.append(
                        label.item() if hasattr(label, "item") else label
                    )

        # Use the normalized client2class instead of creating a new one
        client2class = {c: self.client2class.get(c, {}) for c in self.client2model}

        logging.debug("client2class: %s", client2class)

        correct_tracing = 0

        true_labels = []
        predicted_labels = []

        faulty_responsible = sorted(self._faulty_client_strings)
        faulty_client_set = set(faulty_responsible)

        for idx, prov_r in enumerate(input2prov):
            traced_client = prov_r["traced_client"]  # Keep as integer
            client2prov = prov_r["client2prov"]

            target_l = (
                target_labels[idx]
                if idx < len(target_labels)
                else idx % self.cfg.data_dist.num_classes
            )

            if self._faulty_mode:
                # FAULTY CLIENT DETECTION MODE (matches original TraceFL's
                # FederatedProvFalse)
                #
                # In this mode, we're testing if TraceFL can identify ANY faulty client
                # that is poisoning the model, not necessarily the specific client with
                # the target label. This matches the paper's experimental design.
                #
                # Responsible clients = pre-configured faulty clients (from config)
                # Tracing is "correct" if traced_client is in the faulty set
                #
                # Note: This differs from normal mode where responsible clients are
                # determined by label distribution. Here, we're testing fault
                # localization,
                # not data provenance for specific labels.
                responsible_clients = faulty_responsible
                res_c_string = ",".join(f"c{c}" for c in responsible_clients)
                logging.info(
                    "            *********** Input Label: %s, "
                    "Responsible Client(s): %s  *************",
                    target_l,
                    res_c_string,
                )

                if str(traced_client) in faulty_client_set:
                    logging.info(
                        "     Traced Client: c%s || Tracing = Correct",
                        traced_client,
                    )
                    correct_tracing += 1
                    predicted_labels.append(1)
                    true_labels.append(1)
                else:
                    logging.info(
                        "     Traced Client: c%s || Tracing = Wrong",
                        traced_client,
                    )
                    predicted_labels.append(0)
                    true_labels.append(1)
            else:
                # Normal mode: Find responsible clients
                responsible_clients = [
                    cid
                    for cid, c_labels in client2class.items()
                    if target_l in c_labels
                ]
                res_c_string = ",".join(
                    map(str, [f"c{c}" for c in responsible_clients])
                )

                logging.info(
                    "            *********** Input Label: %s, "
                    "Responsible Client(s): %s  *************",
                    target_l,
                    res_c_string,
                )

                # Check if traced client has the target label
                if target_l in client2class[traced_client]:
                    logging.info(
                        "     Traced Client: c%s || Tracing = Correct",
                        traced_client,
                    )
                    correct_tracing += 1
                    predicted_labels.append(1)
                    true_labels.append(1)
                else:
                    logging.info(
                        "     Traced Client: c%s || Tracing = Wrong",
                        traced_client,
                    )
                    predicted_labels.append(0)
                    true_labels.append(1)

            c2nk_label = {
                f"c{c}": client2class.get(c, {}).get(target_l, 0)  # type: ignore
                for c in client2prov.keys()
            }
            c2nk_label = {c: v for c, v in c2nk_label.items() if v > 0}

            client2prov_score = {f"c{c}": round(p, 2) for c, p in client2prov.items()}
            logging.info(
                "    TraceFL Clients Contributions Rank:     %s",
                client2prov_score,
            )
            logging.info("\n")

        eval_metrics = get_prov_eval_metrics(true_labels, predicted_labels)
        a = correct_tracing / len(input2prov)
        assert a == eval_metrics["Accuracy"], "Accuracy mismatch"
        return eval_metrics
