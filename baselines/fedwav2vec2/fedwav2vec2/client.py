"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import gc
import logging
from math import exp

import flwr as fl
import speechbrain as sb
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    NDArrays,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from omegaconf import DictConfig

from fedwav2vec2.models import int_model
from fedwav2vec2.sb_recipe import get_weights, set_weights


class SpeechBrainClient(fl.client.Client):
    """Flower client for SpeechBrain."""

    def __init__(self, cid: str, asr_brain, dataset):
        self.cid = cid
        self.params = asr_brain.hparams
        self.modules = asr_brain.modules
        self.asr_brain = asr_brain
        self.dataset = dataset

        fl.common.logger.log(logging.DEBUG, "Starting client %s", cid)

    def get_parameters(self, _: GetParametersIns) -> GetParametersRes:
        """Return the parameters of the current net."""
        weights: NDArrays = get_weights(self.modules)
        parameters = ndarrays_to_parameters(weights)
        gc.collect()
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        """Implement distributed fit function for a given client."""
        weights: NDArrays = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config

        # Read training configuration
        epochs = int(config["epochs"])

        (_, num_examples, avg_loss, avg_wer) = self._train_speech_recogniser(
            weights, epochs
        )
        metrics = {"train_loss": avg_loss, "wer": avg_wer}

        parameters = self.get_parameters(GetParametersIns(config={})).parameters
        del self.asr_brain.modules
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        status = Status(code=Code.OK, message="Success")

        return FitRes(
            status=status,
            parameters=parameters,
            num_examples=num_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Implement distributed evaluation for a given client."""
        weights = parameters_to_ndarrays(ins.parameters)

        num_examples, loss, wer = self.evaluate_train_speech_recogniser(
            server_params=weights,
            epochs=1,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        status = Status(code=Code.OK, message="Success")
        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            status=status,
            num_examples=num_examples,
            loss=float(loss),
            metrics={"Error rate": float(wer)},
        )

    def evaluate_train_speech_recogniser(self, server_params, epochs):
        """Evaluate aggerate/server model."""
        _, _, test_data = self._setup_task(server_params, epochs)
        self.params.wer_file = self.params.output_folder + "/wer_test.txt"

        batch_count, loss, wer = self.asr_brain.evaluate(
            test_data,
            test_loader_kwargs=self.params.test_dataloader_options,
        )

        return batch_count, float(loss), float(wer)

    def _setup_task(
        self,
        server_params,
        epochs,
    ):
        self.params.epoch_counter.limit = epochs
        self.params.epoch_counter.current = 0

        train_data, valid_data, test_data = self.dataset
        # Set the parameters to the ones given by the server
        if server_params is not None:
            set_weights(server_params, self.modules, self.params.device)
        return train_data, valid_data, test_data

    def _train_speech_recogniser(self, server_params, epochs):
        train_data, valid_data, _ = self._setup_task(server_params, epochs)

        # Training
        count_sample, avg_loss, avg_wer = self.asr_brain.fit(
            self.params.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=self.params.dataloader_options,
            valid_loader_kwargs=self.params.test_dataloader_options,
        )
        # exp operation to avg_loss and avg_wer
        avg_wer = 100 if avg_wer > 100 else avg_wer
        avg_loss = exp(-avg_loss)
        avg_wer = exp(100 - avg_wer)

        # retrieve the parameters to return
        params_list = get_weights(self.modules)

        # Manage when last batch isn't full w.r.t batch size
        train_set = sb.dataio.dataloader.make_dataloader(
            train_data, **self.params.dataloader_options
        )
        if count_sample > len(train_set) * self.params.batch_size * epochs:
            count_sample = len(train_set) * self.params.batch_size * epochs

        del train_data, valid_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return (params_list, count_sample, avg_loss, avg_wer)


def get_client_fn(config: DictConfig, save_path: str):
    """Return a function that creates a Flower client."""

    def client_fn(cid: str) -> fl.client.Client:
        """Generate the simulated clients."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        asr_brain, dataset = int_model(cid, config, device=device, save_path=save_path)
        return SpeechBrainClient(cid, asr_brain, dataset)

    return client_fn
