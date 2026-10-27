"""Neuron provenance analysis for TraceFL."""

import logging

import torch
import torch.nn.functional as F

from .neuron_activation import get_all_layers
from .utils import compute_importance


class NeuronProvenance:
    """Neuron-level provenance analysis for TraceFL.

    This class performs detailed neuron-level analysis to trace client contributions
    through neural network layers.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(self, cfg, arch, test_data, gmodel, c2model, num_classes, c2nk):
        """Initialize a NeuronProvenance instance.

        Parameters
        ----------
        cfg : object
            Configuration object.
        arch : str
            Model architecture type.
        test_data : object
            Test dataset.
        gmodel : torch.nn.Module
            Global model.
        c2model : dict
            Dictionary mapping client identifiers to their models.
        num_classes : int
            Number of classes.
        c2nk : dict
            Dictionary mapping client identifiers to number of examples.
        """
        self.arch = arch
        self.cfg = cfg
        self.test_data = test_data
        self.gmodel = gmodel
        self.c2model = c2model
        self.num_classes = num_classes
        self.device = cfg.device
        self.c2nk = c2nk  # client2num_examples
        self.client_ids = list(self.c2model.keys())
        self.layer_importance = compute_importance(len(get_all_layers(gmodel)))

        if not cfg.enable_beta:
            self.layer_importance = [1 for _ in range(len(get_all_layers(gmodel)))]

        logging.info("client ids: %s", self.client_ids)
        self.pk = {
            cid: self.c2nk[cid] / sum(self.c2nk.values()) for cid in self.c2nk.keys()
        }
        if self.cfg.client_weights_normalization:
            logging.debug(">> clients weights are normaized")
            self._inplace_scale_client_ws()

        # Initialize attributes that will be set later
        self.global_neurons_inputs_outputs_batch = None
        self.inputs2layer_grads = None

    def _check_anomalies(self, t):
        """Check the tensor for inf or NaN values.

        Parameters
        ----------
        t : torch.Tensor
            The tensor to be checked.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If inf or NaN values are detected in the tensor.
        """
        inf_mask = torch.isinf(t)
        nan_mask = torch.isnan(t)
        if inf_mask.any() or nan_mask.any():
            logging.error("Inf values: %s", torch.sum(inf_mask))
            logging.error("NaN values: %s", torch.sum(nan_mask))
            logging.error("Total values: %s", torch.numel(t))
            raise ValueError("Anomalies detected in tensor")

    def _calculate_layer_contribution(
        self,
        global_neurons_outputs: torch.Tensor,
        global_layer_grads: torch.Tensor,
        client2outputs,
        layer_id: int,
    ):
        """Calculate the contribution of each client for a specific layer.

        Parameters
        ----------
        global_neurons_outputs : torch.Tensor
            Global neurons outputs for a given input.
        global_layer_grads : torch.Tensor
            Gradients of the global layer.
        client2outputs : dict
            Dictionary mapping client IDs to their outputs.
        layer_id : int
            Identifier of the layer.

        Returns
        -------
        dict
            Dictionary mapping client IDs to their contribution for the layer.
        """
        client2avg = {cid: 0.0 for cid in self.client_ids}
        self._check_anomalies(global_neurons_outputs)
        self._check_anomalies(global_layer_grads)
        global_layer_grads = global_layer_grads.flatten()

        for cli in self.client_ids:
            cli_acts = client2outputs[cli].to(self.device).flatten()
            self._check_anomalies(cli_acts)
            cli_part = torch.dot(cli_acts, global_layer_grads)

            client2avg[cli] = cli_part.item() * self.layer_importance[layer_id]
            cli_acts = cli_acts.cpu()

        # cli with max contribution
        max_contributor = max(client2avg, key=client2avg.get)  # type: ignore
        logging.debug("Max contributor: %s", max_contributor)
        return client2avg

    def _map_client_layer_contributions(self, layer_id: int):
        """Map the contributions of clients for a specific layer.

        Parameters
        ----------
        layer_id : int
            Identifier of the layer.

        Returns
        -------
        dict
            Dictionary mapping input indices to client contributions.
        """
        client2layers = {cid: get_all_layers(cm) for cid, cm in self.c2model.items()}
        global_neurons_inputs = self.global_neurons_inputs_outputs_batch[layer_id][0]
        global_neurons_inputs = global_neurons_inputs.to(self.device)
        global_neurons_outputs = self.global_neurons_inputs_outputs_batch[layer_id][1]

        # check if global_neurons_outputs is a tuple
        if isinstance(global_neurons_outputs, tuple | list):
            assert (
                len(global_neurons_outputs) == 1
            ), f"Expected 1 element in tuple, got {len(global_neurons_outputs)}"
            global_neurons_outputs = global_neurons_outputs[0]
        global_neurons_outputs = global_neurons_outputs.to(self.device)

        c2l = {cid: client2layers[cid][layer_id] for cid in self.client_ids}
        clinet2outputs = {
            c: self._evaluate_layer(layer, global_neurons_inputs)
            for c, layer in c2l.items()
        }

        input2client2contribution = {}
        for input_id in range(len(self.test_data)):  # for per input in the batch
            logging.debug(
                "Mapping client contributions for input %s for layer %s",
                input_id,
                layer_id,
            )
            c2out_per_input = {
                cid: clinet2outputs[cid][input_id] for cid in self.client_ids
            }

            glayer_grads = torch.squeeze(
                self.inputs2layer_grads[input_id][layer_id][1]
            ).to(self.device)

            c2contribution = self._calculate_layer_contribution(
                global_neurons_outputs=global_neurons_outputs[input_id],
                global_layer_grads=glayer_grads,
                client2outputs=c2out_per_input,
                layer_id=layer_id,
            )
            input2client2contribution[input_id] = c2contribution

        return input2client2contribution

    def _inplace_scale_client_ws(self):
        """Scale client model weights based on the number of data points per client.

        Returns
        -------
        None
        """
        logging.debug(
            "Scaling client weights based on the number of data points each client has."
        )

        logging.debug("Total clients in c2nk: %s", len(self.c2nk))
        logging.debug("Total clients in c2model: %s", len(self.c2model))

        ids1 = list(self.c2model.keys())
        ids2 = list(self.c2nk.keys())
        logging.debug("ids1: %s", ids1)
        logging.debug("ids2: %s", ids2)

        for cid in self.c2model.keys():
            scale_factor = self.c2nk[cid] / sum(self.c2nk.values())
            logging.debug(
                "Scaling client %s by %s, nk = %s",
                cid,
                scale_factor,
                self.c2nk[cid],
            )
            with torch.no_grad():
                for cparam in self.c2model[cid].parameters():
                    cparam.data = scale_factor * cparam.data
                self.c2model[cid] = self.c2model[cid].eval().cpu()

    def _capture_layer_io(self):
        """Capture the inputs and outputs of all layers in the global model.

        Returns
        -------
        None
        """
        hook_manager = HookManager()
        glayers = get_all_layers(self.gmodel)
        hooks_forward = [hook_manager.insert_forward_hook(layer) for layer in glayers]
        self.gmodel.eval()
        self.gmodel = self.gmodel.to(self.device)

        # Forward pass through test data
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(
                self.test_data, batch_size=len(self.test_data)
            ):
                if isinstance(batch, dict):
                    if "pixel_values" in batch:
                        # CNN input
                        self.gmodel(batch["pixel_values"].to(self.device))
                    elif "input_ids" in batch:
                        # Transformer input
                        inputs = {
                            k: v.to(self.device)
                            for k, v in batch.items()
                            if k in ["input_ids", "attention_mask"]
                        }
                        self.gmodel(**inputs)
                else:
                    # Direct tensor input
                    self.gmodel(batch.to(self.device))
                break  # Only process first batch

        hook_manager.remove_hooks(hooks_forward)
        self.global_neurons_inputs_outputs_batch = hook_manager.forward_hooks_storage
        hook_manager.clear_storages()

    def _capture_layer_gradients(self):
        """Capture the gradients for each layer of the global model for the test data.

        Returns
        -------
        None
        """
        self.inputs2layer_grads = []
        for m_input in torch.utils.data.DataLoader(self.test_data, batch_size=1):
            hook_manager = HookManager()
            self._set_gradients_of_model(m_input, hook_manager)
            self.inputs2layer_grads.append(hook_manager.backward_hooks_storage)
            hook_manager.clear_storages()

    def _set_gradients_of_model(self, input_for_model, hook_manager):
        """Set up hooks to capture gradients of the model during backward pass.

        Parameters
        ----------
        input_for_model : dict or tensor
            Input data for the model.
        hook_manager : HookManager
            Instance of HookManager to manage hooks.

        Returns
        -------
        None
        """
        if self.arch == "transformer":
            self._set_gradients_transformer_model(input_for_model, hook_manager)
        else:
            self._set_gradients_cnn_model(input_for_model, hook_manager)

    def _set_gradients_cnn_model(self, input_for_model, hook_manager):
        """Set up hooks and compute gradients for a CNN model.

        Parameters
        ----------
        input_for_model : dict
            Dictionary containing input data under key 'pixel_values'.
        hook_manager : HookManager
            Instance of HookManager to manage hooks.

        Returns
        -------
        None
        """
        self.gmodel.zero_grad()
        all_layers = get_all_layers(self.gmodel)
        all_hooks = [hook_manager.insert_backward_hook(layer) for layer in all_layers]

        self.gmodel = self.gmodel.to(self.device)

        if isinstance(input_for_model, dict) and "pixel_values" in input_for_model:
            img_input = input_for_model["pixel_values"]
        else:
            img_input = input_for_model

        outs = self.gmodel(img_input.to(self.device))

        if isinstance(outs, dict) and "logits" in outs:
            logits = outs["logits"]
        else:
            logits = outs

            _, predicted = torch.max(logits, dim=1)
            logits[0, predicted].backward()  # computing the gradients
        hook_manager.remove_hooks(all_hooks)
        hook_manager.backward_hooks_storage.reverse()

    def _set_gradients_transformer_model(self, text_input_tuple, hook_manager):
        """Set up hooks and compute gradients for a transformer model.

        Parameters
        ----------
        text_input_tuple : dict
            Dictionary containing input data for the transformer.
        hook_manager : HookManager
            Instance of HookManager to manage hooks.

        Returns
        -------
        None
        """
        self.gmodel.zero_grad()
        all_layers = get_all_layers(self.gmodel)
        all_hooks = [hook_manager.insert_backward_hook(layer) for layer in all_layers]

        self.gmodel.to(self.device)

        # Prepare input for transformer
        if isinstance(text_input_tuple, dict):
            inputs = {
                k: v.to(self.device)
                for k, v in text_input_tuple.items()
                if k in ["input_ids", "attention_mask"]
            }
        else:
            # Fallback for tensor input
            inputs = {"input_ids": text_input_tuple.to(self.device)}

        outs = self.gmodel(**inputs)

        if isinstance(outs, dict) and "logits" in outs:
            logits = outs["logits"]
        else:
            logits = outs.logits

            _, predicted = torch.max(logits, dim=1)
            predicted = predicted.cpu().detach().item()
            logits[0, predicted].backward()  # computing the gradients
        hook_manager.remove_hooks(all_hooks)
        hook_manager.backward_hooks_storage.reverse()

    def _evaluate_layer(
        self, client_layer: torch.nn.Module, global_neurons_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate a client's layer using the global model's layer inputs.

        Parameters
        ----------
        client_layer : torch.nn.Module
            The client's layer.
        global_neurons_inputs : torch.Tensor
            The inputs captured from the global model for the layer.

        Returns
        -------
        torch.Tensor
            The output of the client's layer.
        """
        client_layer = client_layer.eval().to(device=self.device)
        client_neurons_outputs = client_layer(global_neurons_inputs)

        if isinstance(client_neurons_outputs, tuple | list):
            client_neurons_outputs = client_neurons_outputs[0].cpu()
        else:
            client_neurons_outputs = client_neurons_outputs.cpu()

        client_layer = client_layer.cpu()
        return client_neurons_outputs

    def _aggregate_client_contributions(self, input_id: int, layers2prov):
        """Aggregate the contributions of all clients for a given input across layers.

        Parameters
        ----------
        input_id : int
            The index of the input.
        layers2prov : list
            List of client contributions per layer.

        Returns
        -------
        dict
            Dictionary mapping client IDs to aggregated contribution for the input.
        """
        client2totalcont = {c: 0.0 for c in self.client_ids}
        clients_prov = [i2prov[input_id] for i2prov in layers2prov]
        for c2lsum in clients_prov:
            for cid in self.client_ids:
                client2totalcont[cid] += c2lsum[cid]
        return client2totalcont

    def _normalize_contributions(self, contributions):
        """Normalize client contributions using softmax.

        Parameters
        ----------
        contributions : dict
            Dictionary mapping client IDs to contributions.

        Returns
        -------
        dict
            Dictionary mapping client IDs to normalized contributions.
        """
        conts = F.softmax(torch.tensor(list(contributions.values())), dim=0)
        client2prov = {
            cid: v.item() for cid, v in zip(self.client_ids, conts, strict=False)
        }
        return dict(sorted(client2prov.items(), key=lambda item: item[1], reverse=True))

    def _aggregate_input_contributions(self, layers2prov):
        """Aggregate client contributions for each input across all layers.

        Parameters
        ----------
        layers2prov : list
            List of client contributions per layer.

        Returns
        -------
        list
            List of dictionaries mapping each input to its provenance data.
        """
        input2prov = []
        for input_id in range(len(self.test_data)):
            client_conts = self._aggregate_client_contributions(input_id, layers2prov)
            normalized_conts = self._normalize_contributions(client_conts)
            traced_client = max(
                normalized_conts, key=normalized_conts.get
            )  # type: ignore

            input2prov.append(
                {"traced_client": traced_client, "client2prov": normalized_conts}
            )
        return input2prov

    def compute_input_provenance(self):
        """Compute the provenance of each input by aggregating client contributions
        across layers.

        Returns
        -------
        list
            List of dictionaries mapping each input to its provenance data.
        """
        self._capture_layer_io()
        self._capture_layer_gradients()

        layers2prov = []
        for layer_id in range(len(self.global_neurons_inputs_outputs_batch)):
            client2cont = self._map_client_layer_contributions(layer_id)
            layers2prov.append(client2cont)

        input2prov = self._aggregate_input_contributions(layers2prov)
        return input2prov


class HookManager:
    """Hook manager for neural network layer monitoring.

    This class manages forward and backward hooks for capturing layer inputs, outputs,
    and gradients during model evaluation.
    """

    def __init__(self):
        """Initialize a HookManager instance."""
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def insert_forward_hook(self, layer):
        """Insert a forward hook into the specified layer.

        Parameters
        ----------
        layer : torch.nn.Module
            The layer to attach the hook to.

        Returns
        -------
        handle
            The hook handle.
        """

        def forward_hook(_module, input_tensor, output_tensor):
            try:
                input_tensor = input_tensor[0]
                input_tensor = input_tensor.detach()
            except (AttributeError, TypeError, IndexError):
                pass

            input_tensor = input_tensor.detach()
            self.forward_hooks_storage.append((input_tensor, output_tensor))

        hook = layer.register_forward_hook(forward_hook)
        return hook

    def insert_backward_hook(self, layer):
        """Insert a backward hook into the specified layer.

        Parameters
        ----------
        layer : torch.nn.Module
            The layer to attach the hook to.

        Returns
        -------
        handle
            The hook handle.
        """

        def backward_hook(_module, input_tensor, output_tensor):
            try:
                input_tensor = input_tensor[0]
                output_tensor = output_tensor[0]
                input_tensor = input_tensor.detach()
                output_tensor = output_tensor.detach()

            except (AttributeError, TypeError, IndexError):
                pass
            try:
                input_tensor = input_tensor.detach()
            except (AttributeError, TypeError):
                pass
            try:
                output_tensor = output_tensor.detach()
            except (AttributeError, TypeError):
                pass

            self.backward_hooks_storage.append((input_tensor, output_tensor))

        hook = layer.register_full_backward_hook(backward_hook)
        return hook

    def clear_storages(self):
        """Clear the storage for forward and backward hooks.

        Returns
        -------
        None
        """
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def remove_hooks(self, hooks):
        """Remove the provided hooks.

        Parameters
        ----------
        hooks : list
            List of hook handles to be removed.

        Returns
        -------
        None
        """
        for hook in hooks:
            hook.remove()
