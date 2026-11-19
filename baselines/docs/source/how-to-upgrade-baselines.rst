Upgrading your Baselines
========================

As of ``Flower 1.13``, the way users interact with the underlying Flower simulation framework has changed. Specifically:

1. Simulations are no longer launched via the ``start_simulation`` command (which has since been deprecated), but rather through using ``flwr run`` through the command-line. 
2. Client and server object definitions have been refactored.

More information on the specific breaking changes can be found `here <https://flower.ai/docs/framework/how-to-upgrade-to-flower-1.13.html>`_ .

This document is meant to serve as a practical guide on how one can approach upgrading pre-existing baseline implementations from the previous Flower versions,
to ones that are currently supported. Specifically, this is meant to compliment pre-existing material that denotes how to make use of the Flower framework effectively.

To make this guide more compelling, we will use the `following PR that upgrades FedBN <https://github.com/adap/flower/pull/5115>`_ to illustrate the necessary changes that should be made allowing with some code examples. 
This guide will provide a motivation behind the design decisions made in the refactoring of the codebase. Broadly speaking, the below changes are typically the main changes that need to be made
to successfully migrate the baseline to a higher Flower version, where all other files (such as models, utils etc.) can remain the same.


Moving pre-simulation code from ``main.py`` 
*******************************************
One of the first changes that we should take care of is migrating existing code that is located in the ``main.py`` file of the baseline implementation. The reason for this is that the ``start_simulation`` function 
is longer supported (which was located in ``main.py``), and clients and servers are now defined as ``ClientApps`` and ``ServerApps``, respectively.  As such, we can no longer define, and launch
our experiments from ``main.py`` with a call analogous to ``python -m main.py``. For example, a typical definition
of the the main.py file as was done pre-``Flower 1.13`` can be seen as follows:

.. code-block:: python

    @hydra.main(config_path="conf", config_name="base", version_base=None)
    def main(cfg: DictConfig) -> None:
        """Run the baseline.
        Parameters
        ----------
        cfg : DictConfig
            An omegaconf object that stores the hydra config.
        """
        # 1. Print parsed config
        print(OmegaConf.to_yaml(cfg))

        # Hydra automatically creates an output directory
        # Let's retrieve it and save some results there
        save_path = Path(HydraConfig.get().runtime.output_dir)

        # 2. Prepare your dataset
        # please ensure you followed the README.md and you downloaded the
        # pre-processed dataset suplied by the authors of the FedBN paper
        client_data_loaders = get_data(cfg.dataset)

        # 3. Define your client generation function
        client_fn = gen_client_fn(client_data_loaders, cfg.client, cfg.model, save_path)

        # 4. Define your strategy
        strategy = instantiate(cfg.strategy)

        # 5. Start Simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            client_resources={
                "num_cpus": cfg.client_resources.num_cpus,
                "num_gpus": cfg.client_resources.num_gpus,
            },
            strategy=strategy,
        )

        # 6. Save your results
        print("................")
        print(history)

        # Save results as a Python pickle using a file_path
        # the directory created by Hydra for each run
        data = {"history": history}
        history_path = f"{str(save_path)}/history.pkl"
        with open(history_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Simple plot
        quick_plot(history_path)


    if __name__ == "__main__":
        main()

Broadly, we can view the above main function as having two components: i) pre-simulation setup, which occurs prior to the call of ``start_simulation`` and ii) post-simulation clean-up, occurring after
the call to ``start_simulation``. As a general rule of thumb, all functionality that is placed prior to the launch of the simulation should be placed within the ``server_fn`` that is used by the ``ServerApp``.
This is with the exception of any configuration that is specific to the client setup; this should be moved to the definition of the ``client_fn`` accordingly. 

In the FedBN refactoring, the configuration specification and the definition of the relevant strategy of this baseline was moved to the ``server-app`` function (located in ``server_app.py``), whilst the dataset and client definitions have
been moved to the ``client_fn`` (located in ``client_app.py``).

.. code-block:: python

    # Function in server_app.py
    def server_fn(context: Context):
        """Construct components that set the ServerApp behaviour."""
        # Read from context
        print("### BEGIN: RUN CONFIG ####")
        run_config = context.run_config
        print(json.dumps(run_config, indent=4))
        print("### END: RUN CONFIG ####")
        num_rounds = context.run_config["num-server-rounds"]

        ndarrays = extract_weights(
            CNNModel(num_classes=run_config["num-classes"]),
            run_config["algorithm-name"],
        )
        parameters = ndarrays_to_parameters(ndarrays)
        # Define Strategy
        strategy = FedAvg(
            fraction_fit=float(run_config["fraction-fit"]),
            fraction_evaluate=float(run_config["fraction-evaluate"]),
            min_available_clients=int(run_config["num-clients"]),
            on_fit_config_fn=get_on_fit_config(),
            initial_parameters=parameters,
            fit_metrics_aggregation_fn=get_metrics_aggregation_fn(),
            evaluate_metrics_aggregation_fn=get_metrics_aggregation_fn(),
        )
        config = ServerConfig(num_rounds=int(num_rounds))
        client_manager = SimpleClientManager()
        server = ResultsSaverServer(
            client_manager=client_manager,
            strategy=strategy,
            results_saver_fn=save_results_and_config,
            run_config=run_config,
        )
        return ServerAppComponents(server=server, config=config)
    
    # Function in client_app.py
    def client_fn(context: Context):
        """Construct a Client that will be run in a ClientApp."""
        # Load model and data
        run_config = context.run_config
        net = CNNModel(num_classes=run_config["num-classes"])
        partition_id = int(context.node_config["partition-id"])
        trainloader, valloader, dataset_name = (get_data(context))[partition_id]

        # Return Client instance
        client_type, client_state = (
            (FlowerClient, None)
            if run_config["algorithm-name"] == "FedAvg"
            else (FedBNFlowerClient, context.state)
        )
        return client_type(
            net=net,
            trainloader=trainloader,
            testloader=valloader,
            dataset_name=dataset_name,
            learning_rate=run_config["learning-rate"],
            client_state=client_state,
        ).to_client()

Ensuring Stateful Clients
*************************
With newer Flower versions, a pertinent change that introduced was the ability to ensure stateful clients within the simulation without relying saving temporary files to disk. 
Specifically, clients are now able to preserve any related information (such as model parameters, statistics etc.) within the unique ``Context`` that is assigned to each client. You can find more information about how to do this `at the following link <https://flower.ai/docs/framework/how-to-design-stateful-clients.html>`_.

In the case of our working example, we see that stateful clients were ensured by writing to a specific directory on the disk in order to preserve relevant batch norm statistics per client.

.. code-block:: python

    class FedBNFlowerClient(FlowerClient):	
        """Similar to FlowerClient but this is used by FedBN clients."""	

        def __init__(self, save_path: Path, client_id: int, *args, **kwargs) -> None:	
            super().__init__(*args, **kwargs)	
            # For FedBN clients we need to persist the state of the BN	
            # layers across rounds. In Simulation clients are statess	
            # so everything not communicated to the server (as it is the	
            # case as with params in BN layers of FedBN clients) is lost	
            # once a client completes its training. An upcoming version of	
            # Flower suports stateful clients	
            bn_state_dir = save_path / "bn_states"	
            bn_state_dir.mkdir(exist_ok=True)	
            self.bn_state_pkl = bn_state_dir / f"client_{client_id}.pkl"	

        def _save_bn_statedict(self) -> None:	
            """Save contents of state_dict related to BN layers."""	
            bn_state = {	
                name: val.cpu().numpy()	
                for name, val in self.model.state_dict().items()	
                if "bn" in name	
            }	

            with open(self.bn_state_pkl, "wb") as handle:	
                pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)	

        def _load_bn_statedict(self) -> Dict[str, torch.tensor]:	
            """Load pickle with BN state_dict and return as dict."""	
            with open(self.bn_state_pkl, "rb") as handle:	
                data = pickle.load(handle)	
            bn_stae_dict = {k: torch.tensor(v) for k, v in data.items()}	
            return bn_stae_dict	

        def get_parameters(self, config) -> NDArrays:	
            """Return model parameters as a list of NumPy ndarrays w or w/o using BN.	
            layers.	
            """	
            # First update bn_state_dir	
            self._save_bn_statedict()	
            # Excluding parameters of BN layers when using FedBN	
            return [	
                val.cpu().numpy()	
                for name, val in self.model.state_dict().items()	
                if "bn" not in name	
            ]	

        def set_parameters(self, parameters: NDArrays) -> None:	
            """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if.	
            available.	
            """	
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]	
            params_dict = zip(keys, parameters)	
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})	
            self.model.load_state_dict(state_dict, strict=False)	

            # Now also load from bn_state_dir	
            if self.bn_state_pkl.exists():  # It won't exist in the first round	
                bn_state_dict = self._load_bn_statedict()	
                self.model.load_state_dict(bn_state_dict, strict=False)


However, with the new version of ``Flower``, this can be done directly with the client's simulation ``Context``, as seen below:

    .. code-block:: python

        class FedBNFlowerClient(FlowerClient):
        """Similar to FlowerClient but this is used by FedBN clients."""

        def __init__(self, client_state, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.client_state = client_state
            # For FedBN clients we need to persist the state of the BN
            # layers across rounds. In Simulation clients are states
            # so everything not communicated to the server (as it is the
            # case as with params in BN layers of FedBN clients) is lost
            # once a client completes its training. This is the case unless
            # we preserve the batch norm states in the Context.
            if not self.client_state.array_records:
                # Ensure statefulness of error feedback buffer.
                self.client_state.array_records["local_batch_norm"] = ArrayRecord(
                    OrderedDict({"initialisation": Array(np.array([-1]))})
                )

        def _save_bn_statedict(self) -> None:
            """Save contents of state_dict related to BN layers."""
            bn_state = OrderedDict(
                {
                    name: Array(val.cpu().numpy())
                    for name, val in self.net.state_dict().items()
                    if "bn" in name
                }
            )
            self.client_state.array_records["local_batch_norm"] = ArrayRecord(
                bn_state
            )

        def get_weights(self) -> NDArrays:
            """Return model parameters as a list of NumPy ndarrays without BN.
            layers.
            """
            # First update bn_state_dir
            self._save_bn_statedict()
            return extract_weights(self.net, "FedBN")

        def set_weights(self, parameters: NDArrays) -> None:
            """Set model parameters from a list of NumPy ndarrays Exclude the bn.
            layer if available.
            """
            keys = [k for k in self.net.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=False)

            # Now also load from bn_state_dir
            if (
                "initialisation"
                not in self.client_state.array_records["local_batch_norm"].keys()
            ):  # It won't exist in the first round
                batch_norm_state = {
                    k: torch.tensor(v.numpy())
                    for k, v in self.client_state.array_records[
                        "local_batch_norm"
                    ].items()
                }
                self.net.load_state_dict(batch_norm_state, strict=False)


Moving post-simulation code from ``main.py`` 
********************************************
As we saw in the initial ``main.py`` file, typical post-simulation code is reserved to result handling. In our example specifically, the results following the end of the simulation are saved,
and the relevant results are plotted. In the base definition of Flower run, by default results are not saved to disk, but rather written to the standard output. However, we can define this functionality for ourselves by creating our own custom ``ResultsSaverServer`` that inherits from the
base ``Server`` definition that is used for simulations. Concretely, we define said server as follows:

.. code-block:: python

    """fedbn: A Flower Baseline."""

    import json
    import os
    import pickle
    from logging import INFO
    from pathlib import Path
    from secrets import token_hex
    from typing import Dict, Optional, Union

    from flwr.common import log
    from flwr.server import Server
    from flwr.server.history import History

    PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent


    class ResultsSaverServer(Server):
        """Server to save history to disk."""

        def __init__(
            self,
            *,
            client_manager,
            strategy=None,
            results_saver_fn=None,
            run_config=None,
        ):
            super().__init__(client_manager=client_manager, strategy=strategy)
            self.results_saver_fn = results_saver_fn
            self.run_config = run_config

        def fit(self, num_rounds, timeout):
            """Run federated averaging for a number of rounds."""
            history, elapsed = super().fit(num_rounds, timeout)
            if self.results_saver_fn:
                log(INFO, "Results saver function provided. Executing")
                self.results_saver_fn(history, self.run_config)
            return history, elapsed


    def save_results_as_pickle(
        history: History,
        file_path: Union[str, Path],
        extra_results: Optional[Dict] = None,
        default_filename: str = "results.pkl",
    ) -> None:
        """Save results from simulation to pickle.
        Parameters
        ----------
        history: History
            History returned by start_simulation.
        file_path: Union[str, Path]
            Path to file to create and store both history and extra_results.
            If path is a directory, the default_filename will be used.
            path doesn't exist, it will be created. If file exists, a
            randomly generated suffix will be added to the file name. This
            is done to avoid overwritting results.
        extra_results : Optional[Dict]
            A dictionary containing additional results you would like
            to be saved to disk. Default: {} (an empty dictionary)
        default_filename: Optional[str]
            File used by default if file_path points to a directory instead
            to a file. Default: "results.pkl"
        """
        path = Path(file_path)

        # ensure path exists
        path.mkdir(exist_ok=True, parents=True)

        def _add_random_suffix(path_: Path):
            """Add a randomly generated suffix to the file name (so it doesn't.
            overwrite the file).
            """
            print(f"File `{path_}` exists! ")
            suffix = token_hex(4)
            print(f"New results to be saved with suffix: {suffix}")
            return path_.parent / (path_.stem + "_" + suffix + ".pkl")

        def _complete_path_with_default_name(path_: Path):
            """Append the default file name to the path."""
            print("Using default filename")
            return path_ / default_filename

        if path.is_dir():
            path = _complete_path_with_default_name(path)

        if path.is_file():
            # file exists already
            path = _add_random_suffix(path)

        print(f"Results will be saved into: {path}")

        data = {"history": history}
        if extra_results is not None:
            data = {**data, **extra_results}

        # save results to pickle
        with open(str(path), "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def save_results_and_config(history, run_config):
        """Save history and clean scaler dir."""
        results_path = (
            PROJECT_DIR
            / run_config["results-save-dir"]
            / run_config["algorithm-name"]
        )
        save_results_as_pickle(
            history=history, file_path=results_path, default_filename="history.pkl"
        )
        save_path = results_path / "config.json"
        if os.path.exists(save_path):
            log(INFO, "Config for this run has already been saved before")
        else:
            with open(save_path, "w", encoding="utf8") as fp:
                json.dump(run_config, fp)

Observing the changes relative to the initial Server definition,  we receive the history at the end of the simulation in the ``fit`` method. Instead of immediately returning these results,
we are able to save this to disk by injecting a function that specifies how saving should be done, which is ``save_results_and_config`` in this case. Once this has be defined, 
we can now specify that we wish to use this ``Server`` definition in the simulation explicitly instead of relying on the default instantiation of the `Server` object.

.. code-block:: python

    config = ServerConfig(num_rounds=int(num_rounds))
    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        client_manager=client_manager,
        strategy=strategy,
        results_saver_fn=save_results_and_config,
        run_config=run_config,
    )
    return ServerAppComponents(server=server, config=config)

How to do multiple runs?
************************
Previous Flower versions that used ``Hydra`` as a configuration management solution allowed for multiple runs across seeds, for example, via the ``--multirun`` call. However, since newer Flower
versions have migrated away from using ``Hydra``, one would now have to invoke the same ``flwr run`` call multiple times (for example through a ``bash`` script) to generate an experiment trial over multiple seeds.
The ``ResultsSaverServer`` above allows for this functionality by preventing each completed simulation from being overwritten the previous results that were obtained. In addition to saving multiple trails, this function
also saves the exact run configuration to be used at a later stage. This is useful for plotting results, as we will see below.

Finally, plotting 
*****************
As a general rule of thumb with baselines, it is preferable to generate the result plots via a Python Notebook, rather than this being done directly during the simulation runtime. Effectively, 
there is no difference between the two, but the notebook provides an easier interface and is more flexible to use. As such, any code in the ``main.py`` that involves plotting should be moved either
to a ``utils`` file and then used directly within the notebook via an import statement, or defined in the notebook directly.