.. _quickstart-ios:

################
 Quickstart iOS
################

.. meta::
    :description: Read this Federated Learning quickstart tutorial for creating an iOS app using Flower to train a neural network on MNIST.

.. warning::

    The experimental Flower iOS SDK is not compatible with the latest version of Flower.
    iOS support is currently being reworked and will be released in 2025.

    This quickstart tutorial is kept for historical purposes and will be updated once
    the new iOS SDK is released.

In this tutorial we will learn how to train a Neural Network on MNIST using Flower and
CoreML on iOS devices.

First of all, for running the Flower Python server, it is recommended to create a
virtual environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`. For the Flower client implementation in iOS,
it is recommended to use Xcode as our IDE.

Our example consists of one Python *server* and two iPhone *clients* that all have the
same model.

*Clients* are responsible for generating individual weight updates for the model based
on their local datasets. These updates are then sent to the *server* which will
aggregate them to produce a better model. Finally, the *server* sends this improved
version of the model back to each *client*. A complete cycle of weight updates is called
a *round*.

Now that we have a rough idea of what is going on, let's get started to setup our Flower
server environment. We first need to install Flower. You can do this by using pip:

.. code-block:: shell

    $ pip install flwr[simulation]

Or Poetry:

.. code-block:: shell

    $ poetry add flwr

***************
 Flower Client
***************

Now that we have all our dependencies installed, let's run a simple distributed training
using CoreML as our local training pipeline and MNIST as our dataset. For simplicity
reasons we will use the complete Flower client with CoreML, that has been implemented
and stored inside the Swift SDK. The client implementation can be seen below:

.. code-block:: swift

    /// Parses the parameters from the local model and returns them as GetParametersRes struct
    ///
    /// - Returns: Parameters from the local model
    public func getParameters() -> GetParametersRes {
      let parameters = parameters.weightsToParameters()
      let status = Status(code: .ok, message: String())

      return GetParametersRes(parameters: parameters, status: status)
    }

    /// Calls the routine to fit the local model
    ///
    /// - Returns: The result from the local training, e.g., updated parameters
    public func fit(ins: FitIns) -> FitRes {
      let status = Status(code: .ok, message: String())
      let result = runMLTask(configuration: parameters.parametersToWeights(parameters: ins.parameters), task: .train)
      let parameters = parameters.weightsToParameters()

      return FitRes(parameters: parameters, numExamples: result.numSamples, status: status)
      }

    /// Calls the routine to evaluate the local model
    ///
    /// - Returns: The result from the evaluation, e.g., loss
    public func evaluate(ins: EvaluateIns) -> EvaluateRes {
      let status = Status(code: .ok, message: String())
      let result = runMLTask(configuration: parameters.parametersToWeights(parameters: ins.parameters), task: .test)

      return EvaluateRes(loss: Float(result.loss), numExamples: result.numSamples, status: status)
    }

Let's create a new application project in Xcode and add ``flwr`` as a dependency in your
project. For our application, we will store the logic of our app in ``FLiOSModel.swift``
and the UI elements in ``ContentView.swift``. We will focus more on ``FLiOSModel.swift``
in this quickstart. Please refer to the `full code example
<https://github.com/flwrlabs/flower/tree/main/examples/ios>`_ to learn more about the
app.

Import Flower and CoreML related packages in ``FLiOSModel.swift``:

.. code-block:: swift

    import Foundation
    import CoreML
    import flwr

Then add the mlmodel to the project simply by drag-and-drop, the mlmodel will be bundled
inside the application during deployment to your iOS device. We need to pass the url to
access mlmodel and run CoreML machine learning processes, it can be retrieved by calling
the function ``Bundle.main.url``. For the MNIST dataset, we need to preprocess it into
``MLBatchProvider`` object. The preprocessing is done inside ``DataLoader.swift``.

.. code-block:: swift

    // prepare train dataset
    let trainBatchProvider = DataLoader.trainBatchProvider() { _ in }

    // prepare test dataset
    let testBatchProvider = DataLoader.testBatchProvider() { _ in }

    // load them together
    let dataLoader = MLDataLoader(trainBatchProvider: trainBatchProvider,
                                  testBatchProvider: testBatchProvider)

Since CoreML does not allow the model parameters to be seen before training, and
accessing the model parameters during or after the training can only be done by
specifying the layer name, we need to know this information beforehand, through looking
at the model specification, which are written as proto files. The implementation can be
seen in ``MLModelInspect``.

After we have all of the necessary information, let's create our Flower client.

.. code-block:: swift

    let compiledModelUrl = try MLModel.compileModel(at: url)

    // inspect the model to be able to access the model parameters
    // to access the model we need to know the layer name
    // since the model parameters are stored as key value pairs
    let modelInspect = try MLModelInspect(serializedData: Data(contentsOf: url))
    let layerWrappers = modelInspect.getLayerWrappers()
    self.mlFlwrClient = MLFlwrClient(layerWrappers: layerWrappers,
                                     dataLoader: dataLoader,
                                     compiledModelUrl: compiledModelUrl)

Then start the Flower gRPC client and start communicating to the server by passing our
Flower client to the function ``startFlwrGRPC``.

.. code-block:: swift

    self.flwrGRPC = FlwrGRPC(serverHost: hostname, serverPort: port)
    self.flwrGRPC.startFlwrGRPC(client: self.mlFlwrClient)

That's it for the client. We only have to implement ``Client`` or call the provided
``MLFlwrClient`` and call ``startFlwrGRPC()``. The attribute ``hostname`` and ``port``
tells the client which server to connect to. This can be done by entering the hostname
and port in the application before clicking the start button to start the federated
learning process.

***************
 Flower Server
***************

For simple workloads we can start a Flower server and leave all the configuration
possibilities at their default values. In a file named ``server.py``, import Flower and
start the server:

.. code-block:: python

    import flwr as fl

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))

*****************************
 Train the model, federated!
*****************************

With both client and server ready, we can now run everything and see federated learning
in action. FL systems usually have a server and multiple clients. We therefore have to
start the server first:

.. code-block:: shell

    $ python server.py

Once the server is running we can start the clients in different terminals. Build and
run the client through your Xcode, one through Xcode Simulator and the other by
deploying it to your iPhone. To see more about how to deploy your app to iPhone or
Simulator visit `here
<https://developer.apple.com/documentation/xcode/running-your-app-in-simulator-or-on-a-device>`_.

Congratulations! You've successfully built and run your first federated learning system
in your ios device. The full `source code
<https://github.com/flwrlabs/flower/blob/main/examples/ios>`_ for this example can be
found in ``examples/ios``.
