---
fed-number: 0002
title: secure aggregation
authors: ["@FANTOME-PAN"]
creation-date: 2023-04-25
last-updated: 2023-06-19
status: provisional
---

# FED Template

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Proposal](#proposal)
  - [Data types for SA](#data-types-for-sa)
  - [Server-side components](#server-side-components)
  - [Client-side components](#client-side-components)

## Summary

The current Flower framework does not have built-in modules for Secure Aggregation (SA).
However, flower users may want to use SA in their FL solutions or
implement their own SA protocols easily.

Based on the previous SA implementation, I intend to build the SA 
for flower on the Driver API.



## Proposal

### Data types for SA

Judging from the SecAgg protocol, the SecAgg+ protocol, the LightSecAgg protocol,
and the FastSecAgg protocol, the following fields can better facilitate
SA implementations.

1. bytes, List of bytes

    SA protocols often use encryption and send ciphertext in bytes.
Besides, cryptography-related information, such as public keys, are normally stored as bytes.
Sharing these info will require transmitting bytes.

    Currently, both FitIns and FitRes contain one dictionary field,
mapping strings to scalars (including bytes).
    Though it is possible to store lists of bytes in the dictionary using tricks,
it can be easier to implement SA if TaskIns and TaskRes have fields supporting bytes and lists of bytes

2. arrays

    In many protocols, the server and the clients need to send 
additional but necessary information to complete SA.
These info are usually single or multiple lists of integers or floats.
We now need to store them in the parameters field.


Considering all above, if possible, I would suggest adding a more general dictionary field,
i.e., `Dict[str, Value]`,
where `Value` can be a scalar or a list of scalars.
(As defined in `transport.proto`, Scalar means `Union[double, int, bool, str, bytes]`. 
In the current design, flower users would need to manually convert model parameters to bytes, 
as NDArrays is no longer a supported data type in this message.)


Example code:

``` protobuf
message Task {
  Node producer = 1;
  Node consumer = 2;
  string created_at = 3;
  string delivered_at = 4;
  string ttl = 5;
  repeated string ancestry = 6;
  SecureAggregation sa = 7;

  ServerMessage legacy_server_message = 101 [ deprecated = true ];
  ClientMessage legacy_client_message = 102 [ deprecated = true ];
}

message Value {
  message DoubleList { repeated double vals = 1; }
  message Sint64List { repeated sint64 vals = 1; }
  message BoolList { repeated bool vals = 1; }
  message StringList { repeated string vals = 1; }
  message BytesList { repeated bytes vals = 1; }

  oneof value {
    // Single element
    double double = 1;
    sint64 sint64 = 2;
    bool bool = 3;
    string string = 4;
    bytes bytes = 5;

    // List types
    DoubleList double_list = 21;
    Sint64List sint64_list = 22;
    BoolList bool_list = 23;
    StringList string_list = 24;
    BytesList bytes_list = 25;
  }
}

message SecureAggregation { map<string, Value> named_values = 1; }
```


### Server-side components

The server actively coordinates the SA protocols.
Its responsibilities include:
1. help broadcast SA configs for initialisation.
2. forward messages from one client to another.
3. gathering information from clients to obtain aggregate output.

In short, other then serving as a relay, the server is a controller and decryptor.
It controls the workflow. Since SA protocols are rather different from each other,
we may want to allow customising the workflow, i.e., allowing users to define 
arbitrary rounds of communication in a single FL fit round.

Example code as follows.


**user_messages.py** :
```python
from dataclasses import dataclass
from typing import List, Dict, Union, Optional

import numpy as np

from flwr.common import Scalar


@dataclass
class FitIns:
    parameters: List[np.ndarray]
    config: Dict[str, Scalar]


@dataclass
class FitRes:
    parameters: List[np.ndarray]
    num_examples: int
    metrics: Dict[str, Scalar]


@dataclass
class EvaluateIns:
    parameters: List[np.ndarray]
    config: Dict[str, Scalar]


@dataclass
class EvaluateRes:
    loss: float
    num_examples: int
    metrics: Dict[str, Scalar]


@dataclass
class ServerMessage:
    fit_ins: Optional[FitIns] = None
    evaluate_ins: Optional[EvaluateIns] = None


@dataclass
class ClientMessage:
    fit_res: Optional[FitRes] = None
    evaluate_res: Optional[EvaluateRes] = None


@dataclass
class SecureAggregationMessage:
    named_arrays: Dict[str, Union[np.ndarray, List[np.ndarray]]] = None
    named_bytes: Dict[str, Union[bytes, List[bytes]]] = None
    named_scalars: Dict[str, Union[Scalar, List[Scalar]]] = None


@dataclass
class Task:
    legacy_server_message: Optional[ServerMessage] = None
    legacy_client_message: Optional[ClientMessage] = None
    secure_aggregation_message: Optional[SecureAggregationMessage] = None
```


**workflows.py** :
```python
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict, Generator

from flwr.common import Parameters


def workflow_without_sec_agg(parameters: Parameters, sampled_node_ids: List[int]) \
        -> Generator[Dict[int, Task], Dict[int, Task], None]:
    # configure fit
    fit_ins = FitIns(parameters=parameters, config={})
    task = Task(legacy_server_message=ServerMessage(fit_ins=fit_ins))
    yield {node_id: task for node_id in sampled_node_ids}

    # aggregate fit
    node_messages: Dict[int, Task] = yield
    print(f'updating parameters with received messages {node_messages}...')
    # todo
    
    
def workflow_with_sec_agg(parameters: Parameters, sampled_node_ids: List[int]) \
        -> Generator[Dict[int, Task], Dict[int, Task], None]:
            
    yield request_keys_ins(sampled_node_ids)
    
    node_messages: Dict[int, Task] = yield
    yield share_keys_ins(node_messages)
    
    node_messages: Dict[int, Task] = yield
    yield request_parameters_ins(node_messages)
    
    node_messages: Dict[int, Task] = yield
    yield request_key_shares_ins(sampled_node_ids, node_messages)
    
    node_messages: Dict[int, Task] = yield
    print(f'trying to decrypt and update parameters...')
    # todo
```

**driver.py** :

```python
import random
import time
from typing import List, Dict

from flwr.common import ndarrays_to_parameters
from flwr.driver import Driver
from flwr.proto import driver_pb2, task_pb2, node_pb2
from task import Net, get_parameters
from workflows import workflow_with_sec_agg


def user_task_to_proto(task: Task) -> task_pb2.Task:
    ...


def user_task_from_proto(proto: task_pb2.Task) -> Task:
    ...


# -------------------------------------------------------------------------- Driver SDK
driver = Driver(driver_service_address="0.0.0.0:9091", certificates=None)
# -------------------------------------------------------------------------- Driver SDK

anonymous_client_nodes = True
num_client_nodes_per_round = 1
sleep_time = 1
num_rounds = 1
parameters = ndarrays_to_parameters(get_parameters(net=Net()))
workflow = workflow_with_sec_agg(None)  # should specify a strategy instance

# -------------------------------------------------------------------------- Driver SDK
driver.connect()
# -------------------------------------------------------------------------- Driver SDK

for server_round in range(num_rounds):
    print(f"Commencing server round {server_round + 1}")

    # List of sampled node IDs in this round
    sampled_node_ids: List[int] = []

    # Sample node ids
    if anonymous_client_nodes:
        # If we're working with anonymous clients, we don't know their identities, and
        # we don't know how many of them we have. We, therefore, have to assume that
        # enough anonymous client nodes are available or become available over time.
        #
        # To schedule a TaskIns for an anonymous client node, we set the node_id to 0
        # (and `anonymous` to True)
        # Here, we create an array with only zeros in it:
        sampled_node_ids = [0] * num_client_nodes_per_round
    else:
        # If our client nodes have identiy (i.e., they are not anonymous), we can get
        # those IDs from the Driver API using `get_nodes`. If enough clients are
        # available via the Driver API, we can select a subset by taking a random
        # sample.
        #
        # The Driver API might not immediately return enough client node IDs, so we
        # loop and wait until enough client nodes are available.
        while True:
            # Get a list of node ID's from the server
            get_nodes_req = driver_pb2.GetNodesRequest()

            # ---------------------------------------------------------------------- Driver SDK
            get_nodes_res: driver_pb2.GetNodesResponse = driver.get_nodes(
                req=get_nodes_req
            )
            # ---------------------------------------------------------------------- Driver SDK

            all_node_ids: List[int] = get_nodes_res.node_ids
            print(f"Got {len(all_node_ids)} node IDs")

            if len(all_node_ids) >= num_client_nodes_per_round:
                # Sample client nodes
                sampled_node_ids = random.sample(
                    all_node_ids, num_client_nodes_per_round
                )
                break

            time.sleep(3)

    # Log sampled node IDs
    print(f"Sampled {len(sampled_node_ids)} node IDs: {sampled_node_ids}")
    time.sleep(sleep_time)

    node_responses = sampled_node_ids

    while True:
        try:
            ins: Dict[int, Task] = workflow.send(node_messages)
            next(workflow)
        except StopIteration:
            break
        task_ins_list: List[task_pb2.TaskIns] = []
        # Schedule a task for all sampled nodes
        for node_id, user_task in ins.items():
            new_task = user_task_to_proto(user_task)
            new_task_ins = task_pb2.TaskIns(
                task_id="",  # Do not set, will be created and set by the DriverAPI
                group_id="",
                workload_id="",
                task=task_pb2.Task(
                    producer=node_pb2.Node(
                        node_id=0,
                        anonymous=True,
                    ),
                    consumer=node_pb2.Node(
                        node_id=node_id,
                        anonymous=anonymous_client_nodes,  # Must be True if we're working with anonymous clients
                    ),
                    legacy_server_message=new_task.legacy_server_message,
                    sec_agg=new_task.sec_agg
                ),
            )
            task_ins_list.append(new_task_ins)

        push_task_ins_req = driver_pb2.PushTaskInsRequest(task_ins_list=task_ins_list)

        # ---------------------------------------------------------------------- Driver SDK
        push_task_ins_res: driver_pb2.PushTaskInsResponse = driver.push_task_ins(
            req=push_task_ins_req
        )
        # ---------------------------------------------------------------------- Driver SDK

        print(
            f"Scheduled {len(push_task_ins_res.task_ids)} tasks: {push_task_ins_res.task_ids}"
        )

        time.sleep(sleep_time)

        # Wait for results, ignore empty task_ids
        task_ids: List[str] = [
            task_id for task_id in push_task_ins_res.task_ids if task_id != ""
        ]
        all_task_res: List[task_pb2.TaskRes] = []
        while True:
            pull_task_res_req = driver_pb2.PullTaskResRequest(
                node=node_pb2.Node(node_id=0, anonymous=True),
                task_ids=task_ids,
            )

            # ------------------------------------------------------------------ Driver SDK
            pull_task_res_res: driver_pb2.PullTaskResResponse = driver.pull_task_res(
                req=pull_task_res_req
            )
            # ------------------------------------------------------------------ Driver SDK

            task_res_list: List[task_pb2.TaskRes] = pull_task_res_res.task_res_list
            print(f"Got {len(task_res_list)} results")

            time.sleep(sleep_time)

            all_task_res += task_res_list

            # in secure aggregation, this may changed to a timer:
            # when reaching time limit, the server will assume the nodes have lost connection.
            if len(all_task_res) == len(task_ids):
                break

        # "Aggregate" results
        node_responses = {task_res.task.producer: user_task_from_proto(task_res.task) for task_res in all_task_res}
        print(f"Received {len(node_responses)} results")

        time.sleep(sleep_time)

    # Repeat

# -------------------------------------------------------------------------- Driver SDK
driver.disconnect()
# -------------------------------------------------------------------------- Driver SDK
```

### Client-side components

The key responsibilities of a client are:
1. generate (cryptography-related) information
2. sharing information via the server
3. encrypt its output
4. help the server decrypt the aggregate output

In summary, a client is an encryptor. It requires additional information from 
other encryptors for initialisation and also provides other encryptors with its information.
Then, it can independently encrypt its output. 
In the end of the fit round, it provides the server with necessary information that allows
and only allows the server to decrypt aggregate output, learning nothing of individual outputs.

Example code is as follows.

**client_workflow.py**
```python
from abc import abstractmethod
from typing import Generator

import flwr as fl
import user_messages as usr


class ClientWorkflow:
    def __init__(self):
        self.wf = (_ for _ in range(0))

    def handle(self, task: usr.Task) -> usr.Task:
        try:
            next(self.wf)
        except StopIteration:
            self.wf.close()
            self.wf = self.workflow()
            next(self.wf)
        return self.wf.send(task)

    @abstractmethod
    def workflow(self) -> Generator[usr.Task, usr.Task, None]:
        ...


class FitEvalClientWorkflow(ClientWorkflow):

    def workflow(self: fl.client.NumPyClient) -> Generator[usr.Task, usr.Task, None]:
        # fit round
        task: usr.Task = yield
        fit_ins = task.legacy_server_message.fit_ins
        yield usr.ClientMessage(fit_res=usr.FitRes(*self.fit(fit_ins.parameters, fit_ins.config)))

        # eval round
        task: usr.Task = yield
        eval_ins = task.legacy_server_message.evaluate_ins
        yield usr.ClientMessage(fit_res=usr.EvaluateRes(*self.evaluate(eval_ins.parameters, eval_ins.config)))


class SecAggClientWorkFlow(ClientWorkflow):

    def workflow(self: fl.client.NumPyClient) -> Generator[usr.Task, usr.Task, None]:
        # setup configurations for SA and upload own public keys
        task: usr.Task = yield
        yield setup(self, task)

        # receive other public keys and upload own secret key shares
        task: usr.Task = yield
        yield share_keys(self, task)

        # receive other secret key shares, train the model, and upload masked updates
        task: usr.Task = yield
        # need training in parallel
        fit_ins = task.legacy_server_message.fit_ins
        self.fit(fit_ins.parameters, fit_ins.config)
        yield ask_vectors(self, task)

        # receive list of dropped clients and active clients and upload relevant info
        task: usr.Task = yield
        yield unmask_vectors(self, task)
```

**client.py**

```python
import flwr as fl

from client_workflow import FitEvalClientWorkflow, SecAggClientWorkFlow
from task import (
    Net,
    DEVICE,
    load_data,
    get_parameters,
    set_parameters,
    train,
    test,
)


# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient, SecAggClientWorkFlow):
    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader, epochs=1)
        return get_parameters(net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="0.0.0.0:9093",
    client=FlowerClient(),
    rest=True,
)
```
