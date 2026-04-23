---
ep-number: 0002
title: Decouple Parameter Serialization
authors: ["@orlandohohmeier"]
creation-data: 2022-03-10
status: provisional
---

# Decouple Parameter Serialization

## Table of Contents

- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
- [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)
- [Additional Remarks](#additional-remarks)

## Summary

Ease the implementation of Flower clients in languages other than Python to enable broad FL use-cases by supporting different parameter formats across strategies.

## Motivation

Flower currently uses the [NumPy NYP](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html) format to represent tensors (parameter values). The format includes all of the necessary information to reconstruct arrays, including shape and dtype on a machine of a different architecture. Re-implementing the NYP format in other languages and for architectures is a non-trivial task and impedes adoption beyond Python clients. Platform-specific strategies like [FedAvgAndroid](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg_android.py) to support formats other than NYP hinder experimentation with different strategies and emphasize the need for a better mechanism.

### Goals

- Ease the implementation of Flower clients in languages other than Python.
- Support different platforms and architectures without the need for platform-specific strategies.

### Non-Goals

- Develop a universal data exchange format for Flower as it's not feasible to track all future data type advancements in the format, i.e., btfloat16.
- Develop a mechanism to negotiate supported value types between clients and servers – this is a topic for future work and not required to achieve the goals above.
- Implement a plugin mechanism to discover and load serializers autmoatically.

## Proposal

Introduce a mechanism to allow the use of different `Parameters` exchange formats in Flower without the need for implementing platform-specific strategies. Flower will by default, ship with two `Parameters` serializers supporting NumPy NYP and a custom value format utilizing Protobuffers to solve platform/architecture compatibility issues.

The mechanism will consist of an abstract Serializer interface and a concrete implementation of the interface for each supported format. Serializers will be injected into strategies so that they can be used to serialize and deserialize `Parameters`. This moves control over the supported Serializers to the strategy and client code. The server will not need to know the serialized `Parameters` format.

### Parameter types

The type property is used to determine the format of a value and how to serialize and deserialize it SHOULD describe both the value format as well as the data type and needs to be unique for each input/output pair.

Example:

- `numpy.ndarray` - NumPy NYP Archive represnetaion for a multi-dimensional array. Serialized from and deserialized to `List[np.ndarray]`.
- `proto.ndarray` - Protobuffer representation for a multi-dimensional array. Serialized from and deserialized to `List[np.ndarray]`.
- `proto.tftensor` - Protobuffer representation for a single TensorFlow tensor. Serialized from and deserialized to TensorFlow Tensor.
- ...

N.B. Only `numpy.ndarray` and `proto.ndarray` are supported out of the box.

### Abstract Serializer

Abstract base class for serializer implementations. `can_handle` method is used to determine if a serializer can handle a given `Parameters` type. `serialize` and `deserialize` methods are used to serialize and deserialize `Parameters`.

```Python
from abc import ABC, abstractmethod
import string
from typing import List, Any
import numpy as np

class Serializer(ABC):
    """Abstract base class for serializer implementations."""

    @abstractmethod
    def can_handle(self, type: string) -> bool:
    """Return True if this serializer can handle the given type."""

    @abstractmethod
    def serialize(self, parameters: List[Any]) -> Parameters:
        """Serialize the given values into a Parameters."""

    @abstractmethod
    def deserialize(self, parameters: Parameters) -> List[Any]:
        """Deserialize the given Parameters into a values."""

```

Considerations:

- Strategies MUST not be bound to a `np.ndarray` or any other parameter value type for maximum flexibility, in line with the current design.
- The `can_handle` method is used to determine if a serializer can handle a given `Parameters` type and will eventually be used to determine which serializer to use during client/server negotiation.
- `Parameters` may be replaced with a more general type to use the same serializers for different messages i.e. `PropertiesRes` bytes.

### NumPy NDArray Serializer

The NumPy NDArray serializer is a direct port of the current functions (see `src/py/flwr/common/parameter.py`) used to serialize and deserialize `np. ndarray's from and to NumPy NYP format.

```Python
from io import BytesIO
from typing import cast
import numpy as np

def ndarray_to_nyp(self, value: np.ndarray) -> bytes:
    """Serialize NumPy ndarray to NYP archive."""
    bytes_io = BytesIO()

    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, value, allow_pickle=False)

    return bytes_io.getvalue()

def nyp_to_ndarray(self, value: bytes) -> np.ndarray:
    """Deserialize NumPy ndarray from NYP archive."""
    bytes_io = BytesIO(value)

    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)

    return cast(np.ndarray, ndarray_deserialized)

class NumPyNDArraySerializer(Serializer):
    """NumPy NYP <> NDArray Serializer."""

    def can_handle(self, type: str) -> bool:
        return type == 'numpy.ndarray'

    def serialize(self, parameters: List[np.ndarray]) -> Parameters:
        return Parameters(tensor_type="numpy.ndarray", tensors=[ndarray_to_nyp(parameter) for parameter in parameters])

    def deserialize(self, parameters: Parameters) -> List[np.ndarray]:
        if not self.can_handle(parameters.tensor_type):
            raise ValueError("Unsupported type: {}".format(parameters.tensor_type))

        return [nyp_to_ndarray(value) for value in parameters.tensors]

```

### Protobuf Serializer

The protobuf serializer will be used to serialize and deserialize `np.ndarray` into a new `Value` proto message, which only uses type-specific representations for maximum compatibility.

```python
def ndarray_to_proto(ndarray: np.ndarray) -> Value:
    """Serialize NumPy ndarray to Protobuf."""

    # TODO: Add support for other dtypes
    return Value(dtype=Value.DT_DOUBLE,
                 double=ndarray.reshape(-1, order="c"), shape=ndarray.shape)


def proto_to_ndarry(value: Value) -> Value:
    """Serialize NumPy ndarray to Protobuf."""

    # TODO: Add support for other dtypes
    return np.array(value.double).reshape(value.shape, order="c")


class ProtoNDArraySerializer(Serializer):
    """Protobuf <> NDArray Serializer."""

    def can_handle(self, type: str) -> bool:
        return type == 'proto.ndarray'

    def serialize(self, parameters: List[np.ndarray]) -> Parameters:
        return Parameters(tensor_type="proto.ndarray",
                          values=[ndarray_to_proto(parameter) for parameter in parameters])

    def deserialize(self, parameters: Parameters) -> List[np.ndarray]:
        if not self.can_handle(parameters.tensor_type):
            raise ValueError("Unsupported type: {}".format(parameters.tensor_type))

        return [proto_to_ndarry(value) for value in parameters.values]
```

#### Value message

```proto
message Parameters {
  repeated bytes tensors = 1;
  string tensor_type = 2;
  repeated Value values = 3;
}

message Value {
  // Only one of the value representations can be set and must match the "dtype".
  //
  // The message format is similar to the ones foiund at:
  // - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
  // - https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
  //
  //It's not using `oneof` as that would require extra messages and
  // compute when serializing/deserializing.
  //
  // Value representations hold the flattened data in row-major order.

  enum DataType {
      DT_DOUBLE = 0;
      // DT_FLOAT = 1;
      // DT_INT32 = 2;
      // DT_INT64 = 3;
      // DT_UINT32 = 4;
      // DT_UINT64 = 5;
      // DT_SINT32 = 6;
      // DT_SINT64 = 7;
      // DT_FIXED32 = 8;
      // DT_FIXED64 = 9;
      // DT_SFIXED32 = 10;
      // DT_SFIXED64 = 11;
      // DT_BOOL = 12;
      DT_STRING = 13;
  }

  DataType dtype = 1;

   // DT_DOUBLE
  repeated double double = 2 [packed = true];
  // DT_STRING
  repeated string string = 15;

  // Commented-out types are listed for reference and
  // might be enabled in future releases. Source:
  // https://developers.google.com/protocol-buffers/docs/proto3#scalar
  // repeated float float = 3 [packed = true];
  // repeated int32 int32 = 4 [packed = true];
  // repeated int64 int64 = 5 [packed = true];
  // repeated uint32 uint32 = 6 [packed = true];
  // repeated uint64 uint64 = 7 [packed = true];
  // repeated sint32 sint32 = 8 [packed = true];
  // repeated sint64 sint64 = 9 [packed = true];
  // repeated fixed32 fixed32 = 10 [packed = true];
  // repeated fixed64 fixed64 = 11 [packed = true];
  // repeated sfixed32 sfixed32 = 12 [packed = true];
  // repeated sfixed64 sfixed64 = 13 [packed = true];
  // repeated bool bool = 14 [packed = true];

  // Shape of the value
  repeated int64 shape = 16 [packed = true];

  // Optional name to identify the value.
  string name = 17;
}
```

The Value message is included in the transport proto to remove the need for extra serialization from/to string messages.

## Drawbacks

The current design builds on the assumption that strategies may use operate on data other then multi-dimensional arrays, and thus, the serialization/deserialization must be owned by the strategy. Breaking with this assumption would allow for a much simpler design where serialization/deserialization is handled by the server instead.

The proposal also only add to the `Parameters` message, but changing for the data format exchanged between strategies and all data between clients and servers is represented as a number of multi-dimensional arrays. Following that assumption, it's rather easy to envision how one might implement type negotiation between clients and servers in the future. However, if this doesn't hold true for all strategies, we would need to adjust the implementation so that serializers get injected into the strategy with strategies announcing the supported types, which in turn would still allow for a general negotiation mechanism in the server.

## Alternatives Considered

Change the Parameters (and properties) proto message as follows and only allow type-specific values.

```proto
message Parameters {
  repeated Value values = 3;
}

```

_N.B. Value refers to the message presented in the above section._

This design has the benefit that it provides more insight into the data exchanged and allows for new features building on that knowledge but reduces flexibility and reduces the number of supported data types to the ones part of the protobuf spec.

## Additional Remarks

- There is no added value having Parameters typings; instead the serializer should return proto messages directly. This simplifies the design and removes the need for additional conversion steps.
- The `Prameters` and `Properties` messages should be aligned into a general `Values` message to utilize the same seriliazation/deserialization for all messages. Dicts/Maps can easily be represented as set of key value pairs and thus fit the value (Parameter) format.
- When implementing a mechanism to negotiate the supported types between clients and servers, the grpc metadata should be used to pass the supported types.
