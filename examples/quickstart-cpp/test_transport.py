"""Cross-language regression tests using real C++ clients and a loopback Fleet."""

import argparse
import hashlib
import struct
import subprocess
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import grpc
from flwr.app.message import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.app.message.arraychunk import ArrayChunk
from flwr.common import FitIns, Parameters
from flwr.common.serde import recorddict_to_proto
from flwr.compat.common.recorddict_compat import (
    fitins_to_recorddict,
    recorddict_to_fitres,
)
from flwr.proto import fleet_pb2, fleet_pb2_grpc, heartbeat_pb2, message_pb2
from flwr.supercore.inflatable.inflatable_object import (
    get_object_children_ids_from_object_content,
    get_object_type_from_object_content,
)

TRANSPORT_CLIENT = Path("build/test_transport_client").resolve()
CPP_CLIENT = Path("build/flwr_client").resolve()


class FleetFixture(fleet_pb2_grpc.FleetServicer):
    """Send one instruction, capture real wire objects, then inject an outage."""

    def __init__(self, tensors=None, fail_registration=False):
        self.tensors = tensors
        self.fail_registration = fail_registration
        self.sent = False
        self.record_id = None
        self.objects = {}
        self.calls = []

    def RegisterNode(self, request, context):
        self.calls.append("register")
        if self.fail_registration:
            context.abort(grpc.StatusCode.UNAVAILABLE, "injected registration failure")
        return fleet_pb2.RegisterNodeFleetResponse(node_id=123)

    def ActivateNode(self, request, context):
        self.calls.append("activate")
        return fleet_pb2.ActivateNodeResponse(node_id=123)

    def SendNodeHeartbeat(self, request, context):
        return heartbeat_pb2.SendNodeHeartbeatResponse(success=True)

    def PullMessages(self, request, context):
        if self.sent or self.tensors is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "injected polling failure")
        self.sent = True
        ins = FitIns(Parameters(tensors=self.tensors, tensor_type="cpp_double"), {})
        message = message_pb2.Message(
            metadata=message_pb2.Metadata(
                run_id=99,
                message_id="test-instruction",
                src_node_id=1,
                dst_node_id=123,
                message_type="train",
                created_at=time.time(),
                ttl=60,
            ),
            content=recorddict_to_proto(fitins_to_recorddict(ins, keep_input=True)),
        )
        return fleet_pb2.PullMessagesResponse(messages_list=[message])

    def PushMessages(self, request, context):
        tree = request.message_object_trees[0]
        self.record_id = tree.children[0].object_id

        def ids(node):
            yield node.object_id
            for child in node.children:
                yield from ids(child)

        return fleet_pb2.PushMessagesResponse(
            objects_to_push=list(dict.fromkeys(ids(tree)))
        )

    def PushObject(self, request, context):
        self.objects[request.object_id] = request.object_content
        return message_pb2.PushObjectResponse(stored=True)

    def DeactivateNode(self, request, context):
        self.calls.append("deactivate")
        return fleet_pb2.DeactivateNodeResponse()

    def UnregisterNode(self, request, context):
        self.calls.append("unregister")
        return fleet_pb2.UnregisterNodeFleetResponse()

    def received_parameters(self):
        classes = {
            cls.__name__: cls
            for cls in (
                Array,
                ArrayRecord,
                ArrayChunk,
                RecordDict,
                MetricRecord,
                ConfigRecord,
            )
        }

        def inflate(object_id):
            content = self.objects[object_id]
            if hashlib.sha256(content).hexdigest() != object_id:
                raise ValueError("Object hash mismatch")
            children = {
                child: inflate(child)
                for child in get_object_children_ids_from_object_content(content)
            }
            cls = classes[get_object_type_from_object_content(content)]
            return cls.inflate(content, children)

        return recorddict_to_fitres(inflate(self.record_id), keep_input=True).parameters


@contextmanager
def serve(fleet):
    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        fleet_pb2_grpc.add_FleetServicer_to_server(fleet, server)
        port = server.add_insecure_port("127.0.0.1:0")
        if not port:
            raise RuntimeError("Could not allocate a loopback test port")
        server.start()
        try:
            yield f"127.0.0.1:{port}"
        finally:
            server.stop(grace=0).wait(timeout=5)


def run_client(binary, *args):
    return subprocess.run(
        [str(binary), *args],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


class TransportTest(unittest.TestCase):
    def test_tensor_order_at_python_receiver(self):
        for count in (1, 2, 9, 10, 11, 12, 21, 100):
            with self.subTest(tensors=count):
                tensors = [struct.pack("<d", float(i)) for i in range(count)]
                fleet = FleetFixture(tensors)
                with serve(fleet) as address:
                    result = run_client(TRANSPORT_CLIENT, address)
                self.assertIsNotNone(fleet.record_id, result.stdout + result.stderr)
                received = fleet.received_parameters()
                self.assertEqual(received.tensors, tensors)
                self.assertEqual(received.tensor_type, "cpp_double")
                self.assertIn("injected polling failure", result.stderr)
                self.assertEqual(fleet.calls[-2:], ["deactivate", "unregister"])

    def test_fatal_polling_error_exits_nonzero(self):
        fleet = FleetFixture()
        with serve(fleet) as address:
            result = run_client(CPP_CLIENT, "0", address)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[flwr-cpp] fatal: PullMessages failed:", result.stderr)
        self.assertIn("injected polling failure", result.stderr)
        self.assertEqual(fleet.calls[-2:], ["deactivate", "unregister"])

    def test_fatal_registration_error_exits_nonzero(self):
        fleet = FleetFixture(fail_registration=True)
        with serve(fleet) as address:
            result = run_client(CPP_CLIENT, "0", address)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[flwr-cpp] fatal: RegisterNode failed:", result.stderr)
        self.assertEqual(fleet.calls, ["register"])

    def test_invalid_arguments_exit_nonzero(self):
        self.assertNotEqual(run_client(CPP_CLIENT).returncode, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-client", type=Path, default=TRANSPORT_CLIENT)
    parser.add_argument("--client", type=Path, default=CPP_CLIENT)
    args, unittest_args = parser.parse_known_args()
    TRANSPORT_CLIENT = args.transport_client.resolve()
    CPP_CLIENT = args.client.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_args])
