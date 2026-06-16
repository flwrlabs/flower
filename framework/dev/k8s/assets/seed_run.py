# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import hashlib
from pathlib import Path

import grpc

from flwr.cli.build import build_fab_from_disk
from flwr.common.serde import fab_to_proto
from flwr.proto.control_pb2 import StartRunRequest
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.constant import NOOP_FEDERATION
from flwr.supercore.fab import Fab

_PROBE_APP_DIR = Path("/opt/flower-local-k8s/probe_app")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-api-address", required=True)
    args = parser.parse_args()

    fab_bytes = build_fab_from_disk(_PROBE_APP_DIR)
    fab_hash = hashlib.sha256(fab_bytes).hexdigest()
    channel = grpc.insecure_channel(args.control_api_address)
    grpc.channel_ready_future(channel).result(timeout=60)
    stub = ControlStub(channel)
    response = stub.StartRun(
        StartRunRequest(
            fab=fab_to_proto(Fab(fab_hash, fab_bytes, {})),
            federation=NOOP_FEDERATION,
        )
    )
    if not response.HasField("run_id"):
        raise RuntimeError("Control API did not return a run_id")
    print(f"K8s launch seed created run_id={response.run_id}")


if __name__ == "__main__":
    main()
