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
"""Probe ServerApp used by the local k8s launch-path harness."""

import time

import flwr as fl

app = fl.serverapp.ServerApp()
_PROBE_HOLD_SECONDS_CONFIG_KEY = "local-k8s.probe-hold-seconds"
_PROBE_CRASH_CONFIG_KEY = "local-k8s.probe-crash"


@app.main()
def main(grid, context):
    """Run the probe ServerApp and optionally stay active for capacity
    tests."""
    run_id = context.run_id
    print(f"K8s launch probe ServerApp starting run_id={run_id}", flush=True)
    print(f"K8s launch probe ServerApp ran run_id={run_id}", flush=True)
    if context.run_config.get(_PROBE_CRASH_CONFIG_KEY, False):
        print(
            f"K8s launch probe ServerApp crashing run_id={run_id}",
            flush=True,
        )
        raise RuntimeError("Intentional local k8s probe crash")
    hold_seconds = context.run_config.get(_PROBE_HOLD_SECONDS_CONFIG_KEY, 0.0)
    if isinstance(hold_seconds, (float, int)) and hold_seconds > 0:
        print(
            f"K8s launch probe ServerApp sleeping run_id={run_id} "
            f"seconds={float(hold_seconds)}",
            flush=True,
        )
        time.sleep(float(hold_seconds))
    print(f"K8s launch probe ServerApp exiting run_id={run_id}", flush=True)
