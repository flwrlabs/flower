"""ServerApp for heartbeat E2E tests."""

import time

import flwr as fl

app = fl.serverapp.ServerApp()


@app.main()
def main(grid, context):
    """Keep the task alive long enough for heartbeat interruption tests."""
    print("Sleep for 30 seconds")
    time.sleep(30)
    print("Done sleeping")
