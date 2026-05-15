"""Subprocess executor for SuperExec TaskExecutor processes."""

import os
import subprocess

from .types import ExecutionSpec


class SubprocessExecutor:
    """Run TaskExecutor processes as local subprocesses."""

    def launch(self, spec: ExecutionSpec) -> None:
        """Start the TaskExecutor process described by the execution spec."""
        args = [
            spec.command,
            f"--{spec.appio_api_kind}-api-address",
            spec.appio_api_address,
            "--token",
            spec.token,
        ]

        if spec.insecure:
            args.append("--insecure")
        elif spec.root_certificates_path is not None:
            args.extend(["--root-certificates", spec.root_certificates_path])

        if spec.parent_pid is not None:
            args.extend(["--parent-pid", str(os.getpid())])

        if spec.runtime_dependency_install:
            args.append("--runtime-dependency-install")

        if spec.suppress_output:
            subprocess.Popen(  # pylint: disable=consider-using-with
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        subprocess.Popen(args)  # pylint: disable=consider-using-with
