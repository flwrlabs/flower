# Python foundation images

This directory contains reusable Python runtime foundations for framework
container images.

The Ubuntu foundation image:

- compiles the requested Python version with PyEnv;
- installs pinned pip and setuptools versions;
- creates the shared Python virtual environment; and
- includes the runtime libraries and non-root app user needed by downstream
  images.

The Dockerfile is platform-neutral and can be built for both Linux AMD64 and
Linux ARM64 with Docker Buildx.
