# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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

# Multi-stage base image build
#
# Building the base image for ARM64 requires installing some additional system dependencies to
# compile the grpcio package, as they do not provide a pre-built package. However, we don't want
# the dependencies in the final base image as they are only needed to compile the package.
# That's why we're creating a multi-stage build. When installing the flwr, we are using a
# virtual environment to keep all files in a single isolated directory as described here:
# https://pythonspeed.com/articles/multi-stage-docker-python/

# hadolint global ignore=DL3018
ARG PYTHON_VERSION=3.12
ARG DISTRO=alpine
ARG DISTRO_VERSION=3.22
FROM python:${PYTHON_VERSION}-${DISTRO}${DISTRO_VERSION} AS compile

# Install system dependencies
RUN apk add --no-cache \
    # required for compiling grpcio on ARM64
    g++ \
    libffi-dev \
    # required for installing flwr via git
    git \
    # create virtual env
    && python -m venv /python/venv

# Make sure we use the virtualenv
ENV PATH=/python/venv/bin:$PATH

# Install specific version of pip, setuptools and flwr
ARG PIP_VERSION
ARG SETUPTOOLS_VERSION
RUN pip install -U --no-cache-dir \
    pip==${PIP_VERSION} \
    setuptools==${SETUPTOOLS_VERSION}

ARG FLWR_VERSION
ARG FLWR_VERSION_REF
ARG FLWR_PACKAGE=flwr
# hadolint ignore=DL3013
RUN if [ -z "${FLWR_VERSION_REF}" ]; then \
    pip install -U --no-cache-dir ${FLWR_PACKAGE}==${FLWR_VERSION}; \
    else \
    pip install -U --no-cache-dir ${FLWR_PACKAGE}@${FLWR_VERSION_REF}; \
    fi

FROM python:${PYTHON_VERSION}-${DISTRO}${DISTRO_VERSION} AS base

# Keep the version of system Python pip and setuptools in sync with those installed in the
# virtualenv.
ARG PIP_VERSION
ARG SETUPTOOLS_VERSION
RUN pip install -U --no-cache-dir pip==${PIP_VERSION} setuptools==${SETUPTOOLS_VERSION}

# required by the grpc package
RUN apk add --no-cache \
    libstdc++ \
    ca-certificates \
    # add non-root user
    && adduser \
    --no-create-home \
    --home /app \
    --disabled-password \
    --gecos "" \
    --uid 49999 app \
    && mkdir -p /app \
    && chown -R app:app /app

COPY --from=compile --chown=app:app /python/venv /python/venv

# Make sure we use the virtualenv
ENV PATH=/python/venv/bin:$PATH \
    # Send stdout and stderr stream directly to the terminal. Ensures that no
    # output is retained in a buffer if the application crashes.
    PYTHONUNBUFFERED=1 \
    # Typically, bytecode is created on the first invocation to speed up following invocation.
    # However, in Docker we only make a single invocation (when we start the container).
    # Therefore, we can disable bytecode writing.
    PYTHONDONTWRITEBYTECODE=1 \
    # Ensure that python encoding is always UTF-8.
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app
USER app
ENV HOME=/app
