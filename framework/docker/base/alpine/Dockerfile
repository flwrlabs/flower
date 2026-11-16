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
ARG PYTHON_VERSION=3.11
ARG DISTRO=alpine
ARG DISTRO_VERSION=3.23
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
RUN pip_wheel='pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' \
    && test "${PIP_VERSION}" = "26.2.2" \
    && python -m pip install -U --no-cache-dir "${pip_wheel}" \
    && python -m pip install -U --no-cache-dir setuptools==${SETUPTOOLS_VERSION}

ARG FLWR_VERSION
ARG FLWR_VERSION_REF
ARG FLWR_PACKAGE=flwr
# hadolint ignore=DL3013
RUN if [ -z "${FLWR_VERSION_REF}" ]; then \
    pip install -U --no-cache-dir ${FLWR_PACKAGE}==${FLWR_VERSION}; \
    else \
    pip install -U --no-cache-dir ${FLWR_PACKAGE}@${FLWR_VERSION_REF}; \
    fi \
    # Keep pip and its bundled dependencies out of the runtime virtualenv.
    && python -m pip uninstall --yes pip

FROM python:${PYTHON_VERSION}-${DISTRO}${DISTRO_VERSION} AS base

# Keep system setuptools aligned with the virtualenv, then remove system pip
# and the bundled wheel that could reinstall it.
ARG PIP_VERSION
ARG SETUPTOOLS_VERSION
RUN pip_wheel='pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' \
    && test "${PIP_VERSION}" = "26.2.2" \
    && python -m pip install -U --no-cache-dir "${pip_wheel}" \
    && python -m pip install -U --no-cache-dir setuptools==${SETUPTOOLS_VERSION} \
    && python -m pip uninstall --yes pip \
    && rm -rf /usr/local/lib/python*/ensurepip

# Upgrade inherited packages before adding runtime dependencies.
RUN apk upgrade --no-cache \
    && apk add --no-cache \
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
