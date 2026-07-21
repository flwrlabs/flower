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
# ===============================================================================

# hadolint global ignore=DL3008
ARG DISTRO=ubuntu
ARG DISTRO_VERSION=24.04
FROM $DISTRO:$DISTRO_VERSION AS python

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required to compile Python.
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    clang-format git unzip ca-certificates openssh-client liblzma-dev \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev wget \
    libsqlite3-dev curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyEnv and Python.
ARG PYTHON_VERSION=3.13
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PATH
# https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# hadolint ignore=DL3003
RUN git clone https://github.com/pyenv/pyenv.git \
    && cd pyenv/plugins/python-build || exit \
    && ./install.sh

# Issue: python-build only accepts the exact Python version e.g. 3.13.1 but
# we want to allow more general versions like 3.13.
# Solution: first use pyenv to get the exact version and then pass it to python-build.
RUN LATEST=$(pyenv latest -k ${PYTHON_VERSION}) \
    && python-build "${LATEST}" /usr/local/bin/python

ENV PATH=/usr/local/bin/python/bin:$PATH

ARG PIP_VERSION
ARG SETUPTOOLS_VERSION
# Keep the versions of pip and setuptools in sync with those installed in the
# virtual environment.
RUN pip install -U --no-cache-dir pip==${PIP_VERSION} setuptools==${SETUPTOOLS_VERSION} \
    # Use a virtual environment so packages have the same location regardless
    # of whether the subsequent image build runs as root or app.
    && python -m venv /python/venv \
    && /python/venv/bin/python -m pip install -U --no-cache-dir \
    pip==${PIP_VERSION} \
    setuptools==${SETUPTOOLS_VERSION}

FROM $DISTRO:$DISTRO_VERSION AS foundation

COPY --from=python /usr/local/bin/python /usr/local/bin/python

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/bin/python/bin:$PATH

RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    libsqlite3-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    # Add a non-root user for downstream framework images.
    && useradd \
    --no-create-home \
    --home-dir /app \
    -c "" \
    --uid 49999 app \
    && mkdir -p /app \
    && chown -R app:app /app

COPY --from=python --chown=app:app /python/venv /python/venv

ENV PATH=/python/venv/bin:$PATH \
    # Send stdout and stderr directly to the terminal.
    PYTHONUNBUFFERED=1 \
    # Docker images normally invoke Python once, so do not write bytecode.
    PYTHONDONTWRITEBYTECODE=1 \
    # Ensure that Python encoding is always UTF-8.
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app
USER app
ENV HOME=/app
