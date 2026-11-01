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

# Install the certificate bundle required to download the managed Python build.
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y --no-install-recommends install \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python from uv's managed, prebuilt CPython distributions.
COPY --from=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /uvx /bin/

ARG PYTHON_VERSION=3.13
ENV UV_PYTHON_INSTALL_DIR=/usr/local/bin/python \
    UV_PYTHON_BIN_DIR=/usr/local/bin/python/bin \
    UV_PYTHON_INSTALL_BIN=1 \
    PATH=/usr/local/bin/python/bin:$PATH
RUN uv python install --managed-python --default --no-cache "${PYTHON_VERSION}"

ARG PIP_VERSION
ARG SETUPTOOLS_VERSION
# Use a virtual environment so packages have the same location regardless of
# whether the subsequent image build runs as root or app.
# Update the bundled installer too, since the managed Python tree is copied
# into the runtime image alongside the virtual environment.
RUN pip_wheel='pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' \
    && test "${PIP_VERSION}" = "26.2.2" \
    && python -m pip install -U --no-cache-dir --break-system-packages "${pip_wheel}" \
    && python -m pip install -U --no-cache-dir --break-system-packages \
    pip==${PIP_VERSION} \
    setuptools==${SETUPTOOLS_VERSION} \
    && python -m venv /python/venv \
    && /python/venv/bin/python -m pip install -U --no-cache-dir "${pip_wheel}" \
    && /python/venv/bin/python -m pip install -U --no-cache-dir \
    pip==${PIP_VERSION} \
    setuptools==${SETUPTOOLS_VERSION}

FROM $DISTRO:$DISTRO_VERSION AS foundation

COPY --from=python /usr/local/bin/python /usr/local/bin/python

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/bin/python/bin:$PATH

RUN apt-get update \
    && apt-get -y upgrade \
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
