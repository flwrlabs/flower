FROM ubuntu:22.04

# https://code.visualstudio.com/docs/remote/containers-advanced#_creating-a-nonroot-user
ARG USERNAME=flwr-vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo bash \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# A persistent volume will be configured in devcontainer.json so we don't loose the commandhistory
# after rebuilding the container
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo $SNIPPET >> "/home/$USERNAME/.bashrc"

# Install system dependencies
RUN apt-get update
RUN apt-get install -y curl wget gnupg python3 python-is-python3 python3-pip git \
    build-essential tmux vim

RUN python -m pip install \
    pip==26.0.1 \
    setuptools==82.0.0 \
    poetry==2.3.2 \
    maturin==1.8.6
# Install taplo from the main branch on GitHub so that maturin can build the wheel
RUN python -m pip install \
    "git+https://github.com/tamasfe/taplo.git@1bb2c1881995a819d2bda7681d145709c783be1b#egg=taplo"

USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
