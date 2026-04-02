# Setting up your Embedded Device

> [!NOTE]
> This guide is applicable to many embedded devices such as Raspberry Pi. This guide assumes you have a fresh install of Raspberry Pi OS Lite or Ubuntu Server (e.g. 22.04) and that you have successfully `ssh`-ed into your device.

## Setting up your device for Python developemnet

We are going to use [`pyenv`](https://github.com/pyenv/pyenv) to manage different Python versions and to create an environment. First, we need to install some system dependencies

```shell
sudo apt-get update
# Install python deps relevant for this and other examples
sudo apt-get install build-essential zlib1g-dev libssl-dev \
                libsqlite3-dev libreadline-dev libbz2-dev \
                git libffi-dev liblzma-dev libsndfile1 -y

# Install some good to have
sudo apt-get install htop tmux -y

# Add mouse support for tmux
echo "set-option -g mouse on" >> ~/.tmux.conf
```

It is recommended to work on virtual environments instead of in the global Python environment. Let's install `pyenv` with the `virtualenv` plugin.

### Install `pyenv` and `virtualenv` plugin

```shell
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Now reload .bashrc
source ~/.bashrc

# Install pyenv virtual env plugin
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
# Restart your shell
exec "$SHELL"
```

## Create a Python environment and activate it

> [!TIP]
> If you are using a Raspberry Pi Zero 2 or another embedded device with a small amount of RAM (e.g. \<1GB), you probably need to extend the size of the SWAP partition. See the guide at the end of this readme.

Now all is ready to create a virtualenvironment. But first, let's install a recent version of Python:

```shell
# Install python 3.10+
pyenv install 3.10.14

# Then create a virtual environment
pyenv virtualenv 3.10.14 my-env
```

Finally, activate your environment and install the dependencies for your project:

```shell
# Activate your environment
pyenv activate my-env

# Then, install flower
pip install flwr

# Install any other dependency needed for your device
# Likely your embedded device will run a Flower SuperNode
# This means you'll likely want to install dependencies that
# your Flower `ClientApp` needs.

pip install <your-clientapp-dependencies>
```

## Extening SWAP for `RPi Zero 2`

> [!NOTE]
> This mini-guide is useful if your RPi Zero 2 cannot complete installing some packages (e.g. TensorFlow or even Python) or do some processing due to its limited RAM.

A workaround is to create a `swap` disk partition (non-existant by default) so the OS can offload some elements to disk. I followed the steps described [in this blogpost](https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04) that I copy below. You can follow these steps if you often see your RPi Zero running out of memory:

```shell
# Let's create a 1GB swap partition
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
# Enable swap
sudo swapon /swapfile # you should now be able to see the swap size on htop.
# make changes permanent after reboot
sudo cp /etc/fstab /etc/fstab.bak
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Please note using swap as if it was RAM comes with a large penalty in terms of data movement.
