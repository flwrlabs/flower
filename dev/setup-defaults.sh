#!/bin/bash
set -e

version=${1:-3.11.14}

# To install pyenv and virtualenv plugin
function install_pyenv(){
    curl https://pyenv.run | bash
}

if [ ! -d $HOME/.pyenv ]
then
    # Install pyenv with the virtualenv plugin
    echo 'Installing pyenv...'
    install_pyenv &>/dev/null

    # To add the config to the right file (depends on the shell used)
    rcfile=$HOME/.$(basename $SHELL)rc

    # Add $PYENV_ROOT environment variable
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $rcfile
    # Add pyenv to $PATH
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> $rcfile

    # Init pyenv with the shell
    echo 'eval "$(pyenv init -)"' >> $rcfile
    echo 'eval "$(pyenv virtualenv-init -)"' >> $rcfile
else
    [[ ! $PYENV_ROOT ]] && echo "You must restart your shell for env variables to be set" && exit

    # If pyenv is already installed, check for a newer version
    read -p 'Pyenv already installed, do you want to updating it y/[n]? ' update
    update="${update:-"n"}"

    if [ $update == "y" ]
    then
        # If the pyenv-update plugin isn't installed do the update manually
        if [ ! -d $HOME/.pyenv/plugins/pyenv-update ]
        then
            if [ ! -d $HOME/.pyenv/.git ]
            then
                echo "Couldn't perform the update, continuing..."
            else
                git -C $HOME/.pyenv pull &>/dev/null
                git -C $HOME/.pyenv/plugins/pyenv-virtualenv pull &>/dev/null
            fi
        else
            pyenv update &>/dev/null
        fi
    fi
fi

# Create the virtual environment for Flower
function create_venv(){
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    $( dirname "${BASH_SOURCE[0]}" )/venv-create.sh $version
}

if [ ! -d $HOME/.pyenv/versions/flower-$version ]
then
    echo 'Creating the virtual environment for Flower...'
    create_venv &>/dev/null
else
    echo 'Virtual env already installed, nothing to do.'
    echo "If not already done, "\
    "you must run dev/bootstrap.sh $version to install all the dependencies"
    exit
fi

echo "$(tput bold)Virtual env flower-$version created, "\
"you must now run dev/bootstrap.sh $version to install all dependencies.$(tput sgr0)"

exec "$SHELL"
