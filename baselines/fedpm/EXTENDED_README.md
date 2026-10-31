
# Extended Readme

> The baselines are expected to run in a machine running Ubuntu 22.04

While `README.md` should include information about the baseline you implement and how to run it, this _extended_ readme provides info on what's the expected directory structure for a new baseline and more generally the instructions to follow before your baseline can be merged into the Flower repository. Please follow closely these instructions. It is likely that you have already completed steps 1-2.

1. Fork the Flower repository and clone it.
2. Navigate to the `baselines/` directory and from there run:
    ```bash
    # This will create a new directory with the same structure as this `baseline_template` directory.
    ./dev/create-baseline.sh <baseline-name>
    ``` 
3. All your code and configs should go into a sub-directory with the same name as the name of your baseline.
    *    The sub-directory contains a series of Python scripts that you can edit. Please stick to these files and consult with us if you need additional ones.
    *    There is also a basic config structure in `<baseline-name>/conf` ready be parsed by [Hydra](https://hydra.cc/) when executing your `main.py`.
4. Therefore, the directory structure in your baseline should look like:
    ```bash
    baselines/<baseline-name>
                    ├── README.md # describes your baseline and everything needed to use it
                    ├── EXTENDED_README.md # to remove before creating your PR
                    ├── pyproject.toml # details your Python environment
                    └── <baseline-name>
                                ├── *.py # several .py files including main.py and __init__.py
                                └── conf
                                     └── *.yaml # one or more Hydra config files

    ```
> :warning: Make sure the variable `name` in `pyproject.toml` is set to the name of the sub-directory containing all your code.

5. Add your dependencies to the `pyproject.toml` (see below a few examples on how to do it). Read more about Poetry below in this `EXTENDED_README.md`.
6. Regularly check that your coding style and the documentation you add follow good coding practices. To test whether your code meets the requirements, please run the following:
    ```bash
    # After activating your environment and from your baseline's directory
    cd .. # to go to the top-level directory of all baselines
    ./dev/test-baseline.sh <baseline-name>
    ./dev/test-baseline-structure.sh <baseline-name>
    ```
    Both `test-baseline.sh` and `test-baseline-structure.sh` will also be automatically run when you create a PR, and both tests need to pass for the baseline to be merged.
    To automatically solve some formatting issues and apply easy fixes, please run the formatting script:
    ```bash
    # After activating your environment and from your baseline's directory
    cd .. # to go to the top-level directory of all baselines
    ./dev/format-baseline.sh <baseline-name>
    ```
7. Ensure that the Python environment for your baseline can be created without errors by simply running `poetry install` and that this is properly described later when you complete the `Environment Setup` section in `README.md`. This is specially important if your environment requires additional steps after doing `poetry install`.
8. Ensure that your baseline runs with default arguments by running `poetry run python -m <baseline-name>.main`. Then, describe this and other forms of running your code in the `Running the Experiments` section in `README.md`.
9. Once your code is ready and you have checked:
    *    that following the instructions in your `README.md` the Python environment can be created correctly

    *    that running the code following your instructions can reproduce the experiments in the paper
   
   , then you just need to create a Pull Request (PR) to kickstart the process of merging your baseline into the Flower repository.

> Once you are happy to merge your baseline contribution, please delete this `EXTENDED_README.md` file.


## About Poetry

We use Poetry to manage the Python environment for each individual baseline. You can follow the instructions [here](https://python-poetry.org/docs/) to install Poetry in your machine. 


### Specifying a Python Version (optional)
By default, Poetry will use the Python version in your system. In some settings, you might want to specify a particular version of Python to use inside your Poetry environment. You can do so with [`pyenv`](https://github.com/pyenv/pyenv). Check the documentation for the different ways of installing `pyenv`, but one easy way is using the [automatic installer](https://github.com/pyenv/pyenv-installer):
```bash
curl https://pyenv.run | bash # then, don't forget links to your .bashrc/.zshrc
```

You can then install any Python version with `pyenv install <python-version>` (e.g. `pyenv install 3.9.17`). Then, in order to use that version for your baseline, you'd do the following:

```bash
# cd to your baseline directory (i.e. where the `pyproject.toml` is)
pyenv local <python-version>

# set that version for poetry
poetry env use <python-version>

# then you can install your Poetry environment (see the next setp)
```

### Installing Your Environment
With the Poetry tool already installed, you can create an environment for this baseline with commands:
```bash
# run this from the same directory as the `pyproject.toml` file is
poetry install
```

This will create a basic Python environment with just Flower and additional packages, including those needed for simulation. Next, you should add the dependencies for your code. It is **critical** that you fix the version of the packages you use using a `=` not a `=^`. You can do so via [`poetry add`](https://python-poetry.org/docs/cli/#add). Below are some examples:

```bash
# For instance, if you want to install tqdm
poetry add tqdm==4.65.0

# If you already have a requirements.txt, you can add all those packages (but ensure you have fixed the version) in one go as follows:
poetry add $( cat requirements.txt )
```
With each `poetry add` command, the `pyproject.toml` gets automatically updated so you don't need to keep that `requirements.txt` as part of this baseline.


More critically however, is adding your ML framework of choice to the list of dependencies. For some frameworks you might be able to do so with the `poetry add` command. Check [the Poetry documentation](https://python-poetry.org/docs/cli/#add) for how to add packages in various ways. For instance, let's say you want to use PyTorch:

```bash
# with plain `pip`  you'd run a command such as:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# to add the same 3 dependencies to your Poetry environment you'd need to add the URL to the wheel that the above pip command auto-resolves for you.
# You can find those wheels in `https://download.pytorch.org/whl/cu117`. Copy the link and paste it after the `poetry add` command.
# For instance to add `torch==1.13.1+cu117` and a x86 Linux system with Python3.8 you'd:
poetry add https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl
# you'll need to repeat this for both `torchvision` and `torchaudio`
```
The above is just an example of how you can add these dependencies. Please refer to the Poetry documentation to extra reference.

If all attempts fail, you can still install packages via standard `pip`. You'd first need to source/activate your Poetry environment.
```bash
# first ensure you have created your environment
# and installed the base packages provided in the template
poetry install 

# then activate it
poetry shell
```
Now you are inside your environment (pretty much as when you use `virtualenv` or `conda`) so you can install further packages with `pip`. Please note that, unlike with `poetry add`, these extra requirements won't be captured by `pyproject.toml`. Therefore, please ensure that you provide all instructions needed to: (1) create the base environment with Poetry and (2) install any additional dependencies via `pip` when you complete your `README.md`.