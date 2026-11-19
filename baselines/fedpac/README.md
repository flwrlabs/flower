---
title: title of the paper # TODO
url: https://arxiv.org/abs/2007.14390 # TODO: update with the link to your paper
labels: [label1, label2] # TODO: please add between 4 and 10 single-word (maybe two-words) labels (e.g. system heterogeneity, image classification, asynchronous, weight sharing, cross-silo). Do not use "". Remove this comment once you are done.
dataset: [dataset1, dataset2] # TODO: list of datasets you include in your baseline. Do not use "". Remove this comment once you are done.
---

> [!IMPORTANT]
> This is the template for your `README.md`. Please fill-in the information in all areas with a :warning: symbol.
> Please refer to the [Flower Baselines contribution](https://flower.ai/docs/baselines/how-to-contribute-baselines.html) and [Flower Baselines usage](https://flower.ai/docs/baselines/how-to-use-baselines.html) guides for more details.
> Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.
> Please remove this [!IMPORTANT] block once you are done with your `README.md` as well as all the `:warning:` symbols and the comments next to them.

> [!IMPORTANT]
> To help having all baselines similarly formatted and structured, we have included two scripts in `baselines/dev` that when run will format your code and run some tests checking if it's formatted.
> These checks use standard packages such as `isort`, `black`, `pylint` and others. You as a baseline creator will need to install additional packages. These are already specified in the `pyproject.toml` of
> your baseline. Follow these steps:

```bash
# Create a python env
pyenv virtualenv 3.10.14 fedpac

# Activate it
pyenv activate fedpac

# Install project including developer packages
# Note the `-e` this means you install it in editable mode 
# so even if you change the code you don't need to do `pip install`
# again. However, if you add a new dependency to `pyproject.toml` you
# will need to re-run the command below
pip install -e ".[dev]"

# Even without modifying or adding new code, you can run your baseline
# with the placeholder code generated when you did `flwr new`. If you
# want to test this to familiarise yourself with how flower apps are
# executed, execute this from the directory where you `pyproject.toml` is:
flwr run .

# At anypoint during the process of creating your baseline you can 
# run the formatting script. For this do:
cd .. # so you are in the `flower/baselines` directory

# Run the formatting script (it will auto-correct issues if possible)
./dev/format-baseline.sh fedpac

# Then, if the above is all good, run the tests.
./dev/test-baseline.sh fedpac
```

> [!IMPORTANT]
> When you open a PR to get the baseline merged into the main Flower repository, the `./dev/test-baseline.sh` script will run. Only if test pass, the baseline can be merged. 
> Some issues highlighted by the tests script are easier than others to fix. Do not hesitate in reaching out for help to us (e.g. as a comment in your PR) if you are stuck with these.
> Before opening your PR, please remove the code snippet above as well all the [!IMPORTANT] message blocks. Yes, including this one.

# :warning: *_Title of your baseline_* # Also copy this title to the `description` in the `[project]` section of your `pyproject.toml`.

> [!NOTE] 
> If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** :warning: *_add the URL of the paper page (not to the .pdf). For instance if you link a paper on ArXiv, add here the URL to the abstract page (e.g. [paper](https://arxiv.org/abs/1512.03385)). If your paper is in from a journal or conference proceedings, please follow the same logic._*

**Authors:** :warning: *_list authors of the paper_*

**Abstract:** :warning: *_add here the abstract of the paper you are implementing_*


## About this baseline

**What’s implemented:** :warning: *_Concisely describe what experiment(s) (e.g. Figure 1, Table 2, etc.) in the publication can be replicated by running the code. Please only use a few sentences. ”_*

**Datasets:** :warning: *_List the datasets you used (if you used a medium to large dataset, >10GB please also include the sizes of the dataset). We highly recommend using [FlowerDatasets](https://flower.ai/docs/datasets/index.html) to download and partition your dataset. If you have other ways to download the data, you can also use `FlowerDatasets` to partition it._*

**Hardware Setup:** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Indicate how long it took to run the experiments. Someone out there might not have access to the same resources you have so, could you list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

**Contributors:** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_*


## Experimental Setup

**Task:** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

**Model:** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

**Dataset:** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

**Training Hyperparameters:** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

:warning: _Specify the steps to create and activate your environment and install the baseline project. Most baselines are expected to require minimal steps as shown below. These instructions should be comprehensive enough so anyone can run them (if non standard, describe them step-by-step)._

:warning: _The dependencies for your baseline are listed in the `pyproject.toml`, extend it with additional packages needed for your baseline._

:warning: _Baselines should use Python 3.10, [pyenv](https://github.com/pyenv/pyenv), and the [virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugging. 

```bash
# Create the virtual environment
pyenv virtualenv 3.10.14 <name-of-your-baseline-env>

# Activate it
pyenv activate <name-of-your-baseline-env>

# Install the baseline
pip install -e .
```

:warning: _If your baseline requires running some script before starting an experiment, please indicate so here_.

## Running the Experiments

:warning: _Make sure you have adjusted the `client-resources` in the federation in `pyproject.toml` so your simulation makes the best use of the system resources available._

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

:warning: _You might want to add more hyperparameters and settings for your baseline. You can do so by extending `[tool.flwr.app.config]` in `pyproject.toml`. In addition, you can create a new `.toml` file that can be passed with the `--run-config` command (see below an example) to override several config values **already present** in `pyproject.toml`._
```bash
# it is likely that for one experiment you need to override some arguments.
flwr run . --run-config learning-rate=0.1,coefficient=0.123

# or you might want to load different `.toml` configs all together:
flwr run . --run-config <my-big-experiment-config>.toml
```

:warning: _It is preferable to show a single command (or multiple commands if they belong to the same experiment) and then a table/plot with the expected results, instead of showing all the commands first and then all the results/plots._
:warning: _If you present plots or other figures, please include either a Jupyter notebook showing how to create them or include a utility function that can be called after the experiments finish running._
:warning: If you include plots or figures, save them in `.png` format and place them in a new directory named `_static` at the same level as your `README.md`.
