# Evaluation for Code challenge

We leverage the code generation evaluation metrics provided by [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main) to evaluate our fine-tuned LLMs.
Three datasets have been selected for this evaluation: [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (Python), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) (Python), and [MultiPL-E](https://github.com/nuprl/MultiPL-E) (JavaScript, C++). 

> [!WARNING]
> The evaluation process takes ~30 GB VRAM. On a 40GB A100 it requires 15-30mins depending on the dataset to complete.

## Environment Setup

```shell
git clone --depth=1 https://github.com/flwrlabs/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/code ./flowertune-eval-code && rm -rf flower && cd flowertune-eval-code
```

Create a new Python environment (we recommend Python 3.11), activate it, then install dependencies with:

```shell
# From a new python environment, run:
pip install -r requirements.txt

# Log in HuggingFace account
hf auth login
```

After that, install `Node.js` and `g++` for the evaluation of JavaScript, C++:

```shell
# Install nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Restart your terminal

# Download and install Node.js (you may need to restart the terminal)
nvm install 20

# Install g++
sudo apt-get install g++
```

Then, download the `main.py` script from `bigcode-evaluation-harness` repository.

```shell
git clone https://github.com/yan-gao-GY/bigcode-evaluation-harness.git && cd bigcode-evaluation-harness && mv main.py ../ && cd .. && rm -rf bigcode-evaluation-harness
```


## Generate model answers & calculate pass@1 score

> [!NOTE]
> Evaluation needs to be run on MBPP, HumanEval, MultiPL-E (JS) and MultiPL-E (C++).

```bash
python main.py \
--model=your-base-model-name \ # e.g., mistralai/Mistral-7B-v0.3
--peft_model=/path/to/fine-tuned-peft-model-dir/  \ # e.g., ./peft_1
--max_length_generation=1024  \ # change to 2048 when running mbpp
--batch_size=4 \
--use_auth_token \
--allow_code_execution \
--save_generations  \
--save_references \
--tasks=humaneval \ # chosen from [mbpp, humaneval, multiple-js, multiple-cpp]
--metric_output_path=./evaluation_results_humaneval.json # change dataset name based on your choice
```

The model answers and pass@1 scores will be saved to `generations_{dataset_name}.json` and `evaluation_results_{dataset_name}.json`, respectively.


> [!NOTE]
> Please ensure that you provide all **four pass@1 scores** for the evaluation datasets when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/flwrlabs/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
