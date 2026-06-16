# Evaluation for General NLP challenge

We build up a multi-task language understanding pipeline to evaluate our fined-tuned LLMs.
The [MMLU](https://huggingface.co/datasets/lukaemon/mmlu) dataset is used for this evaluation, encompassing three categories: STEM, social sciences (SS), and humanities.


## Environment Setup

```shell
git clone --depth=1 https://github.com/flwrlabs/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/general-nlp ./flowertune-eval-general-nlp && rm -rf flower && cd flowertune-eval-general-nlp
```

Create a new Python environment (we recommend Python 3.11), activate it, then install dependencies with:

```shell
# From a new python environment, run:
pip install -r requirements.txt

# Log in HuggingFace account
hf auth login
```

## Generate model decision & calculate accuracy

> [!NOTE]
> Please ensure that you use `quantization=4` to run the evaluation if you wish to participate in the LLM Leaderboard.

```bash
python eval.py \
--base-model-name-path=your-base-model-name \ # e.g., mistralai/Mistral-7B-v0.3
--peft-path=/path/to/fine-tuned-peft-model-dir/ \ # e.g., ./peft_1
--run-name=fl  \ # specified name for this run  
--batch-size=16 \
--quantization=4 \
--category=stem,social_sciences,humanities
```

The model answers and accuracy values will be saved to `benchmarks/generation_{dataset_name}_{category_name}_{run_name}.jsonl` and `benchmarks/acc_{dataset_name}_{category_name}_{run_name}.txt`, respectively.


> [!NOTE]
> Please ensure that you provide all **three accuracy values (STEM, SS, Humanities)** for three evaluation categories when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/flwrlabs/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
