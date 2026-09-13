# Evaluation for Finance challenge

We build a sentiment classification pipeline on finance-related text to evaluate our fine-tuned LLMs.
Three datasets have been selected for this evaluation: [FPB](https://huggingface.co/datasets/takala/financial_phrasebank), [FIQA](https://huggingface.co/datasets/pauri32/fiqa-2018), and [TFNS](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment). 


## Environment Setup

```shell
git clone --depth=1 https://github.com/flwrlabs/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/finance ./flowertune-eval-finance && rm -rf flower && cd flowertune-eval-finance
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
--batch-size=32 \
--quantization=4 \
--datasets=fpb,fiqa,tfns
```

The model answers and accuracy values will be saved to `benchmarks/generation_{dataset_name}_{run_name}.jsonl` and `benchmarks/acc_{dataset_name}_{run_name}.txt`, respectively.

> [!NOTE]
> Please ensure that you provide all **three accuracy values (FPB, FIQA, TFNS)** for three evaluation datasets when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/flwrlabs/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
