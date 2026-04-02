# This python file is adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import argparse

import torch
from fastchat.conversation import get_conv_template
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument("--question", type=str, default="How are you")
parser.add_argument("--template", type=str, default="vicuna_v1.1")
args = parser.parse_args()

# Load model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    args.peft_path, torch_dtype=torch.float16
).to("cuda")
base_model = model.peft_config["default"].base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Generate answers
temperature = 0.7
choices = []
conv = get_conv_template(args.template)

conv.append_message(conv.roles[0], args.question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer([prompt]).input_ids

output_ids = model.generate(
    input_ids=torch.as_tensor(input_ids).cuda(),
    do_sample=True,
    temperature=temperature,
    max_new_tokens=1024,
)

output_ids = (
    output_ids[0]
    if model.config.is_encoder_decoder
    else output_ids[0][len(input_ids[0]) :]
)

# Be consistent with the template's stop_token_ids
if conv.stop_token_ids:
    stop_token_ids_index = [
        i for i, id in enumerate(output_ids) if id in conv.stop_token_ids
    ]
    if len(stop_token_ids_index) > 0:
        output_ids = output_ids[: stop_token_ids_index[0]]

output = tokenizer.decode(
    output_ids,
    spaces_between_special_tokens=False,
)

if conv.stop_str and output.find(conv.stop_str) > 0:
    output = output[: output.find(conv.stop_str)]

for special_token in tokenizer.special_tokens_map.values():
    if isinstance(special_token, list):
        for special_tok in special_token:
            output = output.replace(special_tok, "")
    else:
        output = output.replace(special_token, "")
output = output.strip()

conv.update_last_message(output)

print(f">> prompt: {prompt}")
print(f">> Generated: {output}")
