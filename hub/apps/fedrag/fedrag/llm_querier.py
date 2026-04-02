"""fedrag: A Flower Federated RAG app."""

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid deadlocks during tokenization


class LLMQuerier:

    def __init__(self, model_name, use_gpu=False):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # set pad token if empty
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )

    def answer(self, question, documents, options, dataset_name, max_new_tokens=10):
        # Format options as A) ... B) ... etc.
        formatted_options = "\n".join([f"{k}) {v}" for k, v in options.items()])

        prompt = self.__format_prompt(
            question, documents, formatted_options, dataset_name
        )

        inputs = self.tokenizer(
            prompt, padding=True, return_tensors="pt", truncation=True
        ).to(self.device)

        # Perform element-wise comparison and create attention mask tensor
        attention_mask = (inputs.input_ids != self.tokenizer.pad_token_id).long()
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            early_stopping=False,
            pad_token_id=self.tokenizer.pad_token_id,  # set explicitly to avoid open-end generation print statement
            eos_token_id=self.tokenizer.eos_token_id,  # set explicitly to avoid open-end generation print statement
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = self.__extract_answer(generated_text, prompt)
        return prompt, generated_answer

    @classmethod
    def __format_prompt(cls, question, documents, options, dataset_name):
        instruction = "You are a helpful medical expert, and your task is to answer a medical question using the relevant documents."
        if dataset_name == "pubmedqa":
            instruction = "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe."
        elif dataset_name == "bioasq":
            instruction = "You are an advanced biomedical AI assistant trained to understand and process medical and scientific texts. Given a biomedical question, your goal is to provide a concise and accurate answer based on relevant scientific literature."

        ctx_documents = "\n".join(
            [f"Document {i + 1}: {doc}" for i, doc in enumerate(documents)]
        )
        prompt = f"""{instruction}

            Here are the relevant documents:
            {ctx_documents}

            Question: 
            {question}

            Options:
            {options}

            Answer only with the correct option: """
        return prompt

    @classmethod
    def __extract_answer(cls, generated_text, original_prompt):
        # Extract only the new generated text
        response = generated_text[len(original_prompt) :].strip()

        # Find first occurrence of A-D (case-insensitive)
        option = re.search(r"\b([A-Da-d])\b", response)
        if option:
            return option.group(1).upper()
        return None
