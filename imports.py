import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, get_dataset_config_names
ANSWERS = ['A', 'B', 'C', 'D']

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_orig_model(model_id="Models/Qwen3-8B"):
  seed_everything(42)
  orig_model = AutoModelForCausalLM.from_pretrained(
      model_id,
      device_map="cpu",
      trust_remote_code=True
  )
  orig_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, fix_mistral_regex=True)

  return orig_model, orig_tokenizer

def compress_model(model_id="Models/Qwen3-8B"):
    seed_everything(42)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    comp_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cpu",
        trust_remote_code=True
    )
    comp_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, fix_mistral_regex=True)

    return comp_model, comp_tokenizer

def get_model_size(model):
    total_bytes = 0
    for name, param in model.named_parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    return total_bytes / (1024 * 1024 * 1024)

def load_mmlu_dataset(split="test"):
    print("Загрузка датасета")
    dataset_topics = get_dataset_config_names("cais/mmlu")
    dataset_topics.remove("all")
    dataset_topics.remove("auxiliary_train")

    dataset = {}
    for topic_name in tqdm(dataset_topics):
        dataset[topic_name] = load_dataset("cais/mmlu", topic_name, split=split)

    return dataset

def get_question_prompt(raw, with_answer=False):
    prompt = raw["question"]
    possible_answers = enumerate([raw["choices"][i] for i in range(len(ANSWERS))])

    for i, option in possible_answers:
        choice_letter = ANSWERS[i]
        prompt += f"\n{choice_letter}. {option}"

    prompt += "\nAnswer:"

    if with_answer:
        prompt += f" {ANSWERS[raw['answer']]}\n\n"

    return prompt