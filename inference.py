import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer
from imports import seed_everything, get_orig_model, get_question_prompt, load_mmlu_dataset, ANSWERS

def get_topic_score(model, tokenizer, topic_data, max_tests=10):
    topic_correct = 0
    topic_samles = 0
    target_token_ids = [tokenizer.convert_tokens_to_ids(answer) for answer in ANSWERS]
    
    for raw in topic_data:
        topic_samles += 1
        if topic_samles > max_tests:
            break

        question_prompt = get_question_prompt(raw)
        inputs = tokenizer(question_prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        next_token_logits = logits[0, prompt_len - 1, :]

        log_probs = {}
        for i, token_id in enumerate(target_token_ids):
            choice_letter = ANSWERS[i]
            score = next_token_logits[token_id].item()
            log_probs[choice_letter] = score

        predicted_choice = max(log_probs, key=log_probs.get)
        predicted_index = ANSWERS.index(predicted_choice)

        if predicted_index == raw['answer']:
            topic_correct += 1
    
    return topic_correct, topic_samles

def get_model_score(model, tokenizer, dataset, prohibited_topics=[]):
  seed_everything(42)
  num_correct = 0
  num_samles = 0

  print("Тестирование")
  model.eval()
  for topic_name, topic_data in tqdm(dataset.items()):
      if topic_name in prohibited_topics:
          continue
      topic_correct, topic_samples = get_topic_score(model, tokenizer, topic_data)
      num_correct += topic_correct
      num_samles += topic_samples

  return num_correct / num_samles

if __name__ == "__main__":
    dataset = load_mmlu_dataset()

    print("Загрузка моделей")
    orig_model, orig_tokenizer = get_orig_model("Models/Qwen3-8B-full")
    orig_acc = get_model_score(orig_model, orig_tokenizer, dataset)

    comp_model, comp_tokenizer = get_orig_model("Models/Qwen3-8B-4bit")
    comp_acc = get_model_score(comp_model, comp_tokenizer, dataset)

    performance_drop = (orig_acc - comp_acc) / orig_acc

    print(f'Падение качества: {performance_drop}')

    comp_model_final = PeftModel.from_pretrained(
        comp_model,
        "Models/Qwen3-8B-4bit-final",
        device_map="cpu",
        trust_remote_code=True
        )
    comp_tokenizer_final = AutoTokenizer.from_pretrained(
        "Models/Qwen3-8B-4bit-final", trust_remote_code=True)
    comp_final_acc = get_model_score(comp_model_final, comp_tokenizer_final, dataset)

    final_performance_drop = (orig_acc - comp_final_acc) / orig_acc

    print(f'Падение качества с дообучением: {final_performance_drop}')