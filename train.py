from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from imports import get_orig_model, seed_everything
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def map_for_training(raw):
    prompt = raw["question"]
    possible_answers = enumerate(raw["choices"])

    for i, answer in possible_answers:
        choice_letter = chr(65 + i)
        prompt += f"\n{choice_letter}. {answer}"

    columns = [
            {"role": "user", "content": prompt},
            {"role": "model", "content": f"Answer: {chr(65 + raw["answer"])}\n\n"}
        ]

    prompt += f"\nAnswer: {chr(65 + raw["answer"])}\n\n"

    full_text = comp_tokenizer.apply_chat_template(columns, tokenize=False)
        
    tokenized = comp_tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding=False,
        add_special_tokens=False
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

def train_model(model, tokenizer, train_dataset):
    config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir="./model-steps",
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        num_train_epochs=1,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        optim="adamw_torch",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    print('Дообучение')
    trainer.train()
    trainer.save_model("Models/Qwen3-8B-4bit-final")
    tokenizer.save_pretrained("Models/Qwen3-8B-4bit-final")

if __name__ == "__main__":
    seed_everything(42)
    comp_model, comp_tokenizer = get_orig_model("Models/Qwen3-8B-4bit")

    dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    dataset = dataset.shuffle(seed=42).select(range(20))
    processed_dataset = dataset.map(map_for_training, remove_columns=dataset.column_names)

    train_model(comp_model, comp_tokenizer, processed_dataset)