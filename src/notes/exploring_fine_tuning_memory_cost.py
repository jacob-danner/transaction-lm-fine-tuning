import torch
from torch import mps
from functools import partial
from textwrap import dedent
from typing import Dict, Literal
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from src.shared.config import settings

"""
# Exploring fine tuning memory and computational cost (02/27/25)

---

## Topics learned about:

- LoRA
- bfloat 16
- mps

### LoRA

- Training neural nets require calculating gradients and storing optimizer states for each parameter, this is memory intensive. LoRA uses vastly fewer parameters, and so uses much less memory.
- `output = FrozenLayer(input) + LoRA_Adapter(input)` We need to store activations at each hidden state in order to be able to calculate gradients for the LoRA parameters.
- When training a LoRA, bottlenecks will be activations (and batch size - which just scales the number of activations)

### bfloat16

- Loading model and training in bfloat16 halves the memory needed for model weights and activations

### mps

- torch.mps exposes a few ways to monitor memory usage on the gpu, and importantly, clear the cache

---

## Memory Usage Experiments

All configurations used batch size = 16.

| Configuration               | Pre-Train Current Mem (Weights) | Peak Driver Mem (During Train) | Post-Cache Current Mem (Weights+Optim) | Post-Cache Driver Mem |
| :-------------------------- | :------------------------------ | :----------------------------- | :------------------------------------- | :-------------------- |
| LoRA (fp32)                 | ~0.48 GB                        | ~11.3 GB                       | ~0.49 GB                               | ~2.30 GB              |
| Full FT (fp32)              | ~0.48 GB                        | ~14.9 GB                       | ~1.90 GB                               | ~8.40 GB              |
| Full FT (bf16 Load+Arg)     | ~0.24 GB                        | ~9.75 GB                       | ~0.95 GB                               | ~5.29 GB              |
| LoRA (bf16 Load+Arg)        | ~0.24 GB                        | ~9.0 GB                        | ~0.25 GB                               | ~2.46 GB              |
"""


HF_GPT_ID = "openai-gpt"


def tokenizer_init():
    tokenizer = AutoTokenizer.from_pretrained(HF_GPT_ID)
    tokenizer.add_special_tokens(
        {"pad_token": "<pad>", "eos_token": "<eos>"}
    )  # gpt-1 tokenizer lacks these by default
    return tokenizer


def format_dataset_row(
    row, for_test: bool, unique_from_accounts: set, eos_token: str
) -> Dict[Literal["text"], str]:
    # currently not using unique_from_accounts
    row = dict(row)
    from_account = row.pop("from_account")
    formatted = dedent(
        f"""
        Transaction
        -----------
        Description: {row['description']}
        Amount: {row['amount']}
        Category: {row['category']} (Source: {row['category_source']})
        Transaction Date: {row['transaction_date']}
        Day of Week: {row['day_of_week']}
        Card: {row['card']}

        Question: Which account initiated this transaction?
        Answer: {'' if for_test else f"{from_account}{eos_token}"}
        """
    ).strip()
    return {"text": formatted}


def training_dataset_init(tokenizer) -> DatasetDict:
    dataset = load_dataset(
        f"{settings.hf_user_name}/{settings.hf_dataset_repo_name}"
    ).shuffle(0)

    unique_from_accounts = set(
        dataset["train"]["from_account"] + dataset["test"]["from_account"]
    )
    eos_token = tokenizer.eos_token
    format_for_train = partial(
        format_dataset_row,
        for_test=False,
        unique_from_accounts=unique_from_accounts,
        eos_token=eos_token,
    )

    dataset["train"] = dataset["train"].map(format_for_train)
    dataset["validation"] = dataset["test"].map(format_for_train)
    del dataset["test"]

    remove_columns = [
        "transaction_date",
        "description",
        "amount",
        "category",
        "category_source",
        "card",
        "day_of_week",
        "from_account",
        "text",
    ]

    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"]),
        batched=True,
        remove_columns=remove_columns,
    )
    dataset.set_format("pt")
    return dataset


def model_init():
    tokenizer = tokenizer_init()
    model = AutoModelForCausalLM.from_pretrained(HF_GPT_ID, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(
        len(tokenizer), mean_resizing=False
    )  # extend the embedding layer to handle padding and eos tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model = get_peft_model(
        model,
        LoraConfig(
            target_modules=["c_attn", "c_proj"],
            inference_mode=False,
            task_type="CAUSAL_LM",
        ),
    )
    return model.to(settings.device)


def bytes_to_gb(bytes: int) -> float:
    return bytes / 1e9


n_epochs = 1
best_learning_rate = 1e-4

tokenizer = tokenizer_init()
dataset = training_dataset_init(tokenizer)

model = model_init()

training_args = TrainingArguments(
    output_dir="/tmp/gpt_1_causal_finetune_lora",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,  # effective batch size of 64
    num_train_epochs=n_epochs,
    learning_rate=best_learning_rate,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=False,
    disable_tqdm=False,
    bf16=True,
)

print(f"pre train:              {bytes_to_gb(mps.current_allocated_memory()) = }")
print(f"pre train:              {bytes_to_gb(mps.driver_allocated_memory()) = }")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

print(f"post train:             {bytes_to_gb(mps.current_allocated_memory()) = }")
print(f"post train:             {bytes_to_gb(mps.driver_allocated_memory()) = }")

mps.empty_cache()

print(f"post train empty cache: {bytes_to_gb(mps.current_allocated_memory()) = }")
print(f"post train empty cache: {bytes_to_gb(mps.driver_allocated_memory()) = }")