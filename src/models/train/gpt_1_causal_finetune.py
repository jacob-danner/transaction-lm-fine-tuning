from typing import Dict, Literal
from datasets import DatasetDict
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from textwrap import dedent
from src.models.utils import transactions_dataset_init, GPT1
from src.shared.config import settings
import json


def finetune():
    n_epochs = 20  # will likely stop early
    batch_size = 64

    tokenizer = GPT1.tokenizer_init()
    dataset = GPT1.training_dataset_init(tokenizer)

    training_args = TrainingArguments(
        output_dir=f"{settings.base_training_output_dir}/{settings.gpt_1_causal_finetune}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        learning_rate=1e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=True,
        hub_model_id=f"{settings.hf_user_name}/{settings.gpt_1_causal_finetune}",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model_init=GPT1.model_init,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    finetune()
