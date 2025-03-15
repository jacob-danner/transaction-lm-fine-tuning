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
from src.models.utils import transactions_dataset_init
from src.shared.config import settings
import json


HF_GPT_ID = "openai-gpt"
DEVICE = "mps"


def tokenizer_init():
    tokenizer = AutoTokenizer.from_pretrained(HF_GPT_ID)
    tokenizer.add_special_tokens(
        {"pad_token": "<pad>", "eos_token": "<eos>"}
    )  # gpt-1 tokenizer lacks these by default
    return tokenizer


def model_init():
    tokenizer = tokenizer_init()
    model = AutoModelForCausalLM.from_pretrained(HF_GPT_ID)
    model.resize_token_embeddings(
        len(tokenizer_init()), mean_resizing=False
    )  # extend the embedding layer to handle padding and eos tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model.to(DEVICE)


def format_dataset_row(
    row, for_test: bool, unique_from_accounts: set, eos_token: str
) -> Dict[Literal["text"], str]:
    row = dict(row)
    from_account = row.pop("from_account")
    formatted = dedent(
        f"""
        <possible accounts>{', '.join(unique_from_accounts)}</possible accounts>
        <transaction>{json.dumps(row)}</transaction>
        which account did this transaction come from?
        answer: {'' if for_test else f"{from_account}{eos_token}" }
    """
    ).strip()
    return {"text": formatted}


def dataset_init(tokenizer) -> DatasetDict:
    dataset = transactions_dataset_init()

    unique_from_accounts = set(
        dataset["train"]["from_account"] + dataset["test"]["from_account"]
    )
    eos_token = tokenizer.eos_token
    format_not_for_test = partial(
        format_dataset_row,
        for_test=False,
        unique_from_accounts=unique_from_accounts,
        eos_token=eos_token,
    )

    dataset["train"] = dataset["train"].map(format_not_for_test)
    dataset["validation"] = dataset["test"].map(format_not_for_test)
    del dataset["test"]

    remove_columns = [
        "amount",
        "month",
        "day",
        "year",
        "vendor",
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


def finetune():
    n_epochs = 20  # will likely stop early
    batch_size = 64

    tokenizer = tokenizer_init()
    dataset = dataset_init(tokenizer)

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
        model_init=model_init,
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
