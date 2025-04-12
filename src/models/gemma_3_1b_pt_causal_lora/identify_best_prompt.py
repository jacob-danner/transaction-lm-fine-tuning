import torch
import json
from torch import mps
from textwrap import dedent
from typing import TypedDict, Callable, List, Set, Dict, Any, Literal
from datasets.formatting.formatting import LazyRow
from functools import wraps, partial
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from src.shared.config import settings

# -------
# DATASET
# -------


def dataset_init() -> DatasetDict:
    return load_dataset(
        f"{settings.hf_user_name}/{settings.hf_dataset_repo_name}"
    ).shuffle(0)


class Transaction(TypedDict):
    transaction_date: str
    description: str
    amount: str
    category: str
    category_source: str
    card: str
    day_of_week: str
    from_account: str


type TransactionLazyRow = LazyRow
type PromptFunction = Callable[[TransactionLazyRow], str]


def create_prompt_functions(
    eos_token: str, possible_accounts: Set[str]
) -> Dict[str, PromptFunction]:
    def put_in_expected_format(prompt: str) -> Dict[Literal["text"], str]:
        return {"text": prompt}

    def a(t: TransactionLazyRow) -> str:
        """
        results:
        loss: 2.67
        eval_loss: 2.48
        """
        t: Transaction = dict(t)
        from_account = t.pop("from_account")
        input_json = json.dumps(t, indent=2)
        prompt = dedent(
            f"""\
            Input Transaction Details:
            ```json
            {{input_json}}
            ```
            Output Classification:
            {from_account}{eos_token}
            """
        )
        return put_in_expected_format(prompt.format(input_json=input_json))

    def b(t: TransactionLazyRow) -> str:
        """
        results:
        loss: 2.1
        eval_loss: 1.93
        """
        t: Transaction = dict(t)
        from_account = t.pop("from_account")
        input_json = json.dumps(t, indent=2)
        prompt = dedent(
            f"""\
            Input Transaction Details:
            ```json
            {{input_json}}
            ```
            Possible Accounts: {list(possible_accounts)}
            Output Classification:
            {from_account}{eos_token}
            """
        )
        return put_in_expected_format(prompt.format(input_json=input_json))

    return {"a": a, "b": b}


def prompt_dataset_init(tokenizer) -> DatasetDict:
    dataset = dataset_init()
    possible_accounts = set(
        dataset["train"]["from_account"] + dataset["test"]["from_account"]
    )
    prompt_functions = create_prompt_functions(tokenizer.eos_token, possible_accounts)

    tokenize = lambda batch: tokenizer(batch["text"])
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

    for id, function in prompt_functions.items():
        dataset[id] = DatasetDict(
            {
                "train": (
                    dataset["train"]
                    .map(function)
                    .map(tokenize, batched=True, remove_columns=remove_columns)
                ),
                "validation": (
                    dataset["train"]
                    .map(function)
                    .map(tokenize, batched=True, remove_columns=remove_columns)
                ),
            }
        )

    del dataset["train"]
    del dataset["test"]
    return dataset


# --------
# MODELING
# --------


HF_GEMMA_ID = "google/gemma-3-1b-pt"


def tokenizer_init():
    tokenizer = AutoTokenizer.from_pretrained(HF_GEMMA_ID)
    return tokenizer


def model_init():
    model = AutoModelForCausalLM.from_pretrained(
        HF_GEMMA_ID, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
            task_type="CAUSAL_LM",
        ),
    )

    return model.to(settings.device)


class EmptyMPSCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        mps.empty_cache()

    def on_evaluate(self, args, state, control, **kwargs):
        mps.empty_cache()


if __name__ == "__main__":
    training_args = TrainingArguments(
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        output_dir="/tmp/gemma_3_1b_causal_finetune_lora",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # effective batch size of 32
        num_train_epochs=1,
        weight_decay=0.01,
        learning_rate=2.5e-4,
        eval_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=False,
        disable_tqdm=False,
        bf16=True,
        save_strategy="no",
    )

    tokenizer = tokenizer_init()
    dataset = prompt_dataset_init(tokenizer)
    for shard in dataset.keys():
        print(shard)

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            ),
            train_dataset=dataset[shard]["train"],
            eval_dataset=dataset[shard]["validation"],
            callbacks=[EmptyMPSCacheCallback],
        )

        trainer.train()

        print(f"-----", end="\n\n")
