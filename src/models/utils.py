from src.shared.config import settings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Literal
from datasets import DatasetDict
from textwrap import dedent
from functools import partial
import json


def transactions_dataset_init():
    return load_dataset(f"{settings.hf_user_name}/{settings.hf_dataset_repo_name}")


class GPT1:
    HF_GPT_ID = "openai-gpt"

    @classmethod
    def tokenizer_init(cls):
        tokenizer = AutoTokenizer.from_pretrained(cls.HF_GPT_ID)
        tokenizer.add_special_tokens(
            {"pad_token": "<pad>", "eos_token": "<eos>"}
        )  # gpt-1 tokenizer lacks these by default
        return tokenizer

    @classmethod
    def model_init(cls):
        tokenizer = GPT1.tokenizer_init()
        model = AutoModelForCausalLM.from_pretrained(cls.HF_GPT_ID)
        model.resize_token_embeddings(
            len(tokenizer), mean_resizing=False
        )  # extend the embedding layer to handle padding and eos tokens
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        return model.to(settings.device)

    @classmethod
    def format_dataset_row(
        cls, row, for_test: bool, unique_from_accounts: set, eos_token: str
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

    @classmethod
    def training_dataset_init(cls, tokenizer) -> DatasetDict:
        dataset = transactions_dataset_init()

        unique_from_accounts = set(
            dataset["train"]["from_account"] + dataset["test"]["from_account"]
        )
        eos_token = tokenizer.eos_token
        format_for_train = partial(
            GPT1.format_dataset_row,
            for_test=False,
            unique_from_accounts=unique_from_accounts,
            eos_token=eos_token,
        )

        dataset["train"] = dataset["train"].map(format_for_train)
        dataset["validation"] = dataset["test"].map(format_for_train)
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

    @classmethod
    def test_dataset_init(cls, tokenizer) -> DatasetDict:
        dataset = transactions_dataset_init()

        unique_from_accounts = set(
            dataset["train"]["from_account"] + dataset["test"]["from_account"]
        )
        eos_token = tokenizer.eos_token
        format_for_test = partial(
            GPT1.format_dataset_row,
            for_test=True,
            unique_from_accounts=unique_from_accounts,
            eos_token=eos_token,
        )

        dataset["test"] = dataset["test"].map(format_for_test)
        del dataset["train"]

        return dataset
