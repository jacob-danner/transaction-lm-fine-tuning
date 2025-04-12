import torch
import json
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
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from src.shared.config import settings


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


def format_dataset_row(row, eos_token: str) -> Dict[Literal["text"], str]:
    row = dict(row)
    from_account = row.pop("from_account")
    formatted = dedent(
        f"""
        input: {json.dumps(row)}
        label: {f"{from_account}{eos_token}"}
        """
    ).strip()
    return {"text": formatted}


def training_dataset_init(tokenizer) -> DatasetDict:
    dataset = load_dataset(
        f"{settings.hf_user_name}/{settings.hf_dataset_repo_name}"
    ).shuffle(0)

    eos_token = tokenizer.eos_token
    format_for_train = partial(
        format_dataset_row,
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


class EmptyMPSCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        mps.empty_cache()

    def on_evaluate(self, args, state, control, **kwargs):
        mps.empty_cache()


tokenizer = tokenizer_init()
dataset = training_dataset_init(tokenizer)

n_epochs = 15

training_args = TrainingArguments(
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    output_dir="/tmp/gemma_3_1b_causal_finetune_lora",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,  # effective batch size of 64
    num_train_epochs=n_epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    disable_tqdm=False,
    bf16=True,
    save_strategy="epoch",
    push_to_hub=True,
    hub_model_id=f"{settings.hf_user_name}/{settings.gemma_3_1b_causal_finetune_lora}",
    learning_rate=2.5e-4,
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    callbacks=[EmptyMPSCacheCallback, EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
trainer.push_to_hub(commit_message="feat: baseline run. dead simple prompt")

"""
memory peaked at: 102 GB

{'loss': 3.2323, 'grad_norm': 0.698011577129364, 'learning_rate': 0.00023472222222222224, 'epoch': 1.0}                                                                             
{'eval_loss': 2.9388952255249023, 'eval_runtime': 6.3199, 'eval_samples_per_second': 30.539, 'eval_steps_per_second': 2.057, 'epoch': 1.0}                                          
{'loss': 2.6433, 'grad_norm': 1.2237988710403442, 'learning_rate': 0.00021805555555555556, 'epoch': 2.0}                                                                            
{'eval_loss': 2.26686429977417, 'eval_runtime': 6.0827, 'eval_samples_per_second': 31.729, 'eval_steps_per_second': 2.137, 'epoch': 2.0}                                            
{'loss': 1.9319, 'grad_norm': 1.937229037284851, 'learning_rate': 0.0002013888888888889, 'epoch': 3.0}                                                                              
{'eval_loss': 1.599684476852417, 'eval_runtime': 6.0418, 'eval_samples_per_second': 31.944, 'eval_steps_per_second': 2.152, 'epoch': 3.0}                                           
{'loss': 1.4135, 'grad_norm': 0.8454684615135193, 'learning_rate': 0.00018472222222222224, 'epoch': 4.0}                                                                            
{'eval_loss': 1.2554705142974854, 'eval_runtime': 6.4099, 'eval_samples_per_second': 30.11, 'eval_steps_per_second': 2.028, 'epoch': 4.0}                                           
{'loss': 1.193, 'grad_norm': 0.3112393319606781, 'learning_rate': 0.00016805555555555557, 'epoch': 5.0}                                                                             
{'eval_loss': 1.1551096439361572, 'eval_runtime': 7.1387, 'eval_samples_per_second': 27.036, 'eval_steps_per_second': 1.821, 'epoch': 5.0}                                          
{'loss': 1.1195, 'grad_norm': 0.43125247955322266, 'learning_rate': 0.0001513888888888889, 'epoch': 6.0}                                                                            
{'eval_loss': 1.104691982269287, 'eval_runtime': 6.9698, 'eval_samples_per_second': 27.691, 'eval_steps_per_second': 1.865, 'epoch': 6.0}                                           
{'loss': 1.0782, 'grad_norm': 0.2784508168697357, 'learning_rate': 0.00013472222222222222, 'epoch': 7.0}                                                                            
{'eval_loss': 1.0753082036972046, 'eval_runtime': 6.966, 'eval_samples_per_second': 27.706, 'eval_steps_per_second': 1.866, 'epoch': 7.0}                                           
{'loss': 1.0507, 'grad_norm': 0.3005714416503906, 'learning_rate': 0.00011805555555555556, 'epoch': 8.0}                                                                            
{'eval_loss': 1.0514860153198242, 'eval_runtime': 7.0283, 'eval_samples_per_second': 27.461, 'eval_steps_per_second': 1.85, 'epoch': 8.0}                                           
{'loss': 1.0263, 'grad_norm': 0.3211475610733032, 'learning_rate': 0.00010138888888888889, 'epoch': 9.0}                                                                            
{'eval_loss': 1.0319091081619263, 'eval_runtime': 7.0023, 'eval_samples_per_second': 27.563, 'eval_steps_per_second': 1.857, 'epoch': 9.0}                                          
{'loss': 1.0072, 'grad_norm': 0.35849425196647644, 'learning_rate': 8.472222222222222e-05, 'epoch': 10.0}                                                                           
{'eval_loss': 1.0154170989990234, 'eval_runtime': 6.9904, 'eval_samples_per_second': 27.609, 'eval_steps_per_second': 1.86, 'epoch': 10.0}                                          
{'loss': 0.9911, 'grad_norm': 0.3441905081272125, 'learning_rate': 6.805555555555555e-05, 'epoch': 11.0}                                                                            
{'eval_loss': 1.0053327083587646, 'eval_runtime': 7.1467, 'eval_samples_per_second': 27.006, 'eval_steps_per_second': 1.819, 'epoch': 11.0}                                         
{'loss': 0.9781, 'grad_norm': 0.6558969616889954, 'learning_rate': 5.138888888888889e-05, 'epoch': 12.0}                                                                            
{'eval_loss': 0.9918075203895569, 'eval_runtime': 7.1543, 'eval_samples_per_second': 26.977, 'eval_steps_per_second': 1.817, 'epoch': 12.0}                                         
{'loss': 0.9675, 'grad_norm': 0.512424886226654, 'learning_rate': 3.472222222222222e-05, 'epoch': 13.0}                                                                             
{'eval_loss': 0.9833388924598694, 'eval_runtime': 7.1204, 'eval_samples_per_second': 27.105, 'eval_steps_per_second': 1.826, 'epoch': 13.0}                                         
{'loss': 0.9597, 'grad_norm': 0.5348716378211975, 'learning_rate': 1.8055555555555555e-05, 'epoch': 14.0}                                                                           
{'eval_loss': 0.9789060354232788, 'eval_runtime': 7.2558, 'eval_samples_per_second': 26.599, 'eval_steps_per_second': 1.792, 'epoch': 14.0}                                         
{'loss': 0.9552, 'grad_norm': 0.32860100269317627, 'learning_rate': 1.388888888888889e-06, 'epoch': 15.0}                                                                           
{'eval_loss': 0.9766164422035217, 'eval_runtime': 7.2062, 'eval_samples_per_second': 26.783, 'eval_steps_per_second': 1.804, 'epoch': 15.0}                                         
{'train_runtime': 846.5703, 'train_samples_per_second': 13.537, 'train_steps_per_second': 0.213, 'train_loss': 1.3698330190446641, 'epoch': 15.0}                                   
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 180/180 [14:06<00:00,  4.70s/it]
"""
