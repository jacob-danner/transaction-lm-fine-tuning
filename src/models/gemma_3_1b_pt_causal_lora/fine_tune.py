import torch
import json
from torch import mps
from functools import partial
from textwrap import dedent
from typing import Dict, Literal, Set
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


def format_dataset_row(
    row, eos_token: str, possible_accounts: Set[str]
) -> Dict[Literal["text"], str]:
    t = dict(row)
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
    return {"text": prompt.format(input_json=input_json)}


def training_dataset_init(tokenizer) -> DatasetDict:
    dataset = load_dataset(
        f"{settings.hf_user_name}/{settings.hf_dataset_repo_name}"
    ).shuffle(0)

    eos_token = tokenizer.eos_token
    format_for_train = partial(
        format_dataset_row,
        eos_token=eos_token,
        possible_accounts=set(
            dataset["train"]["from_account"] + dataset["test"]["from_account"]
        ),
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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # effective batch size of 32. memory usage peaked at about 105 (system)
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
trainer.push_to_hub(commit_message="feat: improved prompt")

'''
{'loss': 1.9444, 'grad_norm': 0.5543580651283264, 'learning_rate': 0.00023402777777777777, 'epoch': 1.0}                                                                                                                                  
{'eval_loss': 1.467583179473877, 'eval_runtime': 13.4158, 'eval_samples_per_second': 14.386, 'eval_steps_per_second': 1.863, 'epoch': 1.0}                                                                                                

{'loss': 0.9846, 'grad_norm': 0.374586284160614, 'learning_rate': 0.00021736111111111112, 'epoch': 2.0}                                                                                                                                   
{'eval_loss': 0.6282725930213928, 'eval_runtime': 16.9685, 'eval_samples_per_second': 11.374, 'eval_steps_per_second': 1.473, 'epoch': 2.0}                                                                                               

{'loss': 0.553, 'grad_norm': 0.20275253057479858, 'learning_rate': 0.00020069444444444445, 'epoch': 3.0}                                                                                                                                  
{'eval_loss': 0.5021805167198181, 'eval_runtime': 15.2557, 'eval_samples_per_second': 12.651, 'eval_steps_per_second': 1.639, 'epoch': 3.0}                                                                                               

{'eval_loss': 0.475649893283844, 'eval_runtime': 15.152, 'eval_samples_per_second': 12.738, 'eval_steps_per_second': 1.65, 'epoch': 4.0}

{'loss': 0.4596, 'grad_norm': 0.21350014209747314, 'learning_rate': 0.0001673611111111111, 'epoch': 5.0}                                
{'eval_loss': 0.45479676127433777, 'eval_runtime': 15.0746, 'eval_samples_per_second': 12.803, 'eval_steps_per_second': 1.658, 'epoch': 5.0}                                                                                                                                    

{'loss': 0.4389, 'grad_norm': 0.21676509082317352, 'learning_rate': 0.00015069444444444443, 'epoch': 6.0}                               
{'eval_loss': 0.43736299872398376, 'eval_runtime': 15.4443, 'eval_samples_per_second': 12.497, 'eval_steps_per_second': 1.619, 'epoch': 6.0}                                                                                                                                    

{'loss': 0.4206, 'grad_norm': 0.2708960771560669, 'learning_rate': 0.00013402777777777778, 'epoch': 7.0}                                
{'eval_loss': 0.4226933717727661, 'eval_runtime': 15.3915, 'eval_samples_per_second': 12.539, 'eval_steps_per_second': 1.624, 'epoch': 7.0}                                                                                                                                     

{'loss': 0.4037, 'grad_norm': 0.3346748650074005, 'learning_rate': 0.00011736111111111112, 'epoch': 8.0}                                
{'eval_loss': 0.4092947244644165, 'eval_runtime': 15.5528, 'eval_samples_per_second': 12.409, 'eval_steps_per_second': 1.607, 'epoch': 8.0}                                                                                                                                     

{'loss': 0.3879, 'grad_norm': 0.43560776114463806, 'learning_rate': 0.00010069444444444445, 'epoch': 9.0}                                                                                                                                                                                                                                                                                     
{'eval_loss': 0.39710062742233276, 'eval_runtime': 15.2618, 'eval_samples_per_second': 12.646, 'eval_steps_per_second': 1.638, 'epoch': 9.0}                                                                                                                                                                                                                                                  

{'loss': 0.3749, 'grad_norm': 0.3679960072040558, 'learning_rate': 8.402777777777778e-05, 'epoch': 10.0}                                                                                                                                                                                                                                                                                      
{'eval_loss': 0.3877725601196289, 'eval_runtime': 15.3819, 'eval_samples_per_second': 12.547, 'eval_steps_per_second': 1.625, 'epoch': 10.0}                                                                                                                                                                                                                                                  

{'loss': 0.365, 'grad_norm': 0.39415737986564636, 'learning_rate': 6.736111111111111e-05, 'epoch': 11.0}                                                                                                                                                                                                                                                                                      
{'eval_loss': 0.38242390751838684, 'eval_runtime': 15.2876, 'eval_samples_per_second': 12.625, 'eval_steps_per_second': 1.635, 'epoch': 11.0}                                                                                                                                                                                                                                                 

{'loss': 0.3571, 'grad_norm': 0.4076540470123291, 'learning_rate': 5.069444444444444e-05, 'epoch': 12.0}                                                                                                                                                                                                                                                                                      

{'loss': 0.3517, 'grad_norm': 0.5568859577178955, 'learning_rate': 3.4027777777777775e-05, 'epoch': 13.0}                                                                                                                                                                                                                                                                                     
{'eval_loss': 0.3717840313911438, 'eval_runtime': 15.4551, 'eval_samples_per_second': 12.488, 'eval_steps_per_second': 1.618, 'epoch': 13.0}                                                                                                                                                                                                                                                  

{'loss': 0.3474, 'grad_norm': 0.3303230404853821, 'learning_rate': 1.736111111111111e-05, 'epoch': 14.0}                                                                                                                                                                                                                                                                                      
{'eval_loss': 0.36912745237350464, 'eval_runtime': 15.3032, 'eval_samples_per_second': 12.612, 'eval_steps_per_second': 1.634, 'epoch': 14.0}                                                                                                                                                                                                                                                 

{'loss': 0.3446, 'grad_norm': 0.3771592080593109, 'learning_rate': 6.944444444444445e-07, 'epoch': 15.0}                                                                                                                                                                                                                                                                                      
{'eval_loss': 0.36838680505752563, 'eval_runtime': 15.452, 'eval_samples_per_second': 12.49, 'eval_steps_per_second': 1.618, 'epoch': 15.0}                                                                                                                                                                                                                                                   

{'train_runtime': 1833.8445, 'train_samples_per_second': 6.249, 'train_steps_per_second': 0.196, 'train_loss': 0.5478116220898098, 'epoch': 15.0}   
'''