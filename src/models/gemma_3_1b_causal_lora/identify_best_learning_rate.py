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

n_epochs = 1

training_args = TrainingArguments(
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    output_dir="/tmp/gemma_3_1b_causal_finetune_lora",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # effective batch size of 32
    num_train_epochs=n_epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=False,
    disable_tqdm=False,
    bf16=True,
    save_strategy="no",
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    callbacks=[EmptyMPSCacheCallback],
)

best_run = trainer.hyperparameter_search(
    hp_space=lambda trial: {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    },
    n_trials=10,
    direction="minimize",
)

print("Best trial found:")
print(f"  Score ({training_args.metric_for_best_model}): {best_run.objective}")
print(f"  Hyperparameters: {best_run.hyperparameters}")


"""
peak memory used: 60 GB

[I 2025-03-29 15:23:57,138] A new study created in memory with name: no-name-b9519386-e3a4-45d0-9997-08c100822bba
{'loss': 3.3909, 'grad_norm': 0.41674989461898804, 'learning_rate': 1.2773283207529082e-06, 'epoch': 1.0}                                                                           
{'eval_loss': 3.3699090480804443, 'eval_runtime': 7.669, 'eval_samples_per_second': 25.166, 'eval_steps_per_second': 3.26, 'epoch': 1.0}                                            
{'train_runtime': 53.1364, 'train_samples_per_second': 14.378, 'train_steps_per_second': 0.452, 'train_loss': 3.3909193674723306, 'epoch': 1.0}                                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:53<00:00,  2.21s/it]
[I 2025-03-29 15:24:51,003] Trial 0 finished with value: 3.3699090480804443 and parameters: {'learning_rate': 3.06558796980698e-05}. Best is trial 0 with value: 3.3699090480804443.
{'loss': 3.202, 'grad_norm': 0.6403970122337341, 'learning_rate': 7.477810365275432e-06, 'epoch': 1.0}                                                                              
{'eval_loss': 3.042307138442993, 'eval_runtime': 7.1994, 'eval_samples_per_second': 26.808, 'eval_steps_per_second': 3.473, 'epoch': 1.0}                                           
{'train_runtime': 51.4454, 'train_samples_per_second': 14.851, 'train_steps_per_second': 0.467, 'train_loss': 3.2020384470621743, 'epoch': 1.0}                                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:51<00:00,  2.14s/it]
[I 2025-03-29 15:25:43,243] Trial 1 finished with value: 3.042307138442993 and parameters: {'learning_rate': 0.00017946744876661037}. Best is trial 1 with value: 3.042307138442993.
{'loss': 3.3581, 'grad_norm': 0.4712085425853729, 'learning_rate': 2.6232883788310676e-06, 'epoch': 1.0}                                                                            
{'eval_loss': 3.309656858444214, 'eval_runtime': 7.3679, 'eval_samples_per_second': 26.195, 'eval_steps_per_second': 3.393, 'epoch': 1.0}                                           
{'train_runtime': 51.7577, 'train_samples_per_second': 14.761, 'train_steps_per_second': 0.464, 'train_loss': 3.358084042867025, 'epoch': 1.0}                                      
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:51<00:00,  2.16s/it]
[I 2025-03-29 15:26:35,771] Trial 2 finished with value: 3.309656858444214 and parameters: {'learning_rate': 6.295892109194562e-05}. Best is trial 1 with value: 3.042307138442993.
{'loss': 3.4014, 'grad_norm': 0.38545897603034973, 'learning_rate': 5.944755683320883e-07, 'epoch': 1.0}                                                                            
{'eval_loss': 3.3907337188720703, 'eval_runtime': 7.9183, 'eval_samples_per_second': 24.374, 'eval_steps_per_second': 3.157, 'epoch': 1.0}                                          
{'train_runtime': 54.4059, 'train_samples_per_second': 14.043, 'train_steps_per_second': 0.441, 'train_loss': 3.401378631591797, 'epoch': 1.0}                                      
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:54<00:00,  2.27s/it]
[I 2025-03-29 15:27:31,005] Trial 3 finished with value: 3.3907337188720703 and parameters: {'learning_rate': 1.4267413639970121e-05}. Best is trial 1 with value: 3.042307138442993.
{'loss': 3.3772, 'grad_norm': 0.442158579826355, 'learning_rate': 1.8831156947658586e-06, 'epoch': 1.0}                                                                             
{'eval_loss': 3.3453242778778076, 'eval_runtime': 8.1379, 'eval_samples_per_second': 23.716, 'eval_steps_per_second': 3.072, 'epoch': 1.0}                                          
{'train_runtime': 60.1607, 'train_samples_per_second': 12.699, 'train_steps_per_second': 0.399, 'train_loss': 3.3771610260009766, 'epoch': 1.0}                                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [01:00<00:00,  2.51s/it]
[I 2025-03-29 15:28:32,106] Trial 4 finished with value: 3.3453242778778076 and parameters: {'learning_rate': 4.5194776674380606e-05}. Best is trial 1 with value: 3.042307138442993.
{'loss': 3.1007, 'grad_norm': 0.7134320139884949, 'learning_rate': 1.0301330374684006e-05, 'epoch': 1.0}                                                                            
{'eval_loss': 2.868825674057007, 'eval_runtime': 7.9566, 'eval_samples_per_second': 24.257, 'eval_steps_per_second': 3.142, 'epoch': 1.0}                                           
{'train_runtime': 58.7779, 'train_samples_per_second': 12.998, 'train_steps_per_second': 0.408, 'train_loss': 3.10074520111084, 'epoch': 1.0}                                       
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:58<00:00,  2.45s/it]
[I 2025-03-29 15:29:32,668] Trial 5 finished with value: 2.868825674057007 and parameters: {'learning_rate': 0.00024723192899241613}. Best is trial 5 with value: 2.868825674057007.
{'loss': 3.21, 'grad_norm': 0.6327525973320007, 'learning_rate': 7.241928897555146e-06, 'epoch': 1.0}                                                                               
{'eval_loss': 3.057427167892456, 'eval_runtime': 8.2279, 'eval_samples_per_second': 23.457, 'eval_steps_per_second': 3.038, 'epoch': 1.0}                                           
{'train_runtime': 59.5978, 'train_samples_per_second': 12.819, 'train_steps_per_second': 0.403, 'train_loss': 3.2100019454956055, 'epoch': 1.0}                                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:59<00:00,  2.48s/it]
[I 2025-03-29 15:30:33,231] Trial 6 finished with value: 3.057427167892456 and parameters: {'learning_rate': 0.00017380629354132353}. Best is trial 5 with value: 2.868825674057007.
{'loss': 3.3472, 'grad_norm': 0.48501309752464294, 'learning_rate': 2.986886992417095e-06, 'epoch': 1.0}                                                                            
{'eval_loss': 3.2914392948150635, 'eval_runtime': 8.087, 'eval_samples_per_second': 23.865, 'eval_steps_per_second': 3.091, 'epoch': 1.0}                                           
{'train_runtime': 59.0599, 'train_samples_per_second': 12.936, 'train_steps_per_second': 0.406, 'train_loss': 3.3472251892089844, 'epoch': 1.0}                                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:59<00:00,  2.46s/it]
[I 2025-03-29 15:31:34,410] Trial 7 finished with value: 3.2914392948150635 and parameters: {'learning_rate': 7.168528781801029e-05}. Best is trial 5 with value: 2.868825674057007.
{'loss': 3.3682, 'grad_norm': 0.45607656240463257, 'learning_rate': 2.241545712956988e-06, 'epoch': 1.0}                                                                            
{'eval_loss': 3.3281683921813965, 'eval_runtime': 8.1476, 'eval_samples_per_second': 23.688, 'eval_steps_per_second': 3.068, 'epoch': 1.0}                                          
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [01:01<00:00,  2.55s/it]
[I 2025-03-29 15:32:36,385] Trial 8 pruned.                                                                                                                                         
{'loss': 3.4026, 'grad_norm': 0.3812362849712372, 'learning_rate': 4.7500136822249003e-07, 'epoch': 1.0}                                                                            
{'eval_loss': 3.3923892974853516, 'eval_runtime': 8.5522, 'eval_samples_per_second': 22.567, 'eval_steps_per_second': 2.923, 'epoch': 1.0}                                          
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [01:02<00:00,  2.62s/it]
[I 2025-03-29 15:33:39,983] Trial 9 pruned.                                                                                                                                         
Best trial found:
  Score (eval_loss): 2.868825674057007
  Hyperparameters: {'learning_rate': 0.00024723192899241613}
"""
