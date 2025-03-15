from datasets import Dataset, DatasetDict
from pathlib import Path
from src.shared.config import settings
import json


hf_dataset_repo_name = f"{settings.hf_user_name}/{settings.hf_dataset_repo_name}"
json_path = Path(__file__).parent / "processed_data.json"

with open(json_path, "r") as f:
    data = json.load(f)

dataset = DatasetDict(
    {"train": Dataset.from_list(data["train"]), "test": Dataset.from_list(data["test"])}
)

dataset.push_to_hub(hf_dataset_repo_name, commit_message="feat: initial dataset upload")
