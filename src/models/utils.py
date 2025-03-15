from src.shared.config import settings
from datasets import load_dataset


def transactions_dataset_init():
    return load_dataset(f"{settings.hf_user_name}/{settings.hf_dataset_repo_name}")
