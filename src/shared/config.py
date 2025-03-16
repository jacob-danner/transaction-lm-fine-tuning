from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_user_name: str
    hf_dataset_repo_name: str
    gpt_1_causal_finetune: str
    base_training_output_dir: str
    device: str

    class Config:
        env_file = ".env"


settings = Settings()  # type: ignore
