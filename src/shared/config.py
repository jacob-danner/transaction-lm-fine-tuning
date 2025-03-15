from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_user_name: str
    hf_dataset_repo_name: str

    class Config:
        env_file = ".env"


settings = Settings()  # type: ignore
