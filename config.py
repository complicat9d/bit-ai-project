import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    MODEL_PATH: str = os.getcwd() + "/data/model.h5"
    TEST_PHOTO_PATH: str = os.getcwd() + "/data/photo/test"
    TRAIN_PHOTO_PATH: str = os.getcwd() + "/data/photo/train"
    JSON_PATH: str = os.getcwd() + "/data/json/.json"


settings = Settings()
