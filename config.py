import os
from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    MODEL_PATH: str = os.getcwd() + "/data/model.h5"
    TEST_PHOTO_PATH: str = os.getcwd() + "/data/photo/test"
    TRAIN_PHOTO_PATH: str = os.getcwd() + "/data/photo/train"
    JSON_PATH: str = os.getcwd() + "/data/json/.json"

    LABELS: Dict[str, int] = {
        "fashion_Jacket": 0,
        "fashion_Trousers": 1,
        "fashion_Bag": 2,
        "fashion_Sweater": 3,
        "fashion_Evening dress": 4,
        "fashion_Boots": 5,
        "fashion_Leggings": 6,
        "fashion_Heels": 7,
        "fashion_Jewelry": 8,
        "fashion_Sandals": 9,
        "fashion_Blouse": 10,
        "fashion_Skirt": 11,
        "fashion_Midi dress": 12,
        "fashion_Belt": 13,
        "fashion_Casual dress": 14,
        "fashion_Shorts": 15,
        "fashion_Tank top": 16,
        "fashion_Loafers": 17,
        "fashion_Gloves": 18,
        "fashion_Scarf": 19,
        "fashion_Trench coat": 20,
        "fashion_Sneakers": 21,
        "fashion_Cardigan": 22,
        "fashion_Coat": 23,
        "fashion_Shirt": 24,
        "fashion_Hoodie": 25,
        "fashion_T-shirt": 26,
        "fashion_Jeans": 27,
        "common_Shirt": 28,
        "common_Jeans": 29,
        "common_Shorts": 30,
        "common_Sandals": 31,
        "common_T-shirt": 32,
        "common_Loafers": 33,
        "common_Sneakers": 34,
        "common_Sweater": 35,
        "common_Trousers": 36,
        "common_Tank top": 37,
        "common_Belt": 38,
        "common_Boots": 39,
        "common_Cardigan": 40,
        "common_Jacket": 41,
        "common_Scarf": 42,
        "common_Skirt": 43,
        "common_Heels": 44,
        "common_Blouse": 45,
        "common_Flats": 46,
        "common_Bag": 47,
        "common_Evening dress": 48,
        "common_Casual dress": 49,
    }


settings = Settings()
