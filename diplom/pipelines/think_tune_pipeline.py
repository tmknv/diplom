"""
Использование логики diplom/model/fine_tune/think_tune.py
    По сути вынос if __name__ == "__main__":
        в отдельный файл
    Нужно для dvc пайплайна
"""
import os 

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from diplom.utils.load_params import get_params
from diplom.utils.logger import get_logger, setup_logging
from diplom.model.fine_tune.think_tune import ThinkFineTune

params = get_params()
use_dataset_size = params["fine_tune_params"]["use_dataset_size"]
use_dataset_size_eval = params["fine_tune_params"]["use_dataset_size_eval"]

logger = get_logger(__name__)
setup_logging()


def fine_tune_classic_model():

    logger.info(f"Запуск пайплайна дистилляции {params["model"]["base_llm_path_with_think_tokens"]}")

    
    model_path = params["model"]["base_llm_path_with_think_tokens"]
    logger.info(f"Загрузка tokenizer из: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    logger.info(f"Загрузка модели из: {model_path}")
    model_ = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )


    processed_path = params["datasets"]["reasoning_destil_path"]

    train_path = os.path.join(
        processed_path,
        params["datasets"]["train_file"],
    )

    test_path = os.path.join(
        processed_path,
        params["datasets"]["test_file"],
    )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    model = ThinkFineTune(
                        model = model_,
                        tokenizer = tokenizer,
                        mode = "train",
                           )
    
    model.fit(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        use_dataset_size=use_dataset_size,
        use_dataset_size_eval=use_dataset_size_eval,
    )

if __name__ == "__main__":

    fine_tune_classic_model()

    