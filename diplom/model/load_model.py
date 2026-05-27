# diplom/diplom/model/load_model.py
"""
Качаю и сохранаю локально модель с HF.
Name, save path из params.yaml

Сохраняю в:
    diplom/artifactc/models/model_name
"""
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

from diplom.utils.logger import get_logger, setup_logging
from diplom.utils.load_params import get_params
from diplom.utils.init_secrets import get_settings

setup_logging()

logger = get_logger(__name__)
params = get_params()
settings = get_settings()


def load_model():
    """
    Returns:
        tokenizer, model
    """
    token = settings.HF_TOKEN

    model_name = params["model"]["name"]
    save_path = params["model"]["save_path"]

    logger.info(f"Проверка наличия модели в: {save_path}")

    # проверка: есть ли уже модель локально
    if os.path.exists(save_path) and os.listdir(save_path):
        logger.info(f"Модель найдена локально. Загружаю из {save_path}")

        tokenizer = AutoTokenizer.from_pretrained(save_path)
        model = AutoModelForCausalLM.from_pretrained(save_path)

        logger.info("Модель успешно загружена из локального хранилища")
        return tokenizer, model

    logger.info(f"Локальной модели нет. Начинается загрузка: {model_name}")

    # загрузка с HF
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

    logger.info(f"Загрузка окончена: {model_name}")

    # сохранение локально
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    logger.info(f"Модель {model_name} сохранена в {save_path}")
    print(f"Модель сохранена в: {save_path}")

    return tokenizer, model

if __name__ == "__main__":

    tokenizer, model = load_model()

    # проверю на спец токены
    for token in ["<think>", "</think>", "<answer>", "</answer>"]:
        tokens = tokenizer.tokenize(token)
        print(tokens)

