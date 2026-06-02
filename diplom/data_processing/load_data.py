# diplom/data/load_data.py
"""
Загружаем датасет с HF и сохраняем локально.
Name, save path из params.yaml

Сохраняем в:
    diplom/artifacts/datasets/dataset_name
"""
import os

from datasets import load_dataset, load_from_disk

from diplom.utils.logger import get_logger
from diplom.utils.logger import setup_logging
from diplom.utils.load_params import get_params
from diplom.utils.init_secrets import get_settings

setup_logging()

logger = get_logger(__name__)
params = get_params()
settings = get_settings()


def load_and_store_data() -> None:
    token = settings.HF_TOKEN

    dataset_name = params["datasets"]["math"]

    full_path = params["datasets"]["math_full_path"]


    logger.info(f"Проверка наличия датасета в: {full_path}")

    # проверка: есть ли датасет локально
    if os.path.exists(full_path) and os.listdir(full_path):
        logger.info(f"Датасет найден локально. Загружаю из {full_path}")

        dataset = load_from_disk(full_path)

        logger.info("Датасет успешно загружен из локального хранилища")
        return dataset

    logger.info(f"Локального датасета нет. Начинается загрузка: {dataset_name}")

    # загрузка с HF
    try:
        dataset = load_dataset(dataset_name, "main", token=token)
    except Exception:
        dataset = load_dataset(dataset_name, token=token)

    logger.info(f"Загрузка окончена: {dataset_name}")

    # сохранение локально
    try:
        os.makedirs(full_path, exist_ok=True)
        dataset.save_to_disk(full_path)

        logger.info(f"Датасет {dataset_name} сохранен в {full_path}")
        print(f"Датасет сохранен в: {full_path}")
        return dataset

    except Exception as e:
        logger.error(f"Не смогли сохранить датасет: {e}")


if __name__ == "__main__":
    load_and_store_data()