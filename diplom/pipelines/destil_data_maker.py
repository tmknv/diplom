import os

from diplom.data_processing.data_preprocessing import DataProcessor
from diplom.data_processing.load_data import load_and_store_data
from diplom.utils.logger import setup_logging, get_logger
from diplom.utils.load_params import get_params

setup_logging()

logger = get_logger(__name__)
params = get_params()


def run_df_maker():

    processed_path = params["datasets"]["reasoning_destil_path"]


    os.makedirs(processed_path, exist_ok=True)

    train_path = os.path.join(
        processed_path,
        params["datasets"]["train_file"],
    )

    test_path = os.path.join(
        processed_path,
        params["datasets"]["test_file"],
    )

    if (
        os.path.isfile(train_path)
        and os.path.isfile(test_path)
    ):
        logger.info("Processed dataset already exists")
        return

    # Скачивает если нет
    dataset = load_and_store_data()
    if dataset:
        logger.info(f"Датасет получен")
    else:
        logger.error("Не получилось загрузить датасет")
        return


    processor = DataProcessor(dataset=dataset)

    dataset = processor.get_df_reasoning_destil()

    if "train" in dataset:
        dataset["train"].to_parquet(train_path)
        logger.info(f"Saved train -> {train_path}")

    if "test" in dataset:
        dataset["test"].to_parquet(test_path)
        logger.info(f"Saved test -> {test_path}")


if __name__ == "__main__":
    run_df_maker()