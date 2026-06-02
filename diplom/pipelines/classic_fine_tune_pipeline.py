"""
Использование логики diplom/model/fine_tune/classic_model_fine_tune.py
    По сути вынос if __name__ == "__main__":
        в отдельный файл
    Нужно для dvc пайплайна
"""

from diplom.utils.load_params import get_params

from diplom.utils.logger import get_logger, setup_logging

from diplom.model.load_model import load_model
from diplom.data_processing.load_data import load_data

from diplom.data_processing.data_preprocessing import DataProcessor
from diplom.model.fine_tune.classic_model_fine_tune import ClassicFineTune

params = get_params()
use_dataset_size = params["fine_tune_params"]["use_dataset_size"]
use_dataset_size_eval = params["fine_tune_params"]["use_dataset_size_eval"]

logger = get_logger(__name__)
setup_logging()


def fine_tune_classic_model():

    logger.info(f"Запуск пайплайна классического обучения для {params["model"]["name"]}")

    dataset = load_data()
    tokenizer, model_ = load_model()

    DP = DataProcessor(dataset = dataset)
    proccessed_dataset = DP.get_processed_dataset()

    model = ClassicFineTune(
                        model = model_,
                        tokenizer = tokenizer,
                        mode = "train",
                           )
    
    model.fit(
        dataset = proccessed_dataset,
        use_dataset_size = use_dataset_size,
        use_dataset_size_eval = use_dataset_size_eval
    )

if __name__ == "__main__":

    fine_tune_classic_model()

    