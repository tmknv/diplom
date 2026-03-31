"""
Модуль для:
    1. Генерации ризонинга моделькой учителем
    2. Составления итогового датасета
    3. Итоговый датасет будет указанной размерности
"""


import requests

from diplom.utils.logger import get_logger
from diplom.utils.load_params import get_params
from diplom.utils.init_secrets import get_settings


logger = get_logger(__name__)
params = get_params()
setting = get_settings()

# API_KEY = setting.API_KEY
# url = params["open_router"]["url"]
# model_name = params["open_router"]["model_name"]

# headers = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }

# data = {
#     "model": model_name,
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Напиши короткую шутку"}
#     ]
# }

# response = requests.post(url, headers=headers, json=data)

# print(response.json())


class DataProcessor:

    def __init__(self, dataset):
        self.dataset = dataset
    
    def get_processed_dataset(self):
        """
        Приводит датасет к виду, удобному для подсчёта F1-score.

        Для датасета openai/gsm8k добавляется новая колонка
        `cleaned_answer_for_f1`, содержащая только итоговый числовой ответ,
        извлечённый из строки ответа с рассуждением.

        Извлечение происходит по шаблону:
            text.split("####")[1].strip()

        Args:
            dataset_path (str): Путь до датасета.

        Returns:
            datasets.DatasetDict | None:
                Обработанный датасет с добавленной колонкой
                `cleaned_answer_for_f1`, либо None, если формат датасета
                не поддерживается.
        """


        dataset = self.dataset

        if dataset is None:
            logger.error("dataset is None")
            return None

        def _extract_answer(example):
            """
            Извлекает финальный ответ из поля `answer`.

            Args:
                example (dict): Пример из датасета.

            Returns:
                dict: Обновлённый пример с новым полем.
            """
            try:
                cleaned = example["answer"].split("####")[1].strip().replace(",",'.')
            except Exception:
                cleaned = None

            return {"one_num_answer": cleaned}

        # применяем ко всем сплитам (train/test)
        dataset = dataset.map(_extract_answer)

        return dataset


class DataProcessorTraining:
    """
    Класс для:
        1. Получения данных для разного обучения
        2. Сохранения обработанных данных
    """

