# diplom/data_processing/data_preprocessing.py
"""
Модуль для:
    1. Генерации ризонинга моделькой учителем
    2. Составления итогового датасета
    3. Итоговый датасет будет указанной размерности
"""

import re 

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
    

    def _extract_reasoning(self, sample):
        """
        Извлекает reasoning из answer и удаляет GSM8K markup:
        <<8*3=24>>

        Args:
            sample (dict): Элемент датасета.

        Returns:
            dict: reasoning для элемента датасета.
        """
        try:
            # Всё до ####
            cleaned = sample["answer"].split("####")[0].strip()

            # Удаляем <<...>>
            cleaned = re.sub(r"<<.*?>>", "", cleaned)

            # Нормализуем запятые
            cleaned = cleaned.replace(",", ".")

            # Убираем лишние пробелы
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        except Exception:
            cleaned = None

        return {"reasoning": cleaned}
    

    def _extract_answer(self, example):
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
        
        # применяем ко всем сплитам (train/test)
        dataset = dataset.map(self._extract_answer)

        return dataset
    
    def get_df_reasoning_destil(self):
        """
        Приводит датасет к виду, необходимому для дистилляции ризонинга.

        Для датасета openai/gsm8k добавляется новые колонки:
        1.  `cleaned_answer_for_f1`. Колнака, содержащая только итоговый числовой ответ.
        2. `reasoning`. Колонка, содержащая рассуждения.

        Returns:
            datasets.DatasetDict | None:
                Обработанный датасет с новыми колонками
        """

        dataset = self.dataset

        if dataset is None:
            logger.error("dataset is None")
            return None
        
        # применяем ко всем сплитам (train/test)
        dataset = dataset.map(self._extract_answer)
        dataset = dataset.map(self._extract_reasoning)

        return dataset

        






# class DataProcessorTraining:
#     """
#     Класс для:
#         1. Получения данных для разного обучения
#         2. Сохранения обработанных данных
#     """


