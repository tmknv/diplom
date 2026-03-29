# diplom/diplom/model/validate_model.py
"""
Модуль для получения результатов работы модели

Сохраняем в:
    diplom/artifacts/metrics/model_name
"""
import os
from abc import ABC, abstractmethod

from datasets import load_dataset

from diplom.utils.logger import get_logger
from diplom.utils.logger import setup_logging
from diplom.utils.load_params import get_params
from diplom.utils.init_secrets import get_settings
from diplom.schemas import Metrics

setup_logging()

logger = get_logger(__name__)
params = get_params()
settings = get_settings()

class Validator(ABC):
    
    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def validate(model_answ, dataset) -> Metrics:
        ...


    

class ValidateBaseModel(Validator):

        
    def validate(self, slm_outs: dict) -> Metrics:
        """
        Description:
            Вычисляет f1.
        Note:
            Используется упрощённая формула:
                score = TA / (TA + FA)

            where:
                TA (True Answer) — количество правильных ответов модели
                FA (False Answer) — количество неправильных ответов модели

        Args:
            slm_out (dict): словарь. Выход из diplom/model/pipelines/answer_generation.py
                - question (str): вопрос к модели
                - valid_answer (str): ответ llm с reasoning
                - model_answer (str/dict): грязный выход из slm
                - cleaned_model_answer (dict): dict приведеннный ответ
                - valid_answer_f1 (str): вырезанный f1 ответ из valid_answer

        Returns:
            Metrics:
                - validator_name (str): имя валидатора
                - model_name (str): имя оцениваемой модельки
                - f1-score (float): метрика качества модельки
                - meta (dict): сюда летят причины f1-score = 0 (ошибки технические, авторские, ..)
        """

        model_name = params["model"]["name"]

        if slm_outs is None:
            logger.error("slm_outs не передано")
            return Metrics(
                validator_name=self.name,
                model_name=model_name,
                f1_score=0,
                meta={
                    "error":"slm_outs не передано"
                }
            )

        ta = 0  # True answers
        fa = 0  # False answers

        for out in slm_outs:
            
            cleaned_model_answer = out["cleaned_model_answer"]
            valid_answer_f1 = out["valid_answer_f1"]

            if cleaned_model_answer is None or cleaned_model_answer is None:
                fa +=1 
                continue
            
            if float(cleaned_model_answer["answer"]) == float(valid_answer_f1):

                ta +=1
            else: 
                fa +=1

        total = ta + fa

        score = (ta / total if total != 0 else 0)

        logger.info(f"F1-score (precision-like): {score:.4f} (TC={ta}, FC={fa})")

        return Metrics(
                validator_name=self.name,
                model_name=model_name,
                f1_score=score.__round__(4)
            )
        

if __name__ == "__main__":

    import json 

    llm_type = "base_llm"
    path = params["llm_result"]["answer_generation"][llm_type]

    with open (path, "r", encoding = "utf-8") as f:

        slm_outs = json.load(f)

    validator = ValidateBaseModel()
    answ = validator.validate(slm_outs = slm_outs)

    print (answ)

    