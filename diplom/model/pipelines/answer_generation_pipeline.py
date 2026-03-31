# diplom/diplom/model/pipelines/answer_generation.py  
"""
Пайплайн для генерации ответов в нужном формате моделькой

Мудь поддерживает: 
    1. Чекпоинты 
    2. генерация в нужном формате 
    3. Сохранение результатов 
"""


"""
Пайплайн для генерации ответов моделью.

Поддерживает:
    1. Чекпоинты (resume)
    2. Генерацию в нужном формате
    3. Сохранение результатов
"""


    
    

import os
import json
from typing import List, Dict, Optional

from tqdm import tqdm

from diplom.model.model import Model
from diplom.model.load_model import load_model
from diplom.data_processing.load_data import load_data
from diplom.data_processing.data_preprocessing import DataProcessor
from diplom.utils.logger import get_logger
from diplom.utils.logger import setup_logging
from diplom.utils.load_params import get_params

setup_logging()
logger = get_logger(__name__)

import json
import re

def _postprocess_answer(answer: str) :
    """ 
    Note:
        Хоть и прошу SLM возвращать строго {"answer": string},\n
            но у нее не получается.\n 
                Появляются доп символы.
    Description:
        Привести к dict виду c помощью:
        1. вырезает JSON из текста
        2. убирает ```json блоки
        3. чистит спецсимволы
        4. пытается починить кривой JSON
    Args:
        answer (str): строка с нужным ответ.
    Returns:
        dict: почищенный ответ
    """

    if not answer or not isinstance(answer, str):
        logger.error("Пустой или невалидный ответ")
        return None

    try:
        # убираю markdown ```json ```
        cleaned = re.sub(r"```.*?```", "", answer, flags=re.DOTALL).strip()

        # ищу JSON внутри строки 
        start = cleaned.find("{")
        end = cleaned.find("}")
        # end = cleaned.rfind("}")

        if start == -1 or end == -1:
            logger.warning("JSON не найден в ответе")
            return None

        cleaned = cleaned[start:end + 1]

        # чищу спецсимволы 
        cleaned = cleaned.replace("\n", " ")
        cleaned = cleaned.replace("\t", " ")

        # иногда модель юзает одинарные кавычки
        #   заменяю только ключи/строки
        cleaned = re.sub(r"'", '"', cleaned)

        # убираю trailing commas
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)

        # пробую распарсить 
        parsed = json.loads(cleaned)

        logger.info("Ответ успешно приведён к валидному JSON")
        return parsed

    except json.JSONDecodeError as e:
        logger.warning(f"json decode error: {e}")

        # fallback: вытаскиваю хотя бы answer поле 
        try:
            match = re.search(r'"answer"\s*:\s*"(.+?)"', cleaned)
            if match:
                fallback = {"answer": match.group(1)}
                logger.info("Извлечено поле answer через regex fallback")
                return fallback
        except Exception:
            pass

        logger.error("Не удалось привести ответ к валидному виду")
        return None

    except Exception as e:
        logger.error(f"Ошибка постпроцессинга: {e}")
        return None


def generate_answers_pipeline(
    model,
    dataset,
    system_prompt: str,
    user_prompt_template: str,
    output_path: str,
    max_samples: Optional[int] = None,
    save_every: int = 10,
    resume: bool = True,
    dataset_mode: str = "test"
) -> List[Dict]:
    """
    Генерирует ответы модели и сохраняет их в файл.

    Args:
        model: Объект Model (обёртка над LLM)
        dataset: dataset["test"]
        system_prompt (str): системный промпт
        user_prompt_template (str): шаблон user prompt с {MATH_PROBLEM}
        output_path (str): путь до json файла
        max_samples (int, optional): сколько примеров обработать
        save_every (int): частота сохранения
        resume (bool): продолжать с чекпоинта
        dataset_mode (str): train/test

    Returns:
        None
    """

    logger.info("Запуск пайплайна генерации ответов")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    answers: List[Dict] = []

    #resume логика
    start_idx = 0
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                answers = json.load(f)
                start_idx = len(answers)

            logger.info(f"Найден чекпоинт. Продолжаем с idx={start_idx}")

        except Exception as e:
            logger.warning(f"Не удалось загрузить чекпоинт: {e}")

    # ограничение по датасету 
    total_size = len(dataset[dataset_mode])
    if max_samples:
        total_size = min(total_size, max_samples)

    logger.info(f"Всего примеров к обработке: {total_size}")

    # основной цикл 
    for idx in tqdm(range(start_idx, total_size)):
        example = dataset[dataset_mode][idx]
        try:
            
            question = example["question"]
            valid_answer = example.get("answer", None)
            valid_answer_f1 = example.get("one_num_answer", None)


            # формируем prompt
            user_prompt = user_prompt_template.replace(
                "{MATH_PROBLEM}", question
            )

            answer = model.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            cleaned_model_answer = _postprocess_answer(answer=str(answer))

            if valid_answer:

                result = {
                    "question": question,
                    "valid_answer": valid_answer,
                    "model_answer": answer,
                    "cleaned_model_answer": cleaned_model_answer,
                    "valid_answer_f1": valid_answer_f1
                }

                answers.append(result)
            
            else: # считаю что за 4 генерации хватит на ответ, который можно сделать валидным
                for _ in range(3):

                    answer = model.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                        )

                    valid_answer = _postprocess_answer(answer=str(answer))

                    if valid_answer:

                        result = {
                        "question": question,
                        "valid_answer": valid_answer,
                        "model_answer": answer,
                        "cleaned_model_answer": cleaned_model_answer,
                        "valid_answer_f1": valid_answer_f1
                        }

                        answers.append(result)
                        break



        except Exception as e:
            logger.error(f"Ошибка на idx={idx}: {e}")

            answers.append({
                "question": example.get("question", None),
                "valid_answer": example.get("answer", None),
                "model_answer": None,
                "cleaned_model_answer": None,
                "valid_answer_f1": example.get("one_num_answer", None),
                "error": str(e)
            })

        #промежуточное сохранение (чтобы потом по чекпоинтам прыгать легко можно было)
        if (idx + 1) % save_every == 0:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(answers, f, ensure_ascii=False, indent=2)

                logger.info(f"Сохранено {idx + 1} ответов")

            except Exception as e:
                logger.error(f"Ошибка при сохранении: {e}")

    # финальное сохранение 
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Готово! Сохранено {len(answers)} ответов в {output_path}"
        )

    except Exception as e:
        logger.error(f"Финальная ошибка сохранения: {e}")


def run_generate_answers_pipeline(llm_type: str = "base_llm", max_samples: int = 300):
    
    params = get_params()
    
    save_path = params["llm_result"]["answer_generation"][llm_type]
    system_prompt = params["prompt"]["qwen"]["system_prompt"]
    user_prompt = params["prompt"]["qwen"]["user_prompt"]

    tokenizer, model_ = load_model()

    dataset = load_data()
    DP = DataProcessor(dataset = dataset)
    preprocessed_dataset = DP.get_processed_dataset()

    model = Model(tokenizer = tokenizer, model = model_)

    generate_answers_pipeline(
        model = model,
        dataset = preprocessed_dataset,
        system_prompt= system_prompt,
        user_prompt_template = user_prompt,
        output_path = save_path,
        max_samples= max_samples,
        save_every = 10,
        resume = True,
        dataset_mode= "test"
    )

if __name__ == "__main__":
    run_generate_answers_pipeline(llm_type="base_llm_classic_fine_tuned")
    