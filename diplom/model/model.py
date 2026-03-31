# diplom/diplom/model/model.py
"""
Основной модуль для работы с моделькой.
    1. Генерация ответов в eval режиме
    2. Дообучение
"""

from typing import Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from diplom.utils.logger import get_logger
from diplom.utils.logger import setup_logging
from diplom.utils.load_params import get_params
from diplom.utils.init_secrets import get_settings

setup_logging()

logger = get_logger(__name__)
params = get_params()
settings = get_settings()


class Model:
    def __init__(self, model, tokenizer, mode: str = "eval"):
        """
        Инициализация обертки над LLM моделью.

        Args:
            model_name (str): Название модели.
            mode (str): Режим работы.
                Возможные значения:
                    - "eval"  — инференс (генерация)
                    - "train" — дообучение
        """
        self.mode = mode

        self.__make_valid_mode()

        self.tokenizer: Optional[AutoTokenizer] = tokenizer
        self.model: Optional[AutoModelForCausalLM] = model
        self.model.to(dtype=torch.bfloat16)

    def __make_valid_mode(self) -> None:
        """
        Проверяет корректность режима работы.

        Raises:
            ValueError: Если передан неподдерживаемый режим.
        """
        valid_modes = {"eval", "train"}

        if self.mode not in valid_modes:
            raise ValueError(
                f"Неверный mode={self.mode}. Доступные режимы: {valid_modes}"
            )

    def generate(
    self,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9
    ) -> str:
        """
        Генерирует ответ модели в режиме инференса (eval).

        Метод:
            1. Формирует диалог (system + user)
            2. Применяет chat template токенайзера
            3. Токенизирует вход
            4. Переносит данные на устройство модели (CPU/GPU)
            5. Запускает генерацию
            6. Возвращает ТОЛЬКО сгенерированную часть (без prompt)

        Args:
            system_prompt (str): Системный промпт (инструкции модели).
            user_prompt (str): Пользовательский запрос.
            max_new_tokens (int, optional): Максимальное количество новых токенов. По умолчанию 128.
            temperature (float, optional): Температура сэмплинга (чем выше — тем более случайный ответ). По умолчанию 0.7.
            top_p (float, optional): Параметр nucleus sampling. По умолчанию 0.9.

        Returns:
            str: Сгенерированный ответ модели (только новая часть, без исходного prompt).

        Raises:
            ValueError: Если модель не в режиме "eval".
            RuntimeError: Если модель или токенайзер не загружены.
        """

        # Проверка режима
        if self.mode != "eval":
            raise ValueError("Метод generate доступен только в режиме 'eval'")

        # Проверка инициализации
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Модель или токенайзер не загружены")

        tokenizer = self.tokenizer
        model = self.model

        # Перевод модели в режим инференса
        model.eval()

        # Формируем диалог
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Применяем chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Токенизация + перенос на устройство (ВАЖНО: не превращаем в dict!)
        device = next(model.parameters()).device
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        # Генерация
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                # eos_token_id=tokenizer.eos_token_id  # можно включить при необходимости
            )

        # Отделяем только сгенерированную часть
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]

        # Декодируем
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return answer
    
    
    def fit(self) -> None:
        """
        Запускает процесс дообучения модели.

        ВАЖНО:
            Это каркасный метод. Реальная логика обучения (Trainer, LoRA,
            DPO и т.д.) должна быть реализована здесь или вынесена
            в отдельный модуль.

        Args:
            train_dataset: Датасет для обучения.
            **kwargs: Дополнительные параметры обучения.

        Raises:
            ValueError: Если режим не "train".
            RuntimeError: Если модель не удалось загрузить.
        """
        
        if self.mode != "train":
            raise ValueError("Метод fit доступен только в режиме 'train'")
        
        raise NotImplementedError(
            "Метод fit ещё не реализован. "
            "Переопределение будет в дочернем классе."
        )

    



if __name__ == "__main__":

    import os
    import json 

    from tqdm import tqdm

    from diplom.model.load_model import load_model
    from diplom.model.validate_model import ValidateBaseModel
    from diplom.data_processing.load_data import load_data
    from diplom.data_processing.data_preprocessing import DataProcessor
    tokenizer, model = load_model()
    model_name = "qwen2_5_3b"

    system_prompt = params["prompt"]["qwen"]["system_prompt"]
    user_prompt = params["prompt"]["qwen"]["user_prompt"]

    
    model = Model(model = model, tokenizer=tokenizer, mode = "eval")
    validator = ValidateBaseModel()
    
    dataset = load_data()

    DP = DataProcessor(dataset=dataset)

    processed_dataset = DP.get_processed_dataset()

    print(processed_dataset["test"][0])
    answers = []

    # Создаём директорию, если её нет
    # output_dir = "/diplom/artifacts"
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = "artifacts"
    output_file = os.path.join(output_dir, "answers.json")

    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(ange(101)):

        dataset["test"][0:101]
        question =  dataset["test"][idx]["question"]
        valid_answer =  dataset["test"][idx]["answer"]

        prompt = user_prompt.replace(
            "{MATH_PROBLEM}", question
        )

        answ = model.generate(
            system_prompt=system_prompt, 
            user_prompt=prompt
        )
        answers.append({
            "question": question,
            "answer": answ,
            "valid_answer": valid_answer
        })
        
        # Сохраняем после каждого ответа (или раз в N итераций)
        if (idx + 1) % 10 == 0:  # сохраняем каждые 10 ответов
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(answers, f, ensure_ascii=False, indent=2)
                print(f"Сохранено {idx + 1} ответов")
                print(f"question number {idx + 1}", question)
                print(f"answer number {idx + 1}", answ)

    # Финальное сохранение
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    print(f"Готово! Сохранено {len(answers)} ответов в {output_file}")
