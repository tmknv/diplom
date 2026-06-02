# diplom/diplom/model/pipelines/answer_generation.py
"""
Пайплайн для генерации ответов в формате:
<think>
reasoning
</think>
final answer

Поддерживает:
    1. случайную выборку из test.parquet
    2. resume по уже сохранённому json
    3. генерацию через ваш diplom.model.model.Model
    4. устойчивость к галлюцинациям формата (нет <think>, нет </think>, несколько блоков и т.д.)
    5. расчёт F1 для reasoning и ответа
    6. сохранение результатов
"""

import os
import json
import re
import random
from collections import Counter
from typing import List, Dict, Optional, Any

import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from diplom.model.model import Model
from diplom.utils.logger import get_logger, setup_logging
from diplom.utils.load_params import get_params

setup_logging()
logger = get_logger(__name__)


# =============================
# Text helpers
# =============================
def _strip_code_fences(text: str) -> str:
    """
    Убирает markdown fences вида:
    ```json
    ...
    ```
    не вырезая содержимое.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    cleaned = re.sub(
        r"^\s*```(?:json|text|markdown)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _normalize_for_f1(text: Optional[str]) -> str:
    """
    Нормализация текста перед F1:
    - lower
    - убираем лишние пробелы
    - убираем think-теги
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.lower().strip()
    text = _strip_code_fences(text)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_for_f1(text: Optional[str]) -> List[str]:
    """
    Токенизация для F1.
    """
    text = _normalize_for_f1(text)
    if not text:
        return []

    tokens = re.findall(
        r"\d+(?:[.,]\d+)?|[A-Za-zА-Яа-яЁё]+|[^\w\s]",
        text,
        flags=re.UNICODE,
    )
    return tokens


def calc_f1(pred: Optional[str], gold: Optional[str]) -> Optional[float]:
    """
    Классический token-level F1.
    """
    if pred is None or gold is None:
        return None

    pred_tokens = _tokenize_for_f1(pred)
    gold_tokens = _tokenize_for_f1(gold)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def extract_final_number(text: Optional[str]) -> Optional[str]:
    """
    Достаёт последний числовой токен из ответа.
    Полезно для GSM8K-like задач.
    """
    if text is None:
        return None

    s = str(text).replace(",", ".")
    matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not matches:
        return None
    return matches[-1]


# =============================
# Parsing helpers
# =============================
def parse_reasoning_answer(text: Any) -> Dict[str, Any]:
    """
    Парсит ответ модели в формате:

    <think>
    reasoning
    </think>
    final answer

    Устойчив к:
        - отсутствию <think>
        - отсутствию </think>
        - нескольким think-блокам
        - code fences
        - лишнему тексту вокруг
    """
    if text is None:
        return {
            "reasoning": None,
            "answer": None,
            "raw_response": None,
            "has_think": False,
            "parse_status": "empty_response",
        }

    if not isinstance(text, str):
        text = str(text)

    raw_response = text
    cleaned = _strip_code_fences(text)

    if not cleaned:
        return {
            "reasoning": None,
            "answer": None,
            "raw_response": raw_response,
            "has_think": False,
            "parse_status": "empty_after_cleaning",
        }

    think_pattern = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
    matches = list(think_pattern.finditer(cleaned))

    # Нормальный случай: есть complete think-блок
    if matches:
        reasoning_parts = []
        for m in matches:
            part = m.group(1).strip()
            if part:
                reasoning_parts.append(part)

        reasoning = "\n\n".join(reasoning_parts).strip() if reasoning_parts else None
        final_answer = cleaned[matches[-1].end():].strip()

        parse_status = "ok"
        if len(matches) > 1:
            parse_status = "multiple_think_blocks"
        if not final_answer:
            parse_status = (
                f"{parse_status}_no_final_answer"
                if parse_status != "ok"
                else "no_final_answer"
            )

        return {
            "reasoning": reasoning,
            "answer": final_answer if final_answer else None,
            "raw_response": raw_response,
            "has_think": True,
            "parse_status": parse_status,
        }

    # Есть открывающий тег, но нет закрывающего
    lower_cleaned = cleaned.lower()
    open_tag = lower_cleaned.find("<think>")
    close_tag = lower_cleaned.find("</think>")

    if open_tag != -1 and close_tag == -1:
        reasoning = cleaned[open_tag + len("<think>"):].strip()
        return {
            "reasoning": reasoning if reasoning else None,
            "answer": None,
            "raw_response": raw_response,
            "has_think": True,
            "parse_status": "missing_closing_think",
        }

    # Есть закрывающий тег, но нет открывающего
    if close_tag != -1 and open_tag == -1:
        final_answer = cleaned[close_tag + len("</think>"):].strip()
        if not final_answer:
            final_answer = cleaned.strip()
        return {
            "reasoning": None,
            "answer": final_answer if final_answer else None,
            "raw_response": raw_response,
            "has_think": False,
            "parse_status": "missing_opening_think",
        }

    # Вообще нет think тегов
    return {
        "reasoning": None,
        "answer": cleaned.strip() if cleaned.strip() else None,
        "raw_response": raw_response,
        "has_think": False,
        "parse_status": "no_think",
    }


# =============================
# I/O helpers
# =============================
def _save_answers(output_path: str, answers: List[Dict[str, Any]]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)


def _load_dataset_from_parquet(test_path: str) -> DatasetDict:
    """
    Загружает test.parquet в формат DatasetDict с одним сплитом test.
    """
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Файл датасета не найден: {test_path}")

    df = pd.read_parquet(test_path)
    if df is None or len(df) == 0:
        raise ValueError(f"Пустой датасет: {test_path}")

    dataset = Dataset.from_pandas(df, preserve_index=False)
    return DatasetDict({"test": dataset})


def _load_reasoning_prompts(params: dict) -> tuple[str, str]:
    """
    Поддерживает структуру params.yaml вида:
        params["prompt"]["qwen"]["system_prompt_think"]
        params["prompt"]["qwen"]["user_prompt_think"]
    и fallback на system_prompt/user_prompt.
    """
    if "prompt" in params and "qwen" in params["prompt"]:
        qwen_cfg = params["prompt"]["qwen"]
        system_prompt = qwen_cfg.get("system_prompt_think", qwen_cfg.get("system_prompt"))
        user_prompt = qwen_cfg.get("user_prompt_think", qwen_cfg.get("user_prompt"))
        if system_prompt and user_prompt:
            return system_prompt, user_prompt

    raise KeyError(
        "Не найдены промпты. Ожидаются params['prompt']['qwen']['system_prompt_think'] "
        "и params['prompt']['qwen']['user_prompt_think'] "
        "или fallback system_prompt / user_prompt."
    )


# =============================
# Model loading
# =============================
def load_reasoning_model() -> Model:
    """
    Загружает дообученную модель из:
    diplom/artifacts/fine_tuned_models/qwen2_5_3b_think/checkpoint-750
    и оборачивает её в ваш Model.
    """
    model_path = "diplom/artifacts/fine_tuned_models/qwen2_5_3b_think/checkpoint-750"

    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
    )

    return Model(
        model=model,
        tokenizer=tokenizer,
        mode="eval",
    )


# =============================
# Main pipeline
# =============================
def generate_answers_pipeline(
    model: Model,
    dataset,
    system_prompt: str,
    user_prompt_template: str,
    output_path: str,
    max_samples: Optional[int] = None,
    save_every: int = 10,
    resume: bool = True,
    dataset_mode: str = "test",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Генерирует ответы модели на случайных примерах из test-сплита
    и сохраняет результаты в json.
    """
    logger.info("Запуск пайплайна генерации ответов")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    answers: List[Dict[str, Any]] = []
    processed_indices = set()

    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                answers = json.load(f)

            for item in answers:
                idx = item.get("dataset_idx", None)
                if idx is not None:
                    processed_indices.add(idx)

            logger.info(f"Найден чекпоинт: {len(answers)} записей, resume включён")
        except Exception as e:
            logger.warning(f"Не удалось загрузить чекпоинт: {e}")
            answers = []
            processed_indices = set()

    if dataset_mode not in dataset:
        raise KeyError(
            f"Сплит '{dataset_mode}' не найден в dataset. Доступны: {list(dataset.keys())}"
        )

    split = dataset[dataset_mode]
    total_size = len(split)

    if total_size == 0:
        logger.warning(f"Сплит {dataset_mode} пустой")
        return answers

    # Случайная выборка индексов из test
    rng = random.Random(seed)
    all_indices = list(range(total_size))
    rng.shuffle(all_indices)

    if max_samples is not None:
        selected_indices = all_indices[: min(max_samples, total_size)]
    else:
        selected_indices = all_indices

    logger.info(
        f"Всего в {dataset_mode}: {total_size}, "
        f"выбрано для прогонки: {len(selected_indices)}, seed={seed}"
    )

    new_items_since_save = 0

    for idx in tqdm(selected_indices, desc=f"Generating on {dataset_mode}"):
        if idx in processed_indices:
            continue

        example = split[idx]

        try:
            question = example.get("question", None)
            valid_reasoning = example.get("reasoning", None)
            valid_answer = example.get("answer", None)
            valid_answer_f1 = example.get("one_num_answer", None)

            if question is None:
                raise ValueError(f"В примере idx={idx} нет поля 'question'")

            user_prompt = user_prompt_template.replace("{MATH_PROBLEM}", question)

            # Генерация через ваш Model
            generated = model.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.9,
            )

            parsed = parse_reasoning_answer(generated)

            model_reasoning = parsed["reasoning"]
            model_answer = parsed["answer"]
            full_response = parsed["raw_response"]
            has_think = parsed["has_think"]
            parse_status = parsed["parse_status"]

            reasoning_f1 = calc_f1(model_reasoning, valid_reasoning)

            gold_answer_for_f1 = (
                valid_answer_f1 if valid_answer_f1 is not None else valid_answer
            )
            answer_f1 = calc_f1(model_answer, gold_answer_for_f1)

            model_numeric_answer = extract_final_number(model_answer)
            numeric_answer_f1 = calc_f1(model_numeric_answer, valid_answer_f1)

            result = {
                "dataset_idx": idx,
                "question": question,

                # gold
                "valid_reasoning": valid_reasoning,
                "valid_answer": valid_answer,
                "valid_answer_f1": valid_answer_f1,

                # prediction
                "model_reasoning": model_reasoning,
                "model_answer": model_answer,
                "model_numeric_answer": model_numeric_answer,

                # metrics
                "reasoning_f1": reasoning_f1,
                "answer_f1": answer_f1,
                "numeric_answer_f1": numeric_answer_f1,

                # raw / debug
                "full_response": full_response,
                "has_think": has_think,
                "parse_status": parse_status,
            }

            answers.append(result)
            processed_indices.add(idx)
            new_items_since_save += 1

        except Exception as e:
            logger.error(f"Ошибка на idx={idx}: {e}")

            answers.append({
                "dataset_idx": idx,
                "question": example.get("question", None),

                "valid_reasoning": example.get("reasoning", None),
                "valid_answer": example.get("answer", None),
                "valid_answer_f1": example.get("one_num_answer", None),

                "model_reasoning": None,
                "model_answer": None,
                "model_numeric_answer": None,

                "reasoning_f1": None,
                "answer_f1": None,
                "numeric_answer_f1": None,

                "full_response": None,
                "has_think": False,
                "parse_status": "error",
                "error": str(e),
            })

            processed_indices.add(idx)
            new_items_since_save += 1

        if save_every > 0 and new_items_since_save % save_every == 0:
            try:
                _save_answers(output_path, answers)
                logger.info(f"Промежуточно сохранено {len(answers)} записей в {output_path}")
            except Exception as e:
                logger.error(f"Ошибка при промежуточном сохранении: {e}")

    try:
        _save_answers(output_path, answers)
        logger.info(f"Готово! Сохранено {len(answers)} ответов в {output_path}")
    except Exception as e:
        logger.error(f"Финальная ошибка сохранения: {e}")

    return answers


def run_generate_answers_pipeline(
    max_samples: int = 300,
    seed: int = 7,
):
    """
    Точка входа:
    - грузит дообученную модель из checkpoint-750
    - грузит test.parquet
    - берёт случайные строки из test
    - сохраняет json с предсказаниями и метриками
    """
    params = get_params()

    OUTPUT_PATH = params["llm_result"]["answer_generation"]["reasoning_tuned_llm"]

    system_prompt, user_prompt = _load_reasoning_prompts(params)

    dataset_path = "diplom/artifacts/datasets_for_tune/test.parquet"
    dataset = _load_dataset_from_parquet(dataset_path)

    model = load_reasoning_model()

    generate_answers_pipeline(
        model=model,
        dataset=dataset,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt,
        output_path=OUTPUT_PATH,
        max_samples=max_samples,
        save_every=10,
        resume=True,
        dataset_mode="test",
        seed=seed,
    )


if __name__ == "__main__":
    run_generate_answers_pipeline(
        max_samples=300,
        seed=7,
    )