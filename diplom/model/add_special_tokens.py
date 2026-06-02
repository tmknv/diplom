# diplom/models/add_think_tokens.py
"""
Добавление специальных токенов <think> и </think>
в tokenizer и model embeddings.

Модель загружается из пути в params.yaml,
после чего:
    1. Добавляются специальные токены
    2. Resize embeddings
    3. Модель и tokenizer сохраняются локально

Сохраняем в:
    diplom/artifacts/models/qwen2_5_3b_think
"""

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from diplom.utils.load_params import get_params
from diplom.utils.logger import get_logger
from diplom.utils.logger import setup_logging

setup_logging()

logger = get_logger(__name__)
params = get_params()


def add_think_tokens() -> None:
    """
    Добавляет специальные токены reasoning-разметки
    в tokenizer и embeddings модели.

    Загружает базовую модель из params.yaml,
    добавляет:
        - <think>
        - </think>

    После этого:
        - resize token embeddings
        - сохраняет обновленную модель
        - сохраняет tokenizer

    Raises:
        Exception:
            Ошибка загрузки модели или сохранения.
    """

    base_model_path = params["model"]["base_llm_path"]

    save_path = params["model"]["base_llm_path_with_think_tokens"]

    logger.info(f"Загрузка tokenizer из: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    logger.info(f"Загрузка модели из: {base_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    special_tokens = {
        "additional_special_tokens": [
            "<think>",
            "</think>",
        ]
    }

    logger.info("Добавление специальных токенов")

    num_added_tokens = tokenizer.add_special_tokens(special_tokens)

    logger.info(f"Добавлено токенов: {num_added_tokens}")

    logger.info("Resize embeddings модели")

    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Сохранение tokenizer и модели в: {save_path}")

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    logger.info("Модель успешно сохранена")

    # проверка
    test_text = "<think>hello</think>"

    token_ids = tokenizer.encode(
        test_text,
        add_special_tokens=False,
    )

    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    logger.info(f"Проверка токенизации: {decoded_tokens}")


if __name__ == "__main__":
    add_think_tokens()