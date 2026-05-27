# diplom/diplom/model/reasoning_destil.py
"""
Модуль для дестилляции ризонинга

Сохраняю в:
    diplom/artifactc/models/model_name
"""


# diplom/diplom/model/classic_model_fine_tune.py
"""
Классическое обучение модельки.
Вопрос - ответ
Сохраняем результат в 
    "diplom/artifacts/models/fine_tuned/classic/qwen2_5_3b"
"""

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import re
# import math
import torch
from typing import Any, Dict

from diplom.model.model import Model
from diplom.utils.logger import get_logger, setup_logging
from diplom.utils.load_params import get_params
from diplom.data_processing.load_data import load_data

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

setup_logging()

logger = get_logger(__name__)
params = get_params()


# также борюсь с очищением кеша
class ClearMPSCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.mps.empty_cache()
        return control
    
class ReasoningFineTune(Model):
    def __init__(self, model, tokenizer, mode: str = "eval"):
        """
        Инициализация обертки для классического дообучения.

        Args:
            model: Экземпляр AutoModelForCausalLM.
            tokenizer: Экземпляр AutoTokenizer.
            mode (str): Режим работы ("eval" или "train").
        """
        super().__init__(model, tokenizer, mode)

        if self.mode == "train":
            self._set_train_regime()

    def _set_train_regime(self) -> None:
        """
        Настраивает модель для режима дообучения (train).

        Выполняет:
            - Перевод модели в режим train
            - Включает gradient checkpointing (экономия памяти)
            - Логирует начало подготовки к обучению
        """
        self.model.train()
        # Включаем gradient checkpointing для экономии VRAM (важно для 3B-модели)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing включён")

        logger.info("Модель переведена в режим обучения (train)")

    def __preprocess_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
            Форматирует один пример GSM8K под chat-template + маскирует loss на промпте.
        """
        system_prompt = params["prompt"]["qwen"]["system_prompt"]
        user_prompt_template = params["prompt"]["qwen"]["user_prompt"]

        question = example["question"]
        raw_answer = example["answer"]

        
        assistant_content = example["one_num_answer"]

        #  Промпт (system + user) с add_generation_prompt=True
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.replace("{MATH_PROBLEM}", question)},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ассистанс. маскирую все что в промпте, чтобы он чисто учил в каких ситуациях какие числа использовать
        full_messages = prompt_messages + [{"role": "assistant", "content": assistant_content}]
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Токенизация
        prompt_ids = self.tokenizer(prompt_text, return_tensors=None)["input_ids"]
        full_ids = self.tokenizer(full_text, return_tensors=None)["input_ids"]

        # Маскируем loss на промпте
        labels = full_ids.copy()
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def fit(self, dataset, use_dataset_size: int = 1000, use_dataset_size_eval: int = 30) -> None:
        """
        Запускает процесс  дообучения ризонингу (SFT) модели Qwen2.5-3B
        на датасете openai/gsm8k.

        Особенности обучения:
            1. Формируется chat-формат: system + user (с {MATH_PROBLEM}) + assistant
            2. Assistant отвечает ТОЛЬКО числом (финальный ответ из GSM8K, извлекается по ####)
            3. Loss считается только на ответе модели (prompt маскируется -100)
            4. Используется Trainer + DataCollatorForSeq2Seq
            5. Модель сохраняется в diplom/artifacts/models/classic_fine_tuned_qwen2_5_3b

        Raises:
            ValueError: Если режим не "train".
            RuntimeError: Если модель или токенайзер не загружены.
        """
        if self.mode != "train":
            raise ValueError("Метод fit доступен только в режиме 'train'")

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Модель или токенайзер не загружены")
        
        if dataset is None:
            raise RuntimeError("С датасетом для обучения проблемы")

        logger.info("Запуск классического дообучения Qwen2.5-3B на GSM8K...")

        logger.info("Пытаюсь уменьшить используемую видео память при обучении")
        self.model.config.use_cache = False  
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        train_dataset = dataset["train"].select(range(use_dataset_size))
        eval_dataset = dataset["test"].select(range(use_dataset_size_eval))

        if dataset["train"][0].get("one_num_answer") is None:
                logger.error("В датасете нет one_num_answer")
                return None
        
        logger.info(f"Размер train: {len(train_dataset)} примеров")


        # Токенизируем датасет (num_proc=1 — безопасно для замыкания)
        tokenized_dataset = train_dataset.map(
            self.__preprocess_function,
            remove_columns=train_dataset.column_names,
            num_proc=1,
            desc="Токенизация и форматирование датасета для SFT",
        )

        tokenized_dataset_eval = eval_dataset.map(
            self.__preprocess_function,
            remove_columns=eval_dataset.column_names,
            num_proc=1,
            desc="Токенизация и форматирование тестового датасета для SFT",
        )
        logger.info("train и eval датасеты предобработаны")

        logger.info(f"Датасет подготовлен. Токенизировано {len(tokenized_dataset)} примеров")

        params = get_params()
        
        args = params["fine_tune_params"]["classic"]
        os.makedirs(args["output_dir"], exist_ok=True)

        # steps_per_epoch = math.ceil(
        # len(tokenized_dataset) / args["per_device_train_batch_size"]
        # )

        # steps_per_epoch = math.ceil(
        #     steps_per_epoch / args["gradient_accumulation_steps"]
        # )

        # total_steps = steps_per_epoch * args["num_train_epochs"]
        # warmup_steps = int(0.1 * total_steps)

        training_args = TrainingArguments(
            output_dir = args["output_dir"],
            num_train_epochs = args["num_train_epochs"],
            per_device_train_batch_size=args["per_device_train_batch_size"],
            gradient_accumulation_steps=args["gradient_accumulation_steps"],
            learning_rate=args["learning_rate"],
            lr_scheduler_type=args["lr_scheduler_type"],
            warmup_steps=args["warmup_steps"],
            fp16=args["fp16"],
            bf16=args["bf16"],
            logging_steps=args["logging_steps"],
            save_steps=args["save_steps"],
            save_total_limit=args["save_total_limit"],
            report_to=args["report_to"],
            remove_unused_columns=args["remove_unused_columns"],
            dataloader_num_workers=args["dataloader_num_workers"],
            seed=args["seed"],
            eval_steps=args["eval_steps"],
            save_strategy = args["save_strategy"],
            eval_strategy = args["eval_strategy"],
            load_best_model_at_end = args["load_best_model_at_end"],
            metric_for_best_model=args["metric_for_best_model"],
            greater_is_better=args["greater_is_better"],
            dataloader_pin_memory = args["dataloader_pin_memory"]
        )

        # Коллатор (учитывает уже замаскированные labels)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest",
            label_pad_token_id=-100,
        )

        logger.info("Очищаем MPS cache перед обучением...")
        torch.mps.empty_cache()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset = tokenized_dataset,
            eval_dataset = tokenized_dataset_eval,
            data_collator=data_collator   
        )

        trainer.add_callback(ClearMPSCacheCallback())

        model_name = params["model"]["name"]
        logger.info(f"Начинаетсяс classic обучения для {model_name}")
        trainer.train()

        os.makedirs(args["output_dir"], exist_ok=True)
        trainer.save_model(args["output_dir"])
        self.tokenizer.save_pretrained(args["output_dir"])

        logger.info(f"Дообучение завершено! Модель сохранена в {args["output_dir"]}")

        torch.mps.empty_cache()
        logger.info(f"Освободил кеш: torch.mps.empty_cache()")