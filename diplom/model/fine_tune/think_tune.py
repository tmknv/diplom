# diplom/model/fine_tune/think_tune.py
"""
Первая стадия обучения модели ризонингу.
1. Добавляем <think></think> как специальные токены
2. Учим модельку использовать эти токены в качестве размышления
3. Подкручиваем штраф модельки, чтобы добиться качественных мыслей
4. Штраф за мысли 2, за ответ после <think><think> 0.5
5. Хотим получить очень качественные мысли
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
    
class ThinkFineTune(Model):
    def __init__(self, model, tokenizer, mode: str = "eval"):
        """
        Note:
            Инициализация обертки для дистиляции ризонинга.
            Основная цель - научить модель хорошо думать

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

    def __preprocess_function(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Форматирует один пример GSM8K под chat-template + маскирует loss на промпте.
        """
        system_prompt = params["prompt"]["qwen"]["system_prompt_think"]
        user_prompt_template = params["prompt"]["qwen"]["user_prompt_think"]

        question = sample["question"]
        reasoning = f"<think>{sample['reasoning']}</think>"
        assistant_content = assistant_content = (
            f"{reasoning}\n"
            f"{sample['one_num_answer']}"
        )

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
        # print(full_text)
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

    def fit(self, train_dataset, test_dataset, use_dataset_size: int = 1000, use_dataset_size_eval: int = 50) -> None:
        """
        Запускает процесс дообучения (SFT) модели Qwen2.5-3B
        на датасете openai/gsm8k.

        Особенности обучения:
            1. Формируется chat-формат: system + user (с {MATH_PROBLEM}) + assistant
            2. Assistant генерирует ризонинг в <think>...</think>. Ответом считается все после </think>
            3. Loss считается на ризонинге и ответе модели. (prompt маскируется -100)
            4. <think> часть получает дополнительный штраф, больший чем получает финальный ответ.
                стараемся научить модель кчественно думать.
            5. Используется Trainer + DataCollatorForSeq2Seq
            6. Модель сохраняется в diplom/artifacts/models/reasoning_destil_v2_qwen2_5_3b

        Raises:
            ValueError: Если режим не "train".
            RuntimeError: Если модель или токенайзер не загружены.
        """
        if self.mode != "train":
            raise ValueError("Метод fit доступен только в режиме 'train'")

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Модель или токенайзер не загружены")


        logger.info("Запуск think дообучения Qwen2.5-3B на GSM8K...")

        logger.info("Пытаюсь уменьшить используемую видео память при обучении")
        self.model.config.use_cache = False  
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        train_dataset = train_dataset.select(
            range(min(use_dataset_size, len(train_dataset)))
        )

        eval_dataset = test_dataset.select(
            range(min(use_dataset_size_eval, len(test_dataset)))
        )

        # train_dataset = train_dataset[:use_dataset_size]
        # eval_dataset = test_dataset[:use_dataset_size_eval]

        
        logger.info(f"Размер train: {len(train_dataset)} примеров")
        logger.info(f"Размер test: {len(eval_dataset)} примеров")


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


        # sample = tokenized_dataset[0]

        # print("input_ids:", len(sample["input_ids"]))
        # print("labels:", len(sample["labels"]))

        # valid_labels = [x for x in sample["labels"] if x != -100]

        # print("valid labels:", len(valid_labels))
        # print(valid_labels[:20])

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

        batch = data_collator([tokenized_dataset[0]])

        batch = {
            k: v.to(self.model.device)
            for k, v in batch.items()
        }

        self.model.eval()

        with torch.no_grad():
            out = self.model(**batch)

        print("LOSS =", out.loss)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset = tokenized_dataset,
            eval_dataset = tokenized_dataset_eval,
            data_collator=data_collator   
        )

        trainer.add_callback(ClearMPSCacheCallback())

        model_name = params["model"]["name"]
        logger.info(f"Начинаетсяс think tune обучение для {model_name}")
        trainer.train()

        os.makedirs(args["output_dir"], exist_ok=True)
        trainer.save_model(args["output_dir"])
        self.tokenizer.save_pretrained(args["output_dir"])

        logger.info(f"Дообучение завершено! Модель сохранена в {args["output_dir"]}")

        logger.info(
            f"Best checkpoint: {trainer.state.best_model_checkpoint}"
        )

        logger.info(
            f"Best metric: {trainer.state.best_metric}"
        )

        torch.mps.empty_cache()
        logger.info(f"Освободил кеш: torch.mps.empty_cache()")