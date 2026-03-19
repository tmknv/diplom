# diplom/diplom/model/load_model.py
"""
Качаю и сохранаю локально модель с HF.
Name, save path из params.yaml

Сохраняю в:
    diplom/artifactc/models/model_name
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

from diplom.utils.logger import get_logger
from diplom.utils.load_params import get_params
from diplom.utils.init_secrets import get_settings

logger = get_logger(__name__)
params = get_params()
settings = get_settings()


token = settings.HF_TOKEN

model_name = params["model"]["name"]
save_path = params["model"]["save_path"]

logger.info(f"Начинается загрузка: {model_name}")

# загрузка
tokenizer = AutoTokenizer.from_pretrained(model_name, token = token)
model = AutoModelForCausalLM.from_pretrained(model_name, token = token)

logger.info(f"Загрузка окончена: {model_name}")

# сохраняю
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

logger.info(f"Модель : {model_name} была сохраненра в {save_path}")

print(f"Модель сохранена в: {save_path}")