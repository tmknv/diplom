"""
Загуржаем датасеты с HF, сохраняем локально.
Датасет name из params.yaml
"""
import os 

from datasets import load_dataset

from diplom.utils.logger import get_logger
from diplom.utils.load_params import get_params
from diplom.utils.init_secrets import get_settings

logger = get_logger(__name__)
params = get_params()
setting = get_settings()

token = setting["HF_TOKEN"]
save_path = params["datasets"]["save_path_math"]
dataset_name = params["datasets"]["math"]

logger.info(f"starting dawnload {dataset_name}...")
try:
    # "main"(socratic) для gsm8k
    dataset = load_dataset(dataset_name, "main", token=token)

except:

    dataset = load_dataset(dataset_name,  token=token)

print(dataset)

train_data = dataset['train']
print("Пример из тренировочного набора:")
print(train_data[0])

test_data = dataset['test']
print("Пример из тестового набора:")
print(test_data[0])


logger.info(f"Сохраняем датасет в {save_path}")

try:

    os.mkdir(f"{save_path}/{dataset_name}", exist_ok = True)
    dataset.save_to_disk(f"{save_path}/{dataset_name}")

    logger.info("Датасет успешно сохранен")

except Exception as e:

    logger.error(f"Не смогли сохранить датасет по пути {sace_patn}, ошибка: {e}")

