"""
Модуль для:
    1. Генерации ризонинга моделькой учителем
    2. Составления итогового датасета
    3. Итоговый датасет будет указанной размерности
"""


import requests

from diplom.utils.logger import get_logger
from diplom.utils.load_params import get_params
from diplom.utils.init_secrets import get_settings


logger = get_logger(__name__)
params = get_params()
setting = get_settings()

API_KEY = setting.API_KEY
url = params["open_router"]["url"]
model_name = params["open_router"]["model_name"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Напиши короткую шутку"}
    ]
}

response = requests.post(url, headers=headers, json=data)

print(response.json())
