"""
Загрузка конфигурации приложения из params.yaml
"""

from pathlib import Path
import yaml
from functools import lru_cache


@lru_cache
def get_params() -> dict:
    """
    Загружает конфигурацию приложения из YAML.

    Returns:
        dict: Конфигурация приложения.
    """
    # config_path = Path(__file__).resolve().parent / "params.yaml"
    config_path =  "params.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    params = get_params()
    print(params)