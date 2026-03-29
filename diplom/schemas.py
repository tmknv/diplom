from pydantic import BaseModel

class Metrics(BaseModel):
    validator_name: str
    model_name: str
    f1_score: float = 0
    meta: dict = {}

