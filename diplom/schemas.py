from pydantic import BaseModel

from dataclasses import dataclass, field

# @dataclass
class Metrics(BaseModel):
    validator_name: str
    model_name: str
    f1_score: float = 0
    meta: dict = {}
    # meta: dict = field(default_factory=dict)

