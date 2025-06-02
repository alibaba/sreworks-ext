from pydantic import BaseModel
from enum import Enum
from typing import Optional, Dict, List


class RunnableOutputLoads(Enum):
    JSON = "JSON"
    TEXT = "TEXT"
    YAML = "YAML"

class RunnableValueDefineType(Enum):
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"

class RunnableValueDefine(BaseModel):
    name: str
    type: RunnableValueDefineType
    required: Optional[bool] = False
    defaultValue: Optional[Dict|List|str|bool|float] = None