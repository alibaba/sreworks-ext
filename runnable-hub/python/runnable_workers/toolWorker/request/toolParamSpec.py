
from pydantic import BaseModel
from typing import Optional, Dict, List

from .toolParamSpecType import ToolParamSpecType

class ToolParamSpec(BaseModel):
    name: str
    type: ToolParamSpecType
    required: Optional[bool] = False
    defaultValue: Optional[Dict|List|str|bool|float] = None