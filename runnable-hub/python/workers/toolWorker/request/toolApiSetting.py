
from pydantic import BaseModel
from typing import Optional, Dict, List

from traitlets import default
from .toolParamSpecType import ToolParamSpecType

class ToolApiSetting(BaseModel):
    payload: Optional[Dict] = None
    headers: Optional[Dict] = None
    query: Optional[Dict] = None
    url: str
    method: str