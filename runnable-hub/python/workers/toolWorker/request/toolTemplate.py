
from pydantic import BaseModel
from typing import Optional

class ToolTemplate(BaseModel):
    apiPayload: Optional[str]
    apiQuery: Optional[str]
    apiHeaders: Optional[str]
    result: Optional[str]