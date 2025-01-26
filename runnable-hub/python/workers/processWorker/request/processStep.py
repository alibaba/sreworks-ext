

from pydantic import BaseModel
from typing import List, Dict

class ProcessStep(BaseModel):
    stepCode: str
    runnableCode: str
    request: Dict