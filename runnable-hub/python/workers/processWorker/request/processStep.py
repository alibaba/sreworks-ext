

from pydantic import BaseModel
from typing import List, Dict

class ProcessStep(BaseModel):
    stepId: str
    runnableCode: str
    request: Dict