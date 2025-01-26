

from pydantic import BaseModel
from typing import List, Dict
from processStep import ProcessStep

class ProcessJob(BaseModel):
    jobCode: str
    steps: List[ProcessStep]
    needs: List[str]