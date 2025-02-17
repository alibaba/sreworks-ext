

from pydantic import BaseModel
from typing import List, Dict, Optional
from .processStep import ProcessStep

class ProcessJob(BaseModel):
    steps: List[ProcessStep]
    needs: List[str] = []
    outputs: Optional[Dict|str] = None