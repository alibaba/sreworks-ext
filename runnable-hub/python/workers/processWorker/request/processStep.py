

from pydantic import BaseModel
from typing import List, Dict, Optional

class ProcessStep(BaseModel):
    stepId: str
    runnableCode: Optional[str] = None
    request: Optional[Dict] = None
    shell: Optional[str] = None