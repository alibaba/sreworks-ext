from pydantic import BaseModel
from typing import Dict, Optional

class ChainThoughtRunnable(BaseModel):
    runnableCode: str
    request: Dict