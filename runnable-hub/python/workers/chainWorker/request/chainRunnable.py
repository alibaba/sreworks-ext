from pydantic import BaseModel
from typing import Dict, Optional

class ChainRunnable(BaseModel):
    runnableCode: str
    request: Dict