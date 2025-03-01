

from pydantic import BaseModel
from typing import List, Dict, Optional

class ProcessStep(BaseModel):
    id: str
    runnableCode: Optional[str] = None
    request: Optional[Dict] = None

    shell: Optional[str] = None
    api: Optional[Dict] = None
    jinja: Optional[Dict] = None
    python: Optional[str] = None
    tool: Optional[Dict] = None