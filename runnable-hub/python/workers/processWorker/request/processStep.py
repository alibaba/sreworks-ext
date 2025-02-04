

from pydantic import BaseModel
from typing import List, Dict, Optional

class ProcessStep(BaseModel):
    id: str
    runnableCode: Optional[str] = None
    request: Optional[Dict] = None

    # Common node customization fields
    shell: Optional[str] = None
    api: Optional[Dict|str] = None
    jinja: Optional[Dict] = None
    python: Optional[str] = None