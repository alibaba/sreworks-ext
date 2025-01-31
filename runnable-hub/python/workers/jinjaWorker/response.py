

from runnable import RunnableResponse
from typing import Dict, Optional, List

class JinjaResponse(RunnableResponse):
    runnableCode: str = "JINJA_WORKER"
    result: str
    outputs: Optional[Dict|List|str]
