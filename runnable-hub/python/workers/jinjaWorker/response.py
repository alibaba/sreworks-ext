

from runnable_hub import RunnableResponse
from typing import Dict, Optional, List

class JinjaResponse(RunnableResponse):
    runnableCode: str = "JINJA"
    result: str
    outputs: Optional[Dict|List|str]
