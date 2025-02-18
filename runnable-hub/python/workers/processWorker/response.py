
from runnable_hub import RunnableResponse
from typing import Dict, Optional, Any

class ProcessResponse(RunnableResponse):
    runnableCode: str = "PROCESS"
    outputs: Optional[Dict|str] = None
    