
from runnable import RunnableResponse
from typing import Dict, Optional, Any

class ProcessResponse(RunnableResponse):
    runnableCode: str = "PROCESS_WORKER"
    outputs: Optional[Dict|str] = None
    