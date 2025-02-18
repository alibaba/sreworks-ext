
from runnable_hub import RunnableRequest
from typing import Dict, Optional
from .processJob import ProcessJob

class ProcessRequest(RunnableRequest):
    runnableCode: str = "PROCESS"
    jobs: Dict[str, ProcessJob]
    outputs: Optional[Dict|str] = None