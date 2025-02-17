
from runnable import RunnableRequest
from typing import Dict, Optional
from .processJob import ProcessJob

class ProcessRequest(RunnableRequest):
    runnableCode: str = "PROCESS_WORKER"
    jobs: Dict[str, ProcessJob]
    outputs: Optional[Dict|str] = None