
from runnable import RunnableRequest
from typing import Dict
from .processJob import ProcessJob

class ProcessRequest(RunnableRequest):
    runnableCode: str = "PROCESS_WORKER"
    jobs: Dict[str, ProcessJob]