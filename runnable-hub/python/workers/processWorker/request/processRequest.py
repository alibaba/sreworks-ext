
from runnable import RunnableRequest
from typing import List
from .processJob import ProcessJob

class ProcessRequest(RunnableRequest):
    runnableCode: str = "PROCESS_WORKER"
    jobs: List[ProcessJob]