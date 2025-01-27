
from runnable import RunnableResponse
from typing import Dict

class ProcessResponse(RunnableResponse):
    runnableCode: str = "PROCESS_WORKER"
    outputs: Dict
    