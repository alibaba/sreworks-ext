

from runnable import RunnableResponse
from typing import Dict, Optional, List

class ChainThoughtResponse(RunnableResponse):
    runnableCode: str = "CHAIN_THOUGHT_WORKER"
    finalAnswer: str
    history: List[str]
