

from runnable import RunnableResponse
from typing import Dict, Optional, List

class ChainResponse(RunnableResponse):
    runnableCode: str = "CHAIN_WORKER"
    finalAnswer: str
    history: List[Dict] = []
