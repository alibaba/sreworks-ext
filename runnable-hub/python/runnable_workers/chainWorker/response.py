

from runnable_hub import RunnableResponse
from typing import Dict, Optional, List

class ChainResponse(RunnableResponse):
    runnableCode: str = "CHAIN"
    finalAnswer: str
    history: List[Dict] = []
