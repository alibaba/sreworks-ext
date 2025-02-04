
from runnable import RunnableRequest
from typing import Dict, Optional, List
from .chainRunnable import ChainRunnable

class ChainRequest(RunnableRequest):
    runnableCode: str = "CHAIN_WORKER"
    runnables: List[ChainRunnable] = []
    llmModel: str
    systemPrompt: str
    userPrompt: str
    completionPattern: str
    completionInterpreterScript: str