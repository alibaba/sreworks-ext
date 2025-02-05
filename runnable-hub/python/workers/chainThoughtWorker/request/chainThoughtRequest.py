
from runnable import RunnableRequest
from typing import Dict, Optional, List
from .chainThoughtRunnable import ChainThoughtRunnable

class ChainThoughtRequest(RunnableRequest):
    runnableCode: str = "CHAIN_THOUGHT_WORKER"
    runnables: List[ChainThoughtRunnable] = []
    llmModel: str
    systemPrompt: str
    userPrompt: str
    completionPattern: str
    completionInterpreterScript: str