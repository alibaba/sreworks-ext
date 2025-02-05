
from runnable import RunnableRequest
from typing import Dict, Optional, List
from .chainRunnable import ChainRunnable

class ChainRequest(RunnableRequest):
    runnableCode: str = "CHAIN_WORKER"
    runnables: List[ChainRunnable] = []
    data: Dict[str, Dict|List|str|float|int|bool] = {}
    llmModel: str
    systemPrompt: str
    userPrompt: str
    chainInterpreter: str