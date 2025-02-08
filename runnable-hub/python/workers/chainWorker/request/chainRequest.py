
from runnable import RunnableRequest
from typing import Dict, Optional, List
from .chainFunction import ChainFunction

class ChainRequest(RunnableRequest):
    runnableCode: str = "CHAIN_WORKER"
    functions: List[ChainFunction] = []
    data: Dict[str, Dict|List|str|float|int|bool] = {}
    llmModel: Optional[str]
    llmSecretKey: Optional[str]
    llmEndpoint: Optional[str]
    systemPrompt: str
    userPrompt: str
    onNext: str
    # chainInterpreter: str