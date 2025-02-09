
from re import L
from runnable import RunnableRequest
from typing import Dict, Optional, List
from .chainFunction import ChainFunction
from ...llmWorker.request.llmSetting import LlmSetting

class ChainRequest(RunnableRequest):
    runnableCode: str = "CHAIN_WORKER"
    functions: List[ChainFunction] = []
    data: Dict[str, Dict|List|str|float|int|bool] = {}
    llm: LlmSetting
    systemPrompt: str
    userPrompt: str
    onNext: str
    # chainInterpreter: str