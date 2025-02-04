
from runnable import RunnableRequest, RunnableOutputLoads
from .llmMessage import LlmMessage
from typing import Dict, List

class LlmRequest(RunnableRequest):
    runnableCode: str = "LLM_WORKER"
    model: str
    endpoint: str
    secretKey: str
    systemPrompt: str
    userPrompt: str
    history: List[LlmMessage] = []