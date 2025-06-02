
from runnable_hub import RunnableRequest, RunnableOutputLoads
from .llmMessage import LlmMessage
from .llmSetting import LlmSetting
from typing import Dict, List

class LlmRequest(RunnableRequest):
    runnableCode: str = "LLM"
    setting: LlmSetting
    systemPrompt: str
    userPrompt: str
    history: List[LlmMessage] = []