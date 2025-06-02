
from runnable_hub import RunnableResponse
from typing import Dict, List, Optional
from .llmUsage import LlmUsage
from ..request.llmMessage import LlmMessage

class LlmResponse(RunnableResponse):
    runnableCode: str = "LLM"
    usage: LlmUsage
    content: str
    messages: List[LlmMessage]