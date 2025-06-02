
from pydantic import BaseModel
# from runnable_hub import RunnableValueDefine
from typing import List

class AgentChainTemplate(BaseModel):
    systemPrompt: str
    userPrompt: str
    onNext: str