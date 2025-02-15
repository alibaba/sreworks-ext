
from pydantic import BaseModel
# from runnable import RunnableValueDefine
from typing import List

class AgentChainTemplate(BaseModel):
    systemPrompt: str
    userPrompt: str
    onNext: str