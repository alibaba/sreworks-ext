
from pydantic import BaseModel
from runnable import RunnableValueDefine
from typing import List


class AgentPromptTemplate(BaseModel):
    systemPrompt: str
    userPrompt: str
    inputDefine: List[RunnableValueDefine] = []