
from pydantic import BaseModel
from runnable import RunnableOutputLoads, RunnableValueDefine
from typing import Dict, List, Optional
from .agentPromptTemplate import AgentPromptTemplate


class AgentDefine(BaseModel):
    agentCode: str
    agentVersion: str
    prerun: Optional[Dict] = None
    postrun: Optional[Dict] = None
    promptTemplate: AgentPromptTemplate
    inputDefine: List[RunnableValueDefine] = []
    outputDefine: List[RunnableValueDefine] = []