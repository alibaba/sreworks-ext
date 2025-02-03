
from pydantic import BaseModel
from runnable import RunnableOutputLoads, RunnableValueDefine
from typing import Dict, List, Optional
from .agentPromptChainTemplate import AgentPromptChainTemplate


class AgentDefine(BaseModel):
    agentCode: str
    agentVersion: str
    prerun: Optional[Dict] = None
    postrun: Optional[Dict] = None
    inputDefine: List[RunnableValueDefine] = []
    outputDefine: List[RunnableValueDefine] = []
    reasoningRunnables: List[str] = []
    reasoningPromptTemplate: AgentPromptChainTemplate