
from pydantic import BaseModel
from runnable import RunnableOutputLoads, RunnableValueDefine
from typing import Dict, List, Optional
from .agentChainTemplate import AgentChainTemplate
from ...chainWorker.request.chainFunction import ChainFunction


class AgentDefine(BaseModel):
    agentCode: str
    agentVersion: str
    prerun: Optional[Dict] = None
    postrun: Optional[Dict] = None
    inputDefine: List[RunnableValueDefine] = []
    outputDefine: List[RunnableValueDefine] = []
    chainTemplate: Optional[AgentChainTemplate]
    chainTemplateCode: Optional[str]
    chainFunctions: List[ChainFunction]