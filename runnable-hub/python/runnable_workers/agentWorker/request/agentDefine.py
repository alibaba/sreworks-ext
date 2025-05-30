
from pydantic import BaseModel
from runnable_hub import RunnableValueDefine, RunnableValueDefineType
from typing import Dict, List, Optional
from .agentChainTemplate import AgentChainTemplate
from .agentFunction import AgentFunction
from ...llmWorker.request.llmSetting import LlmSetting


class AgentDefine(BaseModel):
    agentCode: str
    agentVersion: str
    prerun: Optional[Dict] = None
    postrun: Optional[Dict] = None
    inputDefine: List[RunnableValueDefine] = [RunnableValueDefine(name="prompt", type=RunnableValueDefineType.STRING, required=True)]
    outputDefine: List[RunnableValueDefine] = []
    instanceDefine: List[RunnableValueDefine] = []

    functions: List[AgentFunction]
    instruction: str
    
    llm: Optional[LlmSetting] = None
    llmCode: Optional[str] = None

    chainTemplate: Optional[AgentChainTemplate] = None
    chainTemplateCode: Optional[str] = None
    