
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus
from runnable_hub.interface import RunnableFileStore, RunnableDatabaseStore
from .request.agentRequest import AgentRequest
from .request.agentDefine import AgentDefine
from .request.agentChainTemplate import AgentChainTemplate
from ..llmWorker.request.llmSetting import LlmSetting
from .response import AgentResponse
from typing import Dict
from datetime import datetime
import json

class Worker(RunnableWorker):

    runnableCode = "AGENT"
    Request = AgentRequest
    Response = AgentResponse

    def __init__(self, store: RunnableFileStore|RunnableDatabaseStore):
        self.store = store

    def addAgent(self, agentDefine: AgentDefine):
        if isinstance(self.store, RunnableFileStore):
            filePath = f"{agentDefine.agentCode}/{agentDefine.agentVersion}/define.json"
            self.store.saveFile(filePath, agentDefine.model_dump_json())
        else:
            raise Exception("Not supported")

    def addChainTemplate(self, chainTemplateCode: str, template: AgentChainTemplate):
        if isinstance(self.store, RunnableFileStore):
            filePath = f"chainTemplates/{chainTemplateCode}/template.json"
            self.store.saveFile(filePath, template.model_dump_json())
        else:
            raise Exception("Not supported")

    def addLlm(self, llmCode: str, llmSetting: LlmSetting):
        if isinstance(self.store, RunnableFileStore):
            filePath = f"llm/{llmCode}.json"
            self.store.saveFile(filePath, llmSetting.model_dump_json())
        else:
            raise Exception("Not supported")

    def readAgent(self, agentCode:str, agentVersion:str) -> AgentDefine:
        if isinstance(self.store, RunnableFileStore):
            filePath = f"{agentCode}/{agentVersion}/define.json"
            return AgentDefine.model_validate_json(self.store.readFile(filePath))
        else:
            raise Exception("Not supported")

    def readChainTemplate(self, chainTemplateCode: str) -> AgentChainTemplate:
        if isinstance(self.store, RunnableFileStore):
            filePath = f"chainTemplates/{chainTemplateCode}/template.json"
            return AgentChainTemplate.model_validate_json(self.store.readFile(filePath))
        else:
            raise Exception("Not supported")

    def readLllm(self, llmCode: str) -> LlmSetting:
        if isinstance(self.store, RunnableFileStore):
            filePath = f"llm/{llmCode}.json"
            return LlmSetting.model_validate_json(self.store.readFile(filePath))
        else:
            raise Exception("Not supported")

    async def onNext(self, context: RunnableContext[AgentRequest, AgentResponse]) -> RunnableContext:
        agentDefine = self.readAgent(context.request.agentCode, context.request.agentVersion)
        
      
        # chain   
        if context.data.get("sendChainRequest") is None:
            
            if agentDefine.chainTemplate is not None:
                chainTemplate = {
                    "systemPrompt": agentDefine.chainTemplate.systemPrompt,
                    "userPrompt": agentDefine.chainTemplate.userPrompt,
                    "onNext": agentDefine.chainTemplate.onNext
                }
            elif agentDefine.chainTemplateCode is not None:
                chainTemplateByCode = self.readChainTemplate(agentDefine.chainTemplateCode)
                chainTemplate = {
                    "systemPrompt": chainTemplateByCode.systemPrompt,
                    "userPrompt": chainTemplateByCode.userPrompt,
                    "onNext": chainTemplateByCode.onNext
                }
            else:
                raise Exception("No chainTemplate")

            if agentDefine.llm is not None:
                llm = agentDefine.llm
            elif agentDefine.llmCode is not None:    
                llm = self.readLllm(agentDefine.llmCode)
            else:
                raise Exception("No llm")

            # for agentDefine.functions 

            chainRequest = {
                "runnableCode": "CHAIN",
                "data": context.request.inputs,
                "llm": llm,
            }
            chainRequest.update(chainTemplate)
            context.promise.resolve["chain"] = chainRequest
            context.data["sendChainRequest"] = datetime.now()
            return context

        return context




