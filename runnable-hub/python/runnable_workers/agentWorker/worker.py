
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus
from runnable_hub.interface import RunnableFileStore, RunnableDatabaseStore
from .request.agentRequest import AgentRequest
from .request.agentDefine import AgentDefine
from .request.agentChainTemplate import AgentChainTemplate
from .request.agentFunction import AgentFunctionType
from ..llmWorker.request.llmSetting import LlmSetting
from ..toolWorker.worker import Worker as ToolWorker
from .response import AgentResponse
from typing import Dict
from datetime import datetime
import json

# inputs -> prerun -> outputs
# inputs -> chain -> outputs
# inputs -> postrun -> outputs

class Worker(RunnableWorker):

    runnableCode = "AGENT"
    Request = AgentRequest
    Response = AgentResponse

    def __init__(self, store: RunnableFileStore|RunnableDatabaseStore, toolWorker: ToolWorker):
        self.store = store
        self.toolWorker = toolWorker

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
      
        if context.data.get("inputs") is None:
            context.data["inputs"] = context.request.inputs
            context.data["outputs"] = {}
            context.data["stage"] = []
            if agentDefine.prerun is not None:
                context.data["stage"].append("prerun")
            if agentDefine.chainTemplate is not None or agentDefine.chainTemplateCode is not None:
                context.data["stage"].append("chain")
            if agentDefine.postrun is not None:
                context.data["stage"].append("postrun")

        # prerun
        if len(context.data["stage"]) > 0 and context.data["stage"][0] == "prerun" and agentDefine.prerun is not None:
            if context.data.get("sendPrerunRequest") is None:
                prerunRequest = {
                    "runnableCode": "PROCESS",
                    "inputs": context.request.inputs,
                    "jobs": agentDefine.prerun.get("jobs", {}),
                }
                if agentDefine.prerun.get("chainInputs") is not None:
                    prerunRequest["outputs"] = agentDefine.prerun.get("chainInputs")
                elif agentDefine.prerun.get("outputs") is not None:
                    prerunRequest["outputs"] = agentDefine.prerun.get("outputs")
                context.promise.resolve["prerun"] = prerunRequest
                context.data["sendPrerunRequest"] = datetime.now()
                return context
            else:
                if context.promise.result.get("prerun") is None:
                    raise ValueError("prerun response is missing")
                
                # 将prerun的执行outputs更新到inputs或outputs
                if agentDefine.prerun.get("chainInputs") is not None:
                    context.data["inputs"].update(context.promise.result["prerun"]["outputs"])
                elif agentDefine.prerun.get("outputs") is not None:
                    context.data["outputs"] = context.promise.result["prerun"]["outputs"]

                context.data["stage"].remove("prerun")

        # chain

        if len(context.data["stage"]) > 0 and context.data["stage"][0] == "chain" \
            and (agentDefine.chainTemplate is not None or agentDefine.chainTemplateCode is not None):
            
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

                if agentDefine.llm is not None:
                    llm = agentDefine.llm
                elif agentDefine.llmCode is not None:    
                    llm = self.readLllm(agentDefine.llmCode)
                else:
                    raise Exception("No llm")

                chainFunctions = []
                toolCodeVersions = [f"{f.name}:{f.version}" for f in agentDefine.functions if f.type == AgentFunctionType.TOOL]
                toolMap = {tool.toolCode: tool for tool in await self.toolWorker.readTools(toolCodeVersions)}
                for f in agentDefine.functions:
                    if f.type == AgentFunctionType.TOOL:
                        chainFunctions.append({
                            "type": "TOOL",
                            "name": f.name,
                            "version": f.version,
                            "description": toolMap[f.name].description,
                            "inputDefine": toolMap[f.name].inputSpec,
                            "presetInputs": f.presetInputs,
                        })

                chainRequest = {
                    "runnableCode": "CHAIN",
                    "data": {
                        "inputs": context.data["inputs"],
                    },
                    "llm": llm,
                    "functions": chainFunctions,
                }
                chainRequest.update(chainTemplate)
                context.promise.resolve["chain"] = chainRequest
                context.data["sendChainRequest"] = datetime.now()
                return context
            else:
                if context.promise.result["chain"] is None:
                    raise ValueError("chain response is missing")

                if agentDefine.postrun is None:
                    context.data["outputs"] = context.promise.result["chain"]["finalAnswer"]
                else:
                    context.data["inputs"]["chainAnswer"] = context.promise.result["chain"]["finalAnswer"]
                
                context.data["stage"].remove("chain")

        # postrun
        if len(context.data["stage"]) > 0 and context.data["stage"][0] == "postrun" and agentDefine.postrun is not None:
            if context.data.get("sendPostrunRequest") is None:
                prerunRequest = {
                    "runnableCode": "PROCESS",
                    "inputs": context.data["inputs"],
                    "jobs": agentDefine.postrun.get("jobs", {}),
                    "outputs": agentDefine.postrun.get("outputs"),
                }
                context.promise.resolve["postrun"] = prerunRequest
                context.data["sendPostrunRequest"] = datetime.now()
                return context
            else:
                if context.promise.result["postrun"] is None:
                    raise ValueError("postrun response is missing")
                
                context.data["outputs"] = context.promise.result["postrun"]["outputs"]
        

        context.status = RunnableStatus.SUCCESS
        context.response = AgentResponse(success=True, outputs=context.data["outputs"])
        return context




