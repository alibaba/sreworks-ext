
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus
from runnable_hub.store import RunnableFileStore, RunnableDatabaseStore
from .request.agentRequest import AgentRequest
from .request.agentDefine import AgentDefine
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

    def readAgent(self, agentCode:str, agentVersion:str) -> AgentDefine:
        if isinstance(self.store, RunnableFileStore):
            filePath = f"{agentCode}/{agentVersion}/define.json"
            return AgentDefine.model_validate_json(self.store.readFile(filePath))
        else:
            raise Exception("Not supported")

    async def onNext(self, context: RunnableContext[AgentRequest, AgentResponse]) -> RunnableContext:
        toolDefine = self.readAgent(context.request.agentCode, context.request.agentVersion)
        if context.data.get("sendProcessRequest") is None:
            if toolDefine.toolType == ToolType.API:
                processStepName = "api"
            else:
                raise Exception("Not supported")

            processRequest = {
                "runnableCode": "PROCESS",
                "outputs": "${{ jobs.call.outputs }}",
                "jobs":{
                    "call": {
                        "outputs": "${{ steps.result.outputs }}",
                        "steps":[{
                            "id": "setting",
                            "jinja":{
                                "data": {
                                    "inputs": context.request.inputs
                                },
                                "outputLoads": "JSON",
                                "template": json.dumps(toolDefine.setting),
                            }
                        },{
                            "id": "call",
                            processStepName: "${{ steps.setting.outputs }}"
                        },{
                            "id": "result",
                            "jinja":{
                                "data": {
                                    "result": "${{ steps.call.outputs }}"
                                },
                                "outputLoads": toolDefine.outputsLoads,
                                "template": toolDefine.outputTemplate
                            }
                        }]
                    }
                }
            }
            context.promise.resolve["todo"] = processRequest
            context.data["sendProcessRequest"] = datetime.now()
        else:
            if context.promise.result.get("todo") is None:
                raise Exception("Process request not finish")
            
            result = context.promise.result["todo"]

            context.response = ToolResponse(
                success=True,
                outputs=result["outputs"]
            )
            context.status = RunnableStatus.SUCCESS

        return context



