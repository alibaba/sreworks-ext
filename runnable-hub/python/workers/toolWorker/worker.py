
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from runnable.store import RunnableFileStore, RunnableDatabaseStore
from .request.toolRequest import ToolRequest
from .request.toolDefine import ToolDefine
from .response import ToolResponse
from typing import Dict

class Worker(RunnableWorker):

    runnableCode = "TOOL_WORKER"
    Request = ToolRequest
    Response = ToolResponse

    def __init__(self, store: RunnableFileStore|RunnableDatabaseStore):
        self.store = store

    def addTool(self, toolDefine: ToolDefine):
        if isinstance(self.store, RunnableFileStore):
            filePath = f"{toolDefine.toolCode}/{toolDefine.toolVersion}/define.json"
            self.store.saveFile(filePath, toolDefine.model_dump_json())
        else:
            raise Exception("Not supported")

    def readTool(self, toolCode:str, toolVersion:str) -> ToolDefine:
        if isinstance(self.store, RunnableFileStore):
            filePath = f"{toolCode}/{toolVersion}/define.json"
            return ToolDefine.model_validate_json(self.store.readFile(filePath))
        else:
            raise Exception("Not supported")

    async def onNext(self, context: RunnableContext[ToolRequest, ToolResponse]) -> RunnableContext:
        
        return context



