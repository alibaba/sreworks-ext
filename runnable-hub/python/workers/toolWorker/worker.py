
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from runnable.store import RunnableFileStore, RunnableDatabaseStore
from .request.toolRequest import ToolRequest
from .response import ToolResponse
from typing import Dict

class Worker(RunnableWorker):

    runnableCode = "TOOL_WORKER"
    Request = ToolRequest
    Response = ToolResponse

    def __init__(self, store: RunnableFileStore|RunnableDatabaseStore):
        self.store = store

    def addTool(self, toolCode:str, toolVersion:str, toolType: str, toolPayload: Dict):
        pass

    def readTool(self, toolCode:str, toolVersion:str):
        pass

    async def onNext(self, context: RunnableContext[ToolRequest, ToolResponse]) -> RunnableContext:

        return context



