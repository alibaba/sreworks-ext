
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from runnable.store import RunnableFileStore, RunnableDatabaseStore
from .request.toolRequest import ToolRequest
from .response import ToolResponse


class Worker(RunnableWorker):

    runnableCode = "TOOL_WORKER"
    Request = ToolRequest
    Response = ToolResponse

    def __init__(self, store: RunnableFileStore|RunnableDatabaseStore):
        self.store = store

    async def onNext(self, context: RunnableContext[ToolRequest, ToolResponse]) -> RunnableContext:

        return context



