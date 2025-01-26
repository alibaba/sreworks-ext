

from datetime import datetime
import asyncio
import uuid
from abc import ABC, abstractmethod
from .context import RunnableRequest, RunnableContext, RunnableStatus
from .store import RunnableStore

class RunnableWorker(ABC):
    runnableCode = None
    Request = None
    Response = None
    @abstractmethod
    async def onNext(self, context: RunnableContext) -> RunnableContext:
        pass

    # def execute(self, context: RunnableContext, req: RunnableRequest, name: str):
    #     context.promise.resolve[name] = req


class RunnableWorkerDispatch():
    worker: RunnableWorker
    queue: asyncio.Queue
    store: RunnableStore

    def __init__(self, worker: RunnableWorker, queue: asyncio.Queue, store: RunnableStore):
        self.worker = worker
        self.queue = queue
        self.store = store
        asyncio.create_task(self.run())

    async def run(self):
        while True:
            print(f"RunnableWorkerDispatch {self.worker.runnableCode} wait")
            contextFile = await self.queue.get()
            print(f"RunnableWorkerDispatch {self.worker.runnableCode} get message {contextFile}")
            context = RunnableContext[self.worker.Request].model_validate_json(self.store.read(contextFile))
            print(context)
            context.status = RunnableStatus.RUNNING
        
            try:
                context = await self.worker.onNext(context)
            except Exception as e:
                context.status = RunnableStatus.ERROR
                context.errorMessage = str(e)

            if context.status in [RunnableStatus.ERROR, RunnableStatus.SUCCESS]:
                context.endTime = datetime.now()
            print(context)
            self.store.save(contextFile, context.model_dump_json())


class RunnableHub():

    def __init__(self, store: RunnableStore):
        self.workers = {}
        self.store = store

    def getExecuteStorePath(self, executeId: str, fileName: str):
        return f"execute/{executeId}/{fileName}"

    def registerWorker(self, worker: RunnableWorker):
        self.workers[worker.runnableCode] = RunnableWorkerDispatch(
            worker=worker,
            queue=asyncio.Queue(),
            store=self.store
        )

    async def executeStart(self, request: RunnableRequest) -> RunnableContext:
        newContext = RunnableContext(
            executeId=str(uuid.uuid4()),
            request=request, 
            response=None, 
            startTime=datetime.now(), 
            status=RunnableStatus.PENDING)
        
        contextStorePath = self.getExecuteStorePath(newContext.executeId, "context.json")
        self.store.save(
            contextStorePath,
            newContext.model_dump_json())
        await self.workers[newContext.request.runnableCode].queue.put(contextStorePath)
        return newContext
    
    def executeCheck(self, executeId: str) -> RunnableContext:
        result = RunnableContext(request=None, response=None, executeId=executeId, startTime=datetime.now(), status=RunnableStatus.PENDING)
        return result
