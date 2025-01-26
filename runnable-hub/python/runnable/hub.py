

from datetime import datetime
import asyncio
import uuid
from abc import ABC, abstractmethod
from .context import RunnableRequest, RunnableContext, RunnableStatus
from .store import RunnableStore
from typing import Dict

class RunnableWorker(ABC):
    runnableCode = None
    Request = None
    Response = None
    @abstractmethod
    async def onNext(self, context: RunnableContext) -> RunnableContext:
        pass


class RunnableHub():

    def __init__(self, store: RunnableStore):
        self.workers = {}
        self.store = store

    # def getExecuteStorePath(self, executeId: str, fileName: str):
        # return f"execute/{executeId}/{fileName}"

    def registerWorker(self, worker: RunnableWorker):
        self.workers[worker.runnableCode] = RunnableWorkerDispatch(
            worker=worker,
            queue=asyncio.Queue(),
            store=self.store,
            hub=self
        )

    # def readExecuteContext(self, executeId: str, runnableCode: str) -> RunnableContext:
    #     contextStorePath = self.getExecuteStorePath(executeId, "context.json")
    #     return RunnableContext[self.workers[runnableCode].Request].model_validate_json(self.store.read(contextStorePath))

    # def saveExecuteContext(self, context: RunnableContext):
        # self.store.save(context.storePath, context.model_dump_json())

    # def saveExecutePromiseResult(self, executeId: str, response: Dict):


    async def executeStart(self, request: RunnableRequest, parentContext: RunnableContext|None, name: str|None) -> RunnableContext:
        executeId=str(uuid.uuid4())
        if parentContext is None:
            storePath = "execute/{executeId}"
            parentExecuteId = None
            parentRunnableCode = None
            callDepth = 0
        else:
            storePath = f"{parentContext.storePath}/{executeId}"
            parentExecuteId = parentContext.executeId
            parentRunnableCode = parentContext.runnableCode
            callDepth = parentContext.callDepth + 1

        newContext = RunnableContext(
            executeId=executeId,
            runnableCode=request.runnableCode,
            request=request, 
            response=None, 
            name=name,
            callDepth=callDepth,
            storePath=storePath,
            parentExecuteId=parentExecuteId,
            parentRunnableCode=parentRunnableCode,
            createTime=datetime.now(), 
            status=RunnableStatus.PENDING)
        
        self.store.save(f"{newContext.storePath}/context.json", newContext.model_dump_json())
        await self.workers[newContext.request.runnableCode].queue.put(f"{newContext.storePath}/context.json")
        return newContext
    
    # def executeCheck(self, executeId: str) -> RunnableContext:
    #     result = RunnableContext(request=None, response=None, executeId=executeId, runnableCode="test", startTime=datetime.now(), createTime=datetime.now(), status=RunnableStatus.PENDING)
    #     return result

class RunnableWorkerDispatch():
    worker: RunnableWorker
    queue: asyncio.Queue
    store: RunnableStore
    hub: RunnableHub

    def __init__(self, worker: RunnableWorker, queue: asyncio.Queue, store: RunnableStore, hub: RunnableHub):
        self.worker = worker
        self.queue = queue
        self.store = store
        self.hub = hub
        asyncio.create_task(self.run())

    async def run(self):
        while True:
            print(f"RunnableWorkerDispatch {self.worker.runnableCode} wait")
            contextFile = await self.queue.get()
            print(f"RunnableWorkerDispatch {self.worker.runnableCode} get message {contextFile}")
            context = RunnableContext[self.worker.Request].model_validate_json(self.store.read(contextFile))
            print(context)
            context.startTime = datetime.now()
            context.status = RunnableStatus.RUNNING
        
            try:
                context = await self.worker.onNext(context)
            except Exception as e:
                context.status = RunnableStatus.ERROR
                context.errorMessage = str(e)

            if context.status in [RunnableStatus.ERROR, RunnableStatus.SUCCESS]:
                context.endTime = datetime.now()
                # if context.parentExecuteId and context.parentRunnableCode and context.name:
                #     parentContextFile = self.hub.getExecuteStorePath(context.parentExecuteId, "context.json")
                #     parentContext = self.hub.readExecuteContext(context.parentExecuteId, context.parentRunnableCode)
                #     if context.response is not None:
                #         parentContext.promise.result[context.name] = context.response.model_dump()
                #     else:
                #         parentContext.promise.reject[context.name] = context.errorMessage or "no error message"
                   
                #     contextStorePath = self.hub.saveExecuteContext(parentContext)
        # await self.workers[newContext.request.runnableCode].queue.put(contextStorePath)

                    # parentContext.promise.resolve.pop(context.name)
                    # if all([x is None for x in parentContext.promise.resolve.values()]):
                    #     parentContext.status = context.status
                    #     parentContext.endTime = context.endTime
                    #     parentContext.promise.result = parentContext.promise.result
                    #     self.store.save(parentContextFile, parentContext.model_dump_json())
            # elif context.status == RunnableStatus.RUNNING:
            #     todo = context.promise.resolve
            #     while todo:
            #         name, req = todo.popitem()
            #         await self.hub.executeStart(RunnableRequest(**req), context, name)

            print(context)
            self.store.save(contextFile, context.model_dump_json())
