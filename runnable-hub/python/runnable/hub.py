

from datetime import datetime
import asyncio
import uuid
from abc import ABC, abstractmethod
from .context import RunnableRequest, RunnableContext, RunnableStatus
from .store import RunnableFileStore
from typing import Dict
import os
import traceback

class RunnableWorker(ABC):
    runnableCode = None
    Request = None
    Response = None
    @abstractmethod
    async def onNext(self, context: RunnableContext) -> RunnableContext:
        pass


class RunnableHub():

    def __init__(self, store: RunnableFileStore):
        self.workers = {}
        self.requests = {}
        self.responses = {}
        self.store = store


    @staticmethod
    def shortExecuteId(executeId):
        return executeId.split("-")[0]

    def registerWorker(self, worker: RunnableWorker):
        self.workers[worker.runnableCode] = RunnableWorkerDispatch(
            worker=worker,
            queue=asyncio.Queue(),
            store=self.store,
            hub=self
        )
        self.requests[worker.runnableCode] = worker.Request
        self.responses[worker.runnableCode] = worker.Response

    def readExecuteContext(self, storePath:str, runnableCode: str) -> RunnableContext:
        return RunnableContext[self.requests[runnableCode], self.responses[runnableCode]].model_validate_json(
            self.store.readFile(f"{storePath}/context.json"))

    async def executeStart(self, request: RunnableRequest, parentContext: RunnableContext|None = None, name: str|None = None) -> RunnableContext:
        executeId=str(uuid.uuid4())
        if parentContext is None:
            storePath = f"execute/{executeId}"
            parentExecuteId = None
            parentRunnableCode = None
            callDepth = 0
        else:
            storePath = f"{parentContext.storePath}/{self.shortExecuteId(executeId)}"
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
        
        self.store.saveFile(f"{newContext.storePath}/context.json", newContext.model_dump_json())
        await self.workers[newContext.runnableCode].queue.put(f"{newContext.storePath}/context.json|")
        return newContext
    
    async def parentExecuteNext(self, context: RunnableContext):
        if context.parentExecuteId is None or context.parentRunnableCode is None or context.name is None:
            return
        parentStorePath = os.path.dirname(context.storePath)
        await self.workers[context.parentRunnableCode].queue.put(f"{parentStorePath}/context.json|{context.runnableCode}#{context.name}={self.shortExecuteId(context.executeId)}")

    async def executeWait(self, context: RunnableContext) -> RunnableContext:
        while context.status not in [RunnableStatus.ERROR, RunnableStatus.SUCCESS]:
            await asyncio.sleep(1)
            context = self.readExecuteContext(context.storePath, context.runnableCode)
        return context

class RunnableWorkerDispatch():
    worker: RunnableWorker
    queue: asyncio.Queue
    store: RunnableFileStore
    hub: RunnableHub

    def __init__(self, worker: RunnableWorker, queue: asyncio.Queue, store: RunnableFileStore, hub: RunnableHub):
        self.worker = worker
        self.queue = queue
        self.store = store
        self.hub = hub
        asyncio.create_task(self.run())

    async def run(self):
        while True:
            print(f"RunnableWorkerDispatch {self.worker.runnableCode} wait")
            message = await self.queue.get()
            contextFile, callbacks = message.split("|", 1)
            if callbacks != "":
                print(f"RunnableWorkerDispatch {self.worker.runnableCode} get message {contextFile} with callback {callbacks}")
            else:
                print(f"RunnableWorkerDispatch {self.worker.runnableCode} get message {contextFile}")

            context = RunnableContext[self.worker.Request, self.worker.Response].model_validate_json(
                self.store.readFile(contextFile))

            if callbacks != "":
                for callback in callbacks.split(","):
                    # runnableCode#name=executeShortId
                    runnableCodeAndName, executeShortId = callback.split('=', 1)
                    callbackRunnableCode, name = runnableCodeAndName.split('#', 1)
                    callbackContext = self.hub.readExecuteContext(f"{context.storePath}/{executeShortId}", callbackRunnableCode)
                    if callbackContext.status == RunnableStatus.SUCCESS:
                        context.promise.result[name] = callbackContext.response.model_dump() if callbackContext.response is not None else {}
                    elif callbackContext.status == RunnableStatus.ERROR:
                        context.promise.reject[name] = callbackContext.errorMessage if callbackContext.errorMessage is not None else ""
                    else:
                        print(f"context {context.storePath}/{executeShortId} not in finish status")


            if context.status == RunnableStatus.PENDING:
                context.startTime = datetime.now()
                context.status = RunnableStatus.RUNNING
            elif context.status in [RunnableStatus.ERROR, RunnableStatus.SUCCESS]:
                print(f"context {context.executeId} has been finished, status={context.status}")
                continue
        
            try:
                context = await self.worker.onNext(context)
            except Exception as e:
                context.status = RunnableStatus.ERROR
                context.errorMessage = f"Error: {str(e)}\nStack Trace:\n{traceback.format_exc()}"

            # 执行器读取完成后将此结果置空
            context.promise.result = {}

            if context.status in [RunnableStatus.ERROR, RunnableStatus.SUCCESS]:
                context.endTime = datetime.now()
                await self.hub.parentExecuteNext(context)
            elif context.status == RunnableStatus.RUNNING:
                todo = context.promise.resolve
                while todo:
                    name, req = todo.popitem()
                    print(f"get todo {name} {req}")
                    await self.hub.executeStart(self.hub.requests[req['runnableCode']](**req), context, name)

            print(context)
            self.store.saveFile(contextFile, context.model_dump_json())
