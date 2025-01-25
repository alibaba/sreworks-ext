

from datetime import datetime
import asyncio
import uuid
from abc import ABC, abstractmethod
from .context import RunnableRequest, RunnableContext, RunnableStatus

class RunnableWorker(ABC):
    runnableCode = None
    @abstractmethod
    async def onNext(self, context: RunnableContext) -> RunnableContext:
        pass


class RunnableDispatcher():
    worker: RunnableWorker
    queue: asyncio.Queue

    def __init__(self, worker: RunnableWorker, queue: asyncio.Queue):
        self.worker = worker
        self.queue = queue
        asyncio.create_task(self.run())

    async def run(self):
        while True:
            await asyncio.sleep(1)
            print(f"RunnableDispatcher {self.worker.runnableCode} next")
            queueMessage = await self.queue.get()
            print(queueMessage)
            break
            # await self.worker.onNext(context)

class RunnableWorkerDispatch():
    worker: RunnableWorker
    queue: asyncio.Queue
    dispatcher: RunnableDispatcher
    def __init__(self, worker: RunnableWorker, queue: asyncio.Queue, dispatcher: RunnableDispatcher):
        self.worker = worker
        self.queue = queue
        self.dispatcher = dispatcher


class RunnableHub():

    def __init__(self, store):
        self.workers = {}
        self.store = store

    def registerWorker(self, worker: RunnableWorker):
        self.workers[worker.runnableCode] = RunnableWorkerDispatch(
            worker=worker,
            queue=asyncio.Queue(),
            dispatcher=RunnableDispatcher(worker, asyncio.Queue())
        )

    def executeStart(self, request: RunnableRequest) -> RunnableContext:
        result = RunnableContext(
            executeId=str(uuid.uuid4()),
            request=request, 
            response=None, 
            promiseRunnable={}, promiseResult={}, 
            startTime=datetime.now(), 
            status=RunnableStatus.PENDING)
        return result
    
    def executeCheck(self, executeId: str) -> RunnableContext:
        result = RunnableContext(request=None, response=None, promiseRunnable={}, promiseResult={}, executeId=executeId, startTime=datetime.now(), status=RunnableStatus.PENDING)
        return result
