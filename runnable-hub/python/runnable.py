
from pydantic import BaseModel
from typing import List, Dict

class RunnableContext(BaseModel):
    request: object
    response: object
    promiseRunnable: Dict[str, object]
    promiseResult: Dict[str, object]
    executeId: str

class RunnableRequest(BaseModel):
    runnableCode: str

class RunnableWorker():
    runnableCode = None
    pass

class RunnableHub():
    def __init__(self):
        self.workers = {}

    def registerWorker(self, worker: RunnableWorker):
        self.workers[worker.runnableCode] = worker

    def executeStart(self, request: RunnableRequest) -> RunnableContext:
        result = RunnableContext(request=request, response=None, promiseRunnable={}, promiseResult={}, executeId="123")
        return result
    
    def executeCheck(self, executeId: str) -> RunnableContext:
        result = RunnableContext(request=None, response=None, promiseRunnable={}, promiseResult={}, executeId=executeId)
        return result