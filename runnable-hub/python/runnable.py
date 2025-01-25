
from tracemalloc import start
from pydantic import BaseModel
from typing import List, Dict, Optional, Generic, TypeVar
from datetime import datetime
from enum import Enum

T = TypeVar('T')

class RunnableStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    ERROR = "ERROR"

class RunnableRequest(BaseModel):
    runnableCode: str

class RunnableResponse(BaseModel):
    runnableCode: str

class RunnableContext(BaseModel, Generic[T]):
    request: T
    response: RunnableResponse
    promiseRunnable: Dict[str, object]
    promiseResult: Dict[str, object]
    executeId: str
    startTime: datetime
    endTime: Optional[datetime] = None
    status: RunnableStatus
    errorMessage: Optional[str] = None

class RunnableWorker():
    runnableCode = None

class RunnableHub():
    def __init__(self):
        self.workers = {}

    def registerWorker(self, worker: RunnableWorker):
        self.workers[worker.runnableCode] = worker

    def executeStart(self, request: RunnableRequest) -> RunnableContext:
        result = RunnableContext(request=request, response=None, promiseRunnable={}, promiseResult={}, executeId="123", startTime=datetime.now(), status=RunnableStatus.PENDING)
        return result
    
    def executeCheck(self, executeId: str) -> RunnableContext:
        result = RunnableContext(request=None, response=None, promiseRunnable={}, promiseResult={}, executeId=executeId, startTime=datetime.now(), status=RunnableStatus.PENDING)
        return result