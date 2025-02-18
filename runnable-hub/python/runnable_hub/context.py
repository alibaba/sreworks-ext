
from pydantic import BaseModel
from typing import List, Dict, Optional, Generic, TypeVar
from datetime import datetime
from enum import Enum

T = TypeVar('T')
R = TypeVar("R")

class RunnableStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"

class RunnableRequest(BaseModel):
    runnableCode: str

class RunnableResponse(BaseModel):
    runnableCode: str

class RunnablePromise(BaseModel):
    resolve: Dict[str, Dict] = {}
    result: Dict[str, Dict] = {}
    reject: Dict[str, str] = {}

class RunnableContext(BaseModel, Generic[T,R]):
    request: T
    response: Optional[R] = None
    promise: RunnablePromise = RunnablePromise()
    executeId: str
    createTime: datetime
    startTime: Optional[datetime] = None
    endTime: Optional[datetime] = None
    status: RunnableStatus
    errorMessage: Optional[str] = None
    callDepth: int = 0
    data: Dict = {}
    runnableCode: str
    parentRunnableCode: Optional[str] = None
    parentExecuteId: Optional[str] = None
    name: Optional[str] = None
    storePath: str