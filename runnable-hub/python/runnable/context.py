
from pydantic import BaseModel
from typing import List, Dict, Optional, Generic, TypeVar
from datetime import datetime
from enum import Enum

T = TypeVar('T')

class RunnableStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"

class RunnableRequest(BaseModel):
    runnableCode: str

class RunnableResponse(BaseModel):
    runnableCode: str
    status: Optional[str] = None

class RunnablePromise(BaseModel):
    resolve: Dict[str, Dict] = {}
    result: Dict[str, Dict] = {}

class RunnableContext(BaseModel, Generic[T]):
    request: T
    response: Optional[RunnableResponse] = None
    promise: RunnablePromise = RunnablePromise()
    executeId: str
    startTime: datetime
    endTime: Optional[datetime] = None
    status: RunnableStatus
    errorMessage: Optional[str] = None
    callDepth: int = 0
    data: Dict = {}