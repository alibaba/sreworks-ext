
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
    response: Optional[RunnableResponse] = None
    promiseRunnable: Dict[str, object] = {}
    promiseResult: Dict[str, object] = {}
    executeId: str
    startTime: datetime
    endTime: Optional[datetime] = None
    status: RunnableStatus
    errorMessage: Optional[str] = None
    callDepth: int = 0