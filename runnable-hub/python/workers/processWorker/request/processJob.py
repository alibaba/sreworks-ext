

from pydantic import BaseModel
from typing import List, Dict

class ProcessJob(BaseModel):
    jobCode: str
    runnableCode: str
    request: Dict
    dependencies: List[str]