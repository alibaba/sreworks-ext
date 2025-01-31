
from runnable import RunnableResponse
from typing import Dict, List, Optional

class ApiResponse(RunnableResponse):
    runnableCode: str = "API_WORKER"
    result: str
    statusCode: int
    outputs: Optional[Dict|List|str]