
from runnable_hub import RunnableResponse
from typing import Dict, List, Optional

class ApiResponse(RunnableResponse):
    runnableCode: str = "API"
    result: str
    statusCode: int
    outputs: Optional[Dict|List|str]