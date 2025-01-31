
from typing import Dict
from runnable import RunnableResponse

class ToolResponse(RunnableResponse):
    runnableCode: str = "TOOL_WORKER"
    outputs: Dict
    errorMessage: str
    success: bool
