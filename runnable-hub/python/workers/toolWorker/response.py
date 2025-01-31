
from typing import Dict, Optional
from runnable import RunnableResponse

class ToolResponse(RunnableResponse):
    runnableCode: str = "TOOL_WORKER"
    outputs: Dict
    errorMessage: Optional[str] = None
    success: bool
