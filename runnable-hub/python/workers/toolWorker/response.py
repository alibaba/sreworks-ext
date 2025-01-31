
from typing import Dict, Optional, List
from runnable import RunnableResponse

class ToolResponse(RunnableResponse):
    runnableCode: str = "TOOL_WORKER"
    outputs: Dict|List|str
    errorMessage: Optional[str] = None
    success: bool
