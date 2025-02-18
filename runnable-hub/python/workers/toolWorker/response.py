
from typing import Dict, Optional, List
from runnable_hub import RunnableResponse

class ToolResponse(RunnableResponse):
    runnableCode: str = "TOOL"
    outputs: Dict|List|str
    errorMessage: Optional[str] = None
    success: bool
