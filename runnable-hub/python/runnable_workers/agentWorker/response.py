
from typing import Dict, Optional, List
from runnable_hub import RunnableResponse

class AgentResponse(RunnableResponse):
    runnableCode: str = "AGENT"
    outputs: Dict|List|str
    errorMessage: Optional[str] = None
    success: bool
