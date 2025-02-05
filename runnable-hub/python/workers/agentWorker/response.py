
from typing import Dict, Optional, List
from runnable import RunnableResponse

class AgentResponse(RunnableResponse):
    runnableCode: str = "AGENT_WORKER"
    outputs: Dict|List|str
    errorMessage: Optional[str] = None
    success: bool
