from runnable_hub import RunnableRequest
from typing import Dict

class AgentRequest(RunnableRequest):
    runnableCode: str = "AGENT"
    agentCode: str
    agentVersion: str
    inputs: Dict
    prompt: str